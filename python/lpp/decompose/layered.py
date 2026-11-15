"""Layered decomposition: image + per-pixel layer labels -> ordered, quantized primitives.

For every layer k, back to front, primitives are added in rounds of decreasing
size and fitted by gradient descent through the differentiable renderer. Pixels
are treated by where they sit relative to layer k:

    region k        must match the image (composited over the layers behind)
    front, near     layers in front of k, close to region k: weakly pulled towards an
                    amodal pseudo-target (region k's colors extrapolated by push-pull),
                    so a layer stays whole when the layers in front are hidden
    front, far      no constraint by default (optional coverage penalty)
    behind          layers behind k: coverage is penalized, layer k must not paint over them

The background (layer 0) is special: the whole canvas counts as "near" and any
remaining transparency is penalized, so it is an opaque plate with no holes
where the subject stands.

Quantization-aware fitting: the last part of every round renders through
codec.snap() (straight-through), and the next layer is fitted on top of the
*quantized* result, so the training targets render exactly as fitted.

Ordering (the autoregressive training order): layers back to front, rounds
coarse to fine inside a layer, Morton (Z-order) inside a round.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..codec import Codec
from ..protocol import Protocol, load_protocol
from ..render.reference import Primitives, flatten_on_base, over, pixel_alpha, render_group
from .fill import dilate, morton_code, push_pull_fill

PARAM_NAMES = ("xy", "log_sigma", "theta", "log2_beta", "rgb", "alpha")


@dataclass
class DecomposeConfig:
    size: int | None = None                       # working resolution; None = canvas size
    round_sigmas: tuple[float, ...] = (48.0, 24.0, 12.0, 6.0, 3.0, 1.5)   # init sigma, canvas px
    round_fracs: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)  # share of layer budget
    steps_per_round: int = 120
    snap_frac: float = 0.3            # last fraction of each round fitted through snap()
    budget_scale: float = 2.5         # budget = cap * min(1, budget_scale * sqrt(area fraction))
    min_budget: int = 4
    stop_mse: float = 2e-4            # skip a layer's remaining rounds below this region MSE
    front_band_px: float = 12.0       # "near" band around region k, canvas px
    w_front_near: float = 0.1
    w_front_far: float = 0.0          # >0 discourages a layer sprawling far under front layers
    w_base_opaque: float = 1.0        # background layer: penalty on remaining transparency
    w_behind: float = 1.0
    prune_alpha: float = 2e-3         # drop primitives whose peak in-canvas opacity is below this
    lr: dict[str, float] = field(default_factory=lambda: dict(
        xy=0.5, log_sigma=0.03, theta=0.03, log2_beta=0.05, rgb=0.05, alpha=0.05))
    seed: int = 0


@dataclass
class Decomposition:
    """One image's result, in training order."""

    tokens: dict[str, torch.Tensor]   # attr -> [N] long
    layer: torch.Tensor               # [N] long, non-decreasing
    round: torch.Tensor               # [N] long, non-decreasing inside each layer
    psnr: float                       # quantized render vs. input image
    layer_leak: list[float]           # per layer: mean coverage over pixels of layers behind it


class LayeredDecomposer:
    def __init__(self, codec: Codec, cfg: DecomposeConfig | None = None, proto: Protocol | None = None):
        self.proto = proto or codec.proto or load_protocol()
        self.codec = codec
        self.cfg = cfg or DecomposeConfig()
        if len(self.cfg.round_sigmas) != len(self.cfg.round_fracs):
            raise ValueError("round_sigmas and round_fracs must have the same length")

    # ---- parameterization ------------------------------------------------
    def _to_prims(self, p: dict[str, torch.Tensor], active: torch.Tensor) -> Primitives:
        proto = self.proto
        log_sigma = p["log_sigma"].clamp(math.log(proto.sigma_min), math.log(proto.sigma_max))
        return Primitives(
            xy=p["xy"],
            sigma=log_sigma.exp(),
            theta=p["theta"],
            beta=2.0 ** p["log2_beta"].clamp(0.0, math.log2(max(proto.beta_levels))),
            rgb=torch.sigmoid(p["rgb"]),
            alpha=torch.sigmoid(p["alpha"]) * proto.max_alpha * active,
        )

    def _snapped(self, prims: Primitives, active: torch.Tensor, straight_through: bool) -> Primitives:
        q = self.codec.snap(prims, straight_through=straight_through)
        return q.with_alpha(q.alpha * active)  # padding slots would otherwise snap to the min alpha bin

    # ---- one batch -------------------------------------------------------
    def __call__(self, img: torch.Tensor, label: torch.Tensor) -> list[Decomposition]:
        """img [B, 3, S, S] sRGB in [0, 1]; label [B, S, S] long layer id per pixel."""
        cfg, proto = self.cfg, self.proto
        B, _, S, _ = img.shape
        size = cfg.size or proto.canvas_size
        if S != size:
            raise ValueError(f"image is {S}px, decomposer works at {size}px")
        dev = img.device
        gen = torch.Generator(device=dev).manual_seed(cfg.seed)
        px_per_canvas = size / proto.canvas_size
        label = label.unsqueeze(1)

        before = (torch.zeros_like(img), torch.ones_like(img[:, :1]))  # empty group
        out_tok, out_keep, out_layer, out_round, leaks = [], [], [], [], []

        for k, layer in enumerate(proto.layers):
            region = label == k
            behind, front = label < k, label > k
            opaque = k == 0
            near = front if opaque else front & dilate(region, round(cfg.front_band_px * px_per_canvas))
            far = front & ~near
            pseudo = push_pull_fill(img, region)
            area = region.float().mean((1, 2, 3))
            budget = torch.where(
                area > 0,
                (layer.max_primitives * (cfg.budget_scale * area.sqrt()).clamp(max=1)).round()
                .clamp(min=cfg.min_budget, max=layer.max_primitives),
                torch.zeros_like(area),
            ).long()

            p, active, rounds = self._fit_layer(img, pseudo, region, near, far, behind, before, budget, gen,
                                                opaque)

            prims = self._to_prims(p, active)
            with torch.no_grad():
                peak = pixel_alpha(prims, size, proto).amax((-1, -2))
                keep = active.bool() & (peak > cfg.prune_alpha)
                tok = self.codec.encode(prims)
                final = self._snapped(prims, keep.float(), straight_through=False)
                group = render_group(final, size, proto)
                leak = ((1 - group[1]) * behind).sum((1, 2, 3)) / behind.sum((1, 2, 3)).clamp(min=1)
                before = over(before, group)
            out_tok.append(tok)
            out_keep.append(keep)
            out_layer.append(torch.full_like(rounds, k))
            out_round.append(rounds)
            leaks.append(leak)

        with torch.no_grad():
            final_img = flatten_on_base(before, proto)
            mse = ((final_img - img) ** 2).mean((1, 2, 3))
            psnr = -10 * torch.log10(mse.clamp(min=1e-10))

        results = []
        for b in range(B):
            keep = torch.cat([kp[b] for kp in out_keep])
            tokens = {n: torch.cat([t[n][b] for t in out_tok])[keep].cpu() for n in self.codec_attrs}
            results.append(Decomposition(
                tokens=tokens,
                layer=torch.cat([l[b] for l in out_layer])[keep].cpu(),
                round=torch.cat([r[b] for r in out_round])[keep].cpu(),
                psnr=psnr[b].item(),
                layer_leak=[l[b].item() for l in leaks],
            ))
        return results

    @property
    def codec_attrs(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.proto.attributes)

    # ---- one layer -------------------------------------------------------
    def _fit_layer(self, img, pseudo, region, near, far, behind, before, budget, gen, opaque: bool):
        cfg, proto = self.cfg, self.proto
        B, _, S, _ = img.shape
        dev = img.device
        px_per_canvas = S / proto.canvas_size

        p = {n: torch.zeros(B, 0, *shape, device=dev) for n, shape in
             (("xy", (2,)), ("log_sigma", (2,)), ("theta", ()), ("log2_beta", ()), ("rgb", (3,)), ("alpha", ()))}
        active = torch.zeros(B, 0, device=dev)
        rounds = torch.zeros(B, 0, dtype=torch.long, device=dev)
        done = budget == 0

        cum = torch.tensor([0.0, *torch.tensor(cfg.round_fracs).cumsum(0).tolist()])
        cum = cum / cum[-1]
        n_per_round = [((budget.float() * cum[r + 1]).round() - (budget.float() * cum[r]).round()).long()
                       for r in range(len(cfg.round_fracs))]

        region_n = region.sum((1, 2, 3)).clamp(min=1)
        near_n, far_n = near.sum((1, 2, 3)).clamp(min=1), far.sum((1, 2, 3)).clamp(min=1)
        behind_n = behind.sum((1, 2, 3)).clamp(min=1)
        allowed = (region | near).flatten(1).float()

        def composite(prims_):
            P, T = render_group(prims_, S, proto)
            return flatten_on_base(over(before, (P, T)), proto), T

        for r, sigma0 in enumerate(cfg.round_sigmas):
            n_new = n_per_round[r] * (~done)
            n_max = int(n_new.max())
            if n_max == 0:
                continue

            # --- place new primitives where the current error is largest
            with torch.no_grad():
                comp, _ = composite(self._to_prims(p, active))
                err = ((comp - img) ** 2).sum(1, keepdim=True) * region + \
                      cfg.w_front_near * ((comp - pseudo) ** 2).sum(1, keepdim=True) * near
                ks = max(1, int(sigma0 * px_per_canvas)) | 1
                err = F.avg_pool2d(err, ks, stride=1, padding=ks // 2, count_include_pad=False)
                w = err.flatten(1) * allowed + 1e-12 * allowed
                w = torch.where(allowed.sum(1, keepdim=True) > 0, w, torch.ones_like(w))
                replace = bool((allowed.sum(1) < n_max).any())
                idx = torch.multinomial(w, n_max, replacement=replace, generator=gen)
                iy, ix = idx // S, idx % S
                jit = torch.rand(B, n_max, 2, device=dev, generator=gen)
                xy = (torch.stack([ix, iy], -1).float() + jit) / px_per_canvas
                col = pseudo.flatten(2).gather(2, idx.unsqueeze(1).expand(B, 3, n_max)).transpose(1, 2)

                slot = torch.arange(n_max, device=dev).expand(B, n_max)
                new_active = (slot < n_new.unsqueeze(1)).float()
                # Morton order inside the round; padding slots sort last
                q = (xy.clamp(0, proto.canvas_size - 1e-3)).long()
                key = torch.where(new_active.bool(), morton_code(q[..., 0], q[..., 1]),
                                  torch.full_like(q[..., 0], 1 << 40))
                order = key.argsort(dim=1)
                g2 = lambda t: t.gather(1, order.view(B, n_max, *([1] * (t.dim() - 2))).expand_as(t))  # noqa: E731
                xy, col, new_active = g2(xy), g2(col), g2(new_active)

                new = {
                    "xy": xy,
                    "log_sigma": torch.full((B, n_max, 2), math.log(sigma0), device=dev)
                                 + 0.1 * torch.randn(B, n_max, 2, device=dev, generator=gen),
                    "theta": math.pi * torch.rand(B, n_max, device=dev, generator=gen),
                    "log2_beta": torch.ones(B, n_max, device=dev),
                    "rgb": torch.logit(col.clamp(0.02, 0.98)),
                    "alpha": torch.full((B, n_max), math.log(0.7 / (proto.max_alpha - 0.7)), device=dev),
                }
                p = {n: torch.cat([p[n], new[n]], 1) for n in PARAM_NAMES}
                active = torch.cat([active, new_active], 1)
                rounds = torch.cat([rounds, torch.full((B, n_max), r, dtype=torch.long, device=dev)], 1)

            # --- fit every primitive of this layer
            leaves = {n: t.detach().requires_grad_(True) for n, t in p.items()}
            opt = torch.optim.Adam([{"params": [leaves[n]], "lr": cfg.lr[n]} for n in PARAM_NAMES])
            snap_from = int(cfg.steps_per_round * (1 - cfg.snap_frac))
            for step in range(cfg.steps_per_round):
                prims = self._to_prims(leaves, active)
                if step >= snap_from:
                    prims = self._snapped(prims, active, straight_through=True)
                comp, T = composite(prims)
                cover = 1 - T
                loss = (
                    (((comp - img) ** 2).sum(1, keepdim=True) * region).sum((1, 2, 3)) / region_n
                    + cfg.w_front_near * (((comp - pseudo) ** 2).sum(1, keepdim=True) * near).sum((1, 2, 3)) / near_n
                    + cfg.w_front_far * (cover * far).sum((1, 2, 3)) / far_n
                    + cfg.w_behind * (cover * behind).sum((1, 2, 3)) / behind_n
                    + (cfg.w_base_opaque * T.mean((1, 2, 3)) if opaque else 0.0)
                ).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            p = {n: t.detach() for n, t in leaves.items()}

            with torch.no_grad():
                comp, _ = composite(self._snapped(self._to_prims(p, active), active, straight_through=False))
                mse = (((comp - img) ** 2).mean(1, keepdim=True) * region).sum((1, 2, 3)) / region_n
                done = done | (mse < cfg.stop_mse)

        return p, active, rounds
