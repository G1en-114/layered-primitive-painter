"""Continuous primitive parameters <-> discrete attribute tokens.

Token order per primitive = protocol "attributes":
  v1  x, y, shape (beta_index * codes_per_beta + nearest log-covariance code), color, alpha
  v2  x, y, sigma_x, sigma_y (log bins), theta (periodic bins), beta (log bins), color, alpha
Shapes are canonicalised first (sigma_x >= sigma_y, theta in [0, pi)).

`snap()` = decode(encode(p)): what a primitive looks like after the round trip.
The decomposer uses it (with a straight-through gradient) so training targets
are fitted *after* quantization rather than quantized after fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .codebook import nearest
from .color import oklab_to_srgb, srgb_to_oklab
from .protocol import REPO_ROOT, CodebookAttr, Protocol, load_protocol
from .render.reference import Primitives
from .shape import canonicalize, from_logcov, to_logcov


@dataclass
class Codec:
    """Encoder/decoder for one protocol version (attribute list read from the protocol).

    v1: x, y, shape (beta level x log-covariance codebook), color, alpha
    v2: x, y, sigma_x, sigma_y, theta, beta (ordinal bins), color, alpha
    """

    proto: Protocol
    shape_codes: torch.Tensor | None   # v1 only: [codes_per_beta, 3] log-covariance
    color_codes: torch.Tensor          # [color_size, 3] Oklab

    @classmethod
    def load(cls, proto: Protocol | None = None, root: Path = REPO_ROOT, device="cpu") -> "Codec":
        proto = proto or load_protocol()
        color_attr = proto.attr("color")
        assert isinstance(color_attr, CodebookAttr)
        color = torch.from_numpy(np.load(root / color_attr.file)).float().to(device)
        if color.shape != (color_attr.size, 3):
            raise ValueError(f"color codebook has shape {tuple(color.shape)}")
        shape = None
        if "shape" in proto.attr_names:
            shape_attr = proto.attr("shape")
            assert isinstance(shape_attr, CodebookAttr)
            shape = torch.from_numpy(np.load(root / shape_attr.file)).float().to(device)
            if shape.shape != (shape_attr.codes_per_beta, 3):
                raise ValueError(f"shape codebook has shape {tuple(shape.shape)}")
        return cls(proto, shape, color)

    def to(self, device) -> "Codec":
        shape = None if self.shape_codes is None else self.shape_codes.to(device)
        return Codec(self.proto, shape, self.color_codes.to(device))

    @property
    def attr_names(self) -> tuple[str, ...]:
        return self.proto.attr_names

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, p: Primitives) -> dict[str, torch.Tensor]:
        """Primitives [B, N] -> {attr: LongTensor [B, N]} in protocol order."""
        proto, a = self.proto, self.proto.attr
        sig = p.sigma.clamp(proto.sigma_min, proto.sigma_max)
        sx, sy, th = canonicalize(sig[..., 0], sig[..., 1], p.theta)   # sx >= sy, theta in [0, pi)
        out = {
            "x": a("x").quantize(p.xy[..., 0]),
            "y": a("y").quantize(p.xy[..., 1]),
            "color": nearest(self.color_codes, srgb_to_oklab(p.rgb.clamp(0, 1))),
            "alpha": a("alpha").quantize(p.alpha),
        }
        if proto.factorized_shape:
            out |= {"sigma_x": a("sigma_x").quantize(sx), "sigma_y": a("sigma_y").quantize(sy),
                    "theta": a("theta").quantize(th), "beta": a("beta").quantize(p.beta)}
        else:
            # beta levels are geometric: pick the nearest in log space
            betas = torch.tensor(proto.beta_levels, device=p.beta.device)
            beta_idx = (torch.log(p.beta.clamp(min=1e-3))[..., None] - torch.log(betas)).abs().argmin(-1)
            out["shape"] = beta_idx * self.shape_codes.shape[0] + nearest(self.shape_codes, to_logcov(sx, sy, th))
        return {n: out[n] for n in proto.attr_names}

    def decode(self, tok: dict[str, torch.Tensor]) -> Primitives:
        proto, a = self.proto, self.proto.attr
        dev = tok["x"].device
        if proto.factorized_shape:
            sx, sy = a("sigma_x").dequantize(tok["sigma_x"]), a("sigma_y").dequantize(tok["sigma_y"])
            theta, beta = a("theta").dequantize(tok["theta"]), a("beta").dequantize(tok["beta"])
        else:
            per_beta = self.shape_codes.shape[0]
            sx, sy, theta = from_logcov(self.shape_codes.to(dev)[tok["shape"] % per_beta])
            beta = torch.tensor(proto.beta_levels, device=dev)[tok["shape"] // per_beta]
        return Primitives(
            xy=torch.stack([a("x").dequantize(tok["x"]), a("y").dequantize(tok["y"])], dim=-1),
            sigma=torch.stack([sx, sy], dim=-1),
            theta=theta,
            beta=beta,
            rgb=oklab_to_srgb(self.color_codes.to(dev)[tok["color"]]),
            alpha=a("alpha").dequantize(tok["alpha"]),
        )

    def snap(self, p: Primitives, straight_through: bool = True) -> Primitives:
        """Quantize then dequantize. With straight_through, gradients pass to `p`
        as if snapping were the identity (used by quantization-aware fitting)."""
        q = self.decode(self.encode(p))
        if not straight_through:
            return q

        def st(cont: torch.Tensor, disc: torch.Tensor) -> torch.Tensor:
            return cont + (disc - cont).detach()

        # Canonicalisation may swap (sx, sy) and rotate theta by pi/2; compare
        # against the canonical form of p so the straight-through residual is small.
        sx, sy, th = canonicalize(p.sigma[..., 0], p.sigma[..., 1], p.theta)
        # keep theta continuous across the [0, pi) wrap: move q.theta next to th
        q_theta = th + torch.remainder(q.theta - th + torch.pi / 2, torch.pi) - torch.pi / 2
        return Primitives(
            xy=st(p.xy, q.xy),
            sigma=st(torch.stack([sx, sy], -1), q.sigma),
            theta=st(th, q_theta.detach()),
            beta=st(p.beta, q.beta),
            rgb=st(p.rgb, q.rgb),
            alpha=st(p.alpha, q.alpha),
        )
