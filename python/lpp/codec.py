"""Continuous primitive parameters <-> discrete attribute tokens.

Token order per primitive (protocol "attributes"):  x, y, shape, color, alpha
  x, y   uniform bins over the canvas
  shape  beta_index * codes_per_beta + nearest log-covariance code
  color  nearest Oklab code
  alpha  uniform bins

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
from .protocol import REPO_ROOT, CodebookAttr, Protocol, UniformAttr, load_protocol
from .render.reference import Primitives
from .shape import canonicalize, from_logcov, to_logcov

ATTR_ORDER = ("x", "y", "shape", "color", "alpha")


@dataclass
class Codec:
    proto: Protocol
    shape_codes: torch.Tensor  # [codes_per_beta, 3] log-covariance
    color_codes: torch.Tensor  # [color_size, 3] Oklab

    @classmethod
    def load(cls, proto: Protocol | None = None, root: Path = REPO_ROOT, device="cpu") -> "Codec":
        proto = proto or load_protocol()
        shape_attr, color_attr = proto.attr("shape"), proto.attr("color")
        assert isinstance(shape_attr, CodebookAttr) and isinstance(color_attr, CodebookAttr)
        shape = torch.from_numpy(np.load(root / shape_attr.file)).float().to(device)
        color = torch.from_numpy(np.load(root / color_attr.file)).float().to(device)
        if shape.shape != (shape_attr.codes_per_beta, 3):
            raise ValueError(f"shape codebook has shape {tuple(shape.shape)}")
        if color.shape != (color_attr.size, 3):
            raise ValueError(f"color codebook has shape {tuple(color.shape)}")
        return cls(proto, shape, color)

    def to(self, device) -> "Codec":
        return Codec(self.proto, self.shape_codes.to(device), self.color_codes.to(device))

    # ------------------------------------------------------------------
    def _uniform(self, name: str) -> UniformAttr:
        a = self.proto.attr(name)
        assert isinstance(a, UniformAttr)
        return a

    @torch.no_grad()
    def encode(self, p: Primitives) -> dict[str, torch.Tensor]:
        """Primitives [B, N] -> {attr: LongTensor [B, N]}."""
        proto = self.proto
        sig = p.sigma.clamp(proto.sigma_min, proto.sigma_max)
        logcov = to_logcov(*canonicalize(sig[..., 0], sig[..., 1], p.theta))
        betas = torch.tensor(proto.beta_levels, device=p.beta.device)
        beta_idx = (p.beta[..., None] - betas).abs().argmin(-1)
        per_beta = self.shape_codes.shape[0]
        return {
            "x": self._uniform("x").quantize(p.xy[..., 0]),
            "y": self._uniform("y").quantize(p.xy[..., 1]),
            "shape": beta_idx * per_beta + nearest(self.shape_codes, logcov),
            "color": nearest(self.color_codes, srgb_to_oklab(p.rgb.clamp(0, 1))),
            "alpha": self._uniform("alpha").quantize(p.alpha),
        }

    def decode(self, tok: dict[str, torch.Tensor]) -> Primitives:
        proto = self.proto
        per_beta = self.shape_codes.shape[0]
        shape = tok["shape"]
        sx, sy, theta = from_logcov(self.shape_codes.to(shape.device)[shape % per_beta])
        betas = torch.tensor(proto.beta_levels, device=shape.device)
        return Primitives(
            xy=torch.stack([self._uniform("x").dequantize(tok["x"]),
                            self._uniform("y").dequantize(tok["y"])], dim=-1),
            sigma=torch.stack([sx, sy], dim=-1),
            theta=theta,
            beta=betas[shape // per_beta],
            rgb=oklab_to_srgb(self.color_codes.to(shape.device)[tok["color"]]),
            alpha=self._uniform("alpha").dequantize(tok["alpha"]),
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
            beta=q.beta,
            rgb=st(p.rgb, q.rgb),
            alpha=st(p.alpha, q.alpha),
        )
