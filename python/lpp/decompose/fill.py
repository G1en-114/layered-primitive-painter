"""Mask-aware image utilities for the decomposer."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def push_pull_fill(img: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fill invalid pixels from valid ones with a masked image pyramid ("push-pull").

    img [B, C, H, W], valid [B, 1, H, W] in {0, 1}. Valid pixels are returned
    unchanged; invalid ones get a smooth extrapolation of the nearby valid colors.
    Used to make an amodal pseudo-target: what a layer probably looks like
    underneath the layers in front of it.
    """
    w = valid.to(img.dtype)
    if w.shape[-1] == 1 or w.shape[-2] == 1:
        mean = (img * w).sum((-1, -2), keepdim=True) / w.sum((-1, -2), keepdim=True).clamp(min=1e-8)
        return torch.where(w > 0, img, mean.expand_as(img))
    c = img * w
    c_lo = F.avg_pool2d(c, 2, ceil_mode=True)
    w_lo = F.avg_pool2d(w, 2, ceil_mode=True)
    filled_lo = push_pull_fill(c_lo / w_lo.clamp(min=1e-8), (w_lo > 0).to(img.dtype))
    up = F.interpolate(filled_lo, size=img.shape[-2:], mode="bilinear", align_corners=False)
    return torch.where(w > 0, img, up)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of a [B, 1, H, W] mask with a square of half-width `radius`."""
    if radius <= 0:
        return mask
    return F.max_pool2d(mask.float(), 2 * radius + 1, stride=1, padding=radius) > 0


def morton_code(ix: torch.Tensor, iy: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Interleave the bits of integer x and y (Z-order curve)."""
    code = torch.zeros_like(ix)
    for b in range(bits):
        code |= ((ix >> b) & 1) << (2 * b)
        code |= ((iy >> b) & 1) << (2 * b + 1)
    return code
