"""Clipped blend modes for shading strokes (protocol v3 candidate, docs/line_strokes.md "Shadows").

A layer is a premultiplied group (P [B,3,H,W], T [B,1,H,W]) with coverage A = 1 - T. Shading strokes act on the
layer's own colour C = P / A, clipped to the layer like a painter's clipping mask:

    multiply (shadow):    C' = C * (1 - a + a c)          only darkens; c may be tinted (cool or purple shadows)
    screen   (highlight): C' = C + a c (1 - C)            only lightens

In premultiplied form, P' = P (1 - a + a c) and P' = P + a c (A - P). Where the layer is empty (A = 0, so P = 0)
both leave P = 0: the clipping needs no mask. T is unchanged, because shading recolours the layer and does not
make it more or less opaque. Strokes apply in order. Multiply strokes commute; screen strokes do not quite.
"""
from __future__ import annotations

import torch


def multiply(P: torch.Tensor, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """P [B,3,H,W]; alpha [B,N,H,W] per-pixel stroke opacity; rgb [B,N,3] -> P'."""
    f = 1 - alpha[:, :, None] * (1 - rgb[..., None, None])       # [B, N, 3, H, W]
    return P * f.prod(1)


def screen(P: torch.Tensor, T: torch.Tensor, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    A = 1 - T
    for i in range(alpha.shape[1]):
        ac = alpha[:, i, None] * rgb[:, i, :, None, None]           # [B, 3, H, W]
        P = P + ac * (A - P)
    return P
