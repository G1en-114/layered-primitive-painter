"""Render a mix of Gaussian (PRIMITIVE) and Bezier line (LINE) strokes, as protocol v3 composites them:
layers back to front; inside a layer, its Gaussians in sequence order, then its lines in sequence order
(the lines of a part outline it on top of its fills, and are hidden by the layers in front)."""
from __future__ import annotations

import torch

from .bezier import composite, curve_alpha_local
from .dispatch import render_group
from .reference import flatten_on_base


@torch.no_grad()
def render_strokes(attrs: torch.Tensor, layer: torch.Tensor, is_line: torch.Tensor, codec, size: int = 256) -> torch.Tensor:
    """attrs [N, A] long tokens, layer [N], is_line [N] bool, all in sequence order -> RGB [1, 3, H, W]."""
    proto, dev = codec.proto, codec.color_codes.device
    attrs, layer, is_line = attrs.to(dev), layer.to(dev), is_line.to(dev)
    img = torch.tensor(proto.base_color, device=dev).view(1, 3, 1, 1).expand(1, 3, size, size)
    tok = lambda sel: {k: attrs[sel, j].unsqueeze(0) for j, k in enumerate(proto.attr_names)}   # noqa: E731
    for k in layer.unique().sort().values.tolist():
        g = ((layer == k) & ~is_line).nonzero()[:, 0]
        ln = ((layer == k) & is_line).nonzero()[:, 0]
        if len(g):
            P, T = render_group(codec.decode(tok(g)), size, proto)
            img = P + T * img
        if len(ln):
            c = codec.decode_lines(tok(ln))
            img = composite(curve_alpha_local(c, size), c.rgb, img)
    return img


def render_gaussians_only(attrs, layer, codec, size: int = 256) -> torch.Tensor:
    """The v2 path (no lines), kept for comparisons."""
    order = torch.sort(layer, stable=True).indices
    p = codec.decode({k: attrs[order, j].unsqueeze(0).to(codec.color_codes.device) for j, k in enumerate(codec.proto.attr_names)})
    return flatten_on_base(render_group(p, size, codec.proto), codec.proto)
