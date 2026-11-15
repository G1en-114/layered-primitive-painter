"""Quality metrics for primitive sequences (training targets and generated samples).

orphan_stats: "orphan" primitives are the stray confetti in generated images:
    nearly invisible     visible mass < min_visible px, or
    isolated and small   no overlap with any other primitive of its own layer, and a
                         footprint under max_isolated_area px (a lone large background
                         blob is legitimate, a lone 5-px speck is not)
"""
from __future__ import annotations

import torch

from .protocol import Protocol, load_protocol
from .render.reference import Primitives, pixel_alpha, visible_mass


@torch.no_grad()
def orphan_stats(prims: Primitives, layer: torch.Tensor, size: int = 256, min_visible: float = 2.0,
                 cover: float = 0.05, max_isolated_area: float = 64.0,
                 proto: Protocol | None = None) -> dict[str, float]:
    """prims [1, N], layer [N] -> {"n", "invisible", "isolated", "orphan"} rates over N primitives."""
    proto = proto or load_protocol()
    n = prims.alpha.shape[1]
    if n == 0:
        return {"n": 0, "invisible": 0.0, "isolated": 0.0, "orphan": 0.0}
    scale = (proto.canvas_size / size) ** 2                    # report areas in canvas pixels
    vis = visible_mass(prims, size, proto)[0] * scale
    masks = pixel_alpha(prims, size, proto)[0] > cover          # [N, H, W] footprint
    area = masks.flatten(1).sum(1).float() * scale
    isolated = torch.zeros(n, dtype=torch.bool, device=masks.device)
    for k in layer.unique().tolist():
        idx = torch.nonzero(layer == k).flatten()
        m = masks[idx]
        others = m.sum(0, dtype=torch.int32)[None] - m.int()     # coverage by the rest of the layer
        isolated[idx] = ((others > 0) & m).flatten(1).sum(1) == 0
    invisible = vis < min_visible
    orphan = invisible | (isolated & (area < max_isolated_area))
    return {"n": n, "invisible": invisible.float().mean().item(),
            "isolated": (isolated & (area < max_isolated_area)).float().mean().item(),
            "orphan": orphan.float().mean().item()}
