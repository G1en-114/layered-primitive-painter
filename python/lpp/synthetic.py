"""Synthetic layered scenes with known ground truth, for tests and demos.

Each layer is a cluster of primitives around its own center (layer 0 covers the
whole canvas). The label map assigns every pixel to the front-most layer that
covers it with opacity > 0.5, like a semantic segmentation would.
"""
from __future__ import annotations

import math

import torch

from .protocol import Protocol, load_protocol
from .render.reference import Primitives, render_layers, stack_layers


def synthetic_scene(batch: int, size: int, seed: int = 0, per_layer=(6, 6, 8, 10, 10, 3),
                    proto: Protocol | None = None, device="cpu"):
    proto = proto or load_protocol()
    g = torch.Generator().manual_seed(seed)
    C = proto.canvas_size
    r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731

    xy, sigma, theta, beta, rgb, alpha, layer = [], [], [], [], [], [], []
    for k, n in enumerate(per_layer[: proto.num_layers]):
        if k == 0:
            xy.append(r(batch, n, 2) * C)
            sigma.append(C * (0.25 + 0.25 * r(batch, n, 2)))
            beta.append(torch.ones(batch, n))
        else:
            center = C * (0.25 + 0.5 * r(batch, 1, 2))
            spread = C * (0.18 - 0.02 * k)
            xy.append(center + spread * (r(batch, n, 2) - 0.5))
            sigma.append(C * (0.02 + 0.08 * r(batch, n, 2)))
            levels = torch.tensor(proto.beta_levels)
            beta.append(levels[torch.randint(len(levels), (batch, n), generator=g)])
        theta.append(r(batch, n) * math.pi)
        base = r(batch, 1, 3)  # each layer has its own palette
        rgb.append((base + 0.25 * (r(batch, n, 3) - 0.5)).clamp(0, 1))
        alpha.append(0.6 + 0.39 * r(batch, n))
        layer.append(torch.full((batch, n), k))

    prims = Primitives(
        xy=torch.cat(xy, 1), sigma=torch.cat(sigma, 1), theta=torch.cat(theta, 1),
        beta=torch.cat(beta, 1), rgb=torch.cat(rgb, 1), alpha=torch.cat(alpha, 1),
    )
    prims = Primitives(**{f: getattr(prims, f).to(device) for f in prims.__dataclass_fields__})
    layer = torch.cat(layer, 1).to(device)

    with torch.no_grad():
        groups = render_layers(prims, layer, size, proto)
        img = stack_layers(groups, proto=proto)
        label = torch.zeros(batch, size, size, dtype=torch.long, device=device)
        for k, (_, T) in enumerate(groups):
            if k > 0:
                label = torch.where((1 - T[:, 0]) > 0.5, k, label)
    return img, label, prims, layer
