import torch

from lpp.metrics import orphan_stats
from lpp.render import Primitives


def prims(rows):
    t = lambda i: torch.tensor([[r[i] for r in rows]], dtype=torch.float32)  # noqa: E731
    return Primitives(xy=torch.stack([t(0), t(1)], -1), sigma=torch.stack([t(2), t(2)], -1),
                      theta=torch.zeros(1, len(rows)), beta=torch.full((1, len(rows)), 2.0),
                      rgb=torch.rand(1, len(rows), 3), alpha=t(3))


def test_orphans():
    p = prims([
        (128, 128, 60, 0.9),   # big blob (layer 0): alone but large -> fine
        (100, 100, 20, 0.9),   # two overlapping shapes in layer 1 -> fine
        (115, 105, 20, 0.9),
        (20, 230, 1.0, 0.9),   # tiny speck far from the rest of layer 1 -> orphan (isolated, small)
        (100, 100, 6, 0.9),    # layer 1 shape painted, then fully covered by the next two
        (100, 100, 25, 0.99),  # in layer 2: covers it completely ...
        (100, 100, 25, 0.99),  # ... twice -> the layer-1 shape is invisible
    ])
    layer = torch.tensor([0, 1, 1, 1, 1, 2, 2])
    s = orphan_stats(p, layer)
    assert s["n"] == 7
    assert abs(s["isolated"] - 1 / 7) < 1e-6        # only the speck
    assert s["invisible"] >= 1 / 7                   # at least the covered shape
    assert abs(s["orphan"] - (s["invisible"] + s["isolated"])) < 1e-6 or s["orphan"] <= s["invisible"] + s["isolated"]


def test_empty():
    p = prims([])
    assert orphan_stats(p, torch.zeros(0, dtype=torch.long))["orphan"] == 0.0
