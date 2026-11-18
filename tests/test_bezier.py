import math

import torch

from lpp.render.bezier import Curves, composite, curve_alpha, curve_alpha_local, distance


def one(p0, p1, p2, width=2.0, rgb=(0.0, 0.0, 0.0), alpha=0.99):
    t = lambda v: torch.tensor([[v]], dtype=torch.float32)          # noqa: E731
    return Curves(p0=t(p0), p1=t(p1), p2=t(p2), width=t(width)[..., 0], rgb=t(rgb), alpha=t(alpha)[..., 0])


def test_straight_line_distance():
    c = one((8.0, 16.0), (16.0, 16.0), (24.0, 16.0))              # horizontal, y = 16
    d = distance(c, 32)[0, 0]
    assert abs(d[16, 16].item() - 0.5) < 1e-4                       # pixel centre (16.5, 16.5)
    assert abs(d[20, 16].item() - 4.5) < 1e-4
    assert abs(d[16, 2].item() - math.hypot(5.5, 0.5)) < 1e-4       # beyond the start point


def test_curve_passes_through_its_midpoint():
    c = one((4.0, 28.0), (16.0, 0.0), (28.0, 28.0))                # B(0.5) = (16, 14)
    d = distance(c, 32, segments=32)[0, 0]
    assert d[13, 15].item() < 0.75                                   # pixel centre (15.5, 13.5)


def test_hard_coverage_and_compositing():
    c = one((8.0, 16.0), (16.0, 16.0), (24.0, 16.0), width=3.0, rgb=(1.0, 0.0, 0.0))
    a = curve_alpha(c, 32)
    assert a[0, 0, 16, 16] > 0.98 and a[0, 0, 15, 16] > 0.98         # d = 0.5 and 0.5 (two rows touch y = 16)
    assert a[0, 0, 19, 16] == 0                                      # d = 3.5 > 1.5 + 0.5
    img = composite(a, c.rgb, torch.ones(1, 3, 32, 32))
    assert img[0, 1, 16, 16] < 0.02 and img[0, 0, 16, 16] > 0.98     # red over white
    assert torch.allclose(img[0, :, 2, 2], torch.ones(3))


def test_soft_edges_give_gradients():
    c = one((8.0, 16.0), (16.0, 18.0), (24.0, 16.0), width=2.0)
    for f in ("p0", "p1", "p2", "width"):
        getattr(c, f).requires_grad_(True)
    target = torch.zeros(1, 3, 32, 32)
    img = composite(curve_alpha(c, 32, soft=0.5), c.rgb, torch.ones(1, 3, 32, 32))
    ((img - target) ** 2).mean().backward()
    assert c.width.grad.abs() > 0 and c.p1.grad.abs().sum() > 0


def test_local_window_matches_full_canvas():
    torch.manual_seed(0)
    n = 6
    p0 = torch.rand(1, n, 2) * 40 + 10
    c = Curves(p0=p0, p1=p0 + torch.randn(1, n, 2) * 5, p2=p0 + torch.randn(1, n, 2) * 8,
               width=torch.rand(1, n) * 3 + 1, rgb=torch.rand(1, n, 3), alpha=torch.full((1, n), 0.9))
    for soft in (None, 0.5):
        full = curve_alpha(c, 64, soft=soft)
        local = curve_alpha_local(c, 64, window=48, soft=soft)
        if soft:   # the soft edge has an infinite tail: compare inside the windows only
            assert (full - local).abs().max() < 1e-3
        else:
            assert torch.allclose(full, local, atol=1e-5)
