import math

import torch

from conftest import random_prims
from lpp.render import Primitives, pixel_alpha, render, render_layers, stack_layers, to_straight_rgba
from lpp.render.reference import flatten_on_base, over


def one(x, y, sx, sy, theta=0.0, beta=1.0, rgb=(1.0, 0.0, 0.0), alpha=0.99, dtype=torch.float32):
    t = lambda v: torch.tensor(v, dtype=dtype)  # noqa: E731
    return Primitives(
        xy=t([[[x, y]]]), sigma=t([[[sx, sy]]]), theta=t([[theta]]),
        beta=t([[beta]]), rgb=t([[list(rgb)]]), alpha=t([[alpha]]),
    )


def test_empty_canvas_is_base_color(proto):
    p = random_prims(1, 0)
    img = render(p, size=16)
    assert torch.allclose(img, torch.tensor(proto.base_color).view(1, 3, 1, 1).expand(1, 3, 16, 16))


def test_peak_and_falloff(proto):
    p = one(128.0, 128.0, 10.0, 10.0, alpha=0.8)
    a = pixel_alpha(p, 256)[0, 0]
    g_cut = math.exp(-0.5 * 9)
    # pixel centers sit at k + 0.5, so the peak is half a pixel off-center
    g = math.exp(-0.5 * (0.5**2 + 0.5**2) / 100)
    assert abs(a[127, 127].item() - 0.8 * (g - g_cut) / (1 - g_cut)) < 1e-5
    # exactly zero beyond the cutoff radius (3 sigma = 30 px), and continuous towards it
    assert a[128, 128 + 31].item() == 0.0
    assert 0.0 < a[128, 128 + 29].item() < 0.002


def test_beta_makes_hard_edges():
    soft = pixel_alpha(one(128.0, 128.0, 40.0, 40.0, beta=1.0), 256)[0, 0, 128]
    hard = pixel_alpha(one(128.0, 128.0, 40.0, 40.0, beta=8.0), 256)[0, 0, 128]
    # inside the 1-sigma ellipse a hard shape is ~opaque, outside ~transparent
    assert hard[128 + 30] > 0.95 and hard[128 + 50] < 0.01
    assert 0.2 < soft[128 + 30] < 0.9


def test_rotation():
    p = one(128.0, 128.0, 40.0, 4.0, theta=math.pi / 2, beta=8.0)
    a = pixel_alpha(p, 256)[0, 0]
    assert a[128 + 30, 128] > 0.9   # long axis now vertical
    assert a[128, 128 + 30] < 0.01


def test_order_matters():
    red = one(128.0, 128.0, 30.0, 30.0, beta=8.0, rgb=(1, 0, 0))
    blue = one(128.0, 128.0, 30.0, 30.0, beta=8.0, rgb=(0, 0, 1))
    cat = lambda a, b: Primitives(**{k: torch.cat([getattr(a, k), getattr(b, k)], 1)  # noqa: E731
                                      for k in a.__dataclass_fields__})
    rb = render(cat(red, blue))[0, :, 128, 128]
    br = render(cat(blue, red))[0, :, 128, 128]
    assert rb[2] > 0.95 and br[0] > 0.95  # the later primitive is on top


def test_chunked_equals_single_pass(prims):
    full = render(prims, size=64)
    for chunk in (1, 7, 16):
        assert torch.allclose(render(prims, size=64, chunk=chunk), full, atol=1e-5)


def test_layers_stack_to_full_render(proto, prims):
    """Exact layered export: per-layer RGBA stacked back to front == the plain render."""
    g = torch.Generator().manual_seed(3)
    layer_id = torch.randint(proto.num_layers, prims.alpha.shape, generator=g).sort(-1).values
    layers = render_layers(prims, layer_id, size=64)
    assert torch.allclose(stack_layers(layers), render(prims, size=64), atol=1e-5)


def test_layer_order_not_sequence_order(proto, prims):
    """REDRAW_k: a layer generated last still composites at its own depth."""
    g = torch.Generator().manual_seed(4)
    layer_id = torch.randint(proto.num_layers, prims.alpha.shape, generator=g)  # interleaved order
    layers = render_layers(prims, layer_id, size=64)
    order = layer_id[0].argsort(stable=True)
    sorted_prims = Primitives(**{k: getattr(prims, k)[:1, order] for k in prims.__dataclass_fields__})
    assert torch.allclose(stack_layers(layers)[:1], render(sorted_prims, size=64), atol=1e-5)


def test_straight_rgba_roundtrip(prims):
    from lpp.render import render_group

    P, T = render_group(prims, 64)
    rgba = to_straight_rgba((P, T))
    rebuilt = (rgba[:, :3] * rgba[:, 3:], 1 - rgba[:, 3:])
    assert torch.allclose(flatten_on_base(rebuilt), flatten_on_base((P, T)), atol=1e-5)


def test_over_is_associative(prims):
    from lpp.render import render_group

    a, b, c = (render_group(prims[s], 32) for s in (slice(0, 10), slice(10, 25), slice(25, 40)))
    left, right = over(over(a, b), c), over(a, over(b, c))
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(left, right))


def test_gradients_finite_and_correct():
    p = random_prims(1, 6, seed=5, dtype=torch.float64)
    leaves = [p.xy, p.sigma, p.theta, p.rgb, p.alpha]
    for t in leaves:
        t.requires_grad_(True)

    def f(xy, sigma, theta, rgb, alpha):
        q = Primitives(xy=xy, sigma=sigma, theta=theta, beta=p.beta, rgb=rgb, alpha=alpha * 0.9)
        return render(q, size=12)

    assert torch.autograd.gradcheck(f, leaves, eps=1e-6, atol=1e-5)
    img = render(Primitives(xy=p.xy, sigma=p.sigma, theta=p.theta, beta=p.beta, rgb=p.rgb, alpha=p.alpha), 32)
    img.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in leaves)


def test_resolution_independent():
    p = one(100.0, 60.0, 20.0, 8.0, theta=0.3, beta=2.0)
    lo = render(p, size=64)
    hi = torch.nn.functional.avg_pool2d(render(p, size=256), 4)
    assert (lo - hi).abs().max() < 0.05
