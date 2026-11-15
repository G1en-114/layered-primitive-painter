"""Triton tile rasterizer vs the dense reference: forward values and every gradient."""

import pytest
import torch

from conftest import random_prims
from lpp.render import Primitives
from lpp.render.reference import render_group as ref_group

from lpp.render.dispatch import triton_available

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and triton_available()),
                                reason="needs a GPU with Triton")

FIELDS = ("xy", "sigma", "theta", "beta", "rgb", "alpha")


def tri_group(p, size):
    from lpp.render.triton_raster import render_group_triton
    return render_group_triton(p, size)


def make(B, N, seed, small=False):
    p = random_prims(B, N, seed=seed, device="cuda")
    if small:  # many tiny primitives: exercises tile edges and sub-pixel shapes
        p.sigma = p.sigma * 0.1
    return p


@pytest.mark.parametrize("B,N,size,small", [
    (1, 1, 16, False), (2, 37, 17, False), (3, 64, 64, True), (2, 150, 100, False),
    (2, 300, 256, False), (1, 500, 256, True), (2, 0, 32, False),
])
def test_forward_matches_reference(B, N, size, small):
    p = make(B, N, seed=size + N, small=small)
    P, T = tri_group(p, size)
    Pr, Tr = ref_group(p, size)
    assert (P - Pr).abs().max() < 2e-5 and (T - Tr).abs().max() < 2e-5


def test_padding_primitives_are_skipped():
    p = make(2, 50, seed=3)
    alpha = p.alpha.clone()
    alpha[:, ::3] = 0
    q = p.with_alpha(alpha)
    P, T = tri_group(q, 64)
    Pr, Tr = ref_group(q, 64)
    assert (P - Pr).abs().max() < 2e-5 and (T - Tr).abs().max() < 2e-5


def test_clamped_alpha_and_hard_edges():
    p = make(1, 40, seed=4)
    p.alpha.fill_(0.99)
    p.beta.fill_(8.0)
    P, T = tri_group(p, 64)
    Pr, Tr = ref_group(p, 64)
    assert (P - Pr).abs().max() < 2e-5 and (T - Tr).abs().max() < 2e-5


def grads(fn, p, size, gP, gT):
    leaves = {f: getattr(p, f).detach().clone().requires_grad_(True) for f in FIELDS}
    P, T = fn(Primitives(**leaves), size)
    ((P * gP).sum() + (T * gT).sum()).backward()
    return {f: leaves[f].grad for f in FIELDS}


@pytest.mark.parametrize("B,N,size,small", [(1, 5, 16, False), (2, 60, 48, False), (2, 200, 128, True),
                                            (1, 120, 256, False)])
def test_backward_matches_reference(B, N, size, small):
    p = make(B, N, seed=7 * N + size, small=small)
    g = torch.Generator(device="cuda").manual_seed(0)
    gP = torch.randn(B, 3, size, size, device="cuda", generator=g)
    gT = torch.randn(B, 1, size, size, device="cuda", generator=g)
    got = grads(tri_group, p, size, gP, gT)
    want = grads(ref_group, p, size, gP, gT)
    for f in FIELDS:
        err = (got[f] - want[f]).abs().max().item()
        scale = want[f].abs().max().item()
        assert err <= 2e-3 * scale + 1e-4, f"{f}: max err {err:.3g} vs scale {scale:.3g}"


def test_decomposer_style_loss_gradients():
    """The decomposer's actual use: composite over a fixed background, MSE to a target."""
    p = make(2, 80, seed=11)
    target = torch.rand(2, 3, 64, 64, device="cuda")

    def loss(fn):
        leaves = {f: getattr(p, f).detach().clone().requires_grad_(True) for f in FIELDS}
        P, T = fn(Primitives(**leaves), 64)
        ((P + T * 0.5 - target) ** 2).mean().backward()
        return {f: leaves[f].grad for f in FIELDS}

    got, want = loss(tri_group), loss(ref_group)
    for f in FIELDS:
        assert torch.allclose(got[f], want[f], rtol=2e-3, atol=1e-6 + 2e-3 * want[f].abs().max().item()), f
