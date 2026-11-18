"""Triton line rasterizer == the dense reference (lpp.render.bezier), forward and gradients."""
import pytest
import torch

from lpp.render.bezier import Curves, curve_alpha

triton_lines = pytest.importorskip("lpp.render.triton_lines")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref_group(c, size, soft):
    a = curve_alpha(c, size, soft=soft)                                   # [B, N, H, W]
    log_t = torch.log1p(-a)
    after = log_t.sum(1, keepdim=True) - torch.cumsum(log_t, 1)
    P = torch.einsum("bnhw,bnc->bchw", a * after.exp(), c.rgb)
    return P, log_t.sum(1, keepdim=True).exp()


def curves(B=2, N=24, seed=0, dev="cuda"):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)                           # noqa: E731
    p0 = r(B, N, 2) * 56 + 4
    return Curves(p0=p0.to(dev), p1=(p0 + (r(B, N, 2) - 0.5) * 14).to(dev), p2=(p0 + (r(B, N, 2) - 0.5) * 20).to(dev),
                  width=(r(B, N) * 3 + 1).to(dev), rgb=r(B, N, 3).to(dev), alpha=(0.3 + 0.6 * r(B, N)).to(dev))


@pytest.mark.parametrize("soft", [None, 0.5])
def test_forward_matches_reference(soft):
    c = curves()
    P, T = triton_lines.render_lines_triton(c, 64, canvas=64, soft=soft)
    Pr, Tr = ref_group(c, 64, soft)
    tol = 1e-4 if soft is None else 2e-3        # soft: the kernel's tiles stop at the edge's 8-tau reach
    assert (P - Pr).abs().max() < tol and (T - Tr).abs().max() < tol


def test_gradients_match_reference():
    c = curves(N=16, seed=1)
    leaves = {}
    for name in ("p0", "p1", "p2", "width", "rgb", "alpha"):
        leaves[name] = getattr(c, name).clone().requires_grad_(True)
    cc = Curves(**leaves)
    w = torch.randn(2, 4, 64, 64, device="cuda")
    P, T = triton_lines.render_lines_triton(cc, 64, canvas=64, soft=0.5)
    (P * w[:, :3]).sum().add((T * w[:, 3:]).sum()).backward()
    got = {k: v.grad.clone() for k, v in leaves.items()}
    for v in leaves.values():
        v.grad = None
    Pr, Tr = ref_group(cc, 64, 0.5)
    (Pr * w[:, :3]).sum().add((Tr * w[:, 3:]).sum()).backward()
    for k, v in leaves.items():
        ref = v.grad
        err = (got[k] - ref).abs().max() / ref.abs().max().clamp(min=1e-6)
        assert err < 2e-2, (k, float(err))
