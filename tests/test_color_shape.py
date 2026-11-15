import math

import torch

from lpp.color import oklab_to_srgb, srgb_to_oklab
from lpp.shape import canonicalize, conic, from_logcov, to_logcov


def test_oklab_reference_values():
    lab = srgb_to_oklab(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert torch.allclose(lab[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(lab[1], torch.zeros(3), atol=1e-6)


def test_oklab_roundtrip():
    rgb = torch.rand(5000, 3, dtype=torch.float64)
    assert torch.allclose(oklab_to_srgb(srgb_to_oklab(rgb)), rgb, atol=1e-5)


def _rand_shapes(n=4000):
    sx = torch.exp(torch.rand(n, dtype=torch.float64) * 5 - 0.7)
    sy = torch.exp(torch.rand(n, dtype=torch.float64) * 5 - 0.7)
    th = (torch.rand(n, dtype=torch.float64) * 4 - 2) * math.pi
    return sx, sy, th


def test_logcov_roundtrip():
    sx, sy, th = canonicalize(*_rand_shapes())
    sx2, sy2, th2 = from_logcov(to_logcov(sx, sy, th))
    assert torch.allclose(sx2, sx, rtol=1e-6) and torch.allclose(sy2, sy, rtol=1e-6)
    # theta is only defined for non-circular shapes
    ok = (sx / sy) > 1.01
    dth = torch.remainder(th2 - th + math.pi / 2, math.pi) - math.pi / 2
    assert dth[ok].abs().max() < 1e-5


def test_logcov_is_unique_per_shape():
    """(sx, sy, theta) and (sy, sx, theta + pi/2) are the same ellipse -> same code."""
    sx, sy, th = _rand_shapes()
    assert torch.allclose(to_logcov(sx, sy, th), to_logcov(sy, sx, th + math.pi / 2), atol=1e-9)
    assert torch.allclose(to_logcov(sx, sy, th), to_logcov(sx, sy, th + math.pi), atol=1e-9)


def test_conic_is_inverse_covariance():
    sx, sy, th = _rand_shapes(100)
    c, s = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    cov = R @ torch.diag_embed(torch.stack([sx * sx, sy * sy], -1)) @ R.transpose(-1, -2)
    A, B, C = conic(sx, sy, th).unbind(-1)
    inv = torch.stack([torch.stack([A, B], -1), torch.stack([B, C], -1)], -2)
    eye = torch.eye(2, dtype=torch.float64).expand_as(cov)
    assert torch.allclose(cov @ inv, eye, atol=1e-8)
