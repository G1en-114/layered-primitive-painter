import pytest
import torch

from conftest import random_prims
from lpp.codec import ATTR_ORDER, Codec
from lpp.render import render


@pytest.fixture(scope="module")
def codec():
    try:
        return Codec.load()
    except FileNotFoundError:
        pytest.skip("run tools/build_prior_codebooks.py first")


def test_tokens_in_vocab(codec, proto):
    tok = codec.encode(random_prims(2, 300, seed=7))
    assert set(tok) == set(ATTR_ORDER)
    for name, t in tok.items():
        assert t.dtype == torch.long
        assert 0 <= t.min() and t.max() < proto.attr_vocab_sizes[name]


def test_decode_encode_is_idempotent(codec):
    """Snapped primitives must re-encode to the same tokens (no drift on re-serialisation)."""
    tok = codec.encode(random_prims(1, 500, seed=8))
    tok2 = codec.encode(codec.decode(tok))
    for name in ATTR_ORDER:
        assert torch.equal(tok[name], tok2[name]), name


def test_beta_level_survives(codec):
    p = random_prims(1, 200, seed=9)
    assert torch.equal(codec.decode(codec.encode(p)).beta, p.beta)


def test_snap_straight_through(codec):
    p = random_prims(1, 30, seed=10)
    for t in (p.xy, p.sigma, p.rgb, p.alpha, p.theta):
        t.requires_grad_(True)
    q = codec.snap(p)
    hard = codec.snap(p, straight_through=False)
    # forward value is the quantized one ...
    assert torch.allclose(render(q, 64), render(hard, 64), atol=1e-4)
    # ... while gradients reach the continuous parameters
    render(q, 64).mean().backward()
    for t in (p.xy, p.sigma, p.rgb, p.alpha, p.theta):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
