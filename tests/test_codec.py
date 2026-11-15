import pytest
import torch

from conftest import random_prims
from lpp.codec import Codec
from lpp.protocol import REPO_ROOT, load_protocol
from lpp.render import render


@pytest.fixture(scope="module", params=["v2", "v1"])
def codec(request):
    proto = load_protocol() if request.param == "v2" else load_protocol(REPO_ROOT / "protocols" / "v1.json")
    try:
        return Codec.load(proto=proto)
    except FileNotFoundError:
        pytest.skip("run tools/build_prior_codebooks.py first")


def test_tokens_in_vocab(codec):
    tok = codec.encode(random_prims(2, 300, seed=7))
    assert tuple(tok) == codec.attr_names
    for name, t in tok.items():
        assert t.dtype == torch.long
        assert 0 <= t.min() and t.max() < codec.proto.attr_vocab_sizes[name]


def test_decode_encode_is_idempotent(codec):
    """Snapped primitives must re-encode to the same tokens (no drift on re-serialisation)."""
    tok = codec.encode(random_prims(1, 500, seed=8))
    tok2 = codec.encode(codec.decode(tok))
    for name in codec.attr_names:
        assert torch.equal(tok[name], tok2[name]), name


def test_beta_level_survives(codec):
    p = random_prims(1, 200, seed=9)
    levels = torch.tensor(codec.proto.beta_levels)
    p.beta = levels[torch.randint(len(levels), p.beta.shape, generator=torch.Generator().manual_seed(1))]
    assert torch.allclose(codec.decode(codec.encode(p)).beta, p.beta, rtol=1e-5)


def test_v2_factorized_error_bounds():
    """v2 ordinal bins: position within half a bin, scales within half a log step,
    angle within half a bin around the circle (shape compared in canonical form)."""
    import math
    from lpp.shape import canonicalize
    proto = load_protocol()
    c = Codec.load(proto=proto)
    p = random_prims(1, 2000, seed=12)
    q = c.decode(c.encode(p))
    assert (q.xy - p.xy).abs().max() <= 256 / 511 / 2 + 1e-4
    sx, sy, th = canonicalize(p.sigma[..., 0], p.sigma[..., 1], p.theta)
    log_step = math.log(128 / 0.5) / 127
    assert (torch.log(q.sigma[..., 0]) - torch.log(sx)).abs().max() <= log_step / 2 + 1e-4
    assert (torch.log(q.sigma[..., 1]) - torch.log(sy)).abs().max() <= log_step / 2 + 1e-4
    dth = torch.remainder(q.theta - th + math.pi / 2, math.pi) - math.pi / 2
    near_round = sx / sy < 1.02            # angle undefined for near-circles
    assert dth[~near_round].abs().max() <= math.pi / 128 / 2 + 1e-4


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
