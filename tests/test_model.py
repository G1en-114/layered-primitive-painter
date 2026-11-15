import random

import pytest
import torch

from lpp.codec import Codec
from lpp.data.sequences import Example, SequenceDataset, collate
from lpp.model import ModelConfig, Painter
from lpp.sample import SampleConfig, generate
from lpp.sequence import validate
from test_sequence import fake_decomposition

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def codec():
    try:
        return Codec.load()
    except FileNotFoundError:
        pytest.skip("codebooks missing")


def tiny(proto, codec, seed=0):
    torch.manual_seed(seed)
    return Painter(ModelConfig.tiny(), proto, codec).to(DEV)


def batch_of(proto, n=4, seed=0, text_dim=None):
    exs = [Example(*fake_decomposition(seed + i), image_id=i) for i in range(n)]
    text_fn = (lambda i: torch.randn(3 + i % 3, text_dim)) if text_dim else None
    ds = SequenceDataset(exs, proto, seed=seed, text_fn=text_fn)
    b = collate([ds[i] for i in range(n)])
    return {k: v.to(DEV) for k, v in b.items()}


def test_full_config_size(proto, codec):
    n = Painter(ModelConfig(), proto, codec).num_params()
    assert 100e6 < n < 160e6, n


def test_forward_losses_finite(proto, codec):
    m = tiny(proto, codec)
    out = m(batch_of(proto, text_dim=m.cfg.text_dim))
    assert set(out) == {"event", "x", "y", "shape", "color", "alpha", "loss"}
    assert all(torch.isfinite(v) for v in out.values())


def test_kv_cache_matches_full_forward(proto, codec):
    """Incremental decoding (one record at a time, text prefix with padding) must give
    the same hidden states as one full pass: the property the exported decode graph relies on."""
    m = tiny(proto, codec).eval()
    b = batch_of(proto, n=3, text_dim=m.cfg.text_dim)
    x = m.embed_records(b["kind"], b["event"], b["layer"], b["attrs"])
    xt = m.embed_text(b["text"])
    valid = torch.cat([b["text_valid"], b["valid"]], 1)
    full = m.backbone(torch.cat([xt, x], 1), valid)
    cache = m.new_cache(3, DEV)
    Lt = xt.shape[1]
    steps = [m.backbone(xt, b["text_valid"], cache)]
    for t in range(x.shape[1]):
        steps.append(m.backbone(x[:, t : t + 1], b["valid"][:, t : t + 1], cache))
    inc = torch.cat(steps, 1)
    v = valid.unsqueeze(-1)
    assert torch.allclose(full * v, inc * v, atol=1e-4), (full * v - inc * v).abs().max()
    assert cache.length == Lt + x.shape[1]


def test_overfits_tiny_dataset(proto, codec):
    m = tiny(proto, codec)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    b = batch_of(proto, n=2)
    first = None
    for _ in range(150):
        loss = m(b)["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first, (first, loss.item())


def test_generation_is_grammatical_and_bounded(proto, codec):
    m = tiny(proto, codec)
    g = torch.Generator(device=DEV).manual_seed(0)
    seqs = generate(m, batch=4, cfg=SampleConfig(max_records=120), generator=g)
    for s in seqs:
        validate(s)
        assert len(s) <= 120


def test_generation_deterministic_given_uniforms(proto, codec):
    m = tiny(proto, codec)
    fixed = lambda step, field: torch.full((2,), ((step * 7919 + hash(field)) % 1000) / 1000, device=DEV)  # noqa
    a = generate(m, batch=2, cfg=SampleConfig(max_records=60), uniforms=fixed)
    b = generate(m, batch=2, cfg=SampleConfig(max_records=60), uniforms=fixed)
    assert all(torch.equal(x.attrs, y.attrs) and torch.equal(x.event, y.event) for x, y in zip(a, b))
