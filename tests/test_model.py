
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
    assert set(out) == {"event", "loss", *proto.attr_names}
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


def test_guided_generation(proto, codec):
    """CFG doubles the rows internally; sequences stay grammatical, and scale 1 is exactly
    the plain text-conditioned path."""
    m = tiny(proto, codec)
    text = torch.randn(2, 5, m.cfg.text_dim, device=DEV)
    valid = torch.ones(2, 5, dtype=torch.bool, device=DEV)
    fixed = lambda step, field: torch.full((2,), ((step * 7919 + len(field)) % 997) / 997, device=DEV)  # noqa
    for scale in (1.0, 3.0):
        for s in generate(m, batch=2, cfg=SampleConfig(max_records=80, cfg_scale=scale), text=text,
                          text_valid=valid, uniforms=fixed):
            validate(s)
    a = generate(m, batch=2, cfg=SampleConfig(max_records=60, cfg_scale=1.0), text=text, text_valid=valid,
                 uniforms=fixed)
    b = generate(m, batch=2, cfg=SampleConfig(max_records=60), text=text, text_valid=valid, uniforms=fixed)
    assert all(torch.equal(x.attrs, y.attrs) for x, y in zip(a, b))


def test_caption_strings_collate(proto):
    import random
    from lpp.text import CaptionStore
    store = CaptionStore()
    store.attrs = {i: {"Male": i % 2 == 0, "Smiling": True, "Blond_Hair": i == 1} for i in range(4)}
    exs = [Example(*fake_decomposition(i), image_id=i) for i in range(4)]
    ds = SequenceDataset(exs, proto, captions=store, p_text_drop=0.0)
    b = collate([ds[i] for i in range(4)])
    assert len(b["captions"]) == 4 and all(isinstance(c, str) for c in b["captions"])
    rng = random.Random(0)
    picks = [store.pick(1, rng, p_drop=0.25) for _ in range(2000)]
    assert 0.2 < sum(p is None for p in picks) / 2000 < 0.3
    assert "金发" in next(p for p in picks if p) and "女性" in next(p for p in picks if p)
