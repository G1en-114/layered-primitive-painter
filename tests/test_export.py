"""Export wrappers equal the training/decoding code paths they re-express."""
import pytest
import torch

from lpp.codec import Codec
from lpp.export import Chunk, DecodeHeads
from lpp.model import ModelConfig, Painter

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    codec = Codec.load(device=DEV)
    cfg = ModelConfig.tiny()
    cfg.canvas = True
    m = Painter(cfg, codec.proto, codec).to(DEV).eval()
    with torch.no_grad():                          # open the canvas gates so the cross path matters
        for c in m.cross.values():
            c.gate.fill_(0.5)
    return m


def _empty_past(m, B, P):
    c = m.cfg
    shape = (c.n_layers, B, c.n_kv_heads, P, c.d_model // c.n_heads)
    return torch.zeros(shape, device=DEV), torch.zeros(shape, device=DEV), torch.zeros(B, P, dtype=torch.bool, device=DEV)


@torch.no_grad()
def test_chunk_matches_cached_backbone(model):
    m, B, d = model, 3, model.cfg.d_model
    torch.manual_seed(1)
    x0 = torch.randn(B, 5, d, device=DEV)                  # a prefix: some text slots invalid
    v0 = torch.ones(B, 5, dtype=torch.bool, device=DEV)
    v0[1, :2] = False
    tok = torch.randn(B, m.canvas_enc.n_tokens, d, device=DEV)
    draw0 = torch.zeros(B, 5, dtype=torch.bool, device=DEV)
    draw0[:, -1] = True
    cache = m.new_cache(B, DEV, max_len=16)
    ref0 = m.backbone(x0, v0, cache, canvas_ctx=("current", tok, draw0))
    pk, pv, pm = _empty_past(m, B, 4)                      # P > 0 but fully masked == empty
    h0, k0, v0n = Chunk(m)(x0, v0, pk, pv, pm, torch.zeros(B, dtype=torch.long, device=DEV), tok, draw0)
    assert torch.allclose(h0, ref0, atol=1e-5)

    # one decoding step on top of the runtime-managed cache
    x1 = torch.randn(B, 1, d, device=DEV)
    live = torch.tensor([True, True, False], device=DEV)
    draw1 = torch.ones(B, 1, dtype=torch.bool, device=DEV)
    ref1 = m.backbone(x1, live[:, None], cache, canvas_ctx=("current", tok, draw1))
    start = v0.long().sum(1)                               # next RoPE position = valid prefix length
    h1, _, _ = Chunk(m)(x1, live[:, None], k0, v0n, v0, start, tok, draw1)
    assert torch.allclose(h1, ref1, atol=1e-5)


@torch.no_grad()
def test_decode_heads_match_truncated_decoder(model):
    m, B = model, 4
    torch.manual_seed(2)
    h = torch.randn(B, m.cfg.d_model, device=DEV)
    event = torch.randint(0, m.n_events, (B,), device=DEV)
    attrs = torch.stack([torch.randint(0, n, (B,), device=DEV) for n in m.proto.attr_vocab_sizes.values()], 1)
    outs = DecodeHeads(m)(h, event, attrs)
    assert torch.allclose(outs[0], m.heads["event"](m.decoder(h, event, attrs, steps=1)[:, 0]), atol=1e-5)
    for i, name in enumerate(m.attr_names):
        ref = m.heads[name](m.decoder(h, event, attrs, steps=2 + i)[:, -1])
        assert torch.allclose(outs[1 + i], ref, atol=1e-5), name


@torch.no_grad()
def test_exportable_copy_is_numerically_identical(model):
    from lpp.export import exportable, ManualRMSNorm
    m2 = exportable(model)
    assert not any(isinstance(x, torch.nn.RMSNorm) for x in m2.modules())
    assert any(isinstance(x, ManualRMSNorm) for x in m2.modules())
    h = torch.randn(4, model.cfg.d_model, device=DEV)
    e = torch.zeros(4, dtype=torch.long, device=DEV)
    a = torch.zeros(4, len(model.attr_names), dtype=torch.long, device=DEV)
    for x, y in zip(DecodeHeads(model)(h, e, a), DecodeHeads(m2)(h, e, a)):
        assert torch.allclose(x, y, atol=1e-6)


@torch.no_grad()
def test_chunk_cached_equals_chunk_plus_write(model):
    from lpp.export import ChunkCached
    m, B, T, P = model, 3, 2, 10
    c = m.cfg
    torch.manual_seed(3)
    shape = (c.n_layers, B, c.n_kv_heads, P, c.d_model // c.n_heads)
    ck, cv = torch.randn(shape, device=DEV), torch.randn(shape, device=DEV)
    cm = torch.zeros(B, P, dtype=torch.bool, device=DEV)
    cm[:, :4] = True
    x = torch.randn(B, T, c.d_model, device=DEV)
    valid = torch.tensor([[True, True], [True, False], [True, True]], device=DEV)
    start = torch.tensor([4, 4, 3], device=DEV)
    tok = torch.randn(B, m.canvas_enc.n_tokens, c.d_model, device=DEV)
    draw = torch.ones(B, T, dtype=torch.bool, device=DEV)
    wp = torch.tensor(4, device=DEV)
    h, k2, v2, m2 = ChunkCached(m)(x, valid, ck, cv, cm, wp, start, tok, draw)
    h_ref, kn, vn = Chunk(m)(x, valid, ck, cv, cm, start, tok, draw)
    assert torch.allclose(h, h_ref, atol=1e-6)
    ck_ref, cv_ref, cm_ref = ck.clone(), cv.clone(), cm.clone()
    ck_ref[:, :, :, 4:6], cv_ref[:, :, :, 4:6], cm_ref[:, 4:6] = kn, vn, valid
    assert torch.equal(k2, ck_ref) and torch.equal(v2, cv_ref) and torch.equal(m2, cm_ref)
