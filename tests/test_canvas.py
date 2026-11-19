"""Canvas feedback: checkpoint canvases, and the model with and without them."""
import pytest
import torch

from lpp.canvas import build_training_canvases, checkpoint_flags, compose, is_checkpoint
from lpp.codec import Codec
from lpp.data.sequences import Example, SequenceDataset, collate
from lpp.model import ModelConfig, Painter
from lpp.render.reference import render_layers
from lpp.sequence import KIND_PRIMITIVE, build
from test_sequence import fake_decomposition

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def codec():
    return Codec.load(device=DEV)


def batch_for(seqs):
    b = collate([(s, None) for s in seqs])
    return {k: v.to(DEV) for k, v in b.items()}


def test_checkpoint_rule_matches_decoding_twin():
    kind = torch.tensor([[1] + [3] * 70 + [2] + [3] * 5 + [2]])
    flags = checkpoint_flags(kind, k=32)[0].tolist()
    since, twin = 0, []
    for kd in kind[0].tolist():
        since += kd == KIND_PRIMITIVE
        hit = is_checkpoint(kd, since, 32)
        twin.append(hit)
        since = 0 if hit else since
    assert flags == twin
    assert sum(flags) == 1 + 2 + 2   # BOS, two 32-primitive marks, two control records


@pytest.mark.parametrize("mode", ["full", "redraw", "refine"])
def test_training_canvases_equal_direct_render(codec, mode):
    """Canvas j must equal rendering every primitive up to checkpoint j in paint order
    (by layer, then sequence order), whatever the generation order of the mode."""
    tokens, layer, rounds = fake_decomposition(4)
    kw = {"full": {}, "redraw": {"target": 2}, "refine": {"target": 3, "split_round": 1}}[mode]
    seq = build(tokens, layer, rounds, mode, **kw)
    b = batch_for([seq])
    canvas, seg = build_training_canvases(b, codec, k=8, size=32)
    flags = checkpoint_flags(b["kind"], 8)[0]
    ck_pos = torch.nonzero(flags).flatten().tolist()
    proto = codec.proto
    base = torch.tensor(proto.base_color, device=DEV)
    for j, t in enumerate(ck_pos):
        sel = (b["kind"][0, : t + 1] == KIND_PRIMITIVE)
        attrs, lay = b["attrs"][0, : t + 1][sel], b["layer"][0, : t + 1][sel]
        if attrs.shape[0] == 0:
            want = torch.cat([base.view(3, 1, 1).expand(3, 32, 32), torch.zeros(proto.num_layers, 32, 32,
                                                                                 device=DEV)])
        else:
            prims = codec.decode({n: attrs[:, i].unsqueeze(0) for i, n in enumerate(proto.attr_names)})
            groups = render_layers(prims, lay.unsqueeze(0), 32, proto)
            P = torch.stack([g[0] for g in groups], 1)
            T = torch.stack([g[1] for g in groups], 1)
            want = compose((P, T), base)[0]
        assert torch.allclose(canvas[0, j], want, atol=2e-4), (mode, j, (canvas[0, j] - want).abs().max())
    # every position sees the latest checkpoint at or before it
    for t in range(b["kind"].shape[1]):
        assert seg[0, t] == sum(1 for p in ck_pos if p <= t) - 1


def test_gate_zero_equals_plain_model(codec):
    proto = codec.proto
    torch.manual_seed(0)
    plain = Painter(ModelConfig.tiny(), proto, codec).to(DEV).eval()
    cfg = ModelConfig.tiny()
    cfg.canvas, cfg.canvas_every, cfg.canvas_size = True, 1, 32
    withc = Painter(cfg, proto, codec).to(DEV).eval()
    missing, unexpected = withc.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("canvas_enc", "cross")) for k in missing)
    seqs = [build(*fake_decomposition(i)) for i in range(3)]
    b = batch_for(seqs)
    b["canvas"], b["seg"] = build_training_canvases(b, codec, k=8, size=32)
    with torch.no_grad():
        a, c = plain(b), withc(b)
    assert all(torch.allclose(a[k], c[k], atol=1e-5) for k in a)


def test_canvas_model_trains(codec):
    proto = codec.proto
    cfg = ModelConfig.tiny()
    cfg.canvas, cfg.canvas_every, cfg.canvas_size = True, 1, 32
    m = Painter(cfg, proto, codec).to(DEV)
    exs = [Example(*fake_decomposition(i), image_id=i) for i in range(3)]
    ds = SequenceDataset(exs, proto)
    b = batch_for([ds[i][0] for i in range(3)])
    b["canvas"], b["seg"] = build_training_canvases(b, codec, k=8, size=32)
    out = m(b)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    # zero-initialised gates (as in Flamingo): the gates get gradient first; the encoder only
    # starts learning once a gate has moved away from 0 (d out / d encoder = tanh(gate) = 0)
    assert all(m.cross[k].gate.grad is not None and m.cross[k].gate.grad.abs() > 0 for k in m.cross)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.canvas_enc.parameters())


def test_decoding_canvases_equal_training_canvases(codec):
    """The canvases a model sees while generating must be exactly the ones teacher forcing
    builds for the same sequences (same checkpoint rule, same rendering)."""
    from lpp.sample import SampleConfig, generate
    proto = codec.proto
    cfg = ModelConfig.tiny()
    cfg.canvas, cfg.canvas_every, cfg.canvas_size, cfg.canvas_k = True, 1, 32, 8
    torch.manual_seed(1)
    m = Painter(cfg, proto, codec).to(DEV).eval()
    for c in m.cross.values():
        c.gate.data.fill_(0.5)                        # make the canvas actually matter
    hist = []
    g = torch.Generator(device=DEV).manual_seed(0)
    seqs = generate(m, batch=3, cfg=SampleConfig(max_records=90, require_before_eos=(0, 2)), generator=g,
                    codec=codec, canvas_history=hist)
    b = batch_for(seqs)
    canvas, _ = build_training_canvases(b, codec, k=8, size=32)
    for i in range(3):
        n = int(checkpoint_flags(b["kind"][i:i + 1], 8)[0][b["valid"][i]].sum())
        assert n >= 2
        for j in range(n):    # decoding also records the EOS checkpoint, which inputs never contain
            assert torch.allclose(hist[i][j], canvas[i, j], atol=2e-4), (i, j)


def _checkpoint_flags_loop(kind, k):
    """The original sequential definition, kept as the reference."""
    from lpp.sequence import KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE
    B, T = kind.shape
    flags = torch.zeros(B, T, dtype=torch.bool)
    since = torch.zeros(B, dtype=torch.long)
    for t in range(T):
        kd = kind[:, t]
        marker = (kd == KIND_BOS) | (kd == KIND_CONTROL)
        since = torch.where(kd == KIND_PRIMITIVE, since + 1, since)
        hit = marker | (since >= k)
        flags[:, t] = hit
        since = torch.where(hit, torch.zeros_like(since), since)
    return flags


def test_checkpoint_flags_vectorized_matches_sequential_rule():
    from lpp.canvas import checkpoint_flags
    g = torch.Generator().manual_seed(0)
    for k in (1, 3, 32):
        # mostly primitives, some markers, text prefix and trailing padding (kind 0)
        kind = torch.multinomial(torch.tensor([0.05, 0.02, 0.05, 0.88]), 64 * 300, replacement=True,
                                 generator=g).view(64, 300)
        kind[:, :5] = 0
        kind[:, 5] = 1
        kind[::3, 250:] = 0
        assert torch.equal(checkpoint_flags(kind, k), _checkpoint_flags_loop(kind, k))


def test_residual_canvas_decoding_equals_training(codec):
    """Residual canvas (cfg.canvas_target): the encoder input while generating towards a target, including the
    target / target-minus-canvas / has-target channels, equals what teacher forcing builds for the same sequences;
    a batch without targets gets zeros in those channels."""
    from lpp.canvas import TARGET_CHANNELS
    from lpp.sample import SampleConfig, generate
    proto = codec.proto
    cfg = ModelConfig.tiny()
    cfg.canvas, cfg.canvas_every, cfg.canvas_size, cfg.canvas_k, cfg.canvas_target = True, 1, 32, 8, True
    torch.manual_seed(2)
    m = Painter(cfg, proto, codec).to(DEV).eval()
    for c in m.cross.values():
        c.gate.data.fill_(0.5)
    target = torch.rand(3, 3, 64, 64, generator=torch.Generator().manual_seed(3)).to(DEV)
    hist = []
    g = torch.Generator(device=DEV).manual_seed(0)
    seqs = generate(m, batch=3, cfg=SampleConfig(max_records=90, require_before_eos=(0, 2)), generator=g,
                    codec=codec, canvas_history=hist, target=target)
    b = batch_for(seqs)
    b["target"], b["target_valid"] = target, torch.ones(3, dtype=torch.bool, device=DEV)
    canvas, _ = build_training_canvases(b, codec, k=8, size=32, canvas_target=True)
    C = 3 + proto.num_layers
    assert canvas.shape[2] == C + TARGET_CHANNELS
    for i in range(3):
        n = int(checkpoint_flags(b["kind"][i:i + 1], 8)[0][b["valid"][i]].sum())
        assert n >= 2
        for j in range(n):
            assert torch.allclose(hist[i][j], canvas[i, j], atol=2e-4), (i, j)
    del b["target"], b["target_valid"]
    plain, _ = build_training_canvases(b, codec, k=8, size=32, canvas_target=True)
    assert plain[:, :, C:].abs().max() == 0 and torch.allclose(plain[:, :, :C], canvas[:, :, :C])
