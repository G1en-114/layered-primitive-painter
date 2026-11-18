"""Protocol v3: LINE strokes (Bezier lines) next to the Gaussians (docs/line_strokes.md)."""
import pytest
import torch

from lpp.canvas import build_training_canvases, checkpoint_flags
from lpp.codec import Codec
from lpp.data.sequences import Example, SequenceDataset, collate, widen
from lpp.model import ModelConfig, Painter
from lpp.protocol import PROTOCOL_V3_PATH, load_protocol
from lpp.render.strokes import render_strokes
from lpp.sample import SampleConfig, generate
from lpp.sequence import KIND_PRIMITIVE, build, strokes, validate
from test_sequence import fake_decomposition

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def v3():
    return load_protocol(PROTOCOL_V3_PATH)


@pytest.fixture(scope="module")
def codec3(v3):
    return Codec.load(proto=v3, device=DEV)


def test_v3_keeps_every_v2_id(v3):
    v2 = load_protocol()
    assert v3.events[: len(v2.events)] == v2.events and v3.events[-1] == "LINE"
    assert v3.attr_names[: len(v2.attr_names)] == v2.attr_names and v3.attr_names[-1] == "bend"
    assert v3.stroke_events == (v3.event_id("PRIMITIVE"), v3.event_id("LINE"))
    assert v3.fixed_tokens(v3.event_id("LINE")) == {"beta": 0}
    assert v3.fixed_tokens(v3.event_id("PRIMITIVE")) == {"bend": v3.attr("bend").bins // 2}


def lined_example(v3, seed=0):
    """A fake decomposition (v2 width) widened to v3, with every third stroke of rounds >= 2 a line."""
    tok, lay, rnd = fake_decomposition(seed)
    tok = widen(tok, v3)
    is_line = (torch.arange(len(lay)) % 3 == 0) & (rnd >= 2)
    tok[is_line, v3.attr_names.index("beta")] = 0
    return tok, lay, rnd, is_line


def test_build_and_validate_with_lines(v3):
    tok, lay, rnd, is_line = lined_example(v3)
    for mode in ({"mode": "full"}, {"mode": "draft", "split_round": 2}):
        s = build(tok, lay, rnd, proto=v3, is_line=is_line, **mode)
        validate(s, v3, draft=mode["mode"] == "draft")
        a, l, ln = strokes(s, v3)
        assert int(ln.sum()) == int(is_line.sum()) and len(a) == len(lay)
    with pytest.raises(ValueError):
        build(tok[:, :8], lay, rnd, is_line=is_line)                   # v2 protocol: no LINE event


def test_v2_data_trains_a_v3_model(v3, codec3):
    exs = [Example(*fake_decomposition(i), image_id=i) for i in range(3)]   # 8-wide v2 tokens
    b = collate([SequenceDataset(exs, v3, seed=0)[i] for i in range(3)])
    assert b["attrs"].shape[-1] == len(v3.attr_names)
    torch.manual_seed(0)
    m = Painter(ModelConfig.tiny(), v3, codec3).to(DEV)
    out = m({k: v.to(DEV) for k, v in b.items()})
    assert torch.isfinite(out["loss"]) and "bend" in out


def test_generation_paints_lines_with_their_fixed_slots(v3, codec3):
    torch.manual_seed(0)
    m = Painter(ModelConfig.tiny(), v3, codec3).to(DEV).eval()
    with torch.no_grad():
        m.heads["event"].bias[v3.event_id("LINE")] = 4.0              # make lines likely
    seqs = generate(m, batch=3, cfg=SampleConfig(max_records=80), generator=torch.Generator(device=DEV).manual_seed(0))
    n_lines = 0
    bi, ki = v3.attr_names.index("beta"), v3.attr_names.index("bend")
    for s in seqs:
        validate(s, v3)
        a, _, ln = strokes(s, v3)
        n_lines += int(ln.sum())
        assert (a[ln, bi] == 0).all() and (a[~ln, ki] == v3.attr("bend").bins // 2).all()
    assert n_lines > 5


def test_lines_composite_over_their_layer_and_under_the_next(v3, codec3):
    """A dark line in layer 2 shows over layer 2's light fill, and an opaque layer-4 Gaussian hides it."""
    def tok(**kw):
        t = {n: 0 for n in v3.attr_names}
        t |= {"bend": v3.attr("bend").bins // 2, "alpha": 15, "beta": 7}
        t |= kw
        return [t[n] for n in v3.attr_names]
    ax, ay = v3.attr("x"), v3.attr("y")
    q = lambda v: int(ax.quantize(torch.tensor(float(v))))                # noqa: E731
    white = int(codec3.encode_lines.__self__.color_codes.shape[0] - 1)    # any colour: compare, not absolute
    fill = tok(x=q(128), y=q(128), sigma_x=110, sigma_y=110, color=white)  # big Gaussian
    line = tok(x=q(128), y=q(128), sigma_x=100, sigma_y=30, theta=0, beta=0, color=0)   # ~39 px half length
    cover = tok(x=q(165), y=q(128), sigma_x=80, sigma_y=80, color=white)                # ~16 px, hard edge
    attrs = torch.tensor([fill, line, cover])
    layer = torch.tensor([2, 2, 4])
    is_line = torch.tensor([False, True, False])
    img = render_strokes(attrs, layer, is_line, codec3)[0]
    no_line = render_strokes(attrs[[0, 2]], layer[[0, 2]], is_line[[0, 2]], codec3)[0]
    diff = (img - no_line).abs().sum(0)
    assert diff[128, 105] > 0.1                                          # the line shows on the fill
    assert diff[128, 165] < 1e-3                                         # and is hidden by layer 4 there


def test_decoding_canvases_equal_training_canvases_with_lines(v3, codec3):
    cfg = ModelConfig.tiny()
    cfg.canvas, cfg.canvas_every, cfg.canvas_size, cfg.canvas_k = True, 1, 32, 8
    torch.manual_seed(1)
    m = Painter(cfg, v3, codec3).to(DEV).eval()
    for c in m.cross.values():
        c.gate.data.fill_(0.5)
    with torch.no_grad():
        m.heads["event"].bias[v3.event_id("LINE")] = 3.0
    hist = []
    seqs = generate(m, batch=3, cfg=SampleConfig(max_records=90, require_before_eos=(0, 2)),
                    generator=torch.Generator(device=DEV).manual_seed(0), codec=codec3, canvas_history=hist)
    assert sum(int(strokes(s, v3)[2].sum()) for s in seqs) > 3
    b = {k: v.to(DEV) for k, v in collate([(s, None) for s in seqs]).items()}
    canvas, _ = build_training_canvases(b, codec3, k=8, size=32)
    for i in range(3):
        n = int(checkpoint_flags(b["kind"][i:i + 1], 8)[0][b["valid"][i]].sum())
        for j in range(n):
            assert torch.allclose(hist[i][j], canvas[i, j], atol=2e-4), (i, j)
