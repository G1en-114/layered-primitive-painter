"""export.DecodeSample (all attributes of a record in one graph call) == the reference sampler's attribute loop,
and export.sample_top_p == lpp.sample._sample. Same uniforms in, same tokens out."""
import torch

from lpp.codec import Codec
from lpp.export import DecodeSample, exportable, sample_top_p
from lpp.model import ModelConfig, Painter
from lpp.protocol import PROTOCOL_V3_PATH, load_protocol
from lpp.sample import SampleConfig, _sample


def test_sample_top_p_matches_reference():
    g = torch.Generator().manual_seed(0)
    for V, p in ((512, 0.6), (33, 0.9), (4096, 0.95)):
        logits = torch.randn(64, V, generator=g) * 3
        u = torch.rand(64, generator=g)
        cfg = SampleConfig(temperature=1.0, top_p=p)
        ref = _sample(logits, cfg, u)
        got = sample_top_p(logits, u, torch.tensor([p]), torch.tensor([1.0]))
        assert torch.equal(ref, got), (V, p, (ref != got).sum())


def reference_attrs(m, h, event, u, cfg):
    """The attribute loop of lpp.sample.generate for one record, guidance on."""
    B, A = event.shape[0], len(m.attr_names)
    ev2 = torch.cat([event, event])
    attrs = torch.full((B, A), -1, dtype=torch.long)
    for i, name in enumerate(m.attr_names):
        dh = m.decoder(h, ev2, torch.cat([attrs, attrs]).clamp(min=0), steps=2 + i)
        logits = m.attr_logits(name, dh[:, -1], None, ev2)
        c, un = logits.float().chunk(2)
        attrs[:, i] = _sample(un + cfg.cfg_scale * (c - un), cfg, u[:, i], name)
    for e in m.proto.stroke_events:
        for n, t in m.proto.fixed_tokens(e).items():
            j = m.attr_names.index(n)
            attrs[:, j] = torch.where(event == e, torch.full_like(attrs[:, j], t), attrs[:, j])
    return attrs


def test_decode_sample_matches_reference_loop():
    proto = load_protocol(PROTOCOL_V3_PATH)
    codec = Codec.load(proto=proto, device="cpu")
    cfg = ModelConfig.tiny()
    cfg.line_heads = True
    torch.manual_seed(0)
    m = Painter(cfg, proto, codec).eval()
    B = 12
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2 * B, cfg.d_model, generator=g)
    line, prim = proto.event_id("LINE"), proto.event_id("PRIMITIVE")
    event = torch.tensor([line, prim] * (B // 2))
    u = torch.rand(B, len(m.attr_names), generator=g)
    sc = SampleConfig(temperature=1.0, top_p=0.6, cfg_scale=5.0)
    with torch.no_grad():
        ref = reference_attrs(m, h, event, u, sc)
        got = DecodeSample(exportable(m)).eval()(h, event, u, torch.tensor([5.0]), torch.tensor([0.6]), torch.tensor([1.0]))
    assert torch.equal(ref, got), (ref != got).nonzero()[:5]
