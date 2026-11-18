"""Relative position heads (ModelConfig.rel_pos): x/y logits = absolute head + offset-from-previous head."""
import pytest
import torch

from lpp.model import KIND_PRIMITIVE, ModelConfig, Painter
from lpp.sample import SampleConfig, generate
from lpp.sequence import validate
from test_model import DEV, batch_of, codec  # noqa: F401  (fixture)


def rel_tiny(proto, codec, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig.tiny()
    cfg.rel_pos = True
    return Painter(cfg, proto, codec).to(DEV)


def test_prev_primitive(proto, codec):
    m = rel_tiny(proto, codec)
    b = batch_of(proto, n=2)
    prev = m.prev_primitive(b["kind"], b["attrs"])
    for r in range(2):
        last = None
        for t in range(b["kind"].shape[1]):
            if b["kind"][r, t] == KIND_PRIMITIVE:
                last = b["attrs"][r, t]
            want = last if last is not None else torch.full_like(b["attrs"][r, t], -1)
            assert torch.equal(prev[r, t], want), (r, t)


def test_zero_init_is_a_no_op(proto, codec):
    """At initialisation the relative model gives exactly the absolute model's losses."""
    rel = rel_tiny(proto, codec)
    torch.manual_seed(0)
    ab = Painter(ModelConfig.tiny(), proto, codec).to(DEV)
    ab.load_state_dict({k: v for k, v in rel.state_dict().items() if not k.startswith("rel_heads.")})
    b = batch_of(proto, n=3)
    lr, la = rel(b), ab(b)
    for k in la:
        assert torch.allclose(lr[k], la[k], atol=1e-6), k


def test_offset_logits(proto, codec):
    m = rel_tiny(proto, codec)
    V = m.vocab["x"]
    with torch.no_grad():
        m.heads["x"].weight.zero_(); m.heads["x"].bias.zero_()
        m.rel_heads["x"].bias[V - 1 + 3] = 10.0            # offset +3
    h = torch.randn(3, m.cfg.d_dec, device=DEV)
    prev = torch.full((3, len(m.attr_names)), -1, device=DEV)
    prev[0, 0], prev[1, 0] = 100, V - 1
    lg = m.attr_logits("x", h, prev)
    assert lg[0].argmax() == 103
    assert lg[1].max() < 1.0                                # offset +3 falls off the end: nothing boosted
    assert torch.allclose(lg[2], m.heads["x"](h[2:3])[0])   # no previous primitive: absolute head only


def test_sampler_uses_the_previous_primitive(proto, codec):
    """A model that only knows "x = previous x + 1" paints strokes whose x climbs by one."""
    m = rel_tiny(proto, codec).eval()
    V = m.vocab["x"]
    with torch.no_grad():
        m.heads["x"].weight.zero_(); m.heads["x"].bias.zero_()
        m.rel_heads["x"].bias[V - 1 + 1] = 50.0
        m.heads["event"].bias[proto.event_id("PRIMITIVE")] = 5.0     # mostly strokes, few control records
    seqs = generate(m, batch=2, cfg=SampleConfig(max_records=80, temperature=1.0, top_p=0.5),
                    generator=torch.Generator(device=DEV).manual_seed(0))
    climbs = 0
    for s in seqs:
        validate(s)
        xs = s.attrs[s.kind == KIND_PRIMITIVE, 0].tolist()
        climbs += sum(b - a == 1 for a, b in zip(xs, xs[1:]))
        assert all(b - a == 1 for a, b in zip(xs, xs[1:]) if a < V - 2)
    assert climbs > 5


def test_teacher_forcing_sees_the_samplers_previous_primitive(proto, codec):
    """Greedy decoding picks every attribute as the argmax of its logits; re-scoring the result with
    teacher forcing (training, and record_logprobs for GRPO) must reproduce those argmaxes, which only
    holds if both paths offset the relative heads by the same previous primitive."""
    from lpp.rl import rollout_batch
    m = rel_tiny(proto, codec, seed=1).eval()
    with torch.no_grad():
        for h in m.rel_heads.values():
            h.weight.normal_(0, 0.3)
    seqs = generate(m, batch=2, cfg=SampleConfig(max_records=60, temperature=0.0),
                    generator=torch.Generator(device=DEV).manual_seed(3))
    b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in rollout_batch(seqs, [None, None], proto).items()}
    with torch.no_grad():
        h = m._hidden(b)
        sel = b["tgt_event"] >= 0
        ev, at = b["tgt_event"][sel], b["tgt_attrs"][sel]
        out = m.decoder(h[sel], ev, at)
        prim = ev == proto.event_id("PRIMITIVE")
        pp = m.prev_primitive(b["kind"], b["attrs"])[sel][prim]
        assert prim.sum() > 5
        for i, name in enumerate(m.attr_names):
            pred = m.attr_logits(name, out[prim, 1 + i], pp).argmax(-1)
            assert torch.equal(pred, at[prim, i]), name
