"""GRPO pieces: grammar replay, exact record log-probs, advantages and the surrogate loss."""
import pytest
import numpy as np
import torch

from lpp.codec import Codec
from lpp.model import ModelConfig, Painter
from lpp.rl import group_advantages, grpo_loss, rollout_batch
from lpp.sample import SampleConfig, generate

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    codec = Codec.load(device=DEV)
    cfg = ModelConfig.tiny()
    cfg.canvas = True
    model = Painter(cfg, codec.proto, codec).to(DEV).eval()
    g = torch.Generator(device=DEV).manual_seed(1)
    sc = SampleConfig(temperature=1.0, top_p=1.0, max_records=60, require_before_eos=(0, 2))
    seqs = generate(model, batch=4, cfg=sc, generator=g, codec=codec)
    return codec, model, seqs


def _to_dev(batch, codec, model):
    from lpp.canvas import build_training_canvases
    b = {k: v.to(DEV) for k, v in batch.items() if torch.is_tensor(v)}
    b["canvas"], b["seg"] = build_training_canvases(b, codec, model.cfg.canvas_k, model.cfg.canvas_size)
    return b


def test_replayed_grammar_allows_every_sampled_event(setup):
    codec, model, seqs = setup
    batch = rollout_batch(seqs, [None] * len(seqs), codec.proto, require_before_eos=(0, 2))
    tgt, allowed = batch["tgt_event"], batch["event_allowed"]
    sel = tgt >= 0
    assert bool(allowed[sel].gather(1, tgt[sel][:, None]).all())
    # the grammar actually restricts: some positions forbid some events
    assert bool((~allowed[sel]).any())


def test_record_logprobs_finite_masking_raises_probability_and_has_grads(setup):
    codec, model, seqs = setup
    b = _to_dev(rollout_batch(seqs, [None] * len(seqs), codec.proto, (0, 2)), codec, model)
    masked = model.record_logprobs(b, b["event_allowed"])
    plain = model.record_logprobs(b)
    sel = b["tgt_event"] >= 0
    assert torch.isfinite(masked[sel]).all() and (masked[sel] <= 1e-5).all()
    assert (masked[~sel] == 0).all()
    assert (masked[sel] >= plain[sel] - 1e-5).all()          # renormalising over a subset
    model.zero_grad()
    masked[sel].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_group_advantages_are_centred_per_group():
    r = torch.tensor([1.0, 2.0, 3.0, 5.0, 5.0, 5.0])
    a = group_advantages(r, 3)
    assert torch.allclose(a.view(2, 3).mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(a[3:], torch.zeros(3))               # identical rewards: no signal


def test_grpo_loss_on_policy_gradient_is_advantage_weighted_logprob():
    torch.manual_seed(0)
    theta = torch.randn(3, 5, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[2, 3:] = False
    adv = torch.tensor([1.0, -0.5, 2.0])
    logp = torch.log_softmax(theta, -1)                        # any differentiable record log-probs
    loss, stats = grpo_loss(logp, logp.detach(), logp.detach(), adv, mask, beta=0.04)
    (g1,) = torch.autograd.grad(loss, theta)
    logp = torch.log_softmax(theta, -1)
    n = mask.sum(1, keepdim=True).float()
    want = -((adv[:, None] * logp * mask) / n).sum(1).mean()   # ratio == 1, KL == 0 and has zero grad
    (g2,) = torch.autograd.grad(want, theta)
    assert torch.allclose(g1, g2, atol=1e-6)
    assert stats["kl"] == pytest.approx(0.0, abs=1e-7)


def test_generate_forced_end_with_unmet_eos_requirement(setup):
    """Running out of records must end the sample even if the EOS requirement is not met."""
    codec, model, _ = setup
    sc = SampleConfig(temperature=1.0, top_p=1.0, max_records=4, require_before_eos=(0, 2))
    seqs = generate(model, batch=3, cfg=sc, generator=torch.Generator(device=DEV).manual_seed(5), codec=codec)
    eos = codec.proto.event_id("EOS")
    assert all(int(s.event[-1]) == eos and len(s) <= 4 for s in seqs)


def test_generate_continues_forced_prefixes(setup):
    """Prefix records are reproduced exactly and generation continues legally from them,
    for a full-mode prefix and a REFINE edit prefix; a BOS-only prefix changes nothing."""
    from lpp.sequence import DrawSeq, build, validate
    from test_sequence import fake_decomposition
    codec, model, _ = setup
    proto = codec.proto
    tokens, layer, rounds = fake_decomposition(5)
    full = build(tokens, layer, rounds, "full", proto=proto)
    refine = build(tokens, layer, rounds, "refine", target=2, split_round=1, proto=proto)

    def cut(s, n):
        return DrawSeq(s.kind[:n], s.event[:n], s.attrs[:n], s.layer[:n])

    eos = proto.event_id("EOS")
    n_ref = int((refine.event == proto.refine_event(2)).nonzero()[0]) + 2      # marker + one primitive
    prefixes = [cut(full, 9), cut(refine, n_ref)]
    sc = SampleConfig(temperature=1.0, top_p=1.0, max_records=n_ref + 40)
    out = generate(model, batch=2, cfg=sc, forced=prefixes, codec=codec,
                   generator=torch.Generator(device=DEV).manual_seed(0))
    for o, p in zip(out, prefixes):
        n = len(p)
        assert torch.equal(o.event[:n], p.event) and torch.equal(o.attrs[:n], p.attrs)
        assert int(o.event[-1]) == eos
        validate(o, proto)
    g = lambda: torch.Generator(device=DEV).manual_seed(3)
    sc2 = SampleConfig(temperature=1.0, top_p=1.0, max_records=30)
    a = generate(model, batch=1, cfg=sc2, codec=codec, generator=g())
    b = generate(model, batch=1, cfg=sc2, codec=codec, generator=g(), forced=[cut(full, 1)])
    assert torch.equal(a[0].event, b[0].event) and torch.equal(a[0].attrs, b[0].attrs)


class _FakeClip:
    """Deterministic stand-in for ClipScorer: image embedding = normalised mean colour."""
    def images(self, arrays):
        e = torch.tensor(np.stack([a.reshape(-1, 3).mean(0) for a in arrays]), dtype=torch.float32) + 1.0
        return torch.nn.functional.normalize(e, dim=-1)

    def texts(self, captions):
        return torch.nn.functional.normalize(torch.ones(len(captions), 3), dim=-1)


def test_rewards_source_likeness_and_missing_attributes():
    from lpp.rl import RewardConfig, rewards
    imgs = [np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 255, np.uint8)]
    clip = _FakeClip()
    src = clip.images(imgs[::-1])                         # each image's "source" is the other one
    attr = torch.tensor([0.8, float("nan")])              # second image has no CelebA labels
    r = rewards(imgs, [0.0, 0.0], [True, True], ["a", "b"], clip, RewardConfig(attr_weight=5.0, src_weight=1.0),
                attr_agreement=attr, source=src)
    assert torch.allclose(r["src"], 100 * (clip.images(imgs) * src).sum(1))
    assert torch.isfinite(r["reward"]).all()                              # NaN attributes stay out of the total
    assert torch.isclose(r["reward"][0] - r["clip"][0] - r["src"][0], torch.tensor(5.0 * 0.8))
    assert torch.isclose(r["reward"][1], r["clip"][1] + r["src"][1])
