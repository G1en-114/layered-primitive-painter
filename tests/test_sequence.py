import random

import pytest
import torch

from lpp.codec import Codec
from lpp.protocol import load_protocol
from lpp.render.reference import render_layers, stack_layers
from lpp.sequence import Grammar, build, paint_order, sample_mode, validate


def fake_decomposition(seed=0, layers=(0, 2, 3, 4)):
    g = torch.Generator().manual_seed(seed)
    lay, rnd = [], []
    for k in layers:
        n = int(torch.randint(3, 12, (1,), generator=g))
        lay += [k] * n
        rnd += sorted(torch.randint(0, 4, (n,), generator=g).tolist())
    N = len(lay)
    vocab = tuple(load_protocol().attr_vocab_sizes.values())
    tokens = torch.stack([torch.randint(v, (N,), generator=g) for v in vocab], 1)
    return tokens, torch.tensor(lay), torch.tensor(rnd)


@pytest.mark.parametrize("seed", range(5))
def test_every_mode_is_grammatical(proto, seed):
    tokens, layer, rounds = fake_decomposition(seed)
    rng = random.Random(seed)
    validate(build(tokens, layer, rounds, "full"))
    for k in torch.unique(layer).tolist():
        validate(build(tokens, layer, rounds, "redraw", target=k))
        for s in torch.unique(rounds[layer == k]).tolist()[1:]:
            validate(build(tokens, layer, rounds, "refine", target=k, split_round=s))
    for _ in range(20):
        validate(build(tokens, layer, rounds, **sample_mode(layer, rounds, rng)))


def test_modes_paint_identically(proto):
    """REDRAW / REFINE change generation order, never the picture."""
    try:
        codec = Codec.load()
    except FileNotFoundError:
        pytest.skip("codebooks missing")
    tokens, layer, rounds = fake_decomposition(3)
    seqs = [build(tokens, layer, rounds, "full"), build(tokens, layer, rounds, "redraw", target=2)]
    k = 3
    s = torch.unique(rounds[layer == k]).tolist()[1]
    seqs.append(build(tokens, layer, rounds, "refine", target=k, split_round=s))
    imgs = []
    for seq in seqs:
        attrs, lay = paint_order(seq)
        prims = codec.decode({n: attrs[:, i].unsqueeze(0) for i, n in enumerate(codec.attr_names)})
        imgs.append(stack_layers(render_layers(prims, lay.unsqueeze(0), 64)))
    assert torch.equal(imgs[0], imgs[1]) and torch.equal(imgs[0], imgs[2])


def test_grammar_rejects_bad_sequences(proto):
    P, EOS = proto.event_id("PRIMITIVE"), proto.event_id("EOS")
    g = Grammar(proto)
    g.bos()
    with pytest.raises(ValueError):
        g.step(P)                                  # primitive before any layer
    g.step(proto.layer_event(2))
    with pytest.raises(ValueError):
        g.step(EOS)                                # empty layer
    g.step(P)
    with pytest.raises(ValueError):
        g.step(proto.layer_event(1))               # layers must increase
    g.step(proto.redraw_event(1))
    g.step(P)
    with pytest.raises(ValueError):
        g.step(proto.refine_event(2))              # only one special marker
    assert g.allowed_events() == [P, EOS]


def test_layer_cap_is_enforced(proto):
    P = proto.event_id("PRIMITIVE")
    g = Grammar(proto)
    g.bos()
    g.step(proto.layer_event(5))
    for _ in range(proto.layers[5].max_primitives):
        g.step(P)
    assert P not in g.allowed_events()


def test_allowed_mask_matches_list(proto):
    g = Grammar(proto)
    g.bos()
    m = g.allowed_mask()
    assert m.sum() == len(g.allowed_events()) and not m[proto.event_id("PRIMITIVE")]


def test_constraints_never_deadlock(proto):
    """A required layer that can no longer be reached must not leave zero legal events."""
    P = proto.event_id("PRIMITIVE")
    g = Grammar(proto, require_before_eos=frozenset({0, 2}))
    g.bos()
    g.step(proto.redraw_event(5))
    for _ in range(proto.layers[5].max_primitives):
        g.step(P)
    assert g.allowed_events() == [proto.event_id("EOS")]


def test_fresh_generation_has_no_special_markers(proto):
    g = Grammar(proto, allow_special=False)
    g.bos()
    names = {proto.events[e] for e in g.allowed_events()}
    assert not any(n.startswith(("REDRAW", "REFINE")) for n in names)


def test_perturbed_sequences_stay_grammatical(proto):
    from lpp.sequence import perturb
    rng = random.Random(0)
    tokens, layer, rounds = fake_decomposition(6)
    for _ in range(30):
        seq = build(tokens, layer, rounds, **sample_mode(layer, rounds, rng))
        p = perturb(seq, rng, proto, frac=0.3)
        validate(p)
        assert p.no_target is not None and len(p.no_target) == len(p)
        assert not bool(p.no_target[p.kind != 3].any())          # only primitives are ever marked


def test_perturbed_records_are_not_targets(proto):
    from lpp.data.sequences import collate
    from lpp.sequence import perturb
    tokens, layer, rounds = fake_decomposition(7)
    p = perturb(build(tokens, layer, rounds), random.Random(1), proto, frac=0.5)
    b = collate([(p, None)])
    bad_next = p.no_target[1:]
    assert bool(bad_next.any())
    assert (b["tgt_event"][0, : len(p) - 1][bad_next] == -1).all()
    assert (b["tgt_event"][0, : len(p) - 1][~bad_next] >= 0).all()


# ---------------------------------------------------------------------------- draft order
def _draft_examples():
    for seed in range(5):
        yield fake_decomposition(seed)
    yield fake_decomposition(9, layers=(0, 1, 2, 3, 4, 5))


def test_draft_is_grammatical_only_for_a_draft_grammar(proto):
    for tokens, layer, rounds in _draft_examples():
        seq = build(tokens, layer, rounds, "draft", split_round=2)
        validate(seq, draft=True)
        names = [proto.events[e] for e in seq.event.tolist()[1:]]
        refines = [int(n.split("_")[1]) for n in names if n.startswith("REFINE")]
        assert refines == sorted(set(refines))                 # each layer refined once, in order
        if len(refines) > 1:
            with pytest.raises(ValueError):
                validate(seq)                                    # chained REFINE needs draft=True


def test_draft_sketches_every_layer_first(proto):
    tokens, layer, rounds = fake_decomposition(9, layers=(0, 1, 2, 3, 4, 5))
    seq = build(tokens, layer, rounds, "draft", split_round=2)
    names = [proto.events[e] for e in seq.event.tolist()[1:]]
    first_refine = next(i for i, n in enumerate(names) if n.startswith("REFINE"))
    opened = [n for n in names[:first_refine] if n.startswith("LAYER")]
    assert opened == [f"LAYER_{k}" for k in torch.unique(layer).tolist()]
    sketch = seq.kind[1:first_refine + 1] == 3
    assert int(sketch.sum()) == int((rounds < 2).sum())       # exactly the coarse rounds before refining


def test_draft_paints_identically(proto):
    try:
        codec = Codec.load()
    except FileNotFoundError:
        pytest.skip("codebooks missing")
    for tokens, layer, rounds in _draft_examples():
        imgs = []
        for seq in (build(tokens, layer, rounds, "full"), build(tokens, layer, rounds, "draft", split_round=2)):
            attrs, lay = paint_order(seq)
            prims = codec.decode({n: attrs[:, i].unsqueeze(0) for i, n in enumerate(codec.attr_names)})
            imgs.append(stack_layers(render_layers(prims, lay.unsqueeze(0), 64)))
        assert torch.equal(imgs[0], imgs[1])


def test_draft_decoding_refines_only_after_required_layers(proto):
    """From BOS, a draft model may switch to refining, but not before background and skin exist."""
    P = proto.event_id("PRIMITIVE")
    g = Grammar(proto, require_before_eos=frozenset({0, 2}), allow_special=False, draft=True)
    g.bos()
    g.step(proto.layer_event(0)); g.step(P)
    names = {proto.events[e] for e in g.allowed_events()}
    assert not any(n.startswith(("REFINE", "REDRAW")) for n in names)    # skin still missing
    g.step(proto.layer_event(2)); g.step(P)
    g.step(proto.layer_event(4)); g.step(P)
    names = {proto.events[e] for e in g.allowed_events()}
    assert {"REFINE_0", "REFINE_2", "REFINE_4"} <= names and not any(n.startswith("REDRAW") for n in names)
    g.step(proto.refine_event(2)); g.step(P)
    names = {proto.events[e] for e in g.allowed_events()}
    assert "REFINE_4" in names and "REFINE_0" not in names and not any(n.startswith("LAYER") for n in names)
    assert "EOS" in names


def test_layer_grammar_unchanged_without_draft(proto):
    g = Grammar(proto, allow_special=False)
    g.bos()
    for k in (0, 2):
        g.step(proto.layer_event(k)); g.step(proto.event_id("PRIMITIVE"))
    assert not any(proto.events[e].startswith("REFINE") for e in g.allowed_events())
