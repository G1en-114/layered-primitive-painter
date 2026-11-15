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
