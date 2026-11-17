"""Drawing sequences: build them from decompositions, check them with the grammar.

A drawing sequence (the part after the text prefix) is a list of records:

    BOS                                    starts the drawing
    LAYER_k                                opens layer k (k strictly increasing)
    PRIMITIVE x y shape color alpha        paints into the open layer
    REDRAW_k / REFINE_k                    the "special" markers, see below
    EOS

Modes (protocol.json "sequence_modes"):
    full     BOS (LAYER_k P+)+ EOS
    redraw   BOS (LAYER_j P+)* REDRAW_k P+ EOS          layer k absent from the prefix
    refine   BOS (LAYER_j P+)+ REFINE_k P+ EOS          layer k present, holding only its coarse rounds
    draft    BOS (LAYER_k P+)+ (REFINE_k P+)* EOS       every layer's coarse rounds first (a sketch of
                                                        the whole picture), then each layer refined in
                                                        turn, REFINE k strictly increasing

Compositing order never depends on the mode: primitives are painted by layer id,
then in their order inside the layer. So all modes of one decomposition render to
the same image (tested). Chained REFINE markers (draft) are legal only for a
Grammar(draft=True), i.e. for models trained in the draft order.

`Grammar` is the state machine that defines what may come next. Training data is
validated with it, and decoding masks the event logits with `allowed_events()`;
the C++ runtime implements the same machine.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .protocol import Protocol, load_protocol

KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE = 1, 2, 3   # protocol position_kinds (0 = TEXT)
NO_ATTR = -1


@dataclass
class DrawSeq:
    kind: torch.Tensor     # [T] long
    event: torch.Tensor    # [T] long, event id (PRIMITIVE for primitives, -1 for BOS)
    attrs: torch.Tensor    # [T, A] long, attribute tokens, NO_ATTR on non-primitive records
    layer: torch.Tensor    # [T] long, layer the record belongs to / opens (-1 for BOS, EOS)
    no_target: torch.Tensor | None = None   # [T] bool: perturbed records, never used as targets

    def __len__(self) -> int:
        return self.kind.numel()


@dataclass
class Grammar:
    """Incremental validator / constraint generator for drawing sequences."""

    proto: Protocol = field(default_factory=load_protocol)
    started: bool = False
    done: bool = False
    current: int = -1              # open layer, -1 before the first marker
    count: int = 0                 # primitives since the last marker
    special: str | None = None     # "redraw" / "refine" once the special marker was used
    seen: dict[int, int] = field(default_factory=dict)   # layer -> primitives painted in it
    # decoding-time constraint (not part of the data grammar): EOS only once these layers
    # have primitives, so a sample cannot stop after the background alone
    require_before_eos: frozenset = frozenset()
    allow_special: bool = True     # False: no REDRAW/REFINE (they edit a given prefix; pointless from BOS)
    # draft order: REFINE may follow REFINE (k strictly increasing), and a draft model may start
    # refining from BOS-started decoding even when allow_special is False (once the required
    # layers have primitives, so ending stays possible)
    draft: bool = False

    # ---- what may come next ---------------------------------------------
    def _cap_left(self) -> int:
        if self.current < 0:
            return 0
        used = self.seen.get(self.current, 0)
        return self.proto.layers[self.current].max_primitives - used

    def allowed_events(self) -> list[int]:
        p = self.proto
        if not self.started or self.done:
            return []
        out = []
        if self.current >= 0 and self._cap_left() > 0:
            out.append(p.event_id("PRIMITIVE"))
        if self.current >= 0 and self.count == 0:
            return out                         # a marker must be followed by a primitive
        required_done = all(self.seen.get(r, 0) > 0 for r in self.require_before_eos)
        if self.special is None:
            for k in range(p.num_layers):
                # a required layer cannot be skipped: layers only go up, so jumping past it
                # would make the EOS requirement impossible to meet
                skips_required = any(r < k and self.seen.get(r, 0) == 0 for r in self.require_before_eos)
                if k > self.current and k not in self.seen and not skips_required:
                    out.append(p.layer_event(k))
                if self.allow_special and k not in self.seen:
                    out.append(p.redraw_event(k))
                elif (self.allow_special or (self.draft and required_done)) and k in self.seen \
                        and k <= self.current and p.layers[k].max_primitives - self.seen[k] > 0:
                    out.append(p.refine_event(k))
        elif self.special == "refine" and self.draft:
            for k in range(self.current + 1, p.num_layers):
                if self.seen.get(k, 0) > 0 and p.layers[k].max_primitives - self.seen[k] > 0:
                    out.append(p.refine_event(k))
        if self.current >= 0 and required_done:
            out.append(p.event_id("EOS"))
        if not out:   # decoding constraints can paint into a corner: ending must stay possible
            out.append(p.event_id("EOS"))
        return sorted(out)

    def allowed_mask(self, device=None) -> torch.Tensor:
        m = torch.zeros(len(self.proto.events), dtype=torch.bool, device=device)
        m[self.allowed_events()] = True
        return m

    # ---- advance ---------------------------------------------------------
    def bos(self) -> None:
        if self.started:
            raise ValueError("BOS twice")
        self.started = True

    def step(self, event: int) -> None:
        if event not in self.allowed_events():
            raise ValueError(f"event {self.proto.events[event]} not allowed here "
                             f"(layer {self.current}, count {self.count}, special {self.special})")
        name = self.proto.events[event]
        if name == "PRIMITIVE":
            self.count += 1
            self.seen[self.current] = self.seen.get(self.current, 0) + 1
        elif name == "EOS":
            self.done = True
        else:
            kind, k = name.split("_")
            self.current, self.count = int(k), 0
            if kind == "REDRAW":
                self.special = "redraw"
            elif kind == "REFINE":
                self.special = "refine"
            self.seen.setdefault(self.current, 0)


def validate(seq: DrawSeq, proto: Protocol | None = None, draft: bool = False) -> None:
    """Raise ValueError if seq is not a complete, legal drawing sequence."""
    g = Grammar(proto or load_protocol(), draft=draft)
    if len(seq) == 0 or seq.kind[0] != KIND_BOS:
        raise ValueError("sequence must start with BOS")
    g.bos()
    for t in range(1, len(seq)):
        g.step(int(seq.event[t]))
    if not g.done:
        raise ValueError("sequence does not end with EOS")


# ---------------------------------------------------------------------------- building
def build(tokens: torch.Tensor, layer: torch.Tensor, rounds: torch.Tensor, mode: str = "full",
          target: int | None = None, split_round: int | None = None,
          proto: Protocol | None = None) -> DrawSeq:
    """Assemble a drawing sequence from one decomposition.

    tokens [N, A] attribute tokens in decomposer order (layer, then round, then Morton),
    layer [N], rounds [N]. mode 'redraw' moves layer `target` to the end; 'refine' keeps
    layer `target`'s rounds < split_round in place and moves the rest to the end; 'draft'
    does that for every layer (rounds < split_round of all layers, then each layer's rest
    after a REFINE marker, layers in order).
    """
    proto = proto or load_protocol()
    A = len(proto.attributes)
    prim_event = proto.event_id("PRIMITIVE")
    kind, event, attrs, lay = [KIND_BOS], [-1], [[NO_ATTR] * A], [-1]

    def marker(ev: int, k: int) -> None:
        kind.append(KIND_CONTROL), event.append(ev), attrs.append([NO_ATTR] * A), lay.append(k)

    def prims(idx: torch.Tensor, k: int) -> None:
        for i in idx.tolist():
            kind.append(KIND_PRIMITIVE), event.append(prim_event), attrs.append(tokens[i].tolist())
            lay.append(k)

    present = [int(k) for k in torch.unique(layer).tolist()]
    if mode not in ("full", "redraw", "refine", "draft"):
        raise ValueError(mode)
    if mode == "draft":
        split = 2 if split_round is None else split_round
        tails = []
        for k in present:
            idx = torch.nonzero(layer == k).flatten()
            coarse, fine = idx[rounds[idx] < split], idx[rounds[idx] >= split]
            if coarse.numel() == 0:          # nothing coarse to sketch: the whole layer is the sketch
                coarse, fine = fine, fine[:0]
            marker(proto.layer_event(k), k)
            prims(coarse, k)
            if fine.numel():
                tails.append((k, fine))
        for k, fine in tails:
            marker(proto.refine_event(k), k)
            prims(fine, k)
        marker(proto.event_id("EOS"), -1)
        return DrawSeq(torch.tensor(kind), torch.tensor(event), torch.tensor(attrs, dtype=torch.long),
                       torch.tensor(lay))
    if mode != "full" and target not in present:
        raise ValueError(f"layer {target} has no primitives")
    tail = None
    for k in present:
        idx = torch.nonzero(layer == k).flatten()
        if mode == "redraw" and k == target:
            tail = (proto.redraw_event(k), idx, k)
            continue
        if mode == "refine" and k == target:
            coarse, fine = idx[rounds[idx] < split_round], idx[rounds[idx] >= split_round]
            if coarse.numel() == 0 or fine.numel() == 0:
                raise ValueError(f"split round {split_round} leaves an empty part of layer {k}")
            marker(proto.layer_event(k), k)
            prims(coarse, k)
            tail = (proto.refine_event(k), fine, k)
            continue
        marker(proto.layer_event(k), k)
        prims(idx, k)
    if tail is not None:
        marker(tail[0], tail[2])
        prims(tail[1], tail[2])
    marker(proto.event_id("EOS"), -1)

    return DrawSeq(torch.tensor(kind), torch.tensor(event), torch.tensor(attrs, dtype=torch.long),
                   torch.tensor(lay))


def sample_mode(layer: torch.Tensor, rounds: torch.Tensor, rng: random.Random,
                p_redraw: float = 0.15, p_refine: float = 0.15) -> dict:
    """Pick a training mode for one example (kwargs for `build`)."""
    present = torch.unique(layer).tolist()
    u = rng.random()
    if u < p_redraw:
        return {"mode": "redraw", "target": rng.choice(present)}
    if u < p_redraw + p_refine:
        options = []
        for k in present:
            r = torch.unique(rounds[layer == k]).tolist()
            options += [(k, s) for s in r[1:]]       # split before any round but the first
        if options:
            k, s = rng.choice(options)
            return {"mode": "refine", "target": k, "split_round": s}
    return {"mode": "full"}


def paint_order(seq: DrawSeq) -> tuple[torch.Tensor, torch.Tensor]:
    """Primitives of a sequence in compositing order -> (attrs [N, A], layer [N]).

    Stable sort by layer: inside a layer, sequence order (coarse part before the
    REFINE tail) is the paint order.
    """
    is_prim = seq.kind == KIND_PRIMITIVE
    attrs, lay = seq.attrs[is_prim], seq.layer[is_prim]
    order = torch.sort(lay, stable=True).indices
    return attrs[order], lay[order]


def perturb(seq: DrawSeq, rng: random.Random, proto: Protocol | None = None, frac: float = 0.05,
            max_ops: int = 40) -> DrawSeq:
    """Corrupt a few primitives so the model sees mistakes in its history (and its canvas)
    and learns to continue well after them (exposure bias, cf. Self Forcing).

    About `frac` of the primitives get one operation each:
        jitter  move / resize / recolour / change opacity a little
        delete  only primitives with primitives on both sides (grammar stays valid)
        insert  a small random primitive in the same layer (if the layer cap allows)
    Jittered and inserted records are marked `no_target`: the model must not learn to
    produce them, only to continue after them. Deleted ones simply vanish.
    """
    proto = proto or load_protocol()
    vocab = proto.attr_vocab_sizes
    names = proto.attr_names
    n = len(seq)
    prim = [i for i in range(n) if int(seq.kind[i]) == KIND_PRIMITIVE]
    if not prim:
        return seq
    k = min(max_ops, max(1, round(frac * len(prim))))
    chosen = set(rng.sample(prim, k))
    counts: dict[int, int] = {}
    for i in prim:
        counts[int(seq.layer[i])] = counts.get(int(seq.layer[i]), 0) + 1

    def jitter(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        for j, nm in enumerate(names):
            v = vocab[nm]
            step = max(1, v // 32)                                  # ~3% of the range
            if nm in ("x", "y"):
                a[j] = min(v - 1, max(0, int(a[j]) + rng.randint(-step, step)))
            elif nm == "theta":                                      # periodic: wrap around
                a[j] = (int(a[j]) + rng.randint(-v // 16, v // 16)) % v
            elif nm == "color":
                if rng.random() < 0.3:
                    a[j] = rng.randrange(v)
            elif rng.random() < 0.5:   # sigma_x, sigma_y, beta, alpha (and v1 shape): small step
                a[j] = min(v - 1, max(0, int(a[j]) + rng.randint(-step, step)))
        return a

    def speck(layer_id: int) -> torch.Tensor:
        a = torch.zeros(len(names), dtype=torch.long)
        for j, nm in enumerate(names):
            v = vocab[nm]
            if nm in ("sigma_x", "sigma_y"):
                a[j] = rng.randrange(0, max(1, v // 4))           # small
            else:
                a[j] = rng.randrange(v)
        return a

    kind, event, attrs, layer, bad = [], [], [], [], []
    for i in range(n):
        rec = (seq.kind[i], seq.event[i], seq.attrs[i], seq.layer[i])
        if i in chosen:
            op = rng.random()
            lay = int(seq.layer[i])
            if op < 0.25 and 0 < i < n - 1 and int(seq.kind[i - 1]) == KIND_PRIMITIVE \
                    and int(seq.kind[i + 1]) == KIND_PRIMITIVE:
                counts[lay] -= 1
                continue                                           # delete
            if op < 0.5 and counts[lay] < proto.layers[lay].max_primitives:
                kind += [rec[0], rec[0]]                           # keep, then insert a speck after it
                event += [rec[1], rec[1]]
                attrs += [rec[2], speck(lay)]
                layer += [rec[3], rec[3]]
                bad += [False, True]
                counts[lay] += 1
                continue
            rec = (rec[0], rec[1], jitter(rec[2]), rec[3])         # jitter
            kind.append(rec[0]), event.append(rec[1]), attrs.append(rec[2]), layer.append(rec[3])
            bad.append(True)
            continue
        kind.append(rec[0]), event.append(rec[1]), attrs.append(rec[2]), layer.append(rec[3])
        bad.append(False)
    return DrawSeq(torch.stack(kind), torch.stack(event), torch.stack(attrs), torch.stack(layer),
                   torch.tensor(bad))
