"""Drawing sequences: build them from decompositions, check them with the grammar.

A drawing sequence (the part after the text prefix) is a list of records:

    BOS                                    starts the drawing
    LAYER_k                                opens layer k (k strictly increasing)
    PRIMITIVE x y shape color alpha        paints into the open layer
    REDRAW_k / REFINE_k                    the one "special" marker, see below
    EOS

Modes (protocol.json "sequence_modes"):
    full     BOS (LAYER_k P+)+ EOS
    redraw   BOS (LAYER_j P+)* REDRAW_k P+ EOS          layer k absent from the prefix
    refine   BOS (LAYER_j P+)+ REFINE_k P+ EOS          layer k present, holding only its coarse rounds

Compositing order never depends on the mode: primitives are painted by layer id,
then in their order inside the layer. So the three modes of one decomposition
render to the same image (tested).

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
        if self.special is None:
            for k in range(p.num_layers):
                if k > self.current and k not in self.seen:
                    out.append(p.layer_event(k))
                if k not in self.seen:
                    out.append(p.redraw_event(k))
                elif k <= self.current and p.layers[k].max_primitives - self.seen[k] > 0:
                    out.append(p.refine_event(k))
        if self.current >= 0:
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


def validate(seq: DrawSeq, proto: Protocol | None = None) -> None:
    """Raise ValueError if seq is not a complete, legal drawing sequence."""
    g = Grammar(proto or load_protocol())
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
    layer `target`'s rounds < split_round in place and moves the rest to the end.
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
    if mode not in ("full", "redraw", "refine"):
        raise ValueError(mode)
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
