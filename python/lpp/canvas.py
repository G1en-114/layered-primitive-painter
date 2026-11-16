"""Canvas feedback: what the model has painted so far, rendered at checkpoints.

Checkpoints (shared by training and decoding, see `is_checkpoint`):
    the BOS record, every CONTROL record (LAYER / REDRAW / REFINE / EOS), and every K-th
    primitive since the previous checkpoint. Position t sees the canvas of the latest
    checkpoint at or before t, i.e. everything painted up to that record (never the
    record it is about to predict).

Canvas = 9 channels at `size` px: the composite RGB over the white base, plus the
accumulated coverage 1 - T_k of each of the 6 layers.

Rendering by layer: inside one layer, sequence order is paint order in every mode
(full, redraw, refine), so each layer keeps its own premultiplied (P_k, T_k), updated
with `over` as primitives arrive, and checkpoints stack the layers back to front. The
same update rule runs in training (all checkpoints at once) and in decoding
(incrementally), so the model sees identical canvases in both.
"""
from __future__ import annotations

import torch

from .codec import Codec
from .render.dispatch import render_group
from .render.reference import Primitives, over
from .sequence import KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE

CANVAS_CHANNELS_BASE = 3


def checkpoint_flags(kind: torch.Tensor, k: int) -> torch.Tensor:
    """kind [B, T] -> bool [B, T]: records that are checkpoints."""
    # Vectorized form of the sequential rule "count primitives since the last checkpoint; a
    # marker or the k-th primitive is a checkpoint and resets the count". Markers reset the
    # count, and between markers the count resets exactly at every k-th primitive, so a
    # primitive is a checkpoint iff its index within its run since the last marker is a
    # multiple of k. A per-position Python loop cost ~5 launches x T (37 ms per batch).
    marker = (kind == KIND_BOS) | (kind == KIND_CONTROL)
    prim = kind == KIND_PRIMITIVE
    count = prim.long().cumsum(1)                                   # primitives up to and incl. t
    at_marker = torch.where(marker, count, torch.zeros_like(count)).cummax(1).values
    since = count - at_marker                                       # primitives since last marker
    return marker | (prim & (since % k == 0))


def is_checkpoint(kind: int, prims_since_last: int, k: int) -> bool:
    """Decoding-time twin of checkpoint_flags (prims_since_last counts this record)."""
    return kind in (KIND_BOS, KIND_CONTROL) or prims_since_last >= k


def empty_state(B: int, n_layers: int, size: int, device):
    P = torch.zeros(B, n_layers, 3, size, size, device=device)
    T = torch.ones(B, n_layers, 1, size, size, device=device)
    return P, T


def compose(state, base_color) -> torch.Tensor:
    """Per-layer (P [B,L,3,H,W], T [B,L,1,H,W]) -> canvas [B, 3 + L, H, W]."""
    P, T = state
    L = P.shape[1]
    rgb = base_color.view(1, 3, 1, 1).expand(P.shape[0], 3, P.shape[-2], P.shape[-1])
    for k in range(L):                                     # back to front
        rgb = P[:, k] + T[:, k] * rgb
    return torch.cat([rgb, 1 - T[:, :, 0]], dim=1)


def add_primitives(state, prims: Primitives, layer: torch.Tensor, size: int, proto):
    """Paint prims [B, n] (in sequence order) into the per-layer state. layer [B, n] (-1 = none)."""
    P, T = state
    if prims.num == 0:
        return state
    P, T = P.clone(), T.clone()
    for k in range(P.shape[1]):
        m = (layer == k)
        if not bool(m.any()):
            continue
        g = render_group(prims.with_alpha(prims.alpha * m), size, proto)
        P[:, k], T[:, k] = over((P[:, k], T[:, k]), g)
    return P, T


@torch.no_grad()
def build_training_canvases(batch: dict, codec: Codec, k: int = 32, size: int = 64):
    """All checkpoint canvases of a teacher-forced batch.

    batch: kind/layer/attrs/valid [B, T] input records (collate output, on the GPU).
    Returns canvas [B, S, 3 + L, size, size] and seg [B, T]: the checkpoint whose canvas
    position t sees (latest checkpoint <= t), -1 for padding.

    Segment j = the records after checkpoint j-1 up to and including checkpoint j, so
    canvas j = segments 0..j painted in order. Each segment is rendered once per layer;
    a prefix scan with `over` then accumulates the per-layer state.
    """
    proto = codec.proto
    kind, layer, attrs, valid = batch["kind"], batch["layer"], batch["attrs"], batch["valid"]
    B, T = kind.shape
    dev = kind.device
    flags = checkpoint_flags(kind, k) & valid
    incl = flags.long().cumsum(1)
    seg = (incl - 1).masked_fill(~valid, -1)              # canvas seen by position t
    member = incl - flags.long()                          # segment record t belongs to
    S = int(incl[:, -1].max())

    is_prim = (kind == KIND_PRIMITIVE) & valid & (member < incl[:, -1:])   # drop the trailing tail
    prims = codec.decode({n: attrs[..., i].clamp(min=0) for i, n in enumerate(proto.attr_names)})
    prims = prims.with_alpha(prims.alpha * is_prim)
    lay = torch.where(is_prim, layer, torch.full_like(layer, -1))

    # pack records into [B*S, max_len] by (segment, position inside the segment)
    member = member.contiguous()
    first = torch.searchsorted(member, member)            # row-wise: index of each segment's first record
    pos = torch.arange(T, device=dev)[None] - first
    keep = is_prim
    max_len = int(pos[keep].max()) + 1 if bool(keep.any()) else 1
    flat = (torch.arange(B, device=dev)[:, None] * S + member) * max_len + pos

    def pack(t: torch.Tensor, fill: float) -> torch.Tensor:
        out = torch.full((B * S * max_len, *t.shape[2:]), fill, dtype=t.dtype, device=dev)
        out[flat[keep]] = t[keep]
        return out.view(B * S, max_len, *t.shape[2:])

    seg_prims = Primitives(xy=pack(prims.xy, 0.0), sigma=pack(prims.sigma, 1.0), theta=pack(prims.theta, 0.0),
                           beta=pack(prims.beta, 1.0), rgb=pack(prims.rgb, 0.0), alpha=pack(prims.alpha, 0.0))
    seg_layer = pack(lay, -1)

    L = proto.num_layers
    Pg = torch.zeros(B * S, L, 3, size, size, device=dev)
    Tg = torch.ones(B * S, L, 1, size, size, device=dev)
    for kk in range(L):
        m = seg_layer == kk
        if bool(m.any()):
            Pg[:, kk], Tg[:, kk] = render_group(seg_prims.with_alpha(seg_prims.alpha * m), size, proto)
    Pg, Tg = Pg.view(B, S, L, 3, size, size), Tg.view(B, S, L, 1, size, size)

    base = torch.tensor(proto.base_color, device=dev)
    P, Tt = empty_state(B, L, size, dev)
    canvases = []
    for j in range(S):                                    # segment j goes on top of everything before
        P, Tt = Pg[:, j] + Tg[:, j] * P, Tg[:, j] * Tt
        canvases.append(compose((P, Tt), base))
    return torch.stack(canvases, 1), seg


class CanvasTracker:
    """Decoding-time canvas: per-layer state updated incrementally at checkpoints.

    update() gets each new record; when a sample hits a checkpoint, the primitives it painted
    since its previous checkpoint are rendered on top of its per-layer state (all samples that
    hit at this step together), and its canvas tokens are re-encoded. Same rule and same
    rendering as build_training_canvases, so training and decoding see identical canvases.
    """

    def __init__(self, model, codec: Codec, batch: int, device, record: bool = False):
        cfg, proto = model.cfg, codec.proto
        self.model, self.codec, self.proto = model, codec, proto
        self.k, self.size = cfg.canvas_k, cfg.canvas_size
        self.state = empty_state(batch, proto.num_layers, self.size, device)
        self.base = torch.tensor(proto.base_color, device=device)
        self.pending: list[list[tuple[torch.Tensor, int]]] = [[] for _ in range(batch)]
        self.since = [0] * batch
        canvas = compose(self.state, self.base)
        self.tokens = model.canvas_enc(canvas)               # BOS is the first checkpoint: blank
        self.history = [[c] for c in canvas] if record else None

    @torch.no_grad()
    def update(self, kind: torch.Tensor, attrs: torch.Tensor, layer: torch.Tensor, live: torch.Tensor,
               kind_cpu: list[int] | None = None, layer_cpu: list[int] | None = None,
               live_cpu: list[bool] | None = None) -> None:
        """The *_cpu lists, when given, are host copies of kind/layer/live (avoids a device sync
        per sample); the tensors stay the source of truth for attrs."""
        kind_cpu = kind.tolist() if kind_cpu is None else kind_cpu
        layer_cpu = layer.tolist() if layer_cpu is None else layer_cpu
        live_cpu = live.tolist() if live_cpu is None else live_cpu
        hit = []
        for b in range(len(kind_cpu)):
            if not live_cpu[b]:
                continue
            kb = kind_cpu[b]
            if kb == KIND_PRIMITIVE:
                self.pending[b].append((attrs[b], layer_cpu[b]))
                self.since[b] += 1
            if is_checkpoint(kb, self.since[b], self.k):
                hit.append(b)
                self.since[b] = 0
        if not hit:
            return
        n = max(len(self.pending[b]) for b in hit)
        if n > 0:
            A = attrs.shape[1]
            dev = attrs.device
            tok = torch.zeros(len(hit), n, A, dtype=torch.long, device=dev)
            lay_cpu = torch.full((len(hit), n), -1, dtype=torch.long)
            for i, b in enumerate(hit):               # one stack per sample, not one write per record
                if self.pending[b]:
                    m = len(self.pending[b])
                    tok[i, :m] = torch.stack([a for a, _ in self.pending[b]]).clamp(min=0)
                    lay_cpu[i, :m] = torch.tensor([layer_id for _, layer_id in self.pending[b]])
            lay = lay_cpu.to(dev, non_blocking=True)
            prims = self.codec.decode({nm: tok[..., i] for i, nm in enumerate(self.proto.attr_names)})
            prims = prims.with_alpha(prims.alpha * (lay >= 0))
            idx = torch.tensor(hit, device=dev)
            sub = (self.state[0][idx], self.state[1][idx])
            P, T = add_primitives(sub, prims, lay, self.size, self.proto)
            self.state[0][idx], self.state[1][idx] = P, T
        idx = torch.tensor(hit, device=self.tokens.device)
        canvas = compose((self.state[0][idx], self.state[1][idx]), self.base)
        self.tokens[idx] = self.model.canvas_enc(canvas)
        for i, b in enumerate(hit):
            self.pending[b] = []
            if self.history is not None:
                self.history[b].append(canvas[i])
