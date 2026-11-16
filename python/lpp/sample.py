"""Reference sampler: grammar-constrained autoregressive generation with a KV cache.

This is the Python oracle for the C++ runtime: same loop, same masks, and the random
numbers can be supplied from outside (`uniforms`), so both implementations can be
made to produce bit-identical sequences.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Painter
from .canvas import CanvasTracker
from .sequence import KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE, NO_ATTR, DrawSeq, Grammar


@dataclass
class SampleConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    cfg_scale: float = 1.0             # >1: classifier-free guidance (needs text)
    attr_temperature: dict | None = None   # per-attribute temperature overrides, e.g. {"x": 0.7}
    renyi_alpha: float | None = None   # adaptive support (POP2-style) instead of top-p, see _sample
    require_before_eos: tuple = ()     # layer ids that must be drawn before EOS
    max_records: int | None = None     # drawing records incl. BOS; default: context minus text


def _sample(logits: torch.Tensor, cfg: SampleConfig, u: torch.Tensor, name: str = "") -> torch.Tensor:
    """Sample one token per row by inverse CDF with given uniforms u [B] (deterministic given u).

    Default: temperature + top-p. With renyi_alpha: the support size is the exponential of the
    Renyi entropy H_a = log(sum p^a) / (1 - a) of the (temperature-scaled) distribution, i.e. an
    "effective number of candidates"; the top k are kept and sampled uniformly (as in POP 2).
    A peaked distribution keeps 1-2 candidates, a flat one many.
    """
    temp = (cfg.attr_temperature or {}).get(name, cfg.temperature)
    if temp <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temp, -1)
    sp, si = probs.sort(-1, descending=True)
    if cfg.renyi_alpha is not None:
        a = cfg.renyi_alpha
        h = torch.log((sp.clamp(min=1e-30) ** a).sum(-1)) / (1 - a)
        legal = (sp > 0).sum(-1)
        k = torch.minimum(torch.exp(h).round().clamp(min=1), legal.float()).long()
        keep = torch.arange(sp.shape[-1], device=sp.device)[None] < k[:, None]
        sp = keep.float()                              # uniform inside the support
    else:
        cum = sp.cumsum(-1)
        keep = (cum - sp) < cfg.top_p                  # always keeps the top token
        sp = sp * keep
    cdf = sp.cumsum(-1) / sp.sum(-1, keepdim=True)
    j = torch.searchsorted(cdf, u.unsqueeze(-1).clamp(max=1 - 1e-7)).clamp(max=cdf.shape[-1] - 1)
    return si.gather(-1, j).squeeze(-1)


@torch.no_grad()
def generate(model: Painter, batch: int = 1, cfg: SampleConfig | None = None, text=None, text_valid=None,
             generator: torch.Generator | None = None, uniforms=None, codec=None,
             canvas_history: list | None = None, image=None, image_valid=None) -> list[DrawSeq]:
    """Generate `batch` drawing sequences from BOS (optionally text- and/or image-conditioned).

    image/image_valid: CLIP image tokens [B, Li, D] for image-conditioned models; guidance (cfg_scale)
    contrasts with and without the text only, both halves keep the image.

    uniforms: optional callable (step, field) -> [B] tensor in [0, 1) replacing the RNG.
    codec: required for models with canvas feedback (decodes primitives to render the canvas).
    canvas_history: if a list is given, it receives each sample's canvases at every checkpoint.
    """
    cfg = cfg or SampleConfig()
    model.eval()
    proto, dev = model.proto, next(model.parameters()).device
    B, A = batch, len(model.attr_names)
    Lt = 0 if text is None else text.shape[1]
    guided = text is not None and cfg.cfg_scale != 1.0
    if guided:   # rows B..2B-1: same records, no text (all text slots invalid)
        text = torch.cat([text, torch.zeros_like(text)])
        text_valid = torch.cat([text_valid, torch.zeros_like(text_valid)])
        if image is not None:
            image, image_valid = torch.cat([image, image]), torch.cat([image_valid, image_valid])
    if image is not None and image.shape[1] > 0:
        if not model.cfg.image:
            raise ValueError("image tokens given to a model without image conditioning")
        Lt += image.shape[1]               # the prefix is [image tokens][text tokens]
    R = 2 * B if guided else B           # rows run through the network
    max_records = cfg.max_records or model.cfg.max_positions - Lt
    cache = model.new_cache(R, dev, max_len=Lt + max_records)

    def guide(logits: torch.Tensor) -> torch.Tensor:
        if not guided:
            return logits.float()
        c, u = logits.float().chunk(2)
        return u + cfg.cfg_scale * (c - u)

    def rows(t: torch.Tensor) -> torch.Tensor:   # duplicate per-sample inputs for both halves
        return torch.cat([t, t]) if guided else t
    rand = uniforms or (lambda step, field: torch.rand(B, device=dev, generator=generator))
    prim_ev = proto.event_id("PRIMITIVE")

    grammars = [Grammar(proto, require_before_eos=frozenset(cfg.require_before_eos), allow_special=False)
                for _ in range(B)]
    for g in grammars:
        g.bos()
    rec_kind = torch.full((B, 1), KIND_BOS, device=dev)
    rec_event = torch.full((B, 1), -1, device=dev)
    rec_layer = torch.full((B, 1), -1, device=dev)
    rec_attrs = torch.full((B, 1, A), NO_ATTR, device=dev)
    x = model.embed_records(rows(rec_kind), rows(rec_event), rows(rec_layer), rows(rec_attrs))
    valid = torch.ones(R, 1, dtype=torch.bool, device=dev)
    prefix, prefix_valid = model.embed_prefix({"text": text, "text_valid": text_valid,
                                               "image": image, "image_valid": image_valid})
    if prefix is not None:
        x = torch.cat([prefix, x], 1)
        valid = torch.cat([prefix_valid, valid], 1)
    tracker = None
    if model.cfg.canvas:
        if codec is None:
            raise ValueError("a canvas-feedback model needs `codec` to render its canvas")
        tracker = CanvasTracker(model, codec, B, dev, record=canvas_history is not None)
        draw = torch.zeros(R, valid.shape[1], dtype=torch.bool, device=dev)
        draw[:, -1] = True                                  # the BOS sees the blank canvas; text does not
        ctx = ("current", rows(tracker.tokens), draw)
    else:
        ctx = None
    h = model.backbone(x, valid, cache, canvas_ctx=ctx)[:, -1]

    out = {k: [v[:, 0]] for k, v in (("kind", rec_kind), ("event", rec_event), ("layer", rec_layer),
                                      ("attrs", rec_attrs))}
    done_cpu = [False] * B
    n_events, eos = len(proto.events), proto.event_id("EOS")
    for step in range(1, max_records):
        # event, masked by each sequence's grammar
        dh = model.decoder(h, torch.zeros(R, dtype=torch.long, device=dev),
                           torch.zeros(R, A, dtype=torch.long, device=dev), steps=1)
        logits = guide(model.heads["event"](dh[:, 0]))
        # grammar masks are built on the CPU and copied once: per-sample GPU masks cost a
        # host<->device round trip each (38% of decoding time at batch 8)
        allowed_cpu = torch.zeros(B, n_events, dtype=torch.bool)
        if step == max_records - 1:      # out of room: force the end (even if constraints are unmet)
            allowed_cpu[:, eos] = True
        else:
            for b, g in enumerate(grammars):
                allowed_cpu[b, [eos] if g.done else g.allowed_events()] = True
        allowed = allowed_cpu.to(dev, non_blocking=True)
        event = _sample(logits.masked_fill(~allowed, float("-inf")), cfg, rand(step, "event"), "event")
        ev = event.tolist()              # the one device->host sync for this record

        attrs = torch.full((B, A), NO_ATTR, dtype=torch.long, device=dev)
        is_prim = event == prim_ev
        if prim_ev in ev:
            for i, name in enumerate(model.attr_names):
                dh = model.decoder(h, rows(event), rows(attrs.clamp(min=0)), steps=2 + i)
                attrs[:, i] = _sample(guide(model.heads[name](dh[:, -1])), cfg, rand(step, name), name)
            attrs[~is_prim] = NO_ATTR

        layer_cpu = []
        for b, g in enumerate(grammars):
            if g.done:
                layer_cpu.append(-1)
                continue
            if step == max_records - 1 and ev[b] == eos:   # forced end: may break decoding-only EOS rules
                g.done = True
            else:
                g.step(ev[b])
            layer_cpu.append(g.current if not g.done else -1)
        layer = torch.tensor(layer_cpu, device=dev)
        kind_cpu = [KIND_PRIMITIVE if e == prim_ev else KIND_CONTROL for e in ev]
        kind = torch.where(is_prim, KIND_PRIMITIVE, KIND_CONTROL)

        live_cpu = [not d for d in done_cpu]
        live = torch.tensor(live_cpu, device=dev)
        for k, v in (("kind", kind), ("event", event), ("layer", layer), ("attrs", attrs)):
            out[k].append(v)
        done_cpu = [d or e == eos for d, e in zip(done_cpu, ev)]
        if all(done_cpu):
            break
        x = model.embed_records(rows(kind)[:, None], rows(event)[:, None], rows(layer)[:, None],
                                rows(attrs)[:, None])
        if tracker is not None:
            tracker.update(kind, attrs, layer, live, kind_cpu=kind_cpu, layer_cpu=layer_cpu, live_cpu=live_cpu)
            ctx = ("current", rows(tracker.tokens), torch.ones(R, 1, dtype=torch.bool, device=dev))
        h = model.backbone(x, rows(live)[:, None], cache, canvas_ctx=ctx)[:, -1]

    if tracker is not None and canvas_history is not None:
        canvas_history.extend(tracker.history)
    kinds = torch.stack(out["kind"], 1)
    events = torch.stack(out["event"], 1)
    layers = torch.stack(out["layer"], 1)
    attrs_all = torch.stack(out["attrs"], 1)
    seqs = []
    eos = proto.event_id("EOS")
    for b in range(B):
        end = int((events[b] == eos).nonzero()[0]) + 1 if (events[b] == eos).any() else events.shape[1]
        seqs.append(DrawSeq(kinds[b, :end].cpu(), events[b, :end].cpu(), attrs_all[b, :end].cpu(),
                            layers[b, :end].cpu()))
    return seqs


__all__ = ["SampleConfig", "generate"]
