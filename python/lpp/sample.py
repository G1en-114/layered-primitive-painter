"""Reference sampler: grammar-constrained autoregressive generation with a KV cache.

This is the Python oracle for the C++ runtime: same loop, same masks, and the random
numbers can be supplied from outside (`uniforms`), so both implementations can be
made to produce bit-identical sequences.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Painter
from .sequence import KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE, NO_ATTR, DrawSeq, Grammar


@dataclass
class SampleConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    cfg_scale: float = 1.0             # >1: classifier-free guidance (needs text)
    max_records: int | None = None     # drawing records incl. BOS; default: context minus text


def _sample(logits: torch.Tensor, cfg: SampleConfig, u: torch.Tensor) -> torch.Tensor:
    """Top-p sampling by inverse CDF with given uniforms u [B] (deterministic given u)."""
    if cfg.temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / cfg.temperature, -1)
    sp, si = probs.sort(-1, descending=True)
    cum = sp.cumsum(-1)
    keep = (cum - sp) < cfg.top_p                     # always keeps the top token
    sp = sp * keep
    cdf = sp.cumsum(-1) / sp.sum(-1, keepdim=True)
    j = torch.searchsorted(cdf, u.unsqueeze(-1).clamp(max=1 - 1e-7)).clamp(max=cdf.shape[-1] - 1)
    return si.gather(-1, j).squeeze(-1)


@torch.no_grad()
def generate(model: Painter, batch: int = 1, cfg: SampleConfig | None = None, text=None, text_valid=None,
             generator: torch.Generator | None = None, uniforms=None) -> list[DrawSeq]:
    """Generate `batch` drawing sequences from BOS (optionally text-conditioned).

    uniforms: optional callable (step, field) -> [B] tensor in [0, 1) replacing the RNG.
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

    grammars = [Grammar(proto) for _ in range(B)]
    for g in grammars:
        g.bos()
    rec_kind = torch.full((B, 1), KIND_BOS, device=dev)
    rec_event = torch.full((B, 1), -1, device=dev)
    rec_layer = torch.full((B, 1), -1, device=dev)
    rec_attrs = torch.full((B, 1, A), NO_ATTR, device=dev)
    x = model.embed_records(rows(rec_kind), rows(rec_event), rows(rec_layer), rows(rec_attrs))
    valid = torch.ones(R, 1, dtype=torch.bool, device=dev)
    if Lt:
        x = torch.cat([model.embed_text(text), x], 1)
        valid = torch.cat([text_valid, valid], 1)
    h = model.backbone(x, valid, cache)[:, -1]

    out = {k: [v[:, 0]] for k, v in (("kind", rec_kind), ("event", rec_event), ("layer", rec_layer),
                                      ("attrs", rec_attrs))}
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(1, max_records):
        # event, masked by each sequence's grammar
        dh = model.decoder(h, torch.zeros(R, dtype=torch.long, device=dev),
                           torch.zeros(R, A, dtype=torch.long, device=dev), steps=1)
        logits = guide(model.heads["event"](dh[:, 0]))
        allowed = torch.stack([g.allowed_mask(dev) if not g.done else
                               torch.nn.functional.one_hot(torch.tensor(proto.event_id("EOS")),
                                                           len(proto.events)).bool().to(dev)
                               for g in grammars])
        if step == max_records - 1:      # out of room: force the end
            allowed = allowed & (torch.arange(len(proto.events), device=dev) == proto.event_id("EOS"))
        event = _sample(logits.masked_fill(~allowed, float("-inf")), cfg, rand(step, "event"))

        attrs = torch.full((B, A), NO_ATTR, dtype=torch.long, device=dev)
        is_prim = event == prim_ev
        if is_prim.any():
            for i, name in enumerate(model.attr_names):
                dh = model.decoder(h, rows(event), rows(attrs.clamp(min=0)), steps=2 + i)
                attrs[:, i] = _sample(guide(model.heads[name](dh[:, -1])), cfg, rand(step, name))
            attrs[~is_prim] = NO_ATTR

        layer = torch.empty(B, dtype=torch.long, device=dev)
        for b, g in enumerate(grammars):
            if g.done:
                layer[b] = -1
                continue
            g.step(int(event[b]))
            layer[b] = g.current if not g.done else -1
        kind = torch.where(is_prim, KIND_PRIMITIVE, KIND_CONTROL)

        live = ~done
        for k, v in (("kind", kind), ("event", event), ("layer", layer), ("attrs", attrs)):
            out[k].append(v)
        done = done | (event == proto.event_id("EOS"))
        if bool(done.all()):
            break
        x = model.embed_records(rows(kind)[:, None], rows(event)[:, None], rows(layer)[:, None],
                                rows(attrs)[:, None])
        h = model.backbone(x, rows(live)[:, None], cache)[:, -1]

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
