"""Export-friendly views of a trained Painter: the graphs a C++ runtime calls in its decode loop.

The training module keeps a Python KV cache object and dispatches on optional inputs; ONNX needs
pure tensor-in / tensor-out functions with static structure. Each wrapper below re-expresses one
piece of Painter.forward / generate with the SAME parameters, and tests check it numerically
against the original (tests/test_export.py):

  embed_record  kind/event/layer [B,T], attrs [B,T,A]           -> x [B,T,d]
  embed_text    text features [B,Lt,Dt]                           -> x [B,Lt,d]
  chunk         x [B,T,d], valid [B,T], past_k/past_v [L,B,kv,P,hd], past_mask [B,P],
                start_pos [B], canvas_tokens [B,n,d], draw [B,T]
                -> h [B,T,d] (final-norm hidden), k_new / v_new [L,B,kv,T,hd]
                The runtime owns the cache: it writes k_new/v_new at its own write offset and sets
                past_mask; attention keys are [past ; new], order-free because RoPE is already
                applied to stored keys.
  decode_heads  h [B,d], event [B], attrs [B,A]                   -> event logits, one logits tensor per attribute
                The attribute decoder is causal, so one full-length pass gives position i exactly
                what the sampler's truncated pass gives; the runtime samples attribute i from
                output i, writes it into attrs and calls again (A + 1 calls per primitive).
  canvas_enc    canvas [B, 3+L, S, S]                             -> canvas tokens [B,n,d]

Attention is written out (GQA by repeating K/V heads, matmul + softmax) instead of
scaled_dot_product_attention(enable_gqa=...), so the graphs export with plain ONNX operators.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import Painter


def _attend(q, k, v, mask):
    """q [B,H,Tq,hd], k/v [B,Hkv,Tk,hd], mask [B,1,Tq,Tk] bool -> [B,H,Tq,hd]."""
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v)


class EmbedRecord(nn.Module):
    def __init__(self, painter: Painter):
        super().__init__()
        self.p = painter

    def forward(self, kind, event, layer, attrs):
        return self.p.embed_records(kind, event, layer, attrs)


class EmbedText(nn.Module):
    def __init__(self, painter: Painter):
        super().__init__()
        self.p = painter

    def forward(self, text):
        return self.p.embed_text(text)


class CanvasEnc(nn.Module):
    def __init__(self, painter: Painter):
        super().__init__()
        self.p = painter

    def forward(self, canvas):
        return self.p.canvas_enc(canvas)


class Chunk(nn.Module):
    """Backbone over T new positions given an explicit past cache (see module docstring)."""

    def __init__(self, painter: Painter):
        super().__init__()
        self.p = painter

    def forward(self, x, valid, past_k, past_v, past_mask, start_pos, canvas_tokens, draw):
        p = self.p
        B, T, _ = x.shape
        pos = start_pos[:, None] + (valid.long().cumsum(1) - 1).clamp(min=0)
        i = torch.arange(T, device=x.device)
        causal = (i[None, :] <= i[:, None])[None]                                  # [1,T,T]
        eye = (i[None, :] == i[:, None])[None]                                       # no bool EyeLike in ORT
        new_mask = (causal & valid[:, None, :]) | eye                              # [B,T,T]
        mask = torch.cat([past_mask[:, None, :].expand(B, T, past_mask.shape[1]), new_mask], -1)[:, None]
        ks, vs = [], []
        for li, blk in enumerate(p.blocks):
            a = blk.attn
            hn = blk.n1(x)
            q, k, v = a.qkv(hn).split([a.h * a.hd, a.kv * a.hd, a.kv * a.hd], dim=-1)
            q = a.rope(q.view(B, T, a.h, a.hd).transpose(1, 2), pos)
            k = a.rope(k.view(B, T, a.kv, a.hd).transpose(1, 2), pos)
            v = v.view(B, T, a.kv, a.hd).transpose(1, 2)
            ks.append(k)
            vs.append(v)
            kk = torch.cat([past_k[li], k], 2)
            vv = torch.cat([past_v[li], v], 2)
            out = _attend(q, kk, vv, mask)
            x = x + a.o(out.transpose(1, 2).reshape(B, T, a.h * a.hd))
            x = x + blk.ff(blk.n2(x))
            if p.cfg.canvas and str(li) in p.cross:
                c = p.cross[str(li)]
                N, Lq, d = x.shape
                qc = c.q(c.norm_q(x)).view(N, Lq, c.h, c.hd).transpose(1, 2)
                kc, vc = c.kv(c.norm_kv(canvas_tokens)).view(N, -1, 2, c.h, c.hd).permute(2, 0, 3, 1, 4)
                full = torch.ones(N, 1, Lq, kc.shape[2], dtype=torch.bool, device=x.device)
                oc = c.o(_attend(qc, kc, vc, full).transpose(1, 2).reshape(N, Lq, d))
                x = x + torch.tanh(c.gate) * oc * draw.unsqueeze(-1).to(x.dtype)
        return p.norm(x), torch.stack(ks), torch.stack(vs)


class ChunkCached(nn.Module):
    """Chunk with the cache update inside the graph (runtime v1: the cache never leaves the GPU).

    cache_k/cache_v [L,B,kv,Tmax,hd] and cache_mask [B,Tmax] are the whole static cache; write_pos
    (0-d int64) is where the T new positions go. Outputs h and the updated cache; the runtime binds
    two device buffers and swaps them each call (ping-pong), so no host copy is involved."""

    def __init__(self, painter: Painter):
        super().__init__()
        self.chunk = Chunk(painter)

    def forward(self, x, valid, cache_k, cache_v, cache_mask, write_pos, start_pos, canvas_tokens, draw):
        T = x.shape[1]
        h, k_new, v_new = self.chunk(x, valid, cache_k, cache_v, cache_mask, start_pos, canvas_tokens, draw)
        end = write_pos + T
        ck = torch.cat([cache_k[:, :, :, :write_pos], k_new, cache_k[:, :, :, end:]], 3)
        cv = torch.cat([cache_v[:, :, :, :write_pos], v_new, cache_v[:, :, :, end:]], 3)
        cm = torch.cat([cache_mask[:, :write_pos], valid, cache_mask[:, end:]], 1)
        return h, ck, cv, cm


class DecodeHeads(nn.Module):
    """Full-length attribute decoder pass -> logits for the event and every attribute."""

    def __init__(self, painter: Painter):
        super().__init__()
        self.p = painter

    def forward(self, h, event, attrs):
        p = self.p
        toks = [p.dec_in(h), p.dec_event_emb(event.clamp(min=0))]
        for i, name in enumerate(p.attr_names[:-1]):
            toks.append(p.dec_attr_emb(name, attrs[:, i].clamp(min=0)))
        x = torch.stack(toks, 1) + p.dec_step
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
        for blk in p.dec_blocks:
            a = blk.attn
            hn = blk.n1(x)
            M = x.shape[0]
            q, k, v = a.qkv(hn).split([a.h * a.hd, a.kv * a.hd, a.kv * a.hd], dim=-1)
            q = q.view(M, T, a.h, a.hd).transpose(1, 2)
            k = k.view(M, T, a.kv, a.hd).transpose(1, 2)
            v = v.view(M, T, a.kv, a.hd).transpose(1, 2)
            out = _attend(q, k, v, mask)
            x = x + a.o(out.transpose(1, 2).reshape(M, T, a.h * a.hd))
            x = x + blk.ff(blk.n2(x))
        x = p.dec_norm(x)
        outs = [p.heads["event"](x[:, 0])]
        for i, name in enumerate(p.attr_names):
            outs.append(p.heads[name](x[:, 1 + i]))
        return tuple(outs)


class ManualRMSNorm(nn.Module):
    """nn.RMSNorm with plain operators (ONNX RMSNormalization needs opset 23; exporters target less).
    Same weight, same eps rule as nn.RMSNorm (finfo(dtype).eps when eps is None)."""

    def __init__(self, ref: nn.RMSNorm):
        super().__init__()
        self.weight, self.eps = ref.weight, ref.eps

    def forward(self, x):
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight


def exportable(painter: Painter) -> Painter:
    """A copy of the model with every nn.RMSNorm replaced by ManualRMSNorm (shared values)."""
    import copy
    m = copy.deepcopy(painter)
    for parent in list(m.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.RMSNorm):
                setattr(parent, name, ManualRMSNorm(child))
    return m


def graphs(painter: Painter) -> dict[str, nn.Module]:
    painter = exportable(painter)
    g = {"embed_record": EmbedRecord(painter), "embed_text": EmbedText(painter), "chunk": Chunk(painter),
         "chunk_cached": ChunkCached(painter),
         "decode_heads": DecodeHeads(painter)}
    if painter.cfg.canvas:
        g["canvas_enc"] = CanvasEnc(painter)
    return g


__all__ = ["Chunk", "ChunkCached", "DecodeHeads", "EmbedRecord", "EmbedText", "CanvasEnc", "graphs"]
