"""Two-level autoregressive painter.

    backbone      decoder-only transformer over records: [text prefix] BOS LAYER P P ... EOS.
                  One position per record; a primitive's attribute embeddings are summed.
                  GQA, RoPE, SwiGLU, RMSNorm, static KV cache (the export target).
    attr decoder  tiny causal transformer run per backbone position t. Input
                  [proj(h_t), e(event), e(x), e(y), e(shape), e(color)], output at step s
                  predicts the s-th field of record t+1:  event, x, y, shape, color, alpha.

Why two levels: the backbone keeps one position per primitive (5x shorter than one
position per attribute) while the attributes of one primitive still depend on each
other; and both halves are plain transformers that export to standard ONNX graphs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import Codec
from .protocol import CodebookAttr, PeriodicAttr, Protocol
from .sequence import KIND_PRIMITIVE

N_KINDS = 4          # TEXT, BOS, CONTROL, PRIMITIVE
SOFT_LABEL_SIGMA = {"alpha": 0.5, "beta": 0.5}   # in bins; other ordinal attributes use 1.0


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    d_dec: int = 512
    dec_layers: int = 2
    dec_heads: int = 8
    dec_ff: int = 1408
    text_dim: int = 768
    max_positions: int = 1024
    rope_base: float = 10000.0
    fourier_freqs: int = 8
    # canvas feedback (docs/canvas_feedback_design.md): a small CNN encodes the canvas rendered at
    # checkpoints; gated cross-attention after every `canvas_every` backbone blocks reads it
    canvas: bool = False
    canvas_k: int = 32          # a checkpoint every k primitives (and at every control record)
    canvas_size: int = 64
    canvas_every: int = 4
    canvas_width: int = 64
    # image conditioning (image -> layered primitives, editing): frozen CLIP image tokens as a
    # prefix before the text, see tools/extract_image_features.py
    image: bool = False
    image_dim: int = 768

    @classmethod
    def tiny(cls) -> "ModelConfig":
        return cls(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128, d_dec=64, dec_layers=1,
                   dec_heads=4, dec_ff=128, text_dim=32)

    @classmethod
    def small(cls) -> "ModelConfig":   # ~20M, for the scaling sweep
        return cls(d_model=384, n_layers=8, n_heads=6, n_kv_heads=2, d_ff=1024, d_dec=256, dec_ff=704)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- building blocks
class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_pos: int, base: float):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_pos).float(), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x [B, H, T, hd], pos [B, T] -> rotated x (half-split layout)."""
        cos, sin = self.cos[pos].unsqueeze(1).to(x.dtype), self.sin[pos].unsqueeze(1).to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class KVCache:
    """Static per-layer K/V buffers [B, n_kv, max_len, hd] plus key validity."""

    def __init__(self, n_layers, B, n_kv, max_len, hd, device, dtype):
        self.k = torch.zeros(n_layers, B, n_kv, max_len, hd, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.valid = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        self.next_pos = torch.zeros(B, dtype=torch.long, device=device)   # next RoPE position
        self.length = 0
        self.max_len = max_len


class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv, rope: Rotary | None):
        super().__init__()
        self.h, self.kv, self.hd = n_heads, n_kv, d // n_heads
        self.qkv = nn.Linear(d, (n_heads + 2 * n_kv) * self.hd, bias=False)
        self.o = nn.Linear(n_heads * self.hd, d, bias=False)
        self.rope = rope

    def forward(self, x, pos, mask, cache: KVCache | None = None, layer: int = 0):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split([self.h * self.hd, self.kv * self.hd, self.kv * self.hd], dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.kv, self.hd).transpose(1, 2)
        v = v.view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        if cache is not None:
            s = cache.length
            cache.k[layer, :, :, s : s + T] = k
            cache.v[layer, :, :, s : s + T] = v
            k, v = cache.k[layer, :, :, : s + T], cache.v[layer, :, :, : s + T]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.kv != self.h)
        return self.o(out.transpose(1, 2).reshape(B, T, self.h * self.hd))


class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv, d_ff, rope):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = Attention(d, n_heads, n_kv, rope)
        self.ff = SwiGLU(d, d_ff)

    def forward(self, x, pos, mask, cache=None, layer=0):
        x = x + self.attn(self.n1(x), pos, mask, cache, layer)
        return x + self.ff(self.n2(x))


class CanvasEncoder(nn.Module):
    """[N, C, S, S] canvas -> [N, (S/8)^2, d] tokens with a learned 2D position embedding."""

    def __init__(self, in_ch: int, d: int, size: int, width: int):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(4 * w, d, 1),
        )
        self.n_tokens = (size // 8) ** 2
        self.pos = nn.Parameter(torch.zeros(self.n_tokens, d))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, canvas):
        return self.net(canvas).flatten(2).transpose(1, 2) + self.pos


class CanvasCross(nn.Module):
    """Gated cross-attention from sequence positions to the canvas tokens of their checkpoint.

    The gate starts at 0, so a model with canvas feedback starts out exactly equal to one
    without it (and can be initialised from such a checkpoint)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h, self.hd = n_heads, d // n_heads
        self.norm_q, self.norm_kv = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def attend(self, q_in, tokens):
        """q_in [N, Lq, d], tokens [N, Lk, d] -> [N, Lq, d]."""
        N, Lq, d = q_in.shape
        q = self.q(self.norm_q(q_in)).view(N, Lq, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(self.norm_kv(tokens)).view(N, -1, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(N, Lq, d))

    def forward(self, x, ctx):
        """ctx = ("segments", tokens [B, S, n, d], seg [B, T]) for teacher-forced training, or
        ("current", tokens [B, n, d], draw [B, T]) for decoding (every drawing position of this
        call sees its row's current canvas)."""
        mode = ctx[0]
        if mode == "current":
            _, tokens, draw = ctx
            out = self.attend(x, tokens) * draw.unsqueeze(-1)
            return x + torch.tanh(self.gate) * out
        _, tokens, seg = ctx
        B, T, d = x.shape
        S = tokens.shape[1]
        has = seg >= 0
        # group positions by the checkpoint they see: [B*S, Lmax, d]
        # running max keeps the index non-decreasing (searchsorted needs that): leading text
        # positions stay 0, trailing padding repeats the last checkpoint; both are masked out
        segc = torch.cummax(seg.clamp(min=0), dim=1).values
        first = torch.stack([torch.searchsorted(segc[b].contiguous(), segc[b].contiguous()) for b in range(B)])
        off = torch.arange(T, device=x.device)[None] - first
        Lmax = int(off[has].max()) + 1 if bool(has.any()) else 1
        flat = (torch.arange(B, device=x.device)[:, None] * S + segc) * Lmax + off
        grouped = x.new_zeros(B * S * Lmax, d)
        grouped[flat[has]] = x[has]
        out = self.attend(grouped.view(B * S, Lmax, d), tokens.reshape(B * S, -1, d)).view(-1, d)
        delta = out.new_zeros(B, T, d)          # attention output dtype (bf16 under autocast)
        delta[has] = out[flat[has]]
        return x + torch.tanh(self.gate) * delta


class Fourier(nn.Module):
    """Ordinal token -> sin/cos features of its normalized value -> Linear.

    Periodic attributes (angles) use integer frequencies on v = idx / bins, so the
    first and last bins are neighbours in feature space."""

    def __init__(self, bins: int, d: int, n_freqs: int, periodic: bool = False):
        super().__init__()
        self.bins, self.periodic = bins, periodic
        freqs = (2 * math.pi * torch.arange(1, n_freqs + 1).float() if periodic
                 else math.pi * 2.0 ** torch.arange(n_freqs).float())
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(2 * n_freqs, d)

    def forward(self, idx):
        v = idx.float().unsqueeze(-1) / (self.bins if self.periodic else max(self.bins - 1, 1))
        ang = v * self.freqs
        return self.proj(torch.cat([ang.sin(), ang.cos()], dim=-1))


class CodeEmbed(nn.Module):
    """Codebook token -> learned row + Linear(code features): similar codes start similar."""

    def __init__(self, features: torch.Tensor, d: int):
        super().__init__()
        self.register_buffer("features", features.float(), persistent=False)
        self.table = nn.Embedding(features.shape[0], d)
        self.proj = nn.Linear(features.shape[1], d)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, idx):
        return self.table(idx) + self.proj(self.features[idx])


class AttrEmbed(nn.Module):
    """Embeddings for the five attribute tokens, in the given width."""

    def __init__(self, proto: Protocol, code_features: dict[str, torch.Tensor], d: int, n_freqs: int):
        super().__init__()
        mods = {}
        for a in proto.attributes:
            mods[a.name] = (CodeEmbed(code_features[a.name], d) if isinstance(a, CodebookAttr)
                            else Fourier(a.bins, d, n_freqs, periodic=isinstance(a, PeriodicAttr)))
        self.mods = nn.ModuleDict(mods)

    def forward(self, name: str, idx):
        return self.mods[name](idx)


def code_features(codec: Codec) -> dict[str, torch.Tensor]:
    """color code -> Oklab (3); v1 shape code -> (log-cov (3), beta one-hot)."""
    feats = {"color": codec.color_codes.cpu()}
    if codec.shape_codes is not None:
        nb = len(codec.proto.beta_levels)
        per_beta = codec.shape_codes.shape[0]
        feats["shape"] = torch.cat([codec.shape_codes.repeat(nb, 1).cpu(),
                                    torch.eye(nb).repeat_interleave(per_beta, 0)], dim=1)
    return feats


# --------------------------------------------------------------------------- the model
class Painter(nn.Module):
    def __init__(self, cfg: ModelConfig, proto: Protocol, codec: Codec):
        super().__init__()
        self.cfg, self.proto = cfg, proto
        self.n_events = len(proto.events)
        self.vocab = proto.attr_vocab_sizes
        self.attr_names = proto.attr_names
        self.dec_steps = 1 + len(self.attr_names)
        feats = code_features(codec)
        d, dd = cfg.d_model, cfg.d_dec

        # backbone input
        self.kind_emb = nn.Embedding(N_KINDS, d)
        self.event_emb = nn.Embedding(self.n_events + 1, d)          # +1: "no event" (BOS / text)
        self.layer_emb = nn.Embedding(proto.num_layers + 1, d)       # +1: "no layer"
        self.attr_emb = AttrEmbed(proto, feats, d, cfg.fourier_freqs)
        self.text_proj = nn.Linear(cfg.text_dim, d)
        if cfg.image:
            self.image_proj = nn.Linear(cfg.image_dim, d)
            self.image_type = nn.Parameter(torch.zeros(d))
        rope = Rotary(d // cfg.n_heads, cfg.max_positions, cfg.rope_base)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff, rope)
                                    for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(d)
        if cfg.canvas:
            self.canvas_enc = CanvasEncoder(3 + proto.num_layers, d, cfg.canvas_size, cfg.canvas_width)
            self.cross = nn.ModuleDict({str(li): CanvasCross(d, cfg.n_heads) for li in range(cfg.n_layers)
                                        if (li + 1) % cfg.canvas_every == 0})

        # attribute decoder
        self.dec_in = nn.Linear(d, dd)
        self.dec_event_emb = nn.Embedding(self.n_events, dd)
        self.dec_attr_emb = AttrEmbed(proto, feats, dd, cfg.fourier_freqs)
        self.dec_step = nn.Parameter(torch.zeros(self.dec_steps, dd))
        self.dec_blocks = nn.ModuleList(Block(dd, cfg.dec_heads, cfg.dec_heads, cfg.dec_ff, None)
                                        for _ in range(cfg.dec_layers))
        self.dec_norm = nn.RMSNorm(dd)
        self.heads = nn.ModuleDict({"event": nn.Linear(dd, self.n_events)} |
                                   {n: nn.Linear(dd, self.vocab[n]) for n in self.attr_names})
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- backbone ------------------------------------------------------
    def embed_records(self, kind, event, layer, attrs):
        """kind/event/layer [B, T], attrs [B, T, A] (-1 = none) -> [B, T, d]."""
        x = self.kind_emb(kind) + self.event_emb(event + 1) + self.layer_emb(layer + 1)
        prim = (kind == KIND_PRIMITIVE).unsqueeze(-1)
        for i, name in enumerate(self.attr_names):
            x = x + self.attr_emb(name, attrs[..., i].clamp(min=0)) * prim
        return x

    def embed_text(self, text):
        return self.text_proj(text) + self.kind_emb.weight[0]

    def embed_image(self, image):
        return self.image_proj(image) + self.image_type

    def embed_prefix(self, batch: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Condition prefix [image tokens][text tokens] -> (x [B, L, d], valid [B, L]) or (None, None)."""
        parts, valids = [], []
        if self.cfg.image and batch.get("image") is not None and batch["image"].shape[1] > 0:
            parts.append(self.embed_image(batch["image"]))
            valids.append(batch["image_valid"])
        if batch.get("text") is not None and batch["text"].shape[1] > 0:
            parts.append(self.embed_text(batch["text"]))
            valids.append(batch["text_valid"])
        if not parts:
            return None, None
        return torch.cat(parts, 1), torch.cat(valids, 1)

    def backbone(self, x, valid, cache: KVCache | None = None, canvas_ctx=None):
        """x [B, T, d], valid [B, T] -> hidden [B, T, d]. With a cache, x continues it.
        canvas_ctx: see CanvasCross.forward (None: no canvas feedback)."""
        B, T, _ = x.shape
        if cache is None:
            pos = (valid.long().cumsum(1) - 1).clamp(min=0)
            i = torch.arange(T, device=x.device)
            causal = i[None, :] <= i[:, None]
            mask = (causal & valid[:, None, :]) | torch.eye(T, dtype=torch.bool, device=x.device)
        else:
            s = cache.length
            pos = cache.next_pos[:, None] + (valid.long().cumsum(1) - 1).clamp(min=0)
            cache.valid[:, s : s + T] = valid
            q = s + torch.arange(T, device=x.device)
            k = torch.arange(s + T, device=x.device)
            mask = ((k[None, :] <= q[:, None]) & cache.valid[:, None, : s + T]) | (k[None, :] == q[:, None])
        mask = mask.unsqueeze(1)
        for li, blk in enumerate(self.blocks):
            x = blk(x, pos, mask, cache, li)
            if canvas_ctx is not None and str(li) in getattr(self, "cross", {}):
                x = self.cross[str(li)](x, canvas_ctx)
        if cache is not None:
            cache.length += T
            cache.next_pos += valid.long().sum(1)
        return self.norm(x)

    def new_cache(self, B, device, dtype=torch.float32, max_len=None) -> KVCache:
        c = self.cfg
        return KVCache(c.n_layers, B, c.n_kv_heads, max_len or c.max_positions, c.d_model // c.n_heads,
                       device, dtype)

    # ---- attribute decoder ----------------------------------------------
    def decoder(self, h, event, attrs, steps: int | None = None):
        """h [M, d]; event [M]; attrs [M, A] (teacher inputs) -> hidden [M, steps, d_dec]."""
        steps = steps or self.dec_steps
        toks = [self.dec_in(h)]
        if steps > 1:
            toks.append(self.dec_event_emb(event.clamp(min=0)))
        for i, name in enumerate(self.attr_names[: steps - 2]):
            toks.append(self.dec_attr_emb(name, attrs[:, i].clamp(min=0)))
        x = torch.stack(toks, 1) + self.dec_step[:steps]
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        for blk in self.dec_blocks:
            x = blk(x, None, mask)
        return self.dec_norm(x)

    # ---- training ---------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """Teacher-forced losses. batch from data.collate()."""
        h = self._hidden(batch)
        tgt_event, tgt_attrs = batch["tgt_event"], batch["tgt_attrs"]
        sel = tgt_event >= 0
        hs, ev, at = h[sel], tgt_event[sel], tgt_attrs[sel]
        out = self.decoder(hs, ev, at)
        losses = {"event": F.cross_entropy(self.heads["event"](out[:, 0]).float(), ev)}
        prim = ev == self.proto.event_id("PRIMITIVE")
        for i, name in enumerate(self.attr_names):
            logits = self.heads[name](out[prim, 1 + i]).float()
            target = at[prim, i]
            attr = self.proto.attr(name)
            if isinstance(attr, CodebookAttr):
                losses[name] = F.cross_entropy(logits, target)
            else:
                losses[name] = soft_ce(logits, target, SOFT_LABEL_SIGMA.get(name, 1.0),
                                       periodic=isinstance(attr, PeriodicAttr))
        losses["loss"] = sum(losses.values())
        return losses

    def record_logprobs(self, batch: dict, event_allowed: torch.Tensor | None = None) -> torch.Tensor:
        """Exact log-probability of each target record under the model: [B, T] float32.

        log p(event) (+ sum of log p(attribute) for primitives) at every position with a
        target, 0 elsewhere. event_allowed [B, T, n_events] masks event logits to the
        grammar's choices, as the sampler does, so the values are the true sampling
        log-probabilities at temperature 1 without truncation or guidance (RL rollouts).
        """
        h = self._hidden(batch)
        tgt_event, tgt_attrs = batch["tgt_event"], batch["tgt_attrs"]
        sel = tgt_event >= 0
        hs, ev, at = h[sel], tgt_event[sel], tgt_attrs[sel]
        out = self.decoder(hs, ev, at)
        ev_logits = self.heads["event"](out[:, 0]).float()
        if event_allowed is not None:
            ev_logits = ev_logits.masked_fill(~event_allowed[sel], float("-inf"))
        lp = F.log_softmax(ev_logits, -1).gather(1, ev[:, None])[:, 0]
        prim = ev == self.proto.event_id("PRIMITIVE")
        attr_lp = torch.zeros_like(lp)
        for i, name in enumerate(self.attr_names):
            logits = self.heads[name](out[prim, 1 + i]).float()
            attr_lp[prim] += F.log_softmax(logits, -1).gather(1, at[prim, i][:, None])[:, 0]
        res = torch.zeros(tgt_event.shape, dtype=torch.float32, device=h.device)
        res[sel] = lp + attr_lp
        return res

    def _hidden(self, batch: dict) -> torch.Tensor:
        """Backbone states at the record positions [B, T, d] (image/text prefix and canvas included)."""
        x = self.embed_records(batch["kind"], batch["event"], batch["layer"], batch["attrs"])
        valid = batch["valid"]
        Lt = 0
        prefix, prefix_valid = self.embed_prefix(batch)
        if prefix is not None:
            Lt = prefix.shape[1]
            x = torch.cat([prefix, x], 1)
            valid = torch.cat([prefix_valid, valid], 1)
        canvas_ctx = None
        if self.cfg.canvas and batch.get("canvas") is not None:
            B, S = batch["canvas"].shape[:2]
            tokens = self.canvas_enc(batch["canvas"].flatten(0, 1)).view(B, S, -1, self.cfg.d_model)
            seg = batch["seg"]
            if Lt:
                seg = torch.cat([torch.full((B, Lt), -1, dtype=seg.dtype, device=seg.device), seg], 1)
            canvas_ctx = ("segments", tokens, seg)
        return self.backbone(x, valid, canvas_ctx=canvas_ctx)[:, Lt:]


def soft_ce(logits, target, sigma: float, periodic: bool = False):
    """Cross-entropy against a Gaussian over neighbouring bins (ordinal targets).
    Periodic attributes measure bin distance around the circle."""
    n = logits.shape[-1]
    bins = torch.arange(n, device=logits.device, dtype=torch.float32)
    dist = bins[None, :] - target[:, None].float()
    if periodic:
        dist = torch.remainder(dist + n / 2, n) - n / 2
    q = torch.exp(-0.5 * (dist / sigma) ** 2)
    q = q / q.sum(-1, keepdim=True)
    return -(q * F.log_softmax(logits, -1)).sum(-1).mean()
