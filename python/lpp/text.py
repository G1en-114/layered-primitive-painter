"""Chinese text conditioning: caption variants and a frozen Chinese-CLIP text encoder.

Captions per image (CaptionStore):
    long      VLM-written detailed description (tools/caption_vlm.py)
    short     VLM-written short description
    template  built from the 40 CelebA-HQ attributes (always available, no VLM needed)
Training draws one variant per example (pick), with a dropout to "no text" so the
model also learns the unconditional distribution needed for classifier-free guidance.

TextEncoder runs inside the training step (frozen, no grad): token-level hidden states
of the Chinese-CLIP text tower, at most `max_positions` per caption. Longer captions are
average-pooled along the sequence instead of truncated, so the end of a description
(background, clothing) is not dropped.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

# CelebA attribute -> Chinese phrase. Order = order in the template caption.
_ATTR_ZH = [
    ("Blond_Hair", "金发"), ("Black_Hair", "黑发"), ("Brown_Hair", "棕发"), ("Gray_Hair", "灰白头发"),
    ("Bald", "光头"), ("Wavy_Hair", "卷发"), ("Straight_Hair", "直发"), ("Bangs", "有刘海"),
    ("Receding_Hairline", "发际线较高"), ("Pale_Skin", "皮肤白皙"), ("Oval_Face", "鹅蛋脸"),
    ("Chubby", "脸型圆润"), ("Arched_Eyebrows", "弯眉"), ("Bushy_Eyebrows", "浓眉"),
    ("Narrow_Eyes", "细长眼睛"), ("Bags_Under_Eyes", "有眼袋"), ("Big_Nose", "大鼻子"),
    ("Pointy_Nose", "尖鼻子"), ("Big_Lips", "厚嘴唇"), ("Smiling", "微笑"),
    ("Mouth_Slightly_Open", "嘴微张"), ("Heavy_Makeup", "浓妆"), ("Wearing_Lipstick", "涂口红"),
    ("Rosy_Cheeks", "脸颊红润"), ("Mustache", "八字胡"), ("Goatee", "山羊胡"),
    ("5_o_Clock_Shadow", "胡茬"), ("Sideburns", "鬓角"), ("Eyeglasses", "戴眼镜"),
    ("Wearing_Hat", "戴帽子"), ("Wearing_Earrings", "戴耳环"), ("Wearing_Necklace", "戴项链"),
    ("Wearing_Necktie", "打领带"),
]


def mirror_caption(caption: str | None) -> str | None:
    """Swap explicit left/right directions for a horizontally mirrored image.

    Keep paired expressions such as 左右/左、右 unchanged. This is a deterministic
    transform of directional words, not an attempt to rewrite the scene description.
    """
    if caption is None:
        return None
    pairs = []

    def protect(match):
        pairs.append(match.group())
        return f"\ue000{len(pairs) - 1}\ue001"

    text = re.sub(r"左右|右左|左[、和及与/]右|右[、和及与/]左", protect, caption)
    text = re.sub(r"[左右](?=侧|边|眼|耳|脸|肩|手|上|下|转|倾)|(?<=[向朝往偏])[左右]",
                  lambda m: "右" if m.group() == "左" else "左", text)

    def english(match):
        word = "right" if match.group().lower() == "left" else "left"
        return word.upper() if match.group().isupper() else word.capitalize() if match.group()[0].isupper() else word

    text = re.sub(r"\b(?:left|right)\b", english, text, flags=re.IGNORECASE)
    for index, pair in enumerate(pairs):
        text = text.replace(f"\ue000{index}\ue001", pair)
    return text


def template_caption(attrs: dict[str, bool], rng: random.Random | None = None) -> str:
    """'金发、卷发、微笑、戴耳环的女性肖像' from CelebA attributes."""
    words = [zh for key, zh in _ATTR_ZH if attrs.get(key)]
    if rng is not None and len(words) > 3:      # vary the length: keep a random subset, in order
        keep = sorted(rng.sample(range(len(words)), rng.randint(3, len(words))))
        words = [words[i] for i in keep]
    who = "男性" if attrs.get("Male") else "女性"
    return ("、".join(words) + "的" if words else "") + who + "肖像"


class CaptionStore:
    def __init__(self, captions_jsonl: Path | None = None, attributes_json: Path | None = None):
        self.vlm: dict[int, dict[str, str]] = {}
        if captions_jsonl:
            for line in Path(captions_jsonl).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("short") and r.get("long"):
                        self.vlm[int(r["id"])] = {"short": r["short"], "long": r["long"]}
        self.attrs: dict[int, dict[str, bool]] = {}
        if attributes_json:
            a = json.loads(Path(attributes_json).read_text())
            self.attrs = {int(k): dict(zip(a["names"], v)) for k, v in a["values"].items()}

    def __len__(self) -> int:
        return len(set(self.vlm) | set(self.attrs))

    def pick(self, image_id: int, rng: random.Random, p_drop: float = 0.1,
             weights=(("long", 0.5), ("short", 0.3), ("template", 0.2))) -> str | None:
        """One caption variant for this example, or None (no text) with probability p_drop."""
        if rng.random() < p_drop:
            return None
        options = [(k, w) for k, w in weights
                   if (k == "template" and image_id in self.attrs) or (k != "template" and image_id in self.vlm)]
        if not options:
            return None
        u, acc = rng.random() * sum(w for _, w in options), 0.0
        for k, w in options:
            acc += w
            if u <= acc:
                break
        return template_caption(self.attrs[image_id], rng) if k == "template" else self.vlm[image_id][k]


class CachedTextEncoder:
    """Memoizes a frozen TextEncoder by caption string (training-time speedup).

    Captions repeat every epoch (VLM long/short, their mirrored forms), so running the
    ~100M-parameter frozen tower each step is wasted work. Features are stored on the CPU
    in bfloat16: the model's first use is text_proj under bf16 autocast, which casts its
    input to bf16 anyway, so the model sees the same values. Returned as float32 like the
    encoder. Least-recently-used entries are evicted beyond max_entries (random template
    captions would otherwise grow the cache without bound).
    """

    def __init__(self, encoder: "TextEncoder", max_entries: int = 250_000):
        from collections import OrderedDict
        self.encoder, self.max_entries = encoder, max_entries
        self.store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = self.misses = 0

    @property
    def dim(self) -> int:
        return self.encoder.dim

    def __call__(self, captions: list[str | None]) -> tuple[torch.Tensor, torch.Tensor]:
        dev = next(self.encoder.model.parameters()).device
        todo = list(dict.fromkeys(c for c in captions if c and c not in self.store))
        if todo:
            feats, valid = self.encoder(todo)
            for i, c in enumerate(todo):
                self.store[c] = feats[i, : int(valid[i].sum())].to("cpu", torch.bfloat16)
        rows = []
        for c in captions:
            if c:
                self.store.move_to_end(c)
                rows.append(self.store[c])
        while len(self.store) > self.max_entries:        # evict only after this batch is gathered
            self.store.popitem(last=False)
        self.misses += len(todo)
        self.hits += len(rows) - len(todo)
        B, D = len(captions), self.encoder.dim
        L = max((r.shape[0] for r in rows), default=0)
        out = torch.zeros(B, L, D, dtype=torch.bfloat16)
        valid = torch.zeros(B, L, dtype=torch.bool)
        j = 0
        for i, c in enumerate(captions):
            if c:
                n = rows[j].shape[0]
                out[i, :n], valid[i, :n] = rows[j], True
                j += 1
        return out.to(dev, non_blocking=True).float(), valid.to(dev, non_blocking=True)


class TextEncoder(torch.nn.Module):
    """Frozen Chinese-CLIP text tower -> per-token features [B, L<=max_positions, 768]."""

    def __init__(self, model_dir: str | Path, max_positions: int = 32, max_tokens: int | None = None):
        super().__init__()
        from transformers import BertTokenizer, ChineseCLIPModel
        self.tok = BertTokenizer.from_pretrained(str(model_dir))
        full = ChineseCLIPModel.from_pretrained(str(model_dir))
        self.model = full.text_model.eval().requires_grad_(False)
        self.dim = self.model.config.hidden_size
        capacity = self.model.config.max_position_embeddings
        self.max_positions = max_positions
        self.max_tokens = capacity if max_tokens is None else max_tokens
        if max_positions < 1 or not 1 <= self.max_tokens <= capacity:
            raise ValueError(f"invalid text limits: max_positions={max_positions}, "
                             f"max_tokens={self.max_tokens}, encoder capacity={capacity}")

    @torch.no_grad()
    def forward(self, captions: list[str | None]) -> tuple[torch.Tensor, torch.Tensor]:
        dev = next(self.model.parameters()).device
        B, P = len(captions), self.max_positions
        feats = torch.zeros(B, P, self.dim, device=dev)
        valid = torch.zeros(B, P, dtype=torch.bool, device=dev)
        idx = [i for i, c in enumerate(captions) if c]
        if not idx:
            return feats[:, :0], valid[:, :0]
        enc = self.tok([captions[i] for i in idx], padding=True, truncation=False, return_tensors="pt")
        if enc["input_ids"].shape[1] > self.max_tokens:
            raise ValueError(f"caption needs {enc['input_ids'].shape[1]} tokens, text encoder limit is "
                             f"{self.max_tokens}; shorten explicitly or increase max_tokens within capacity")
        enc = enc.to(dev)
        h = self.model(**enc).last_hidden_state.float()            # [b, T, dim]
        lengths = enc["attention_mask"].sum(1).tolist()
        for row, (i, n) in enumerate(zip(idx, lengths)):
            x = h[row, :n]                                          # [n, dim]
            if n > P:                                               # pool, don't truncate
                x = F.adaptive_avg_pool1d(x.T.unsqueeze(0), P)[0].T
            feats[i, : x.shape[0]] = x
            valid[i, : x.shape[0]] = True
        L = int(valid.sum(1).max())
        return feats[:, :L], valid[:, :L]
