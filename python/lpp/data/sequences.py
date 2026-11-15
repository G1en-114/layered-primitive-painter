"""Training examples from decomposition shards (tools/decompose_dataset.py output)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..protocol import Protocol, load_protocol
from ..sequence import DrawSeq, build, perturb, sample_mode


@dataclass
class Example:
    tokens: torch.Tensor   # [N, A]
    layer: torch.Tensor    # [N]
    round: torch.Tensor    # [N]
    image_id: int


def load_shards(paths: list[Path]) -> list[Example]:
    out = []
    for p in paths:
        # read each array once: every z["key"] access decompresses the whole array again,
        # which made loading quadratic in shard size
        with np.load(p) as z:
            off, ids = z["offsets"], z["ids"]
            tokens = torch.from_numpy(z["tokens"].astype(np.int64))
            layer = torch.from_numpy(z["layer"].astype(np.int64))
            rounds = torch.from_numpy(z["round"].astype(np.int64))
        for i, image_id in enumerate(ids):
            a, b = int(off[i]), int(off[i + 1])
            out.append(Example(tokens[a:b], layer[a:b], rounds[a:b], int(image_id)))
    return out


class SequenceDataset(Dataset):
    """Each __getitem__ draws a fresh mode (full / redraw / refine) for its example."""

    def __init__(self, examples: list[Example], proto: Protocol | None = None, seed: int = 0,
                 p_redraw: float = 0.15, p_refine: float = 0.15, text_fn=None, captions=None,
                 p_text_drop: float = 0.1, p_perturb: float = 0.0):
        self.examples, self.proto = examples, proto or load_protocol()
        self.rng = random.Random(seed)
        self.p_redraw, self.p_refine = p_redraw, p_refine
        self.text_fn = text_fn          # image_id -> [Lt, text_dim] tensor, or None (precomputed)
        self.captions = captions        # lpp.text.CaptionStore: caption strings, encoded in the train step
        self.p_text_drop = p_text_drop
        self.p_perturb = p_perturb      # share of sequences with corrupted history (see sequence.perturb)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ex = self.examples[i]
        mode = sample_mode(ex.layer, ex.round, self.rng, self.p_redraw, self.p_refine)
        seq = build(ex.tokens, ex.layer, ex.round, proto=self.proto, **mode)
        if self.p_perturb > 0 and self.rng.random() < self.p_perturb:
            seq = perturb(seq, self.rng, self.proto)
        if self.captions is not None:
            text = self.captions.pick(ex.image_id, self.rng, self.p_text_drop)
        else:
            text = self.text_fn(ex.image_id) if self.text_fn else None
        return seq, text


def collate(items: list[tuple[DrawSeq, torch.Tensor | None]], text_dim: int | None = None) -> dict:
    """Right-pad to the longest sequence; targets are the next record."""
    seqs = [s for s, _ in items]
    B, T = len(seqs), max(len(s) for s in seqs) - 1
    A = seqs[0].attrs.shape[1]
    batch = {
        "kind": torch.zeros(B, T, dtype=torch.long),
        "event": torch.full((B, T), -1, dtype=torch.long),
        "layer": torch.full((B, T), -1, dtype=torch.long),
        "attrs": torch.full((B, T, A), -1, dtype=torch.long),
        "valid": torch.zeros(B, T, dtype=torch.bool),
        "tgt_event": torch.full((B, T), -1, dtype=torch.long),
        "tgt_attrs": torch.full((B, T, A), -1, dtype=torch.long),
    }
    for b, s in enumerate(seqs):
        n = len(s) - 1
        batch["kind"][b, :n] = s.kind[:-1]
        batch["event"][b, :n] = s.event[:-1]
        batch["layer"][b, :n] = s.layer[:-1]
        batch["attrs"][b, :n] = s.attrs[:-1]
        batch["valid"][b, :n] = True
        batch["tgt_event"][b, :n] = s.event[1:]
        batch["tgt_attrs"][b, :n] = s.attrs[1:]
        if s.no_target is not None:                     # never learn to produce a perturbed record
            batch["tgt_event"][b, :n][s.no_target[1:]] = -1
    texts = [t for _, t in items]
    if any(isinstance(t, str) for t in texts):      # caption strings: encoded on the GPU later
        batch["captions"] = texts
    elif any(t is not None for t in texts):
        Lt = max(t.shape[0] for t in texts if t is not None)
        dim = text_dim or next(t for t in texts if t is not None).shape[1]
        batch["text"] = torch.zeros(B, Lt, dim)
        batch["text_valid"] = torch.zeros(B, Lt, dtype=torch.bool)
        for b, t in enumerate(texts):
            if t is not None:
                batch["text"][b, : t.shape[0]] = t
                batch["text_valid"][b, : t.shape[0]] = True
    return batch
