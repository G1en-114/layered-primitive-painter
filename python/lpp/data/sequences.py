"""Training examples from decomposition shards (tools/decompose_dataset.py output)."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..protocol import Protocol, load_protocol
from ..sequence import DrawSeq, build, perturb, sample_mode
from ..text import mirror_caption


@dataclass
class Example:
    tokens: torch.Tensor   # [N, A]
    layer: torch.Tensor    # [N]
    round: torch.Tensor    # [N]
    image_id: int
    flip: bool = False


def load_shards(paths: list[Path]) -> list[Example]:
    out = []
    configs = {}
    for p in paths:
        if p.parent not in configs:
            config = p.parent / "config.json"
            configs[p.parent] = json.loads(config.read_text()) if config.exists() else {}
        flipped = bool(configs[p.parent].get("flip", False))
        # read each array once: every z["key"] access decompresses the whole array again,
        # which made loading quadratic in shard size
        with np.load(p) as z:
            off, ids = z["offsets"], z["ids"]
            tokens = torch.from_numpy(z["tokens"].astype(np.int64))
            layer = torch.from_numpy(z["layer"].astype(np.int64))
            rounds = torch.from_numpy(z["round"].astype(np.int64))
        for i, image_id in enumerate(ids):
            a, b = int(off[i]), int(off[i + 1])
            out.append(Example(tokens[a:b], layer[a:b], rounds[a:b], int(image_id), flipped))
    return out


class ImageTokenStore:
    """Precomputed CLIP image tokens (tools/extract_image_features.py), memory-mapped.

    get(image_id, flip) -> float16 array [Li, D], the mirrored image's tokens for flipped
    trajectories. Opened lazily so DataLoader workers each map the files themselves."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.row = {int(i): r for r, i in enumerate(json.loads((self.root / "ids.json").read_text()))}
        self._mm: dict[str, np.ndarray] = {}

    def __contains__(self, image_id: int) -> bool:
        return int(image_id) in self.row

    def get(self, image_id: int, flip: bool) -> np.ndarray:
        key = "flip" if flip else "orig"
        if key not in self._mm:
            self._mm[key] = np.load(self.root / f"tokens_{key}.npy", mmap_mode="r")
        return np.asarray(self._mm[key][self.row[int(image_id)]])


class TargetImageStore:
    """Pixel target images for target-conditioned painting (tools/build_target_store.py):
    images.npy uint8 [N, S, S, 3] memory-mapped + ids.json. get(image_id, flip) mirrors the image
    for flipped trajectories."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.row = {int(i): r for r, i in enumerate(json.loads((self.root / "ids.json").read_text()))}
        self._mm: np.ndarray | None = None

    def __contains__(self, image_id: int) -> bool:
        return int(image_id) in self.row

    def get(self, image_id: int, flip: bool) -> np.ndarray:
        if self._mm is None:
            self._mm = np.load(self.root / "images.npy", mmap_mode="r")
        img = self._mm[self.row[int(image_id)]]
        return np.array(img[:, ::-1] if flip else img)      # a writable copy (torch warns on read-only memmaps)


class SequenceDataset(Dataset):
    """Each __getitem__ draws a fresh mode (full / redraw / refine) for its example."""

    def __init__(self, examples: list[Example], proto: Protocol | None = None, seed: int = 0,
                 p_redraw: float = 0.15, p_refine: float = 0.15, text_fn=None, captions=None,
                 p_text_drop: float = 0.1, p_perturb: float = 0.0, image_tokens: ImageTokenStore | None = None,
                 p_image: float = 0.0, targets: TargetImageStore | None = None, p_target: float = 0.0):
        self.examples, self.proto = examples, proto or load_protocol()
        self.image_tokens, self.p_image = image_tokens, p_image   # image condition with probability p_image
        self.targets, self.p_target = targets, p_target           # pixel target with probability p_target
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
            if ex.flip:
                text = mirror_caption(text)
        else:
            text = self.text_fn(ex.image_id) if self.text_fn else None
        if self.image_tokens is None and self.targets is None:
            return seq, text
        image = target = None
        if self.image_tokens is not None and self.p_image > 0 and ex.image_id in self.image_tokens \
                and self.rng.random() < self.p_image:
            image = self.image_tokens.get(ex.image_id, ex.flip)
        if self.targets is not None and self.p_target > 0 and ex.image_id in self.targets \
                and self.rng.random() < self.p_target:
            target = self.targets.get(ex.image_id, ex.flip)
        return seq, text, image, target


def collate(items: list[tuple[DrawSeq, ...]], text_dim: int | None = None) -> dict:
    """Right-pad to the longest sequence; targets are the next record.

    items: (seq, text), (seq, text, image) or (seq, text, image, target): image a [Li, D] array or
    None, target a uint8 [S, S, 3] image or None (-> batch["target"] [B, 3, S, S] in [0, 1])."""
    seqs = [it[0] for it in items]
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
    images = [it[2] if len(it) > 2 else None for it in items]
    if any(im is not None for im in images):
        Li, D = next(im.shape for im in images if im is not None)
        batch["image"] = torch.zeros(B, Li, D)
        batch["image_valid"] = torch.zeros(B, Li, dtype=torch.bool)
        for b, im in enumerate(images):
            if im is not None:
                batch["image"][b] = torch.as_tensor(np.asarray(im, dtype=np.float32))
                batch["image_valid"][b] = True
    targets = [it[3] if len(it) > 3 else None for it in items]
    if any(t is not None for t in targets):
        S = next(t.shape[0] for t in targets if t is not None)
        batch["target"] = torch.zeros(B, 3, S, S)
        batch["target_valid"] = torch.zeros(B, dtype=torch.bool)
        for b, t in enumerate(targets):
            if t is not None:
                batch["target"][b] = torch.from_numpy(np.ascontiguousarray(t)).permute(2, 0, 1).float() / 255
                batch["target_valid"][b] = True
    texts = [it[1] for it in items]
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
