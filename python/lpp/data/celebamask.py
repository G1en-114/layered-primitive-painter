"""CelebAMask-HQ -> (256 px image, per-pixel layer id) shards.

Source layout (official release):
    CelebA-HQ-img/{idx}.jpg                              1024 x 1024
    CelebAMask-HQ-mask-anno/{idx // 2000}/{idx:05d}_{class}.png   512 x 512, class pixels white
    CelebAMask-HQ-attribute-anno.txt                     count, 40 names, "{idx}.jpg  +-1 x 40"

The 18 annotated classes overlap (hair over skin, glasses over eyes ...). Layers in
protocol.json are ordered back to front, so a pixel takes the *highest* layer of any
class that covers it; pixels with no class are background (layer 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..protocol import Protocol, load_protocol

NUM_IMAGES = 30000
IMG_DIR = "CelebA-HQ-img"
MASK_DIR = "CelebAMask-HQ-mask-anno"
ATTR_FILE = "CelebAMask-HQ-attribute-anno.txt"


def class_to_layer(proto: Protocol | None = None) -> dict[str, int]:
    proto = proto or load_protocol()
    return {c: l.id for l in proto.layers for c in l.celebamask if c != "background"}


def index_masks(root: Path) -> dict[int, list[tuple[str, Path]]]:
    """idx -> [(class, path)] by scanning the mask folders once."""
    out: dict[int, list[tuple[str, Path]]] = {}
    for p in (root / MASK_DIR).glob("*/*.png"):
        stem, _, cls = p.stem.partition("_")
        out.setdefault(int(stem), []).append((cls, p))
    return out


def load_example(root: Path, idx: int, masks: list[tuple[str, Path]], size: int,
                 cls2layer: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(root / IMG_DIR / f"{idx}.jpg").convert("RGB")
    img = np.asarray(img.resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)
    label = np.zeros((size, size), dtype=np.uint8)
    for cls, path in masks:
        if cls not in cls2layer:
            raise KeyError(f"unknown CelebAMask class {cls!r} in {path.name}")
        m = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
        hit = np.asarray(m) > 127
        np.maximum(label, np.where(hit, cls2layer[cls], 0).astype(np.uint8), out=label)
    return img, label


def load_attributes(root: Path) -> tuple[list[str], dict[int, list[int]]]:
    lines = (root / ATTR_FILE).read_text().split("\n")
    count, names = int(lines[0]), lines[1].split()
    attrs = {}
    for line in lines[2 : 2 + count]:
        parts = line.split()
        attrs[int(parts[0].split(".")[0])] = [int(v) > 0 for v in parts[1:]]
    if len(names) != 40 or len(attrs) != count:
        raise ValueError(f"unexpected attribute file: {len(names)} names, {len(attrs)}/{count} rows")
    return names, attrs


def write_attributes(root: Path, out: Path) -> None:
    names, attrs = load_attributes(root)
    out.write_text(json.dumps({"names": names, "values": {str(k): v for k, v in sorted(attrs.items())}}))
