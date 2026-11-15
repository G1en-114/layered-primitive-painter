"""Read explicit source-aware splits instead of silently reassigning validation IDs."""
from __future__ import annotations

import json
from pathlib import Path


def load_split_manifest(path: str | Path) -> dict[int, str]:
    splits = {}
    with Path(path).open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id, split = int(record["id"]), record["split"]
            if image_id in splits:
                raise ValueError(f"duplicate image ID in split manifest: {image_id}")
            if split not in ("train", "val"):
                raise ValueError(f"unsupported training split {split!r} for image {image_id}")
            splits[image_id] = split
    if not splits:
        raise ValueError("empty split manifest")
    return splits
