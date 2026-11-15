"""Convert extracted CelebAMask-HQ into 256 px image + layer-label shards.

    python tools/prepare_celebamask.py --src data/raw/CelebAMask-HQ --out data/celebamask256

Output (uint8 .npy, memory-mappable):
    images_000.npy  [n, S, S, 3]     labels_000.npy  [n, S, S]  (layer id per pixel)
    index.json      shard -> image ids, layer pixel statistics
    attributes.json the 40 CelebA attributes per image id (for template captions)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.data.celebamask import (  # noqa: E402
    NUM_IMAGES, class_to_layer, index_masks, load_example, write_attributes)
from lpp.protocol import load_protocol  # noqa: E402

_MASKS: dict = {}


def _work(idx: int, root: Path, size: int, cls2layer: dict):
    return load_example(root, idx, _MASKS.get(idx, []), size, cls2layer)


def _init(masks):
    global _MASKS
    _MASKS = masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=NUM_IMAGES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    proto = load_protocol()
    cls2layer = class_to_layer(proto)
    t0 = time.perf_counter()
    masks = index_masks(args.src)
    print(f"indexed masks for {len(masks)} images in {time.perf_counter() - t0:.1f}s")
    missing = [i for i in range(args.limit) if i not in masks]
    if missing:
        print(f"warning: {len(missing)} images have no mask files (first: {missing[:5]})")

    ids = list(range(args.limit))
    layer_pixels = np.zeros(proto.num_layers, dtype=np.int64)
    layer_present = np.zeros(proto.num_layers, dtype=np.int64)
    shards = []
    work = partial(_work, root=args.src, size=args.size, cls2layer=cls2layer)
    with Pool(args.workers, initializer=_init, initargs=(masks,)) as pool:
        for s, start in enumerate(range(0, len(ids), args.shard)):
            chunk = ids[start : start + args.shard]
            res = pool.map(work, chunk, chunksize=16)
            imgs = np.stack([r[0] for r in res])
            labs = np.stack([r[1] for r in res])
            np.save(args.out / f"images_{s:03d}.npy", imgs)
            np.save(args.out / f"labels_{s:03d}.npy", labs)
            counts = np.stack([np.bincount(l.ravel(), minlength=proto.num_layers) for l in labs])
            layer_pixels += counts.sum(0)
            layer_present += (counts > 0).sum(0)
            shards.append({"shard": s, "ids": chunk})
            done = start + len(chunk)
            rate = done / (time.perf_counter() - t0)
            print(f"shard {s:03d}: {done}/{len(ids)} images  ({rate:.0f} img/s)", flush=True)

    write_attributes(args.src, args.out / "attributes.json")
    stats = {
        "size": args.size,
        "count": len(ids),
        "layers": [l.name for l in proto.layers],
        "layer_pixel_fraction": (layer_pixels / layer_pixels.sum()).round(4).tolist(),
        "layer_present_fraction": (layer_present / len(ids)).round(4).tolist(),
        "images_without_masks": missing,
    }
    (args.out / "index.json").write_text(json.dumps({"shards": shards, "stats": stats}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
