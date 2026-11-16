"""Rebuild a decomposer input directory from a transfer pack (the slow-link counterpart of
tools/pack_jpeg.py): images_XXX.jpg.npz -> images_XXX.npy (decoded JPEG), labels_XXX.npz ->
labels_XXX.npy (lossless), plus index.json.

    python tools/unpack_jpeg.py --pack data/anime_l6_pack --out data/anime_l6_jpegsrc

Decomposition targets made this way are the JPEG-decoded images (q95, ~42.7 dB from the
originals); sequences record that via the output README, and PSNR is measured against them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_jpeg import decode_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.pack / "index.json").read_text())
    for sh in index["shards"]:
        s = sh["shard"]
        img_out, lab_out = args.out / f"images_{s:03d}.npy", args.out / f"labels_{s:03d}.npy"
        if img_out.exists() and lab_out.exists():
            continue
        imgs, ids = decode_pack(args.pack / f"images_{s:03d}.jpg.npz")
        if [int(i) for i in ids] != [int(i) for i in sh["ids"]]:
            raise ValueError(f"shard {s}: pack ids differ from index.json")
        with np.load(args.pack / f"labels_{s:03d}.npz") as z:
            labels = z["labels"]
            if hashlib.sha256(labels.tobytes()).hexdigest() != str(z["sha256"]):
                raise ValueError(f"shard {s}: label checksum mismatch")
        if labels.shape[:3] != imgs.shape[:3]:
            raise ValueError(f"shard {s}: labels {labels.shape} vs images {imgs.shape}")
        np.save(img_out.with_suffix(".tmp.npy"), imgs)
        np.save(lab_out.with_suffix(".tmp.npy"), labels)
        img_out.with_suffix(".tmp.npy").replace(img_out)
        lab_out.with_suffix(".tmp.npy").replace(lab_out)
        print(f"shard {s:03d}: {len(ids)} images", flush=True)
    shutil.copy(args.pack / "index.json", args.out / "index.json")
    (args.out / "README.txt").write_text(f"Unpacked from {args.pack}: images are JPEG-decoded (lossy), labels exact.\n")


if __name__ == "__main__":
    main()
