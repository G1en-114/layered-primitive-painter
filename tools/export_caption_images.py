"""Export CelebA-HQ images as small JPEGs for captioning on another machine.

    python tools/export_caption_images.py --src data/raw/CelebAMask-HQ --out data/caption_images --size 448
"""
from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from PIL import Image


def one(idx: int, src: Path, out: Path, size: int) -> None:
    dst = out / f"{idx}.jpg"
    if not dst.exists():
        im = Image.open(src / "CelebA-HQ-img" / f"{idx}.jpg").convert("RGB")
        im.resize((size, size), Image.Resampling.LANCZOS).save(dst, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--count", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        pool.map(partial(one, src=args.src, out=args.out, size=args.size), range(args.count), chunksize=64)
    print(f"wrote {args.count} images to {args.out}")


if __name__ == "__main__":
    main()
