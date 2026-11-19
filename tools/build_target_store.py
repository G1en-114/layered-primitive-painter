"""Pixel target images for target-conditioned painting: images.npy uint8 [N, S, S, 3] + ids.json.

    python tools/build_target_store.py --data data/celebamask256 --jpeg data/celebamask256_jpeg \
        --out data/target_images128_v1 --size 128

Every image goes through the same JPEG q95 round trip whether read from images_XXX.npy or a JPEG
pack (as tools/extract_image_features.py), then is resized to --size (area filter). Flipped
trajectories mirror the stored image at load time (lpp.data.sequences.TargetImageStore).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_image_features import jpeg_roundtrip  # noqa: E402
from pack_jpeg import decode_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, nargs="+", help="one or more prepared image dirs (one store)")
    ap.add_argument("--jpeg", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    indexes = [(d, json.loads((d / "index.json").read_text())) for d in args.data]
    ids = [i for _, index in indexes for s in index["shards"] for i in s["ids"]]
    if len(set(ids)) != len(ids):
        raise SystemExit("image ids repeat across --data dirs")
    args.out.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(args.out / "images.npy", "w+", np.uint8, (len(ids), args.size, args.size, 3))
    row = 0
    for data, index in indexes:
        for s in index["shards"]:
            npy = data / f"images_{s['shard']:03d}.npy"
            if npy.exists():
                imgs = jpeg_roundtrip(np.load(npy))
            else:
                imgs, pack_ids = decode_pack(args.jpeg / f"images_{s['shard']:03d}.jpg.npz")
                if pack_ids.tolist() != s["ids"]:
                    raise ValueError(f"JPEG pack ids differ from index for shard {s['shard']}")
            for im in imgs:
                out[row] = np.asarray(Image.fromarray(im).resize((args.size, args.size), Image.Resampling.BOX))
                row += 1
            print(f"shard {s['shard']:03d}: {row}/{len(ids)}", flush=True)
    out.flush()
    (args.out / "ids.json").write_text(json.dumps(ids))
    (args.out / "meta.json").write_text(json.dumps({"size": args.size, "jpeg_quality": 95, "resample": "box",
                                                     "source": [str(d) for d in args.data]}))


if __name__ == "__main__":
    main()
