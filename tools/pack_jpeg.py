"""Pack prepared RGB shards (images_XXX.npy) as JPEG bytes for slow links: images_XXX.jpg.npz.

    python tools/pack_jpeg.py --data data/celebamask256 --out data/celebamask256_jpeg --shards 0 2 4

Each pack holds `jpeg` (object array of bytes), `ids` and the source SHA-256. JPEG is lossy:
packs are for consumers that tolerate it (CLIP image features, previews), never for
decomposition targets or PSNR. tools/extract_image_features.py reads either format.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def decode_pack(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        imgs = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in z["jpeg"]])
        return imgs, z["ids"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, nargs="*", default=None, help="default: all")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.data / "index.json").read_text())
    for s in index["shards"]:
        if args.shards is not None and s["shard"] not in args.shards:
            continue
        src = args.data / f"images_{s['shard']:03d}.npy"
        imgs = np.load(src, mmap_mode="r")
        blobs = []
        for im in imgs:
            buf = io.BytesIO()
            Image.fromarray(np.asarray(im)).save(buf, "JPEG", quality=args.quality, subsampling=0)
            blobs.append(buf.getvalue())
        arr = np.empty(len(blobs), dtype=object)
        arr[:] = blobs
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        np.savez(args.out / f"images_{s['shard']:03d}.jpg.npz", jpeg=arr, ids=np.array(s["ids"]),
                 source_sha256=np.array(sha), quality=np.array(args.quality))
        print(f"shard {s['shard']:03d}: {sum(map(len, blobs)) / 1e6:.1f} MB", flush=True)
    (args.out / "index.json").write_text(json.dumps(index))


if __name__ == "__main__":
    main()
