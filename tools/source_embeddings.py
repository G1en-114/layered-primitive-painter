"""CLIP image embeddings of the source images behind every training prompt (for GRPO image rewards).

    python tools/source_embeddings.py --data data/celebamask256 data/anime_portraits256_v2_l6view \
        --text-encoder data/models/chinese-clip-vit-base-patch16 --out data/source_emb_b16.npz

Writes ids [N] int64 and emb [N, D] float16 (L2-normalised), in shard order. A prompt generated from
image i can then be rewarded for looking like image i in CLIP space ("like its own source", which
works for photos and anime alike), instead of for looking like a photograph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.clip_score import ClipScorer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="prepared shard dirs (index.json, images_*.npy)")
    ap.add_argument("--text-encoder", type=Path, required=True, help="Chinese-CLIP model dir (its image tower is used)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    clip = ClipScorer(args.text_encoder, "cuda")
    ids, embs = [], []
    for d in args.data:
        index = json.loads((d / "index.json").read_text())
        for s in index["shards"]:
            imgs = np.load(d / f"images_{s['shard']:03d}.npy", mmap_mode="r")
            e = clip.images([np.array(imgs[r])[..., :3] for r in range(len(s["ids"]))], batch=args.batch)
            ids += [int(i) for i in s["ids"]]
            embs.append(e.half().cpu().numpy())
            print(f"{d.name} shard {s['shard']:03d}: {len(ids)} images", flush=True)
    np.savez(args.out, ids=np.array(ids, np.int64), emb=np.concatenate(embs))
    print(f"wrote {args.out}: {len(ids)} x {embs[0].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
