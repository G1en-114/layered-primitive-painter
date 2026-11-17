"""CLIP image embeddings of the source images behind every training prompt (for GRPO image rewards).

    python tools/source_embeddings.py --data data/celebamask256 data/anime_portraits256_v2_l6view \
        --text-encoder data/models/chinese-clip-vit-base-patch16 --out data/source_emb_b16.npz

Writes ids [N] int64 and emb [N, D] float16 (L2-normalised), in shard order. A prompt generated from
image i can then be rewarded for looking like image i in CLIP space ("like its own source", which
works for photos and anime alike), instead of for looking like a photograph.

--seq instead of --data: embed the decomposer's reconstruction of each image (one trajectory,
unflipped), for machines that hold the sequences but not the source pixels. In CLIP space these
are close to the originals (FD to the real images 0.08 anime / 0.18 faces, below or near the
real-vs-real noise floor).
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
    ap.add_argument("--data", type=Path, nargs="*", default=[], help="prepared shard dirs (index.json, images_*.npy)")
    ap.add_argument("--seq", type=Path, nargs="*", default=[], help="trajectory dirs of decomp_*.npz (render instead)")
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
    if args.seq:
        import torch
        from lpp.codec import Codec
        from lpp.data.sequences import load_shards
        from lpp.protocol import load_protocol
        from lpp.render.dispatch import render_group
        from lpp.render.reference import flatten_on_base
        proto = load_protocol()
        codec = Codec.load(proto=proto, device="cuda")
        done = set(ids)
        for d in args.seq:
            for f in sorted(d.glob("decomp_*.npz")):
                batch_ids, imgs = [], []
                for e in load_shards([f]):
                    if e.flip or e.image_id in done:
                        continue
                    order = torch.sort(e.layer, stable=True).indices
                    t = e.tokens[order].cuda()
                    pr = codec.decode({k: t[:, j].unsqueeze(0) for j, k in enumerate(proto.attr_names)})
                    with torch.no_grad():
                        img = flatten_on_base(render_group(pr, 256, proto), proto)[0]
                    imgs.append((img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                    batch_ids.append(e.image_id)
                    done.add(e.image_id)
                if imgs:
                    embs.append(clip.images(imgs, batch=args.batch).half().cpu().numpy())
                    ids += batch_ids
                print(f"{d.name}/{f.name}: {len(ids)} images", flush=True)
    np.savez(args.out, ids=np.array(ids, np.int64), emb=np.concatenate(embs))
    print(f"wrote {args.out}: {len(ids)} x {embs[0].shape[1]}", flush=True)
    Path(str(args.out) + ".complete").touch()


if __name__ == "__main__":
    main()
