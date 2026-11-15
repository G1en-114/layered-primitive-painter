"""Decompose prepared image/label shards into primitive sequences.

    python tools/decompose_dataset.py --data data/celebamask256 --out data/seq/v0
    python tools/decompose_dataset.py --data data/celebamask256 --out data/seq/v1 --flip --seed 1
    # split across GPUs / machines:  --part 0/4 ... --part 3/4

Resumable: an output shard is written to a temp file and renamed when complete, and
existing shards are skipped. Output per input shard (decomp_XXX.npz):
    tokens  [M, 5] int16   x, y, shape, color, alpha
    layer   [M]    int8    round [M] int8
    offsets [n+1]  int64   image i owns rows offsets[i]:offsets[i+1]
    ids     [n]    int32   psnr [n] float32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.decompose import DecomposeConfig, LayeredDecomposer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--flip", action="store_true", help="mirror images horizontally (augmentation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="i/n: process shards s with s %% n == i")
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()
    part, nparts = map(int, args.part.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)

    index = json.loads((args.data / "index.json").read_text())
    shards = [s for s in index["shards"] if s["shard"] % nparts == part][: args.limit_shards]
    dev = "cuda"
    codec = Codec.load(device=dev)
    dec = LayeredDecomposer(codec, DecomposeConfig(steps_per_round=args.steps, seed=args.seed))
    (args.out / "config.json").write_text(json.dumps(
        {"flip": args.flip, "seed": args.seed, "steps": args.steps, "data": str(args.data)}, indent=1))

    t_start, done_imgs = time.perf_counter(), 0
    for sh in shards:
        s = sh["shard"]
        final = args.out / f"decomp_{s:03d}.npz"
        if final.exists():
            print(f"shard {s:03d}: exists, skipped")
            continue
        imgs = np.load(args.data / f"images_{s:03d}.npy", mmap_mode="r")
        labs = np.load(args.data / f"labels_{s:03d}.npy", mmap_mode="r")
        toks, lays, rnds, offsets, psnr = [], [], [], [0], []
        for b0 in range(0, len(sh["ids"]), args.batch):
            img = torch.from_numpy(np.ascontiguousarray(imgs[b0 : b0 + args.batch])).to(dev)
            lab = torch.from_numpy(np.ascontiguousarray(labs[b0 : b0 + args.batch])).to(dev).long()
            img = img.permute(0, 3, 1, 2).float().div(255)
            if args.flip:
                img, lab = img.flip(-1), lab.flip(-1)
            for d in dec(img, lab):
                toks.append(torch.stack([d.tokens[n] for n in dec.codec_attrs], 1).numpy().astype(np.int16))
                lays.append(d.layer.numpy().astype(np.int8))
                rnds.append(d.round.numpy().astype(np.int8))
                offsets.append(offsets[-1] + len(d.layer))
                psnr.append(d.psnr)
            done_imgs += img.shape[0]
            rate = done_imgs / (time.perf_counter() - t_start)
            print(f"shard {s:03d}: {b0 + img.shape[0]}/{len(sh['ids'])}  "
                  f"mean PSNR {np.mean(psnr):.2f} dB  {rate:.2f} img/s", flush=True)
        tmp = final.with_suffix(".tmp.npz")
        np.savez(tmp, tokens=np.concatenate(toks), layer=np.concatenate(lays), round=np.concatenate(rnds),
                 offsets=np.array(offsets, dtype=np.int64), ids=np.array(sh["ids"], dtype=np.int32),
                 psnr=np.array(psnr, dtype=np.float32))
        os.replace(tmp, final)
        print(f"shard {s:03d}: written ({offsets[-1]} primitives, {offsets[-1] / len(psnr):.0f}/image)")


if __name__ == "__main__":
    main()
