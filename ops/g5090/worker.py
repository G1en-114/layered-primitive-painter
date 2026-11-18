"""5090 box: one line-decomposition worker (tools/decompose_lines.py), fed by ops/g5090/upload.sh.

    python3 ops/g5090/worker.py --gpu 0 --slot 2 --nslots 12 --traj t0 [--steps 200]

Takes the v2 shards of one trajectory whose index % nslots == slot, in order. Each one starts as soon as the
shard and the image/label shards holding its ids have arrived (the uploader's FILE.ok markers). Finished
outputs are skipped, so a restart resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R = Path("/root/lpp")


def wait(paths, done_flag=None) -> bool:
    """Wait until every path's .ok marker exists; False if done_flag appears first and some are still missing."""
    while True:
        if all(Path(str(p) + ".ok").exists() for p in paths):
            return True
        if done_flag is not None and done_flag.exists() and not all(Path(str(p) + ".ok").exists() for p in paths):
            return False
        time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--nslots", type=int, required=True)
    ap.add_argument("--traj", default="t0")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--deadline", default=None, help="UTC HH:MM: only take as many images of a shard as fit before it")
    ap.add_argument("--rate", type=float, default=0.13, help="images/s of one worker (for --deadline)")
    args = ap.parse_args()
    seq = R / "data/seq" / f"anime_v2_l6_{args.traj}"
    img = R / "data/anime_portraits256_v2_l6view"
    out = R / "out" / f"anime_v3_lines_{args.traj}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(R / "logs" / f"worker_{args.traj}_g{args.gpu}_s{args.slot}.log", "a")
    wait([img / "index.json"])
    index = json.loads((img / "index.json").read_text())
    where = {int(i): s["shard"] for s in index["shards"] for i in s["ids"]}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1")
    done_flag = R / "data/seq" / f"upload_done_{args.traj}"
    for i in range(0, 1000):
        if i % args.nslots != args.slot:
            continue
        shard = seq / f"decomp_{i:03d}.npz"
        if not wait([shard], done_flag):
            break                                          # past the last shard of the trajectory
        if (out / shard.name).exists():
            continue
        ids = np.load(shard)["ids"]
        need = sorted({where[int(x)] for x in ids if int(x) in where})
        wait([img / f"images_{k:03d}.npy" for k in need] + [img / f"labels_{k:03d}.npy" for k in need])
        limit = []
        if args.deadline:
            hh, mm = map(int, args.deadline.split(":"))
            now = time.gmtime()
            left = (hh * 60 + mm - now.tm_hour * 60 - now.tm_min) * 60 - now.tm_sec - 120   # 2 min to write
            n = int(left * args.rate)
            if n < 20:
                print(f"{time.strftime('%H:%M:%S')} deadline: {n} images would fit, stopping", file=log, flush=True)
                break
            limit = ["--limit", str(n)] if n < len(ids) else []
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", "tools/decompose_lines.py", "--seq", str(shard), "--images", str(img),
                              "--out", str(out), "--steps", str(args.steps)] + limit, cwd=R / "code", env=env,
                             stdout=log, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} {shard.name} rc={rc} {time.time() - t0:.0f}s", file=log, flush=True)
    print(f"{time.strftime('%H:%M:%S')} worker done", file=log, flush=True)


if __name__ == "__main__":
    main()
