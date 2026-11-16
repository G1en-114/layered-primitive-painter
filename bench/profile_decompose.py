"""End-to-end profile of the layered decomposer on real CelebAMask faces.

    python bench/profile_decompose.py --batch 16 --steps 80            # timing (img/s), PSNR
    python bench/profile_decompose.py --batch 16 --profile             # + kernel table (top CUDA time, launch count)

Reports images/s, mean PSNR (so speed-ups that change results are visible), the number of CUDA
kernel launches per optimisation step and where GPU time goes. Use the same --seed for A/B runs:
the decomposer is deterministic for a fixed seed, batch and backend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.decompose.layered import DecomposeConfig, LayeredDecomposer  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/celebamask256"))
    ap.add_argument("--protocol", type=Path, default=Path("protocol.json"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=2, help="timed batches (after one warm-up batch)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = ap.parse_args()

    dev = "cuda"
    proto = load_protocol(args.protocol)
    codec = Codec.load(proto=proto, device=dev)
    dec = LayeredDecomposer(codec, DecomposeConfig(steps_per_round=args.steps, seed=args.seed))
    imgs = np.load(args.data / "images_000.npy", mmap_mode="r")
    labs = np.load(args.data / "labels_000.npy", mmap_mode="r")

    def batch(i):
        rows = list(range(i * args.batch, (i + 1) * args.batch))
        img = torch.from_numpy(np.ascontiguousarray(imgs[rows])).to(dev).permute(0, 3, 1, 2).float().div(255)
        lab = torch.from_numpy(np.ascontiguousarray(labs[rows])).to(dev).long()
        return img, lab

    dec(*batch(0))                                     # warm-up: Triton autotune / compile, allocator
    torch.cuda.synchronize()
    psnr, t0 = [], time.perf_counter()
    for i in range(1, 1 + args.repeat):
        psnr += [d.psnr for d in dec(*batch(i))]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = args.repeat * args.batch
    summary = {"batch": args.batch, "steps": args.steps, "images": n, "img_per_s": round(n / dt, 3),
               "mean_psnr": round(float(np.mean(psnr)), 3), "gpu": torch.cuda.get_device_name()}
    print(json.dumps(summary))

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            dec(*batch(1))
            torch.cuda.synchronize()
        ev = sorted(prof.key_averages(), key=lambda e: -getattr(e, "self_device_time_total", 0))
        total = sum(getattr(e, "self_device_time_total", 0) for e in ev)
        launches = sum(e.count for e in prof.key_averages() if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
        rows = []
        for e in ev[:20]:
            t = getattr(e, "self_device_time_total", 0)
            if t <= 0:
                continue
            rows.append({"kernel": e.key[:90], "calls": e.count, "gpu_ms": round(t / 1e3, 2),
                         "share": round(t / max(total, 1), 4)})
        summary |= {"profile": {"gpu_ms_total": round(total / 1e3, 1), "kernel_launches": launches, "top": rows}}
        for r in rows:
            print(f"{r['share'] * 100:5.1f}%  {r['gpu_ms']:9.2f} ms  {r['calls']:7d}  {r['kernel']}")
        print(f"GPU kernel time {total / 1e3:.1f} ms, {launches} kernel launches for one batch of {args.batch}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
