"""Build the data-independent prior codebooks (color in Oklab, shape in log-covariance).

    python tools/build_prior_codebooks.py [--device cuda]

These let the pipeline run before any images are decomposed. Once a dataset is
decomposed, codebooks are refit on its real parameter distribution.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codebook import kmeans, nearest, prior_color_samples, prior_shape_samples  # noqa: E402
from lpp.protocol import REPO_ROOT, load_protocol  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    proto = load_protocol()
    jobs = [
        ("color", proto.attr("color").size, prior_color_samples(args.samples, seed=1)),
        ("shape", proto.attr("shape").codes_per_beta, prior_shape_samples(proto, args.samples, seed=2)),
    ]
    for name, k, samples in jobs:
        t0 = time.perf_counter()
        x = samples.to(args.device)
        codes = kmeans(x, k, iters=args.iters)
        err = (x - codes[nearest(codes, x)]).norm(dim=-1)
        out = REPO_ROOT / proto.attr(name).file
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, codes.cpu().numpy().astype(np.float32))
        print(f"{name}: k={k}  mean err={err.mean():.4f}  p99={err.quantile(0.99):.4f}  "
              f"-> {out.relative_to(REPO_ROOT)}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
