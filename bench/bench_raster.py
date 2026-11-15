"""Benchmark the rasterizer backends on the shapes the project actually runs.

    PYTHONPATH=~/.claude/skills/kernel-opt python bench/bench_raster.py

Scenarios
  decompose-step  one optimizer step of the decomposer: B images x N primitives of one
                  layer at 256 px, forward + backward through an MSE loss
  full-render     forward only, a whole image's worth of primitives (N ~ 1000)
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, os.path.expanduser("~/.claude/skills/kernel-opt"))

from kopt import bench, table  # noqa: E402
from lpp.render import Primitives  # noqa: E402
from lpp.render.reference import render_group as torch_group  # noqa: E402
from lpp.render.triton_raster import TILE, bin_tiles, render_group_triton  # noqa: E402
from lpp.shape import conic  # noqa: E402


def scene(B, N, sigma_lo, sigma_hi, seed=0, dev="cuda") -> Primitives:
    """Log-uniform sizes, like a coarse-to-fine decomposition."""
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.rand(*s, device=dev, generator=g)  # noqa: E731
    lo, hi = math.log(sigma_lo), math.log(sigma_hi)
    return Primitives(
        xy=r(B, N, 2) * 256,
        sigma=torch.exp(lo + (hi - lo) * r(B, N, 2)),
        theta=r(B, N) * math.pi,
        beta=torch.tensor([1.0, 2.0, 4.0, 8.0], device=dev)[(r(B, N) * 4).long().clamp(max=3)],
        rgb=r(B, N, 3),
        alpha=0.3 + 0.69 * r(B, N),
    )


def pair_counts(p: Primitives, size: int) -> tuple[int, int]:
    """(pixel-primitive pairs the tile kernel evaluates, pairs the dense reference evaluates)."""
    con = conic(p.sigma[..., 0], p.sigma[..., 1], p.theta)
    _, counts, _ = bin_tiles(p.xy, con, p.alpha, size, 256, 3.0)
    B, N = p.alpha.shape
    return int(counts.sum()) * TILE * TILE, B * N * size * size


def step_fn(group_fn, p: Primitives, target: torch.Tensor, size: int):
    leaves = {f: getattr(p, f).detach().clone().requires_grad_(True)
              for f in ("xy", "sigma", "theta", "rgb", "alpha")}

    def run():
        for t in leaves.values():
            t.grad = None
        q = Primitives(beta=p.beta, **leaves)
        P, T = group_fn(q, size)
        ((P + T - target) ** 2).mean().backward()
    return run


def fwd_fn(group_fn, p: Primitives, size: int):
    def run():
        with torch.no_grad():
            group_fn(p, size)
    return run


def try_bench(fn, name):
    try:
        return bench(fn, name=name)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {name}: out of memory")
        return None


def main() -> None:
    size = 256
    cases = [
        ("decompose-step", 4, 256, 1.5, 48.0, "step"),
        ("decompose-step", 16, 256, 1.5, 48.0, "step"),
        ("full-render", 4, 960, 1.0, 64.0, "fwd"),
        ("full-render", 16, 960, 1.0, 64.0, "fwd"),
    ]
    for label, B, N, lo, hi, kind in cases:
        p = scene(B, N, lo, hi)
        tile_pairs, dense_pairs = pair_counts(p, size)
        print(f"\n### {label}: B={B} N={N} {size}px  "
              f"(tile kernel evaluates {tile_pairs / dense_pairs:.1%} of the dense pixel-primitive pairs)")
        target = torch.rand(B, 3, size, size, device="cuda")
        rows = []
        for name, fn in (("torch dense reference", torch_group), ("triton tiles v1", render_group_triton)):
            run = step_fn(fn, p, target, size) if kind == "step" else fwd_fn(fn, p, size)
            r = try_bench(run, name)
            if r is not None:
                rows.append(r)
        print(table(rows))
        del p, target
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
