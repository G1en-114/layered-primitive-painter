"""K-means codebooks for the color and shape attributes.

Two stages:
  1. A data-independent *prior* codebook (built here from uniform samples), so
     the rest of the pipeline can run before any dataset is decomposed.
  2. A data-driven codebook refit on parameters from real decompositions.
     (Same `kmeans` function, different samples.)
"""
from __future__ import annotations

import math

import torch

from .color import srgb_to_oklab
from .protocol import Protocol
from .shape import canonicalize, to_logcov


@torch.no_grad()
def nearest(codebook: torch.Tensor, x: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """Index of the nearest code (Euclidean) for each row of x[..., D]."""
    flat = x.reshape(-1, x.shape[-1])
    cb = codebook.to(flat)
    cb_sq = (cb * cb).sum(-1)
    out = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
    for i in range(0, flat.shape[0], chunk):
        part = flat[i : i + chunk]
        # |x - c|^2 = |x|^2 - 2 x.c + |c|^2 ; |x|^2 is constant per row
        out[i : i + chunk] = (cb_sq - 2 * part @ cb.T).argmin(-1)
    return out.reshape(x.shape[:-1])


@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """Plain Lloyd's k-means with k-means++ initialisation. x: [N, D] -> [k, D]."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    n = x.shape[0]
    if n < k:
        raise ValueError(f"need at least k={k} samples, got {n}")

    # k-means++: each new center is sampled proportional to squared distance
    centers = torch.empty(k, x.shape[1], device=x.device, dtype=x.dtype)
    centers[0] = x[torch.randint(n, (1,), generator=g, device=x.device)]
    d2 = ((x - centers[0]) ** 2).sum(-1)
    for i in range(1, k):
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=g)
        centers[i] = x[idx]
        d2 = torch.minimum(d2, ((x - centers[i]) ** 2).sum(-1))

    for _ in range(iters):
        assign = nearest(centers, x)
        sums = torch.zeros_like(centers).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).to(x.dtype).unsqueeze(-1)
        empty = counts.squeeze(-1) == 0
        centers = torch.where(empty.unsqueeze(-1), centers, sums / counts.clamp(min=1))
        if empty.any():  # re-seed dead centers on the worst-fit points
            far = ((x - centers[assign]) ** 2).sum(-1).topk(int(empty.sum())).indices
            centers[empty] = x[far]
    return centers


def prior_color_samples(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return srgb_to_oklab(torch.rand(n, 3, generator=g))


def prior_shape_samples(proto: Protocol, n: int, seed: int = 0) -> torch.Tensor:
    """Log-uniform sigmas in [sigma_min, sigma_max], uniform angle."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = math.log(proto.sigma_min), math.log(proto.sigma_max)
    sx = torch.exp(lo + (hi - lo) * torch.rand(n, generator=g))
    sy = torch.exp(lo + (hi - lo) * torch.rand(n, generator=g))
    theta = math.pi * torch.rand(n, generator=g)
    return to_logcov(*canonicalize(sx, sy, theta))
