"""Pick a rasterizer backend: Triton tiles on CUDA float32, the dense reference otherwise.

Set LPP_RASTER=torch to force the reference (debugging, or comparing results).
"""
from __future__ import annotations

import os

import torch

from ..protocol import Protocol
from .reference import Primitives
from .reference import render_group as render_group_torch


def render_group(prims: Primitives, size: int, proto: Protocol | None = None, backend: str | None = None):
    backend = backend or os.environ.get("LPP_RASTER") or (
        "triton" if prims.xy.is_cuda and prims.xy.dtype == torch.float32 else "torch")
    if backend == "torch":
        return render_group_torch(prims, size, proto)
    if backend == "triton":
        from .triton_raster import render_group_triton
        return render_group_triton(prims, size, proto)
    raise ValueError(f"unknown rasterizer backend {backend!r}")
