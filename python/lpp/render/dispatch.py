"""Pick Triton, native CUDA/HIP tiles, or the dense PyTorch reference.

GPU float32 uses Triton when installed, then native tiles if they can be built.
Other devices/dtypes use the reference. LPP_RASTER=torch|triton|native overrides
automatic selection; explicit backend errors are propagated to the caller.
"""
from __future__ import annotations

import functools
import os

import torch

from ..protocol import Protocol
from .reference import Primitives
from .reference import render_group as render_group_torch


@functools.cache
def triton_available() -> bool:
    """Triton importable here? (Hygon DCU / ROCm builds of PyTorch may ship without it.)"""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def render_group(prims: Primitives, size: int, proto: Protocol | None = None, backend: str | None = None):
    backend = backend or os.environ.get("LPP_RASTER")
    if backend is None:
        backend = "torch"
        if prims.xy.is_cuda and prims.xy.dtype == torch.float32:
            if triton_available():
                backend = "triton"
            else:
                from .native_raster import available
                if available():
                    backend = "native"
    if backend == "torch":
        return render_group_torch(prims, size, proto)
    if backend == "triton":
        from .triton_raster import render_group_triton
        return render_group_triton(prims, size, proto)
    if backend == "native":
        from .native_raster import render_group_native
        return render_group_native(prims, size, proto)
    raise ValueError(f"unknown rasterizer backend {backend!r}")
