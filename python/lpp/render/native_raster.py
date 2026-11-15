"""Native (CUDA C++ / HIP) tile rasterizer: csrc/raster/raster_ext.cu, built on first use.

The same source builds with nvcc on NVIDIA and, after PyTorch's automatic hipify, with
hipcc on Hygon DCU / AMD. Host side (packing, tile binning) is shared with the Triton
backend (tiles.py); only the two kernels differ.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import torch

from ..protocol import Protocol, load_protocol
from ..shape import conic as conic_from_shape
from .tiles import bin_tiles, pack

SRC = Path(__file__).resolve().parents[3] / "csrc" / "raster" / "raster_ext.cu"


def _space_free_torch_lib() -> None:
    """torch writes -L<torch>/lib unquoted into the ninja link line; a venv path with a
    space (e.g. "~/AI infra/.venv") then fails with "cannot find -lc10". Point it at a
    space-free symlink instead."""
    from torch.utils import cpp_extension as ce
    if " " not in ce.TORCH_LIB_PATH:
        return
    link = Path.home() / ".cache" / "lpp" / "torch-lib"
    link.parent.mkdir(parents=True, exist_ok=True)
    real = os.path.realpath(ce.TORCH_LIB_PATH)
    if link.is_symlink() and os.readlink(link) != real:
        link.unlink()
    if not link.exists():
        link.symlink_to(real, target_is_directory=True)
    ce.TORCH_LIB_PATH = str(link)


@cache
def extension(verbose: bool = False):
    from torch.utils.cpp_extension import load
    _space_free_torch_lib()
    build = Path(os.environ.get("LPP_EXT_BUILD", Path.home() / ".cache" / "lpp" / "ext"))
    build.mkdir(parents=True, exist_ok=True)
    return load(name="lpp_raster_native", sources=[str(SRC)], build_directory=str(build),
                extra_cuda_cflags=["-O3"], verbose=verbose)


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        extension()
        return True
    except Exception:  # noqa: BLE001 - no compiler, build error: fall back
        return False


class _Raster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, conic, beta, rgb, alpha, size: int, canvas: int, cutoff: float, max_alpha: float,
                sigma=None, theta=None):
        ext = extension()
        B, N = alpha.shape
        cut2 = cutoff * cutoff
        with torch.no_grad():
            params = pack(xy, conic, beta, rgb, alpha, cut2)
            lists, counts, tiles_x = bin_tiles(xy, conic, alpha, size, canvas, cutoff, sigma, theta)
        P, T = ext.fwd(params, lists, counts, B, N, size, tiles_x, canvas / size, cut2, max_alpha)
        ctx.save_for_backward(params, lists, counts, P, T)
        ctx.meta = (B, N, size, canvas, cut2, max_alpha, tiles_x)
        return P, T

    @staticmethod
    def backward(ctx, gP, gT):
        params, lists, counts, P, T = ctx.saved_tensors
        B, N, size, canvas, cut2, max_alpha, tiles_x = ctx.meta
        gP = torch.zeros_like(P) if gP is None else gP.float().contiguous()
        gT = torch.zeros_like(T) if gT is None else gT.float().contiguous()
        grads = extension().bwd(params, lists, counts, P, T, gP, gT, B, N, size, tiles_x, canvas / size,
                                cut2, max_alpha)
        return (grads[..., 0:2], grads[..., 2:5], grads[..., 5], grads[..., 7:10], grads[..., 6],
                None, None, None, None, None, None)


def render_group_native(prims, size: int, proto: Protocol | None = None):
    """Drop-in replacement for reference.render_group on GPU tensors -> (P, T)."""
    proto = proto or load_protocol()
    con = conic_from_shape(prims.sigma[..., 0], prims.sigma[..., 1], prims.theta)
    return _Raster.apply(prims.xy, con, prims.beta, prims.rgb, prims.alpha, size, proto.canvas_size,
                         proto.cutoff_sigma, proto.max_alpha, prims.sigma.detach(), prims.theta.detach())
