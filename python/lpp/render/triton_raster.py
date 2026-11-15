"""Tile-based Triton rasterizer for ordered generalized 2D Gaussians (forward + backward).

Same math as render/reference.py (the oracle it is tested against), different work:
the dense reference evaluates every primitive at every pixel (B*N*H*W); here each
16x16 tile only visits primitives whose 3-sigma box overlaps it.

Pipeline
--------
1. pack (torch): per primitive x, y, conic A/B/C, beta, alpha, rgb, plus the cutoff
   constants g(c2) and d g(c2) / d beta, into one contiguous [B, N, 12] float32 array
   so a primitive's parameters share one or two cache lines.
2. bin (torch): conservative axis-aligned box of the cutoff ellipse -> tile range;
   per (image, tile) an ordered list of overlapping primitive indices (stable
   compaction keeps paint order). alpha == 0 primitives (batch padding) are skipped,
   which is exact: they change nothing.
3. forward kernel: one program per (image, tile) holds 256 pixels in registers and
   composites its list back to front:  P <- P (1 - a) + c a,  T <- T (1 - a).
4. backward kernel: same tiles, list walked front to back. With S = transmittance of
   everything in front of i and F = paint contributed by i and everything in front:
       dL/dc_i = sum_px gP * a_i S
       dL/da_i = S <gP, c_i> - (<gP, P - F_incl> + gT T) / (1 - a_i)
   P - F_incl is the paint *behind* i, obtained by subtraction from the saved output
   (no recursive division by (1 - a), which would amplify rounding error).
   Per-tile sums are reduced in registers and atomically added per primitive, so
   gradients are not bitwise deterministic (fp32 add order varies between runs).
"""
from __future__ import annotations

import os

# Triton 3.9 warns that the FMA-contraction default flips in 3.10; pin today's behaviour.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from ..protocol import Protocol, load_protocol  # noqa: E402
from ..shape import conic as conic_from_shape  # noqa: E402
from .tiles import NGRAD, NPARAM, TILE  # noqa: E402

FWD_WARPS = 4
BWD_WARPS = 4
GRAD_MODE = "tile_buffer"            # or "atomic" (v1)
TILE_BUFFER_MAX_BYTES = 1 << 30
CULL_OBB = True                      # v3: oriented-box tile culling
D2_MIN = 1e-12


# ----------------------------------------------------------------------------- kernels
@triton.jit
def _eval_alpha(px, py, base, cut2, max_alpha):
    x = tl.load(base + 0)
    y = tl.load(base + 1)
    A = tl.load(base + 2)
    Bc = tl.load(base + 3)
    C = tl.load(base + 4)
    beta = tl.load(base + 5)
    alpha = tl.load(base + 6)
    gcut = tl.load(base + 10)
    dx = px - x
    dy = py - y
    d2 = A * dx * dx + 2.0 * Bc * dx * dy + C * dy * dy
    d2c = tl.minimum(tl.maximum(d2, 1e-12), cut2)
    t = tl.exp2(beta * tl.log2(d2c))          # d2c ** beta
    g = tl.exp(-0.5 * t)
    f = (g - gcut) / (1.0 - gcut)
    a_raw = alpha * f
    a = tl.minimum(a_raw, max_alpha)
    return a, a_raw, f, g, t, d2, d2c, dx, dy, A, Bc, C, beta, alpha, gcut


@triton.jit
def _fwd_kernel(prm, lists, counts, P_out, T_out,
                N, H, W, tiles_x, n_tiles, inv_scale, cut2, max_alpha,
                TILE: tl.constexpr, NPARAM: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // n_tiles
    t = pid % n_tiles
    offs = tl.arange(0, TILE * TILE)
    iy = (t // tiles_x) * TILE + offs // TILE
    ix = (t % tiles_x) * TILE + offs % TILE
    inb = (iy < H) & (ix < W)
    px = (ix.to(tl.float32) + 0.5) * inv_scale
    py = (iy.to(tl.float32) + 0.5) * inv_scale

    acc_r = tl.zeros([TILE * TILE], dtype=tl.float32)
    acc_g = tl.zeros([TILE * TILE], dtype=tl.float32)
    acc_b = tl.zeros([TILE * TILE], dtype=tl.float32)
    trans = tl.full([TILE * TILE], 1.0, dtype=tl.float32)

    count = tl.load(counts + pid)
    lst = lists + pid.to(tl.int64) * N
    for j in range(0, count):
        n = tl.load(lst + j)
        base = prm + (b.to(tl.int64) * N + n) * NPARAM
        a, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _eval_alpha(px, py, base, cut2, max_alpha)
        a = tl.where(inb, a, 0.0)
        one_m = 1.0 - a
        acc_r = acc_r * one_m + tl.load(base + 7) * a
        acc_g = acc_g * one_m + tl.load(base + 8) * a
        acc_b = acc_b * one_m + tl.load(base + 9) * a
        trans = trans * one_m

    pix = iy * W + ix
    img = b.to(tl.int64) * H * W
    tl.store(P_out + img * 3 + pix, acc_r, mask=inb)
    tl.store(P_out + img * 3 + H * W + pix, acc_g, mask=inb)
    tl.store(P_out + img * 3 + 2 * H * W + pix, acc_b, mask=inb)
    tl.store(T_out + img + pix, trans, mask=inb)


@triton.jit
def _bwd_kernel(prm, lists, counts, P_out, T_out, gP, gT, grads,
                N, H, W, tiles_x, n_tiles, inv_scale, cut2, max_alpha,
                TILE: tl.constexpr, NPARAM: tl.constexpr, NGRAD: tl.constexpr, ATOMIC: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // n_tiles
    t = pid % n_tiles
    offs = tl.arange(0, TILE * TILE)
    iy = (t // tiles_x) * TILE + offs // TILE
    ix = (t % tiles_x) * TILE + offs % TILE
    inb = (iy < H) & (ix < W)
    px = (ix.to(tl.float32) + 0.5) * inv_scale
    py = (iy.to(tl.float32) + 0.5) * inv_scale

    pix = iy * W + ix
    img = b.to(tl.int64) * H * W
    Pr = tl.load(P_out + img * 3 + pix, mask=inb, other=0.0)
    Pg = tl.load(P_out + img * 3 + H * W + pix, mask=inb, other=0.0)
    Pb = tl.load(P_out + img * 3 + 2 * H * W + pix, mask=inb, other=0.0)
    Tf = tl.load(T_out + img + pix, mask=inb, other=0.0)
    gr = tl.load(gP + img * 3 + pix, mask=inb, other=0.0)
    gg = tl.load(gP + img * 3 + H * W + pix, mask=inb, other=0.0)
    gb = tl.load(gP + img * 3 + 2 * H * W + pix, mask=inb, other=0.0)
    gt = tl.load(gT + img + pix, mask=inb, other=0.0)
    gtT = gt * Tf

    S = tl.full([TILE * TILE], 1.0, dtype=tl.float32)   # transmittance of primitives in front
    Fr = tl.zeros([TILE * TILE], dtype=tl.float32)      # paint from primitives in front (incl. current)
    Fg = tl.zeros([TILE * TILE], dtype=tl.float32)
    Fb = tl.zeros([TILE * TILE], dtype=tl.float32)

    count = tl.load(counts + pid)
    lst = lists + pid.to(tl.int64) * N
    for jj in range(0, count):
        n = tl.load(lst + (count - 1 - jj))
        base = prm + (b.to(tl.int64) * N + n) * NPARAM
        a, a_raw, f, g, tpow, d2, d2c, dx, dy, A, Bc, C, beta, alpha, gcut = \
            _eval_alpha(px, py, base, cut2, max_alpha)
        a = tl.where(inb, a, 0.0)
        cr = tl.load(base + 7)
        cg = tl.load(base + 8)
        cb = tl.load(base + 9)
        dgcut = tl.load(base + 11)

        w = a * S
        Fr = Fr + cr * w
        Fg = Fg + cg * w
        Fb = Fb + cb * w
        inv1m = 1.0 / (1.0 - a)
        dLda = S * (gr * cr + gg * cg + gb * cb) \
            - (gr * (Pr - Fr) + gg * (Pg - Fg) + gb * (Pb - Fb) + gtT) * inv1m
        S = S * (1.0 - a)

        live = inb & (a_raw < max_alpha)              # clamped pixels pass no gradient to a_raw
        dLda = tl.where(live, dLda, 0.0)
        g_alpha = dLda * f
        dLdf = dLda * alpha
        one_m_gc = 1.0 - gcut
        # f = (g - gcut) / (1 - gcut),  g = exp(-0.5 d2c^beta)
        in_range = (d2 > 1e-12) & (d2 < cut2)
        dfdd2 = tl.where(in_range, g * (-0.5 * beta * tpow / d2c) / one_m_gc, 0.0)
        dgdbeta = g * (-0.5 * tpow * tl.log(d2c))
        dfdbeta = (dgdbeta - dgcut) / one_m_gc + (g - gcut) * dgcut / (one_m_gc * one_m_gc)
        dLdd2 = dLdf * dfdd2

        s0 = tl.sum(dLdd2 * -(2.0 * A * dx + 2.0 * Bc * dy), axis=0)
        s1 = tl.sum(dLdd2 * -(2.0 * Bc * dx + 2.0 * C * dy), axis=0)
        s2 = tl.sum(dLdd2 * dx * dx, axis=0)
        s3 = tl.sum(dLdd2 * 2.0 * dx * dy, axis=0)
        s4 = tl.sum(dLdd2 * dy * dy, axis=0)
        s5 = tl.sum(dLdf * dfdbeta, axis=0)
        s6 = tl.sum(g_alpha, axis=0)
        s7 = tl.sum(gr * w, axis=0)
        s8 = tl.sum(gg * w, axis=0)
        s9 = tl.sum(gb * w, axis=0)
        if ATOMIC:   # v1: one shared row per primitive; contended when a primitive spans many tiles
            out = grads + (b.to(tl.int64) * N + n) * NGRAD
            tl.atomic_add(out + 0, s0)
            tl.atomic_add(out + 1, s1)
            tl.atomic_add(out + 2, s2)
            tl.atomic_add(out + 3, s3)
            tl.atomic_add(out + 4, s4)
            tl.atomic_add(out + 5, s5)
            tl.atomic_add(out + 6, s6)
            tl.atomic_add(out + 7, s7)
            tl.atomic_add(out + 8, s8)
            tl.atomic_add(out + 9, s9)
        else:        # v2: a private row per (image, tile, primitive); summed over tiles afterwards
            out = grads + (pid.to(tl.int64) * N + n) * NGRAD
            tl.store(out + 0, s0)
            tl.store(out + 1, s1)
            tl.store(out + 2, s2)
            tl.store(out + 3, s3)
            tl.store(out + 4, s4)
            tl.store(out + 5, s5)
            tl.store(out + 6, s6)
            tl.store(out + 7, s7)
            tl.store(out + 8, s8)
            tl.store(out + 9, s9)


# ----------------------------------------------------------------------------- host side
from .tiles import bin_tiles as _bin_tiles  # noqa: E402
from .tiles import pack as _pack  # noqa: E402,F401


def bin_tiles(xy, conic, alpha, size, canvas, cutoff, sigma=None, theta=None):
    """tiles.bin_tiles with this module's CULL_OBB switch (kept for the benchmark's v2/v3 comparison)."""
    return _bin_tiles(xy, conic, alpha, size, canvas, cutoff, sigma, theta, cull_obb=CULL_OBB)


class _Raster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, conic, beta, rgb, alpha, size: int, canvas: int, cutoff: float, max_alpha: float,
                sigma=None, theta=None):
        B, N = alpha.shape
        cut2 = cutoff * cutoff
        with torch.no_grad():
            params = _pack(xy, conic, beta, rgb, alpha, cut2)
            lists, counts, tiles_x = bin_tiles(xy, conic, alpha, size, canvas, cutoff, sigma, theta)
        P = torch.empty(B, 3, size, size, device=xy.device, dtype=torch.float32)
        T = torch.empty(B, 1, size, size, device=xy.device, dtype=torch.float32)
        n_tiles = tiles_x * tiles_x
        if B > 0:
            _fwd_kernel[(B * n_tiles,)](params, lists, counts, P, T, N, size, size, tiles_x, n_tiles,
                                        canvas / size, cut2, max_alpha, TILE=TILE, NPARAM=NPARAM,
                                        num_warps=FWD_WARPS)
        ctx.save_for_backward(params, lists, counts, P, T)
        ctx.meta = (B, N, size, canvas, cut2, max_alpha, tiles_x)
        return P, T

    @staticmethod
    def backward(ctx, gP, gT):
        params, lists, counts, P, T = ctx.saved_tensors
        B, N, size, canvas, cut2, max_alpha, tiles_x = ctx.meta
        gP = torch.zeros_like(P) if gP is None else gP.float().contiguous()
        gT = torch.zeros_like(T) if gT is None else gT.float().contiguous()
        n_tiles = tiles_x * tiles_x
        # v2 (default): per-tile partial rows, no atomics, bitwise deterministic. Falls back to
        # v1 atomics when the [B, tiles, N, NGRAD] buffer would be too large.
        atomic = GRAD_MODE == "atomic" or B * n_tiles * N * NGRAD * 4 > TILE_BUFFER_MAX_BYTES
        if atomic:
            grads = torch.zeros(B, N, NGRAD, device=P.device, dtype=torch.float32)
        else:
            grads = torch.zeros(B, n_tiles, N, NGRAD, device=P.device, dtype=torch.float32)
        if B > 0:
            _bwd_kernel[(B * n_tiles,)](params, lists, counts, P, T, gP, gT, grads,
                                        N, size, size, tiles_x, n_tiles, canvas / size, cut2, max_alpha,
                                        TILE=TILE, NPARAM=NPARAM, NGRAD=NGRAD, ATOMIC=atomic,
                                        num_warps=BWD_WARPS)
        if not atomic:
            grads = grads.sum(1)
        g_xy, g_conic = grads[..., 0:2], grads[..., 2:5]
        g_beta, g_alpha, g_rgb = grads[..., 5], grads[..., 6], grads[..., 7:10]
        return g_xy, g_conic, g_beta, g_rgb, g_alpha, None, None, None, None, None, None


def render_group_triton(prims, size: int, proto: Protocol | None = None):
    """Drop-in replacement for reference.render_group on CUDA tensors -> (P, T)."""
    proto = proto or load_protocol()
    con = conic_from_shape(prims.sigma[..., 0], prims.sigma[..., 1], prims.theta)
    return _Raster.apply(prims.xy, con, prims.beta, prims.rgb, prims.alpha,
                         size, proto.canvas_size, proto.cutoff_sigma, proto.max_alpha,
                         prims.sigma.detach(), prims.theta.detach())
