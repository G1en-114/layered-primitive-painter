"""Tile-based Triton rasterizer for quadratic Bezier line strokes (forward + backward), protocol v3.

Same math as render/bezier.py (curve_alpha + over-compositing, the oracle it is tested against), the work of
render/triton_raster.py: each 16x16 tile only visits the lines whose box overlaps it.

Per line, packed [B, N, 12]: P0, P1, P2 (x, y), half width, opacity, rgb, edge softness tau (0 = the hard
one-pixel ramp). Per pixel the kernel finds the nearest point q of the 16-piece polyline through the curve,
d = |pixel - q|, coverage = sigmoid((half - d) / tau) or clamp(half - d + 0.5, 0, 1), a = min(cov * alpha, max).

Backward: compositing as triton_raster (S = transmittance in front, F = paint in front incl. the current
line; dL/da_i = S <gP, c_i> - (<gP, P - F> + gT T) / (1 - a_i)). Geometry by the envelope theorem: at the
nearest point the derivative of the distance along the piece vanishes (or the piece parameter t is clamped
at an end), so only q's dependence on the control points counts:
    q = (1 - t) B(ta) + t B(tb),  B(s) = (1-s)^2 P0 + 2 (1-s) s P1 + s^2 P2,  dd/dq = (q - pixel) / d.
Per (image, tile, line) partial rows, summed afterwards (no atomics, deterministic).
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from .tiles import TILE, cdiv  # noqa: E402

NPARAM_L = 12   # x0 y0 x1 y1 x2 y2 half alpha r g b tau
NGRAD_L = 11    # x0 y0 x1 y1 x2 y2 half alpha r g b
SEGMENTS = 16


@triton.jit
def _nearest(px, py, base, K: tl.constexpr):
    x0 = tl.load(base + 0)
    y0 = tl.load(base + 1)
    x1 = tl.load(base + 2)
    y1 = tl.load(base + 3)
    x2 = tl.load(base + 4)
    y2 = tl.load(base + 5)
    best = tl.full(px.shape, 1e30, dtype=tl.float32)
    bt = tl.zeros(px.shape, dtype=tl.float32)
    bj = tl.zeros(px.shape, dtype=tl.float32)
    bqx = tl.zeros(px.shape, dtype=tl.float32)
    bqy = tl.zeros(px.shape, dtype=tl.float32)
    for j in tl.static_range(K):
        ta = j / K
        tb = (j + 1) / K
        ax = (1 - ta) * (1 - ta) * x0 + 2 * (1 - ta) * ta * x1 + ta * ta * x2
        ay = (1 - ta) * (1 - ta) * y0 + 2 * (1 - ta) * ta * y1 + ta * ta * y2
        bx = (1 - tb) * (1 - tb) * x0 + 2 * (1 - tb) * tb * x1 + tb * tb * x2
        by = (1 - tb) * (1 - tb) * y0 + 2 * (1 - tb) * tb * y1 + tb * tb * y2
        ex = bx - ax
        ey = by - ay
        t = ((px - ax) * ex + (py - ay) * ey) / tl.maximum(ex * ex + ey * ey, 1e-8)
        t = tl.minimum(tl.maximum(t, 0.0), 1.0)
        qx = ax + t * ex
        qy = ay + t * ey
        d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy)
        better = d2 < best
        best = tl.where(better, d2, best)
        bt = tl.where(better, t, bt)
        bj = tl.where(better, j * 1.0, bj)
        bqx = tl.where(better, qx, bqx)
        bqy = tl.where(better, qy, bqy)
    return best, bt, bj, bqx, bqy


@triton.jit
def _coverage(d, half, tau):
    z = (half - d) / tl.maximum(tau, 1e-6)
    sig = 1.0 / (1.0 + tl.exp(-z))
    v = half - d + 0.5
    hard = tl.minimum(tl.maximum(v, 0.0), 1.0)
    soft = tau > 0
    cov = tl.where(soft, sig, hard)
    dcov_dd = tl.where(soft, -sig * (1.0 - sig) / tl.maximum(tau, 1e-6), tl.where((v > 0) & (v < 1), -1.0, 0.0))
    return cov, dcov_dd


@triton.jit
def _fwd_kernel(prm, lists, counts, P_out, T_out, N, H, W, tiles_x, n_tiles, inv_scale, max_alpha,
                TILE: tl.constexpr, NPARAM: tl.constexpr, K: tl.constexpr):
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
        best, _, _, _, _ = _nearest(px, py, base, K)
        cov, _ = _coverage(tl.sqrt(best + 1e-8), tl.load(base + 6), tl.load(base + 11))
        a = tl.minimum(cov * tl.load(base + 7), max_alpha)
        a = tl.where(inb, a, 0.0)
        one_m = 1.0 - a
        acc_r = acc_r * one_m + tl.load(base + 8) * a
        acc_g = acc_g * one_m + tl.load(base + 9) * a
        acc_b = acc_b * one_m + tl.load(base + 10) * a
        trans = trans * one_m
    pix = iy * W + ix
    img = b.to(tl.int64) * H * W
    tl.store(P_out + img * 3 + pix, acc_r, mask=inb)
    tl.store(P_out + img * 3 + H * W + pix, acc_g, mask=inb)
    tl.store(P_out + img * 3 + 2 * H * W + pix, acc_b, mask=inb)
    tl.store(T_out + img + pix, trans, mask=inb)


@triton.jit
def _bwd_kernel(prm, lists, counts, P_out, T_out, gP, gT, grads, N, H, W, tiles_x, n_tiles, inv_scale, max_alpha,
                TILE: tl.constexpr, NPARAM: tl.constexpr, NGRAD: tl.constexpr, K: tl.constexpr):
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
    gtT = tl.load(gT + img + pix, mask=inb, other=0.0) * Tf
    S = tl.full([TILE * TILE], 1.0, dtype=tl.float32)
    Fr = tl.zeros([TILE * TILE], dtype=tl.float32)
    Fg = tl.zeros([TILE * TILE], dtype=tl.float32)
    Fb = tl.zeros([TILE * TILE], dtype=tl.float32)
    count = tl.load(counts + pid)
    lst = lists + pid.to(tl.int64) * N
    for jj in range(0, count):
        n = tl.load(lst + (count - 1 - jj))
        base = prm + (b.to(tl.int64) * N + n) * NPARAM
        best, tq, jq, qx, qy = _nearest(px, py, base, K)
        d = tl.sqrt(best + 1e-8)
        half = tl.load(base + 6)
        alpha = tl.load(base + 7)
        cov, dcov_dd = _coverage(d, half, tl.load(base + 11))
        a_raw = cov * alpha
        a = tl.where(inb, tl.minimum(a_raw, max_alpha), 0.0)
        cr = tl.load(base + 8)
        cg = tl.load(base + 9)
        cb = tl.load(base + 10)
        w = a * S
        Fr = Fr + cr * w
        Fg = Fg + cg * w
        Fb = Fb + cb * w
        dLda = S * (gr * cr + gg * cg + gb * cb) - (gr * (Pr - Fr) + gg * (Pg - Fg) + gb * (Pb - Fb) + gtT) / (1.0 - a)
        S = S * (1.0 - a)
        dLda = tl.where(inb & (a_raw < max_alpha), dLda, 0.0)
        dLdd = dLda * alpha * dcov_dd
        gqx = dLdd * (qx - px) / d
        gqy = dLdd * (qy - py) / d
        ta = jq / K
        tb = (jq + 1.0) / K
        w0 = (1 - tq) * (1 - ta) * (1 - ta) + tq * (1 - tb) * (1 - tb)
        w1 = (1 - tq) * 2 * (1 - ta) * ta + tq * 2 * (1 - tb) * tb
        w2 = (1 - tq) * ta * ta + tq * tb * tb
        out = grads + (pid.to(tl.int64) * N + n) * NGRAD
        tl.store(out + 0, tl.sum(gqx * w0, axis=0))
        tl.store(out + 1, tl.sum(gqy * w0, axis=0))
        tl.store(out + 2, tl.sum(gqx * w1, axis=0))
        tl.store(out + 3, tl.sum(gqy * w1, axis=0))
        tl.store(out + 4, tl.sum(gqx * w2, axis=0))
        tl.store(out + 5, tl.sum(gqy * w2, axis=0))
        tl.store(out + 6, tl.sum(-dLdd, axis=0))                 # cov depends on half - d
        tl.store(out + 7, tl.sum(dLda * cov, axis=0))
        tl.store(out + 8, tl.sum(gr * w, axis=0))
        tl.store(out + 9, tl.sum(gg * w, axis=0))
        tl.store(out + 10, tl.sum(gb * w, axis=0))


def pack_lines(p0, p1, p2, half, rgb, alpha, tau: float) -> torch.Tensor:
    t = torch.full_like(half, float(tau))
    return torch.cat([p0, p1, p2, half[..., None], alpha[..., None], rgb, t[..., None]], -1).float().contiguous()


def bin_lines(p0, p1, p2, half, alpha, size: int, canvas: int, tau: float):
    """Ordered per-tile line lists from the control-point box (a quadratic Bezier stays inside the triangle of
    its control points) widened by the half width and the edge's reach. Same layout as tiles.bin_tiles."""
    B, N = alpha.shape
    tiles_x = cdiv(size, TILE)
    scale = size / canvas
    reach = half + (8 * tau if tau else 0.5) + 1.0                # +1 px slack, as tiles.bin_tiles
    pts = torch.stack([p0, p1, p2], -2)                           # [B, N, 3, 2]
    lo = (pts.amin(-2) - reach[..., None]) * scale - 0.5
    hi = (pts.amax(-2) + reach[..., None]) * scale - 0.5
    x0 = torch.floor(lo[..., 0] / TILE).clamp(min=0)
    x1 = torch.floor(hi[..., 0] / TILE).clamp(max=tiles_x - 1)
    y0 = torch.floor(lo[..., 1] / TILE).clamp(min=0)
    y1 = torch.floor(hi[..., 1] / TILE).clamp(max=tiles_x - 1)
    live = (alpha > 0) & (x0 <= x1) & (y0 <= y1)
    t = torch.arange(tiles_x, device=p0.device, dtype=x0.dtype)
    in_x = (x0[:, None, None, :] <= t[None, None, :, None]) & (t[None, None, :, None] <= x1[:, None, None, :])
    in_y = (y0[:, None, None, :] <= t[None, :, None, None]) & (t[None, :, None, None] <= y1[:, None, None, :])
    mask = (in_x & in_y & live[:, None, None, :]).reshape(B, tiles_x * tiles_x, N)
    pos = mask.cumsum(-1) - 1
    target = torch.where(mask, pos, torch.full_like(pos, N))
    idx = torch.arange(N, device=p0.device, dtype=torch.int32).expand_as(target)
    lists = torch.zeros(B, tiles_x * tiles_x, N + 1, dtype=torch.int32, device=p0.device)
    lists.scatter_(-1, target, idx)
    return lists[..., :N].reshape(B * tiles_x * tiles_x, N).contiguous(), mask.sum(-1, dtype=torch.int32).reshape(-1), tiles_x


class _Lines(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p0, p1, p2, half, rgb, alpha, size: int, canvas: int, tau: float, max_alpha: float):
        B, N = alpha.shape
        with torch.no_grad():
            params = pack_lines(p0, p1, p2, half, rgb, alpha, tau)
            lists, counts, tiles_x = bin_lines(p0, p1, p2, half, alpha, size, canvas, tau)
        P = torch.empty(B, 3, size, size, device=p0.device, dtype=torch.float32)
        T = torch.empty(B, 1, size, size, device=p0.device, dtype=torch.float32)
        n_tiles = tiles_x * tiles_x
        if B and N:
            _fwd_kernel[(B * n_tiles,)](params, lists, counts, P, T, N, size, size, tiles_x, n_tiles, canvas / size,
                                        max_alpha, TILE=TILE, NPARAM=NPARAM_L, K=SEGMENTS, num_warps=4)
        else:
            P.zero_(), T.fill_(1.0)
        ctx.save_for_backward(params, lists, counts, P, T)
        ctx.meta = (B, N, size, canvas, max_alpha, tiles_x)
        return P, T

    @staticmethod
    def backward(ctx, gP, gT):
        params, lists, counts, P, T = ctx.saved_tensors
        B, N, size, canvas, max_alpha, tiles_x = ctx.meta
        gP = torch.zeros_like(P) if gP is None else gP.float().contiguous()
        gT = torch.zeros_like(T) if gT is None else gT.float().contiguous()
        n_tiles = tiles_x * tiles_x
        grads = torch.zeros(B, n_tiles, N, NGRAD_L, device=P.device, dtype=torch.float32)
        if B and N:
            _bwd_kernel[(B * n_tiles,)](params, lists, counts, P, T, gP, gT, grads, N, size, size, tiles_x, n_tiles,
                                        canvas / size, max_alpha, TILE=TILE, NPARAM=NPARAM_L, NGRAD=NGRAD_L,
                                        K=SEGMENTS, num_warps=4)
        g = grads.sum(1)
        return g[..., 0:2], g[..., 2:4], g[..., 4:6], g[..., 6], g[..., 8:11], g[..., 7], None, None, None, None


def render_lines_triton(c, size: int, canvas: int = 256, soft: float | None = None, max_alpha: float = 0.99):
    """Bezier line strokes (lpp.render.bezier.Curves, B >= 1) in order -> premultiplied group (P [B,3,H,W],
    T [B,1,H,W]), differentiable in every curve parameter. soft: sigmoid edge temperature (fitting)."""
    return _Lines.apply(c.p0, c.p1, c.p2, c.width / 2, c.rgb, c.alpha.clamp(0, max_alpha), size, canvas,
                        float(soft or 0.0), max_alpha)
