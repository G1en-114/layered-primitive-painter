"""Backend-independent host side of the tile rasterizers (pure torch, no Triton).

pack       per primitive x, y, conic A/B/C, beta, alpha, rgb, plus the cutoff constants
           g(c2) and d g(c2) / d beta, as one contiguous [B, N, 12] float32 array.
bin_tiles  per (image, 16x16 tile) the ordered list of primitives that can touch it.

Shared by the Triton kernels (NVIDIA) and the native CUDA/HIP extension (NVIDIA + DCU).
"""
from __future__ import annotations

import math

import torch

TILE = 16
NPARAM = 12   # x, y, A, B, C, beta, alpha, r, g, b, gcut, dgcut/dbeta
NGRAD = 10    # x, y, A, B, C, beta, alpha, r, g, b


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def pack(xy, conic, beta, rgb, alpha, cut2: float) -> torch.Tensor:
    c2_beta = torch.pow(torch.full_like(beta, cut2), beta)
    gcut = torch.exp(-0.5 * c2_beta)
    dgcut = gcut * (-0.5 * c2_beta * math.log(cut2))      # d gcut / d beta
    return torch.cat([xy, conic, beta[..., None], alpha[..., None], rgb, gcut[..., None], dgcut[..., None]],
                     dim=-1).float().contiguous()


def bin_tiles(xy, conic, alpha, size: int, canvas: int, cutoff: float, sigma=None, theta=None,
              cull_obb: bool = True):
    """Ordered per-tile primitive lists. Returns lists [B*T, N] int32, counts [B*T] int32, tiles_x.

    Always: the axis-aligned box of the cutoff ellipse. With sigma/theta and cull_obb:
    also a separating-axis test against the ellipse's own oriented box, which drops the
    tiles an elongated, rotated primitive's axis-aligned box covers but the shape never
    reaches. Both tests are conservative: a tile is only dropped if no pixel center in it
    can be inside the cutoff.
    """
    B, N = alpha.shape
    tiles_x = cdiv(size, TILE)
    scale = size / canvas
    A, Bc, C = conic.unbind(-1)
    det = (A * C - Bc * Bc).clamp(min=1e-30)
    rx = cutoff * torch.sqrt(C / det) * scale + 1.0     # +1 px slack: conservative is always safe
    ry = cutoff * torch.sqrt(A / det) * scale + 1.0
    xs, ys = xy[..., 0] * scale - 0.5, xy[..., 1] * scale - 0.5   # pixel-index space
    x0 = torch.floor((xs - rx) / TILE).clamp(min=0)
    x1 = torch.floor((xs + rx) / TILE).clamp(max=tiles_x - 1)
    y0 = torch.floor((ys - ry) / TILE).clamp(min=0)
    y1 = torch.floor((ys + ry) / TILE).clamp(max=tiles_x - 1)
    live = (alpha > 0) & (x0 <= x1) & (y0 <= y1)
    t = torch.arange(tiles_x, device=xy.device, dtype=x0.dtype)
    in_x = (x0[:, None, None, :] <= t[None, None, :, None]) & (t[None, None, :, None] <= x1[:, None, None, :])
    in_y = (y0[:, None, None, :] <= t[None, :, None, None]) & (t[None, :, None, None] <= y1[:, None, None, :])
    mask = in_x & in_y & live[:, None, None, :]                 # [B, TY, TX, N]
    if cull_obb and sigma is not None:
        c, s = torch.cos(theta), torch.sin(theta)                # ellipse axes u = (c, s), v = (-s, c)
        ex, ey = cutoff * sigma[..., 0], cutoff * sigma[..., 1]  # oriented half extents, canvas units
        inv = 1.0 / scale
        tc = (t * TILE + TILE / 2) * inv                         # tile centers, canvas units
        half = (TILE / 2) * inv                                  # tile half size (covers its pixel centers)
        dxc = tc[None, None, :, None] - xy[:, None, None, :, 0]  # [B, 1, TX, N]
        dyc = tc[None, :, None, None] - xy[:, None, None, :, 1]  # [B, TY, 1, N]
        cu, su = c[:, None, None, :], s[:, None, None, :]
        rad_u = ex[:, None, None, :] + half * (cu.abs() + su.abs())
        rad_v = ey[:, None, None, :] + half * (su.abs() + cu.abs())
        mask &= (dxc * cu + dyc * su).abs() <= rad_u
        mask &= (-dxc * su + dyc * cu).abs() <= rad_v
    mask = mask.reshape(B, tiles_x * tiles_x, N)

    pos = mask.cumsum(-1) - 1                                    # stable compaction keeps paint order
    target = torch.where(mask, pos, torch.full_like(pos, N))     # non-members go to a dump column
    idx = torch.arange(N, device=xy.device, dtype=torch.int32).expand_as(target)
    lists = torch.zeros(B, tiles_x * tiles_x, N + 1, dtype=torch.int32, device=xy.device)
    lists.scatter_(-1, target, idx)
    counts = mask.sum(-1, dtype=torch.int32)
    return lists[..., :N].reshape(B * tiles_x * tiles_x, N).contiguous(), counts.reshape(-1), tiles_x
