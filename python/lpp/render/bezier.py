"""Quadratic Bezier line strokes (protocol v3 candidate, docs/line_strokes.md).

A stroke is a quadratic Bezier curve P0 -> P2 with control point P1, a constant width, a colour and an
opacity. Its coverage at a pixel depends on the distance d from the pixel centre to the curve:

    hard (rendering):  cov = clamp(w/2 - d + 0.5, 0, 1)      one-pixel antialiased edge
    soft (fitting):    cov = sigmoid((w/2 - d) / tau)         non-zero gradient around the edge

The distance is measured to a polyline of `segments` pieces through the curve. With 16 pieces the curve
deviates from the polyline by under 0.1 px at the curvatures a 256 px portrait uses. Coverage times opacity is
a per-pixel alpha [B, N, H, W], composited exactly like the Gaussians (reference.render_group): a curve group
can be layered over a Gaussian group, or interleaved with one in any order.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

MAX_ALPHA = 0.99   # as the Gaussians: keeps log(1 - a) finite in the compositing


@dataclass
class Curves:
    """Batched curve parameters, [B, N, ...], canvas pixels."""

    p0: torch.Tensor     # [B, N, 2]
    p1: torch.Tensor     # [B, N, 2] control point
    p2: torch.Tensor     # [B, N, 2]
    width: torch.Tensor  # [B, N]    full width, px
    rgb: torch.Tensor    # [B, N, 3] sRGB in [0, 1]
    alpha: torch.Tensor  # [B, N]    opacity in [0, MAX_ALPHA]

    def __getitem__(self, sl) -> "Curves":
        return Curves(**{f.name: getattr(self, f.name)[:, sl] for f in fields(self)})

    @property
    def n(self) -> int:
        return self.p0.shape[1]


def points(c: Curves, segments: int = 16) -> torch.Tensor:
    """Polyline through each curve: [B, N, segments + 1, 2]."""
    t = torch.linspace(0, 1, segments + 1, device=c.p0.device, dtype=c.p0.dtype)[:, None]
    return ((1 - t) ** 2 * c.p0[:, :, None] + 2 * (1 - t) * t * c.p1[:, :, None] + t ** 2 * c.p2[:, :, None])


def distance(c: Curves, size: int, segments: int = 16, chunk: int = 16) -> torch.Tensor:
    """Distance from every pixel centre to every curve: [B, N, H, W] (px). Chunked over curves."""
    B, N = c.p0.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(size, device=c.p0.device, dtype=c.p0.dtype) + 0.5,
                            torch.arange(size, device=c.p0.device, dtype=c.p0.dtype) + 0.5, indexing="ij")
    pix = torch.stack([xs, ys], -1).view(1, 1, 1, -1, 2)                      # [1, 1, 1, P, 2]
    pts = points(c, segments)
    out = []
    for s in range(0, N, chunk):
        a, b = pts[:, s:s + chunk, :-1, None], pts[:, s:s + chunk, 1:, None]  # [B, n, K, 1, 2]
        ab = b - a
        t = ((pix - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-8)
        q = a + t.clamp(0, 1)[..., None] * ab                                  # nearest point per segment
        d2 = ((pix - q) ** 2).sum(-1).amin(2)                                  # [B, n, P]
        out.append((d2 + 1e-8).sqrt())
    return torch.cat(out, 1).view(B, N, size, size)


def curve_alpha(c: Curves, size: int, soft: float | None = None, segments: int = 16) -> torch.Tensor:
    """Per-pixel alpha [B, N, H, W]. soft: sigmoid edge of that temperature (px) instead of the hard edge."""
    d = distance(c, size, segments)
    half = (c.width / 2)[..., None, None]
    cov = torch.sigmoid((half - d) / soft) if soft else (half - d + 0.5).clamp(0, 1)
    return (cov * c.alpha.clamp(0, MAX_ALPHA)[..., None, None]).clamp(max=MAX_ALPHA)


def composite(alpha: torch.Tensor, rgb: torch.Tensor, under: torch.Tensor) -> torch.Tensor:
    """Paint N strokes in order over an image: alpha [B, N, H, W], rgb [B, N, 3], under [B, 3, H, W]."""
    log_t = torch.log1p(-alpha)
    after = log_t.sum(1, keepdim=True) - torch.cumsum(log_t, 1)               # sum_{j > i} log(1 - a_j)
    w = alpha * after.exp()                                                   # visible share of stroke i
    return torch.einsum("bnhw,bnc->bchw", w, rgb) + log_t.sum(1, keepdim=True).exp() * under


def curve_alpha_local(c: Curves, size: int, window: int | None = None, soft: float | None = None,
                      segments: int = 16) -> torch.Tensor:
    """curve_alpha for B = 1, computed in a window x window patch around each curve and placed into the
    canvas: memory and time scale with N * window^2 instead of N * size^2. window=None sizes it to the largest
    curve of the call (span + width + margins), so no curve is ever cut; the window origin is not
    differentiated (as a rasteriser's tile)."""
    assert c.p0.shape[0] == 1, "batch of one image"
    N, dev, dt = c.n, c.p0.device, c.p0.dtype
    pts = points(c, segments)[0]                                            # [N, K + 1, 2]
    margin = 2 + (8 * soft if soft else 0)                                  # soft tail: sigmoid(-16) ~ 1e-7
    lo = torch.floor(pts.detach().amin(1) - c.width.detach()[0, :, None] / 2 - margin).long()  # [N, 2] (x, y)
    if window is None:      # fit the largest curve of the call, clipped to the canvas (only those pixels count)
        lo = lo.clamp(0, size - 1)
        hi = (pts.detach().amax(1) + c.width.detach()[0, :, None] / 2 + margin).ceil().long().clamp(1, size)
        window = (max(16, int((hi - lo).max()) + 1) + 7) // 8 * 8
    r = torch.arange(window, device=dev)
    gx = lo[:, None, 0] + r[None]                                           # [N, S] canvas x of patch columns
    gy = lo[:, None, 1] + r[None]
    pix = torch.stack([gx[:, None, :].expand(N, window, window), gy[:, :, None].expand(N, window, window)], -1)
    pix = (pix.to(dt) + 0.5).view(N, 1, -1, 2)                              # [N, 1, S*S, 2]
    a, b = pts[:, :-1, None], pts[:, 1:, None]                              # [N, K, 1, 2]
    ab = b - a
    t = ((pix - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-8)
    d = (((pix - (a + t.clamp(0, 1)[..., None] * ab)) ** 2).sum(-1).amin(1) + 1e-8).sqrt()   # [N, S*S]
    half = c.width[0, :, None] / 2
    cov = torch.sigmoid((half - d) / soft) if soft else (half - d + 0.5).clamp(0, 1)
    patch = (cov * c.alpha[0].clamp(0, MAX_ALPHA)[:, None]).clamp(max=MAX_ALPHA).view(N, window, window)
    canvas = torch.zeros(N, size + 2 * window, size + 2 * window, device=dev, dtype=dt)
    iy = (gy + window).clamp(0, size + 2 * window - 1)[:, :, None].expand(N, window, window)
    ix = (gx + window).clamp(0, size + 2 * window - 1)[:, None, :].expand(N, window, window)
    n = torch.arange(N, device=dev)[:, None, None].expand(N, window, window)
    canvas = canvas.index_put((n, iy, ix), patch)
    return canvas[None, :, window:window + size, window:window + size]


@dataclass
class Regions:
    """Filled closed shapes, [B, N, ...]: `segments` quadratic Bezier pieces end to end. pts [B, N, 2S, 2] holds,
    for piece j, its start pts[2j] and control point pts[2j+1]; piece j ends at pts[2j+2] (wrapping to pts[0]).
    Used for cel shadows and highlights (with lpp.render.blend), after LIVE / DiffVG's closed paths."""

    pts: torch.Tensor    # [B, N, 2S, 2]
    rgb: torch.Tensor    # [B, N, 3]
    alpha: torch.Tensor  # [B, N]

    @property
    def n(self) -> int:
        return self.pts.shape[1]


def outline(r: Regions, per_piece: int = 8) -> torch.Tensor:
    """Closed polyline through each region's boundary: [B, N, S * per_piece, 2] (first vertex not repeated)."""
    p0 = r.pts[:, :, 0::2]
    p1 = r.pts[:, :, 1::2]
    p2 = torch.roll(p0, -1, dims=2)
    t = torch.linspace(0, 1, per_piece + 1, device=r.pts.device, dtype=r.pts.dtype)[:-1].view(1, 1, 1, -1, 1)
    q = (1 - t) ** 2 * p0[..., None, :] + 2 * (1 - t) * t * p1[..., None, :] + t ** 2 * p2[..., None, :]
    return q.flatten(2, 3)


def region_alpha_local(r: Regions, size: int, window: int | None = None, soft: float | None = None,
                       per_piece: int = 8) -> torch.Tensor:
    """Per-pixel alpha [1, N, H, W] of filled regions (B = 1), each computed in a window around it.
    Signed distance to the boundary polyline: the magnitude from the nearest edge, the sign from an even-odd
    ray crossing test; coverage clamp(sd + 0.5, 0, 1), or sigmoid(sd / soft) for fitting."""
    assert r.pts.shape[0] == 1, "batch of one image"
    N, dev, dt = r.n, r.pts.device, r.pts.dtype
    poly = outline(r, per_piece)[0]                                        # [N, E, 2]
    margin = 2 + (8 * soft if soft else 0)
    lo = torch.floor(poly.detach().amin(1) - margin).long()               # [N, 2]
    if window is None:
        lo = lo.clamp(0, size - 1)
        hi = (poly.detach().amax(1) + margin).ceil().long().clamp(1, size)
        window = (max(16, int((hi - lo).max()) + 1) + 7) // 8 * 8
    rr = torch.arange(window, device=dev)
    gx, gy = lo[:, None, 0] + rr[None], lo[:, None, 1] + rr[None]
    px = (gx[:, None, :].expand(N, window, window).to(dt) + 0.5).reshape(N, 1, -1)   # [N, 1, P]
    py = (gy[:, :, None].expand(N, window, window).to(dt) + 0.5).reshape(N, 1, -1)
    a, b = poly, torch.roll(poly, -1, dims=1)                              # edges a -> b, [N, E, 2]
    ax, ay, bx, by = a[..., 0, None], a[..., 1, None], b[..., 0, None], b[..., 1, None]
    ex, ey = bx - ax, by - ay
    t = (((px - ax) * ex + (py - ay) * ey) / (ex * ex + ey * ey).clamp(min=1e-8)).clamp(0, 1)
    d = ((((px - ax - t * ex) ** 2 + (py - ay - t * ey) ** 2).amin(1)) + 1e-8).sqrt()   # [N, P]
    with torch.no_grad():                                                  # even-odd rule, horizontal ray to +x
        cross = ((ay > py) != (by > py)) & (px < ax + (py - ay) * ex / torch.where(ey == 0, torch.full_like(ey, 1e-8), ey))
        inside = cross.sum(1) % 2 == 1
    sd = torch.where(inside, d, -d)
    cov = torch.sigmoid(sd / soft) if soft else (sd + 0.5).clamp(0, 1)
    patch = (cov * r.alpha[0].clamp(0, MAX_ALPHA)[:, None]).clamp(max=MAX_ALPHA).view(N, window, window)
    canvas = torch.zeros(N, size + 2 * window, size + 2 * window, device=dev, dtype=dt)
    iy = (gy + window).clamp(0, size + 2 * window - 1)[:, :, None].expand(N, window, window)
    ix = (gx + window).clamp(0, size + 2 * window - 1)[:, None, :].expand(N, window, window)
    n = torch.arange(N, device=dev)[:, None, None].expand(N, window, window)
    canvas = canvas.index_put((n, iy, ix), patch)
    return canvas[None, :, window:window + size, window:window + size]
