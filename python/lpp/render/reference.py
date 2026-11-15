"""Dense PyTorch reference rasterizer for ordered generalized Gaussians.

This is the *correctness oracle*: simple, differentiable, slow. The Triton /
CUDA / SIMD rasterizers written later are tested against it.

Model
-----
Primitives are painted in order (index 0 first, i.e. furthest back). Primitive i
covers pixel p with opacity

    a_i(p) = min(alpha_i * exp(-0.5 * d2^beta_i), max_alpha)   if d2 <= cutoff^2
           = 0                                                   otherwise
    d2     = (p - mu_i)^T Sigma_i^{-1} (p - mu_i)

and is composited with the "over" operator in gamma-encoded sRGB.

Groups and associativity
------------------------
A group of primitives (a layer, or a chunk of the sequence) reduces to a
premultiplied pair (P, T):

    P = sum_i c_i a_i prod_{j>i} (1 - a_j)      color the group adds
    T = prod_i (1 - a_i)                         transmittance through the group

and two groups stack as  (P_back, T_back) under (P_front, T_front)
    = (P_front + T_front * P_back, T_front * T_back).

This operator is associative, so rendering all primitives at once, rendering in
chunks, or rendering per layer and stacking the layers all give the same image.
That is what makes exact layered export (and the REDRAW_k ordering) possible.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..protocol import Protocol, load_protocol
from ..shape import conic


@dataclass
class Primitives:
    """Continuous (decoded) primitive parameters, batched as [B, N, ...].

    Padding primitives use alpha = 0: they then contribute nothing.
    """

    xy: torch.Tensor     # [B, N, 2] center, canvas pixels
    sigma: torch.Tensor  # [B, N, 2] (sx, sy), canvas pixels
    theta: torch.Tensor  # [B, N]    radians
    beta: torch.Tensor   # [B, N]    edge hardness, >= 1
    rgb: torch.Tensor    # [B, N, 3] sRGB in [0, 1]
    alpha: torch.Tensor  # [B, N]    peak opacity in [0, max_alpha]

    def __getitem__(self, sl: slice) -> "Primitives":
        """Slice along the primitive (N) axis."""
        return Primitives(**{f.name: getattr(self, f.name)[:, sl] for f in fields(self)})

    def with_alpha(self, alpha: torch.Tensor) -> "Primitives":
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        kw["alpha"] = alpha
        return Primitives(**kw)

    @property
    def num(self) -> int:
        return self.alpha.shape[1]


def pixel_grid(size: int, proto: Protocol, device, dtype=torch.float32):
    """Pixel-center coordinates in canvas units for a size x size render."""
    scale = proto.canvas_size / size
    c = (torch.arange(size, device=device, dtype=dtype) + 0.5) * scale
    return c.view(1, 1, 1, size), c.view(1, 1, size, 1)  # px [.., W], py [.., H, 1]


def pixel_alpha(prims: Primitives, size: int, proto: Protocol | None = None) -> torch.Tensor:
    """Per-primitive opacity maps, [B, N, H, W]."""
    proto = proto or load_protocol()
    px, py = pixel_grid(size, proto, prims.xy.device, prims.xy.dtype)
    A, B, C = conic(prims.sigma[..., 0], prims.sigma[..., 1], prims.theta).unbind(-1)
    dx = px - prims.xy[..., 0, None, None]
    dy = py - prims.xy[..., 1, None, None]
    d2 = A[..., None, None] * dx * dx + 2 * B[..., None, None] * dx * dy + C[..., None, None] * dy * dy

    cut2 = proto.cutoff_sigma ** 2
    beta = prims.beta[..., None, None]
    # clamp before pow: the upper bound keeps d2**beta finite (and its gradient non-NaN)
    # outside the cutoff; the tiny lower bound removes negative d2 from rounding and keeps
    # d/dbeta = d2**beta * ln(d2) finite at the center, so the decomposer can optimize beta.
    g = torch.exp(-0.5 * d2.clamp(1e-12, cut2) ** beta)
    # Shift-and-rescale so the falloff reaches exactly 0 *at* the cutoff instead of
    # jumping from g(cut) to 0 there (1.1% for beta = 1): no ring artifact, no gradient
    # discontinuity. Pixels beyond the cutoff are clamped to d2 = cut2 and so get 0.
    g_cut = torch.exp(-0.5 * torch.full_like(beta, cut2) ** beta)
    falloff = (g - g_cut) / (1 - g_cut)
    a = prims.alpha[..., None, None] * falloff
    return a.clamp(max=proto.max_alpha)


def render_group(prims: Primitives, size: int, proto: Protocol | None = None):
    """Reduce an ordered group to premultiplied (P [B,3,H,W], T [B,1,H,W])."""
    a = pixel_alpha(prims, size, proto)                      # [B, N, H, W]
    log_t = torch.log1p(-a)                                  # log(1 - a_i), finite since a <= 0.99
    total = log_t.sum(1, keepdim=True)
    # exclusive suffix sum: sum_{j > i} log(1 - a_j)
    after = total - torch.cumsum(log_t, dim=1)
    weight = a * torch.exp(after)                            # visibility of each primitive's paint
    P = torch.einsum("bnhw,bnc->bchw", weight, prims.rgb)
    T = torch.exp(total)
    return P, T


def over(back, front):
    """Stack `front` (P, T) on top of `back` (P, T)."""
    (Pb, Tb), (Pf, Tf) = back, front
    return Pf + Tf * Pb, Tf * Tb


def flatten_on_base(group, proto: Protocol | None = None) -> torch.Tensor:
    """Composite a (P, T) group over the opaque canvas base color -> [B,3,H,W]."""
    proto = proto or load_protocol()
    P, T = group
    base = torch.tensor(proto.base_color, device=P.device, dtype=P.dtype).view(1, 3, 1, 1)
    return P + T * base


def render(prims: Primitives, size: int | None = None, chunk: int | None = None,
           proto: Protocol | None = None) -> torch.Tensor:
    """Render to an sRGB image [B, 3, size, size].

    `chunk` bounds memory to B*chunk*H*W floats by folding chunks with `over`.
    """
    proto = proto or load_protocol()
    size = size or proto.canvas_size
    chunk = chunk or max(prims.num, 1)
    acc = None
    for start in range(0, prims.num, chunk):
        g = render_group(prims[start : start + chunk], size, proto)
        acc = g if acc is None else over(acc, g)
    if acc is None:  # no primitives at all
        B = prims.alpha.shape[0]
        z = prims.xy.new_zeros(B, 3, size, size)
        acc = (z, torch.ones_like(z[:, :1]))
    return flatten_on_base(acc, proto)


def render_layers(prims: Primitives, layer_id: torch.Tensor, size: int | None = None,
                  proto: Protocol | None = None):
    """Render every layer to its own premultiplied (P, T).

    layer_id: [B, N] long. Returns a list of length num_layers, back to front.
    Primitives keep their sequence order inside each layer.
    """
    proto = proto or load_protocol()
    size = size or proto.canvas_size
    return [
        render_group(prims.with_alpha(prims.alpha * (layer_id == k)), size, proto)
        for k in range(proto.num_layers)
    ]


def stack_layers(layers, visible=None, proto: Protocol | None = None) -> torch.Tensor:
    """Composite per-layer groups back-to-front onto the canvas base.

    visible: optional iterable of bools, to toggle layers like an editor would.
    """
    acc = None
    for k, g in enumerate(layers):
        if visible is not None and not visible[k]:
            continue
        acc = g if acc is None else over(acc, g)
    if acc is None:
        P = layers[0][0]
        acc = (torch.zeros_like(P), torch.ones_like(P[:, :1]))
    return flatten_on_base(acc, proto)


def to_straight_rgba(group) -> torch.Tensor:
    """Premultiplied (P, T) -> straight-alpha RGBA [B,4,H,W] for PNG / ORA export."""
    P, T = group
    A = 1 - T
    rgb = torch.where(A > 1e-6, P / A.clamp(min=1e-6), torch.zeros_like(P))
    return torch.cat([rgb.clamp(0, 1), A], dim=1)
