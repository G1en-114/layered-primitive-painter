"""Shape parameterization for 2D generalized Gaussians.

A primitive's footprint is a covariance Sigma = R(theta) diag(sx^2, sy^2) R(theta)^T.
(sx, sy, theta) is a bad space to cluster in: (sx, sy, theta) and
(sy, sx, theta + pi/2) are the same shape, and theta is meaningless for circles.
The matrix logarithm  log Sigma = R diag(2 ln sx, 2 ln sy) R^T  has neither
problem: it is a unique, continuous 3-vector (l11, l22, l12) per shape, and
Euclidean distance in it is scale-invariant. The shape codebook lives there.
"""
from __future__ import annotations

import math

import torch


def canonicalize(sx: torch.Tensor, sy: torch.Tensor, theta: torch.Tensor):
    """Return an equivalent (sx, sy, theta) with sx >= sy and theta in [0, pi)."""
    swap = sy > sx
    sx2 = torch.where(swap, sy, sx)
    sy2 = torch.where(swap, sx, sy)
    theta2 = torch.where(swap, theta + math.pi / 2, theta)
    return sx2, sy2, torch.remainder(theta2, math.pi)


def to_logcov(sx: torch.Tensor, sy: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """(sx, sy, theta) -> (..., 3) tensor (l11, l22, l12) of log Sigma."""
    a, b = 2 * torch.log(sx), 2 * torch.log(sy)
    c, s = torch.cos(theta), torch.sin(theta)
    l11 = a * c * c + b * s * s
    l22 = a * s * s + b * c * c
    l12 = (a - b) * c * s
    return torch.stack([l11, l22, l12], dim=-1)


def from_logcov(l: torch.Tensor):
    """Inverse of to_logcov via the closed-form 2x2 symmetric eigendecomposition.

    Returns canonical (sx, sy, theta) with sx >= sy, theta in [0, pi).
    """
    l11, l22, l12 = l.unbind(-1)
    mean = 0.5 * (l11 + l22)
    radius = torch.sqrt((0.5 * (l11 - l22)) ** 2 + l12 ** 2)
    sx = torch.exp(0.5 * (mean + radius))
    sy = torch.exp(0.5 * (mean - radius))
    theta = 0.5 * torch.atan2(2 * l12, l11 - l22)
    return sx, sy, torch.remainder(theta, math.pi)


def conic(sx: torch.Tensor, sy: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Inverse covariance (A, B, C) so that d2 = A dx^2 + 2 B dx dy + C dy^2."""
    c, s = torch.cos(theta), torch.sin(theta)
    ix, iy = 1.0 / (sx * sx), 1.0 / (sy * sy)
    return torch.stack([ix * c * c + iy * s * s, (ix - iy) * c * s, ix * s * s + iy * c * c], dim=-1)
