"""sRGB <-> Oklab conversion (Björn Ottosson, 2020).

Colors are composited in gamma-encoded sRGB (what paint programs and the ORA
layer format do), but quantized in Oklab, where Euclidean distance tracks
perceived color difference. All functions take tensors with a trailing dim of 3.
"""
from __future__ import annotations

import torch

_LMS_FROM_LINEAR = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_OKLAB_FROM_LMS_ = torch.tensor([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_LMS__FROM_OKLAB = torch.linalg.inv(_OKLAB_FROM_LMS_.double()).float()
_LINEAR_FROM_LMS = torch.linalg.inv(_LMS_FROM_LINEAR.double()).float()


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=0) ** (1 / 2.4) - 0.055)


def _mat(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x @ m.to(x).T


def srgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    lms = _mat(_LMS_FROM_LINEAR, srgb_to_linear(rgb))
    return _mat(_OKLAB_FROM_LMS_, torch.sign(lms) * lms.abs() ** (1 / 3))


def oklab_to_srgb(lab: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    lms = _mat(_LMS__FROM_OKLAB, lab) ** 3
    rgb = linear_to_srgb(_mat(_LINEAR_FROM_LMS, lms))
    return rgb.clamp(0, 1) if clamp else rgb
