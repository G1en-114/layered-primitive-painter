"""Load protocol.json and expose it as typed constants.

protocol.json is the single source of truth shared by the Python side and the
C++ runtime; nothing in this package should hard-code a bin count, a layer id,
an event id or the list of primitive attributes.

Versions: protocol.json is v2 (factorized shape: sigma_x, sigma_y, theta, beta bins).
protocols/v1.json keeps v1 (one shape codebook token) for data and checkpoints made with it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "protocol.json"
PROTOCOL_V3_PATH = REPO_ROOT / "protocol_v3.json"      # + LINE strokes (docs/line_strokes.md)


@dataclass(frozen=True)
class UniformAttr:
    """A scalar attribute quantized to `bins` evenly spaced levels in [min, max].

    Level i decodes to exactly min + i * step, so both ends are representable.
    """

    name: str
    bins: int
    min: float
    max: float

    @property
    def step(self) -> float:
        return (self.max - self.min) / (self.bins - 1)

    def quantize(self, value: torch.Tensor) -> torch.Tensor:
        idx = torch.round((value - self.min) / self.step)
        return idx.clamp_(0, self.bins - 1).long()

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return self.min + idx.to(torch.float32) * self.step


@dataclass(frozen=True)
class LogUniformAttr:
    """Levels evenly spaced in log(value) over [min, max] (scales: sigma, beta)."""

    name: str
    bins: int
    min: float
    max: float

    def _lin(self) -> UniformAttr:
        return UniformAttr(self.name, self.bins, math.log(self.min), math.log(self.max))

    def quantize(self, value: torch.Tensor) -> torch.Tensor:
        return self._lin().quantize(torch.log(value.clamp(min=self.min * 1e-3)))

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._lin().dequantize(idx))


@dataclass(frozen=True)
class PeriodicAttr:
    """`bins` levels i * period / bins on a circle; values wrap (angles)."""

    name: str
    bins: int
    period: float

    def quantize(self, value: torch.Tensor) -> torch.Tensor:
        idx = torch.round(torch.remainder(value, self.period) / self.period * self.bins).long()
        return torch.remainder(idx, self.bins)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return idx.to(torch.float32) * (self.period / self.bins)


OrdinalAttr = UniformAttr | LogUniformAttr | PeriodicAttr


@dataclass(frozen=True)
class CodebookAttr:
    name: str
    size: int
    file: str
    codes_per_beta: int | None = None  # shape only


@dataclass(frozen=True)
class Layer:
    id: int
    name: str
    max_primitives: int
    celebamask: tuple[str, ...]


@dataclass(frozen=True)
class Protocol:
    raw: dict
    canvas_size: int
    base_color: tuple[float, float, float]
    max_positions: int
    max_text_positions: int
    cutoff_sigma: float
    max_alpha: float
    sigma_min: float
    sigma_max: float
    beta_levels: tuple[float, ...]
    attributes: tuple[OrdinalAttr | CodebookAttr, ...]
    events: tuple[str, ...]
    layers: tuple[Layer, ...]
    version: int = 1

    # ---- lookups -------------------------------------------------------
    @property
    def attr_names(self) -> tuple[str, ...]:
        """Primitive attribute tokens in decoding order."""
        return tuple(a.name for a in self.attributes)

    @property
    def factorized_shape(self) -> bool:
        return "sigma_x" in self.attr_names

    def attr(self, name: str) -> OrdinalAttr | CodebookAttr:
        for a in self.attributes:
            if a.name == name:
                return a
        raise KeyError(name)

    def event_id(self, name: str) -> int:
        return self.events.index(name)

    def layer_event(self, layer: int) -> int:
        return self.event_id(f"LAYER_{layer}")

    def redraw_event(self, layer: int) -> int:
        return self.event_id(f"REDRAW_{layer}")

    def refine_event(self, layer: int) -> int:
        return self.event_id(f"REFINE_{layer}")

    # ---- strokes: Gaussians (PRIMITIVE) and, from v3, Bezier lines (LINE) ----------------
    @property
    def has_lines(self) -> bool:
        return "LINE" in self.events

    @property
    def stroke_events(self) -> tuple[int, ...]:
        """Event ids of records that carry attributes (painted strokes)."""
        return (self.event_id("PRIMITIVE"),) + ((self.event_id("LINE"),) if self.has_lines else ())

    def fixed_tokens(self, event: int) -> dict[str, int]:
        """Attributes a stroke type does not use, with the one token they always carry: a LINE has no edge
        hardness (beta = 0), a Gaussian no bend (the middle bin, 0)."""
        if not self.has_lines:
            return {}
        if event == self.event_id("LINE"):
            return {"beta": 0}
        return {"bend": self.attr("bend").bins // 2}

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def attr_vocab_sizes(self) -> dict[str, int]:
        return {a.name: (a.size if isinstance(a, CodebookAttr) else a.bins) for a in self.attributes}

    def worst_case_positions(self) -> int:
        """Text prefix + BOS + one marker per layer + one REDRAW/REFINE marker
        + every layer at its cap + EOS."""
        return (self.max_text_positions + 1 + self.num_layers + 1
                + sum(l.max_primitives for l in self.layers) + 1)


def _parse(raw: dict) -> Protocol:
    prim = raw["primitive"]
    attrs: list[OrdinalAttr | CodebookAttr] = []
    for a in raw["attributes"]:
        if a["type"] == "uniform":
            attrs.append(UniformAttr(a["name"], a["bins"], float(a["min"]), float(a["max"])))
        elif a["type"] == "log_uniform":
            attrs.append(LogUniformAttr(a["name"], a["bins"], float(a["min"]), float(a["max"])))
        elif a["type"] == "periodic":
            attrs.append(PeriodicAttr(a["name"], a["bins"], float(a["period"])))
        elif a["type"] == "codebook":
            attrs.append(CodebookAttr(a["name"], a["size"], a["file"], a.get("codes_per_beta")))
        else:
            raise ValueError(f"unknown attribute type {a['type']!r}")

    proto = Protocol(
        raw=raw,
        canvas_size=raw["canvas"]["size"],
        base_color=tuple(raw["canvas"]["base_color_srgb"]),
        max_positions=raw["context"]["max_positions"],
        max_text_positions=raw["context"]["max_text_positions"],
        cutoff_sigma=float(prim["cutoff_sigma"]),
        max_alpha=float(prim["max_alpha"]),
        sigma_min=float(prim["sigma_px"]["min"]),
        sigma_max=float(prim["sigma_px"]["max"]),
        beta_levels=_beta_levels(raw, attrs),
        attributes=tuple(attrs),
        events=tuple(raw["events"]),
        layers=tuple(
            Layer(l["id"], l["name"], l["max_primitives"], tuple(l["celebamask"])) for l in raw["layers"]
        ),
        version=int(raw.get("version", 1)),
    )
    _validate(proto)
    return proto


def _validate(p: Protocol) -> None:
    if [l.id for l in p.layers] != list(range(p.num_layers)):
        raise ValueError("layer ids must be 0..L-1 in back-to-front order")
    for k in range(p.num_layers):
        p.layer_event(k), p.redraw_event(k), p.refine_event(k)  # raises if missing
    if p.worst_case_positions() > p.max_positions:
        raise ValueError(
            f"layer caps need {p.worst_case_positions()} positions > context {p.max_positions}"
        )
    if "shape" in p.attr_names:
        shape = p.attr("shape")
        assert isinstance(shape, CodebookAttr)
        if shape.codes_per_beta * len(p.beta_levels) != shape.size:
            raise ValueError("shape codebook size must equal codes_per_beta * len(beta_levels)")
    elif not {"sigma_x", "sigma_y", "theta", "beta"} <= set(p.attr_names):
        raise ValueError("protocol needs either a 'shape' codebook or sigma_x/sigma_y/theta/beta")
    if p.has_lines and "bend" not in p.attr_names:
        raise ValueError("a protocol with LINE events needs the bend attribute")
    if len(set(p.attr_names)) != len(p.attr_names):
        raise ValueError("duplicate attribute names")
    seen: set[str] = set()
    for l in p.layers:
        dup = seen.intersection(l.celebamask)
        if dup:
            raise ValueError(f"CelebAMask classes assigned to two layers: {dup}")
        seen.update(l.celebamask)


def _beta_levels(raw: dict, attrs) -> tuple[float, ...]:
    """v1 lists beta levels explicitly; v2 derives them from the beta attribute's bins."""
    if "beta_levels" in raw["primitive"]:
        return tuple(float(b) for b in raw["primitive"]["beta_levels"])
    beta = next(a for a in attrs if a.name == "beta")
    return tuple(float(v) for v in beta.dequantize(torch.arange(beta.bins)))


@lru_cache(maxsize=None)
def load_protocol(path: str | Path = PROTOCOL_PATH) -> Protocol:
    with open(path, encoding="utf-8") as f:
        return _parse(json.load(f))
