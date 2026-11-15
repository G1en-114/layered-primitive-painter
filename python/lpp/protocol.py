"""Load protocol.json and expose it as typed constants.

protocol.json is the single source of truth shared by the Python side and the
C++ runtime; nothing in this package should hard-code a bin count, a layer id
or an event id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "protocol.json"


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
    cutoff_sigma: float
    max_alpha: float
    sigma_min: float
    sigma_max: float
    beta_levels: tuple[float, ...]
    attributes: tuple[UniformAttr | CodebookAttr, ...]
    events: tuple[str, ...]
    layers: tuple[Layer, ...]

    # ---- lookups -------------------------------------------------------
    def attr(self, name: str) -> UniformAttr | CodebookAttr:
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

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def attr_vocab_sizes(self) -> dict[str, int]:
        return {a.name: (a.bins if isinstance(a, UniformAttr) else a.size) for a in self.attributes}

    def worst_case_positions(self) -> int:
        """COND + BOS + one marker per layer + every layer at its cap + EOS."""
        return 2 + self.num_layers + sum(l.max_primitives for l in self.layers) + 1


def _parse(raw: dict) -> Protocol:
    prim = raw["primitive"]
    attrs: list[UniformAttr | CodebookAttr] = []
    for a in raw["attributes"]:
        if a["type"] == "uniform":
            attrs.append(UniformAttr(a["name"], a["bins"], float(a["min"]), float(a["max"])))
        elif a["type"] == "codebook":
            attrs.append(CodebookAttr(a["name"], a["size"], a["file"], a.get("codes_per_beta")))
        else:
            raise ValueError(f"unknown attribute type {a['type']!r}")

    proto = Protocol(
        raw=raw,
        canvas_size=raw["canvas"]["size"],
        base_color=tuple(raw["canvas"]["base_color_srgb"]),
        max_positions=raw["context"]["max_positions"],
        cutoff_sigma=float(prim["cutoff_sigma"]),
        max_alpha=float(prim["max_alpha"]),
        sigma_min=float(prim["sigma_px"]["min"]),
        sigma_max=float(prim["sigma_px"]["max"]),
        beta_levels=tuple(float(b) for b in prim["beta_levels"]),
        attributes=tuple(attrs),
        events=tuple(raw["events"]),
        layers=tuple(
            Layer(l["id"], l["name"], l["max_primitives"], tuple(l["celebamask"])) for l in raw["layers"]
        ),
    )
    _validate(proto)
    return proto


def _validate(p: Protocol) -> None:
    if [l.id for l in p.layers] != list(range(p.num_layers)):
        raise ValueError("layer ids must be 0..L-1 in back-to-front order")
    for k in range(p.num_layers):
        p.layer_event(k), p.redraw_event(k)  # raises if missing
    if p.worst_case_positions() > p.max_positions:
        raise ValueError(
            f"layer caps need {p.worst_case_positions()} positions > context {p.max_positions}"
        )
    shape = p.attr("shape")
    assert isinstance(shape, CodebookAttr)
    if shape.codes_per_beta * len(p.beta_levels) != shape.size:
        raise ValueError("shape codebook size must equal codes_per_beta * len(beta_levels)")
    seen: set[str] = set()
    for l in p.layers:
        dup = seen.intersection(l.celebamask)
        if dup:
            raise ValueError(f"CelebAMask classes assigned to two layers: {dup}")
        seen.update(l.celebamask)


@lru_cache(maxsize=None)
def load_protocol(path: str | Path = PROTOCOL_PATH) -> Protocol:
    with open(path, encoding="utf-8") as f:
        return _parse(json.load(f))
