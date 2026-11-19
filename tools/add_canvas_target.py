"""Turn a canvas-feedback model into a residual-canvas model (docs/plan_v4.md: image -> painting).

    python tools/add_canvas_target.py --ckpt runs/v3_formal/B_ft/model_step012000.pt --out runs/v3_rescanvas_init.pt

The canvas encoder's first convolution gets lpp.canvas.TARGET_CHANNELS more input channels (target, target minus
canvas, has-target plane) with zero weights; nothing else changes. At init the new model is therefore exactly the
old one, with or without a target (checked here on a random canvas); training with tools/train.py --canvas-target
then teaches it to use the target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.canvas import TARGET_CHANNELS, with_target  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402

KEY = "canvas_enc.net.0.weight"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**s["config"])
    if not cfg.canvas:
        raise SystemExit("not a canvas-feedback model")
    if cfg.canvas_target:
        raise SystemExit("already a residual-canvas model")
    proto = _parse(s["protocol"])
    sd = dict(s["model"])
    w = sd[KEY]                                                   # [width, 3 + L, 3, 3]
    sd[KEY] = torch.cat([w, w.new_zeros(w.shape[0], TARGET_CHANNELS, *w.shape[2:])], 1)
    cfg.canvas_target = True
    codec = Codec.load(proto=proto, device="cpu")
    new = Painter(cfg, proto, codec)
    new.load_state_dict(sd)                                       # strict: every other tensor is unchanged
    old = Painter(ModelConfig(**s["config"]), proto, codec)
    old.load_state_dict(s["model"])
    g = torch.Generator().manual_seed(0)
    canvas = torch.rand(2, w.shape[1], cfg.canvas_size, cfg.canvas_size, generator=g)
    target = torch.rand(2, 3, cfg.canvas_size, cfg.canvas_size, generator=g)
    with torch.no_grad():
        ref = old.canvas_enc(canvas)
        for t, has in ((None, None), (target, torch.tensor([True, False]))):
            d = (new.canvas_enc(with_target(canvas, t, has)) - ref).abs().max().item()
            assert d == 0.0, f"converted canvas encoder differs by {d}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": new.state_dict(), "step": 0, "config": cfg.to_dict(), "protocol": s["protocol"],
                "source": str(args.ckpt)}, args.out)
    print(f"{args.out}: {KEY} {tuple(w.shape)} -> {tuple(sd[KEY].shape)}; identical to the source at init")


if __name__ == "__main__":
    main()
