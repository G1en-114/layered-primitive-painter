"""Start a protocol-v3 (LINE strokes) model from a v2 checkpoint (docs/line_strokes.md).

    python tools/init_v3_from_v2.py --ckpt runs/base_v3_snap/model_final.pt --out runs/base_v3_lines_init.pt

v3 appends one event (LINE) and one attribute (bend) to v2 and keeps every other id, so all v2 weights
carry over unchanged. The new pieces:
  - the LINE row of the event head starts as the PRIMITIVE row with a lower bias (the model begins by
    preferring Gaussians and learns where lines go);
  - the bend embedding (backbone and attribute decoder), the decoder's extra step embedding and the bend
    head start fresh (bend is a constant 0 on every v2 stroke, so nothing to transfer);
  - separate LINE attribute heads (cfg.line_heads) start as copies of the Gaussian heads.
The result is a normal checkpoint (model, step 0, config, v3 protocol) for tools/train.py --init.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import PROTOCOL_V3_PATH, load_protocol  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--line-bias", type=float, default=-2.0, help="LINE logit offset relative to PRIMITIVE")
    args = ap.parse_args()
    src = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    v3 = load_protocol(PROTOCOL_V3_PATH)
    cfg = ModelConfig(**src["config"])
    cfg.line_heads = True
    torch.manual_seed(0)
    model = Painter(cfg, v3, Codec.load(proto=v3, device="cpu"))
    dst = model.state_dict()
    copied, new, partial = 0, [], []
    for k, v in src["model"].items():
        if k not in dst:
            raise KeyError(f"{k} not in the v3 model")
        if dst[k].shape == v.shape:
            dst[k] = v
            copied += 1
        else:                                          # grown along dim 0: copy the v2 rows
            n = v.shape[0]
            assert dst[k].shape[1:] == v.shape[1:] and dst[k].shape[0] > n, (k, v.shape, dst[k].shape)
            dst[k][:n] = v
            partial.append(k)
    new = [k for k in dst if k not in src["model"]]
    prim, line = v3.event_id("PRIMITIVE"), v3.event_id("LINE")
    dst["heads.event.weight"][line] = dst["heads.event.weight"][prim]
    dst["heads.event.bias"][line] = dst["heads.event.bias"][prim] + args.line_bias
    for n in model.attr_names:                         # LINE heads start as the Gaussian heads
        if n in v3.attr_names[: len(src["protocol"]["attributes"])]:
            dst[f"line_heads.{n}.weight"] = dst[f"heads.{n}.weight"].clone()
            dst[f"line_heads.{n}.bias"] = dst[f"heads.{n}.bias"].clone()
    model.load_state_dict(dst)
    torch.save({"model": model.state_dict(), "step": 0, "config": cfg.to_dict(), "protocol": v3.raw,
                "init_from": str(args.ckpt)}, args.out)
    print(f"{args.out}: {copied} tensors copied, grown {partial}, new {new}")


if __name__ == "__main__":
    main()
