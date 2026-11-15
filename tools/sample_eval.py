"""Compare sampling settings on a trained checkpoint: orphan rate, early stops, length.

    python tools/sample_eval.py --ckpt runs/x/last.pt --n 48 --out out/sample_eval

Each setting uses the same seeds. "early stop" = the sample ended without the skin layer
(layer 2), i.e. without a face. Writes one image grid per setting and a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.metrics import orphan_stats  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import paint_order  # noqa: E402

SETTINGS = {
    "baseline (T1, top-p .9)": SampleConfig(temperature=1.0, top_p=0.9),
    "+ EOS needs bg & skin": SampleConfig(temperature=1.0, top_p=0.9, require_before_eos=(0, 2)),
    "+ x/y temperature .7": SampleConfig(temperature=1.0, top_p=0.9, require_before_eos=(0, 2),
                                         attr_temperature={"x": 0.7, "y": 0.7}),
    "Renyi a=4 + EOS rule": SampleConfig(renyi_alpha=4.0, require_before_eos=(0, 2)),
    "T .8, top-p .9 + EOS rule": SampleConfig(temperature=0.8, top_p=0.9, require_before_eos=(0, 2)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    proto = _parse(state["protocol"])
    codec = Codec.load(proto=proto, device="cuda")
    model = Painter(ModelConfig(**state["config"]), proto, codec).cuda().eval()
    model.load_state_dict(state["model"])
    print(f"checkpoint step {state['step']}, protocol v{proto.version}, {model.num_params() / 1e6:.1f}M")

    summary = {}
    for name, sc in SETTINGS.items():
        orphan, lengths, early, tiles = [], [], 0, []
        for b0 in range(0, args.n, args.batch):
            g = torch.Generator(device="cuda").manual_seed(1000 + b0)
            for s in generate(model, batch=min(args.batch, args.n - b0), cfg=sc, generator=g, codec=codec):
                attrs, lay = paint_order(s)
                lengths.append(len(s))
                early += int(2 not in lay.tolist())
                if attrs.shape[0] == 0:
                    continue
                prims = codec.decode({k: attrs[:, i].unsqueeze(0).cuda() for i, k in enumerate(proto.attr_names)})
                orphan.append(orphan_stats(prims, lay.cuda(), 128, proto=proto)["orphan"])
                if len(tiles) < 8:
                    img = flatten_on_base(render_group(prims, 256, proto), proto)[0]
                    tiles.append((img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(tiles, 1)).save(args.out / (name.split(" (")[0].replace(" ", "_")
                                                                   .replace("+", "p").replace("/", "") + ".png"))
        summary[name] = {"orphan_rate": round(float(np.mean(orphan)), 4), "early_stop": early / args.n,
                         "mean_length": round(float(np.mean(lengths)), 1)}
        r = summary[name]
        print(f"  {name:28s} orphan {r['orphan_rate']:.1%}  early stop {r['early_stop']:.0%}  "
              f"length {r['mean_length']:.0f}", flush=True)
    (args.out / "summary.json").write_text(json.dumps({"step": state["step"], "settings": summary}, indent=1,
                                                      ensure_ascii=False))


if __name__ == "__main__":
    main()
