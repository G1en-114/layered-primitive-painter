"""How far from converged are the decompositions? Refit a decomposition's Gaussians with Adam, snapped to the
token grid every step (straight-through), and report the PSNR before and after, plus the time per image.

    python tools/refit_probe.py --seq data/seq/v2/decomp_000.npz --images data/celebamask256 --n 16 \
        --steps 100 300 --out out/refit_faces

The decomposer gives each round 80 Adam steps (6 rounds per layer). On anime, refitting all strokes jointly
for 300 more steps raised PSNR from 24.6 to 27.5 dB (docs/line_strokes.md). This measures the same on any
domain, and how many steps it takes. The result is still a valid token sequence (same strokes, same order,
same layers), so a polished decomposition can replace the old one in the training data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import load_shards  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.render.reference import Primitives  # noqa: E402
from lpp.render.reference import render_group as ref_group  # noqa: E402


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(1 / ((a.clamp(0, 1) - b) ** 2).mean().clamp(min=1e-10)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="dir with index.json and images_*.npy")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 300], help="report after each of these")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    proto = load_protocol()
    codec = Codec.load(proto=proto, device=dev)
    base = torch.tensor(proto.base_color, device=dev).view(1, 3, 1, 1)
    index = json.loads((args.images / "index.json").read_text())
    where = {i: (s["shard"], r) for s in index["shards"] for r, i in enumerate(s["ids"])}
    exs = [e for e in load_shards([args.seq]) if not e.flip and e.image_id in where][: args.n]
    rows = []
    for e in exs:
        sh, r = where[e.image_id]
        img = np.array(np.load(args.images / f"images_{sh:03d}.npy", mmap_mode="r")[r])[..., :3]
        target = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
        order = torch.sort(e.layer, stable=True).indices
        tok, lay = e.tokens[order].to(dev), e.layer[order].to(dev)
        g0 = codec.decode({k: tok[:, j].unsqueeze(0) for j, k in enumerate(proto.attr_names)})
        layers = sorted(lay.unique().tolist())
        sel = {k: (lay == k).nonzero()[:, 0] for k in layers}

        def compose(p):
            out = base.expand(1, 3, 256, 256)
            for k in layers:
                P, T = ref_group(p[sel[k]], 256, proto)
                out = P + T * out
            return out

        xy = g0.xy.clone().requires_grad_(True)
        ls = g0.sigma.clamp(min=1e-3).log().clone().requires_grad_(True)
        th = g0.theta.clone().requires_grad_(True)
        rgb = torch.logit(g0.rgb.clamp(0.01, 0.99)).requires_grad_(True)
        al = g0.alpha.clone().requires_grad_(True)
        live = lambda: codec.snap(Primitives(xy=xy, sigma=ls.exp(), theta=th, beta=g0.beta,   # noqa: E731
                                             rgb=torch.sigmoid(rgb), alpha=al.clamp(0, 0.99)))
        opt = torch.optim.Adam([{"params": [xy], "lr": 0.05}, {"params": [ls, th], "lr": 0.01},
                                {"params": [rgb, al], "lr": 0.01}])
        with torch.no_grad():
            row = {"id": e.image_id, "strokes": int(len(lay)), "psnr_0": psnr(compose(g0), target)}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(1, max(args.steps) + 1):
            loss = ((compose(live()) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in args.steps:
                with torch.no_grad():
                    row[f"psnr_{step}"] = psnr(compose(codec.snap(live(), straight_through=False)), target)
        torch.cuda.synchronize()
        row["seconds"] = round(time.perf_counter() - t0, 2)
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if k != "id"}
    print("mean", json.dumps(mean), flush=True)
    (args.out / "refit.json").write_text(json.dumps({"rows": rows, "mean": mean, "args": vars(args)}, indent=1, default=str))


if __name__ == "__main__":
    main()
