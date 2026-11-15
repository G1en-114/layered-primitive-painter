"""Codebook study for protocol v2: how much PSNR does each attribute's quantization cost,
and how much do data-driven codebooks / finer position bins win back?

    # 1. continuous decompositions (no snapping at all, so no bias towards current codes)
    python tools/codebook_study.py collect --data data/celebamask256 --shard 0 --count 640 \
        --out out/codebook_study/faces.npz
    # 2. fit candidate codebooks on the first images, evaluate quantization on held-out ones
    python tools/codebook_study.py evaluate --collected out/codebook_study/faces.npz \
        --images data/celebamask256 --fit-count 512 --out out/codebook_study

Quantizing one attribute at a time and keeping the rest continuous isolates each
attribute's cost. Evaluation images never take part in codebook fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codebook import kmeans, nearest  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.color import oklab_to_srgb, srgb_to_oklab  # noqa: E402
from lpp.protocol import UniformAttr, load_protocol  # noqa: E402
from lpp.render import Primitives  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.shape import canonicalize, from_logcov, to_logcov  # noqa: E402

FIELDS = ("xy", "sigma", "theta", "beta", "rgb", "alpha")


# --------------------------------------------------------------------------- collect
def collect(args) -> None:
    from lpp.decompose import DecomposeConfig, LayeredDecomposer
    codec = Codec.load(device="cuda")
    dec = LayeredDecomposer(codec, DecomposeConfig(steps_per_round=args.steps, quantize=False, seed=args.seed))
    index = json.loads((args.data / "index.json").read_text())
    ids = index["shards"][args.shard]["ids"][: args.count]
    imgs = np.load(args.data / f"images_{args.shard:03d}.npy", mmap_mode="r")
    labs = np.load(args.data / f"labels_{args.shard:03d}.npy", mmap_mode="r")
    rows = {f: [] for f in FIELDS}
    layer, offsets, psnr = [], [0], []
    t0 = time.perf_counter()
    for b0 in range(0, len(ids), args.batch):
        img = torch.from_numpy(np.ascontiguousarray(imgs[b0 : b0 + args.batch])).cuda()
        img = img.permute(0, 3, 1, 2).float() / 255
        lab = torch.from_numpy(np.ascontiguousarray(labs[b0 : b0 + args.batch])).cuda().long()
        for d in dec(img, lab):
            for f in FIELDS:
                rows[f].append(d.continuous[f].numpy())
            layer.append(d.layer.numpy().astype(np.int8))
            offsets.append(offsets[-1] + len(d.layer))
            psnr.append(d.psnr)
        print(f"{b0 + img.shape[0]}/{len(ids)}  continuous PSNR {np.mean(psnr):.2f} dB  "
              f"{(b0 + img.shape[0]) / (time.perf_counter() - t0):.2f} img/s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids, np.int32), offsets=np.array(offsets, np.int64),
                        layer=np.concatenate(layer), psnr=np.array(psnr, np.float32),
                        **{f: np.concatenate(rows[f]).astype(np.float32) for f in FIELDS})
    print(f"wrote {args.out}")


# --------------------------------------------------------------------------- evaluate
class Quantizer:
    """Quantize/dequantize single attributes with a given set of codebooks and bins."""

    def __init__(self, shape_codes, color_codes, xy_bins: int, betas):
        self.shape_codes, self.color_codes = shape_codes, color_codes   # [L, per_beta, 3], [K, 3]
        self.xy = UniformAttr("xy", xy_bins, 256 / xy_bins / 2, 256 - 256 / xy_bins / 2)
        self.betas = betas

    def beta_index(self, beta):
        return (torch.log(beta.clamp(min=1e-3))[..., None] - torch.log(self.betas)).abs().argmin(-1)

    def q_xy(self, p):
        return self.xy.dequantize(self.xy.quantize(p["xy"]))

    def q_shape(self, p):
        sx, sy, th = canonicalize(p["sigma"][..., 0], p["sigma"][..., 1], p["theta"])
        lc = to_logcov(sx.clamp(0.5, 128), sy.clamp(0.5, 128), th)
        bi = self.beta_index(p["beta"])
        out = torch.empty_like(lc)
        for k in range(len(self.betas)):
            m = bi == k
            if m.any():
                out[m] = self.shape_codes[k][nearest(self.shape_codes[k], lc[m])]
        sx, sy, th = from_logcov(out)
        return torch.stack([sx, sy], -1), th, self.betas[bi]

    def q_color(self, p):
        return oklab_to_srgb(self.color_codes[nearest(self.color_codes, srgb_to_oklab(p["rgb"].clamp(0, 1)))])


def apply(p, q: Quantizer, which: set[str]):
    p = dict(p)
    if "xy" in which:
        p["xy"] = q.q_xy(p)
    if "shape" in which:
        p["sigma"], p["theta"], p["beta"] = q.q_shape(p)
    if "color" in which:
        p["rgb"] = q.q_color(p)
    if "alpha" in which:
        a = load_protocol().attr("alpha")
        p["alpha"] = a.dequantize(a.quantize(p["alpha"]))
    return p


def render_psnr(p, offsets, targets) -> np.ndarray:
    """Render each image's primitives in sequence order (= paint order) -> PSNR per image."""
    out = []
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        prims = Primitives(**{f: p[f][a:b].unsqueeze(0) for f in FIELDS})
        img = flatten_on_base(render_group(prims, 256))
        out.append((-10 * torch.log10(((img - targets[i : i + 1]) ** 2).mean())).item())
    return np.array(out)


def evaluate(args) -> None:
    proto = load_protocol()
    z = np.load(args.collected)
    dev = "cuda"
    p_all = {f: torch.from_numpy(z[f]).to(dev) for f in FIELDS}
    offsets = z["offsets"]
    fit_rows = slice(0, offsets[args.fit_count])
    ev_off = offsets[args.fit_count:] - offsets[args.fit_count]
    p_eval = {f: v[offsets[args.fit_count]:] for f, v in p_all.items()}
    ev_ids = z["ids"][args.fit_count:]
    index = json.loads((args.images / "index.json").read_text())
    loc = {i: (s["shard"], k) for s in index["shards"] for k, i in enumerate(s["ids"])}
    targets = torch.from_numpy(np.stack([np.load(args.images / f"images_{loc[int(i)][0]:03d}.npy",
                                                 mmap_mode="r")[loc[int(i)][1]] for i in ev_ids]))
    targets = targets.to(dev).permute(0, 3, 1, 2).float() / 255
    betas = torch.tensor(proto.beta_levels, device=dev)

    v1 = Codec.load(device=dev)
    q_v1 = Quantizer(v1.shape_codes.expand(len(betas), -1, -1), v1.color_codes, 256, betas)

    # data-driven codebooks, fitted on the fit images only
    t0 = time.perf_counter()
    pf = {f: v[fit_rows] for f, v in p_all.items()}
    sx, sy, th = canonicalize(pf["sigma"][..., 0], pf["sigma"][..., 1], pf["theta"])
    lc = to_logcov(sx.clamp(0.5, 128), sy.clamp(0.5, 128), th)
    bi = q_v1.beta_index(pf["beta"])
    per_beta = v1.shape_codes.shape[0]
    shape_v2, beta_share = [], []
    for k in range(len(betas)):
        x = lc[bi == k]
        beta_share.append(round(x.shape[0] / lc.shape[0], 3))
        # too few samples at a level: pad with the prior codes so every level has per_beta codes
        if x.shape[0] < per_beta * 4:
            x = torch.cat([x, v1.shape_codes.to(dev)])
        shape_v2.append(kmeans(x, per_beta, iters=40, seed=k))
    shape_v2 = torch.stack(shape_v2)
    color_v2 = kmeans(srgb_to_oklab(pf["rgb"].clamp(0, 1)), v1.color_codes.shape[0], iters=40, seed=7)
    fit_s = time.perf_counter() - t0

    q_v2 = Quantizer(shape_v2, color_v2, 256, betas)
    q_v2x = Quantizer(shape_v2, color_v2, 512, betas)
    rows = {}
    base = render_psnr(p_eval, ev_off, targets)
    rows["continuous"] = base
    for name, q in (("v1", q_v1), ("v2", q_v2)):
        for which in (["xy"], ["shape"], ["color"], ["alpha"], ["xy", "shape", "color", "alpha"]):
            rows[f"{name}: {'+'.join(which)}"] = render_psnr(apply(p_eval, q, set(which)), ev_off, targets)
    rows["v2 + xy512: xy"] = render_psnr(apply(p_eval, q_v2x, {"xy"}), ev_off, targets)
    rows["v2 + xy512: all"] = render_psnr(apply(p_eval, q_v2x, {"xy", "shape", "color", "alpha"}), ev_off,
                                          targets)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"fit_images": args.fit_count, "eval_images": len(ev_ids), "kmeans_seconds": round(fit_s, 1),
              "beta_level_share": dict(zip(proto.beta_levels, beta_share)),
              "psnr": {k: {"mean": round(float(v.mean()), 3), "delta_vs_continuous": round(float(v.mean() -
                       base.mean()), 3)} for k, v in rows.items()}}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    np.save(args.out / "shape_logcov_v2.npy", shape_v2.cpu().numpy().astype(np.float32))
    np.save(args.out / "color_oklab_v2.npy", color_v2.cpu().numpy().astype(np.float32))
    print(f"fit on {args.fit_count} images, evaluated on {len(ev_ids)} held-out images")
    print(f"beta level share in fit data: {report['beta_level_share']}")
    for k, v in report["psnr"].items():
        print(f"  {k:28s} {v['mean']:6.2f} dB  ({v['delta_vs_continuous']:+.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--count", type=int, default=640)
    c.add_argument("--batch", type=int, default=32)
    c.add_argument("--steps", type=int, default=80)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--collected", type=Path, required=True)
    e.add_argument("--images", type=Path, required=True)
    e.add_argument("--fit-count", type=int, default=512)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"collect": collect, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
