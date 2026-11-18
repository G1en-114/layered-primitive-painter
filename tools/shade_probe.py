"""Shadow strokes, P2 (docs/line_strokes.md "Shadows"): replace the Gaussians that paint shadows with clipped
multiply strokes at an equal stroke budget, and measure what that buys.

    python tools/shade_probe.py --seq data/seq/anime_v2_l6_t0/decomp_003.npz \
        --images data/anime_portraits256_v2_l6view --n 16 --out out/shade_probe

Cel shading puts hard-edged, flat darker regions on a base colour, where the shape and the light put them.
A painter draws them on a multiply layer clipped to the base. Per image and per layer:
  1. base colour: the mean of the layer's brighter pixels (above its 40th luminance percentile);
  2. ratio map image / base: connected regions with luminance ratio < 0.85 (>= 20 px) are shadows;
  3. each region becomes one hard-edged Gaussian (beta = max level; centre, size and angle from the region's
     moments) composited with `blend.multiply` inside its layer: colour = the region's median ratio, so the
     stroke stores "how much darker, which tint", not a colour;
  4. equal budget: remove as many Gaussians of that layer that paint shadows today (rounds >= 2, luminance
     below 0.85 x the layer's median stroke luminance; darkest first, then the smallest);
  5. fit shade strokes and the remaining Gaussians jointly, snapped to the token grid each step; the control
     refits all the original Gaussians alone with the same steps.
Reports PSNR and shadow-edge F1 (edges of the ratio < 0.85 regions of each reconstruction, against the original's).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from line_probe import f1, psnr, to_u8  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import load_shards  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.render.bezier import Regions, region_alpha_local  # noqa: E402
from lpp.render.blend import multiply  # noqa: E402
from line_probe import quantize_curves  # noqa: E402,F401  (same token grid for region points below)
from lpp.render.reference import Primitives, pixel_alpha  # noqa: E402
from lpp.render.reference import render_group as ref_group  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def base_colours(img: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    f = img[..., :3].astype(np.float32) / 255
    for k in np.unique(labels):
        px = f[labels == k]
        lum = px @ LUMA
        out[int(k)] = px[lum >= np.percentile(lum, 40)].mean(0) if len(px) else np.ones(3, np.float32)
    return out


def shadow_mask(img: np.ndarray, labels: np.ndarray, base: dict[int, np.ndarray], thr: float = 0.85) -> np.ndarray:
    f = img[..., :3].astype(np.float32) / 255
    b = np.stack([base.get(int(k), np.ones(3, np.float32)) for k in range(int(labels.max()) + 1)])[labels]
    return ((f @ LUMA) / np.maximum(b @ LUMA, 1e-3)) < thr


def shadow_regions(img, labels, base, max_strokes: int, min_px: int = 20):
    """-> [(layer, cx, cy, sx, sy, theta, rgb_ratio, area)] largest first."""
    f = img[..., :3].astype(np.float32) / 255
    sm = shadow_mask(img, labels, base)
    out = []
    for k in np.unique(labels):
        if k == 0:
            continue                                         # background: no clipping base worth shading
        comp, n = ndimage.label(sm & (labels == k))
        for c in range(1, n + 1):
            ys, xs = np.nonzero(comp == c)
            if len(ys) < min_px:
                continue
            cov = np.cov(np.stack([xs, ys]).astype(np.float64)) + np.eye(2) * 0.25
            ev, evec = np.linalg.eigh(cov)
            sx, sy = np.sqrt(np.maximum(ev[::-1], 0.25)) * 1.2  # hard edge at ~the region's extent
            theta = float(np.arctan2(evec[1, 1], evec[0, 1]))
            ratio = np.clip(np.median(f[ys, xs] / np.maximum(base[int(k)], 1e-3), 0), 0.15, 1.0)
            out.append((int(k), xs.mean() + 0.5, ys.mean() + 0.5, float(sx), float(sy), theta, ratio, len(ys)))
    out.sort(key=lambda r: -r[7])
    return out[:max_strokes]


def quantize_colours(rgb: torch.Tensor, codec) -> torch.Tensor:
    from lpp.codebook import nearest
    from lpp.color import oklab_to_srgb, srgb_to_oklab
    return oklab_to_srgb(codec.color_codes[nearest(codec.color_codes, srgb_to_oklab(rgb.clamp(0, 1)))])


def region_init(img, labels, base, max_regions: int, pieces: int = 6, min_px: int = 20):
    """Shadow components -> closed Bezier outlines: the longest contour resampled to 2 * pieces points by arc
    length; even points are piece ends, odd ones become control points so each piece passes through them.
    -> [(layer, pts [2P, 2], ratio rgb, area, member mask)] largest first."""
    from skimage.measure import find_contours
    f = img[..., :3].astype(np.float32) / 255
    sm = shadow_mask(img, labels, base)
    out = []
    for k in np.unique(labels):
        if k == 0:
            continue
        comp, n = ndimage.label(sm & (labels == k))
        for c in range(1, n + 1):
            m = comp == c
            if m.sum() < min_px:
                continue
            cs = find_contours(np.pad(m, 1).astype(np.float32), 0.5)
            cy = max(cs, key=len) - 1                                     # (row, col), undo the padding
            xy = np.stack([cy[:, 1] + 0.5, cy[:, 0] + 0.5], 1)
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            q = np.stack([np.interp(np.linspace(0, seg[-1], 2 * pieces, endpoint=False), seg, xy[:, i]) for i in (0, 1)], 1)
            pts = q.copy()
            for j in range(pieces):                                       # control so B(0.5) = the odd sample
                a, b = q[2 * j], q[(2 * j + 2) % (2 * pieces)]
                pts[2 * j + 1] = 2 * q[2 * j + 1] - (a + b) / 2
            ratio = np.clip(np.median(f[m] / np.maximum(base[int(k)], 1e-3), 0), 0.15, 1.0)
            out.append((int(k), pts.astype(np.float32), ratio, int(m.sum()), m))
    out.sort(key=lambda r: -r[3])
    return out[:max_regions]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-strokes", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--regions", action="store_true",
                    help="closed Bezier shadow regions (6 pieces, ~3 Gaussians' worth of attributes each) instead of one hard Gaussian per region")
    ap.add_argument("--per-region", type=int, default=3, help="Gaussians removed per region (equal attribute budget)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    proto = load_protocol()
    codec = Codec.load(proto=proto, device=dev)
    base_c = torch.tensor(proto.base_color, device=dev).view(1, 3, 1, 1)
    bmax = float(max(proto.beta_levels))
    index = json.loads((args.images / "index.json").read_text())
    where = {i: (s["shard"], r) for s in index["shards"] for r, i in enumerate(s["ids"])}
    exs = [e for e in load_shards([args.seq]) if not e.flip and e.image_id in where][: args.n]
    rows, sheet = [], []
    empty = lambda: (torch.zeros(1, 3, 256, 256, device=dev), torch.ones(1, 1, 256, 256, device=dev))  # noqa: E731
    for e in exs:
        sh, r = where[e.image_id]
        img = np.array(np.load(args.images / f"images_{sh:03d}.npy", mmap_mode="r")[r])
        labels = np.array(np.load(args.images / f"labels_{sh:03d}.npy", mmap_mode="r")[r])
        target = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
        order = torch.sort(e.layer, stable=True).indices
        tok, lay, rnd = e.tokens[order].to(dev), e.layer[order].to(dev), e.round[order].to(dev)
        prims = codec.decode({k: tok[:, j].unsqueeze(0) for j, k in enumerate(proto.attr_names)})
        layers = sorted(lay.unique().tolist())
        base = base_colours(img, labels)
        regs = region_init(img, labels, base, args.max_strokes) if args.regions else shadow_regions(img, labels, base, args.max_strokes)
        M = len(regs)

        # equal budget: per layer, drop as many of today's shadow-painting Gaussians as it gets shade strokes
        lum = prims.rgb[0] @ torch.tensor(LUMA, device=dev)
        keep = torch.ones(len(lay), dtype=torch.bool, device=dev)
        if args.regions:      # stricter: centre inside the shadow (1 px), darker than the layer's base, round >= 2
            cx = prims.xy[0, :, 0].long().clamp(0, 255).cpu().numpy()
            cyy = prims.xy[0, :, 1].long().clamp(0, 255).cpu().numpy()
            for k in layers:
                need = args.per_region * sum(1 for g in regs if g[0] == k)
                if not need:
                    continue
                shadow_k = ndimage.binary_dilation(np.any([g[4] for g in regs if g[0] == k], 0))
                bl = float(base[int(k)] @ LUMA)
                idx = ((lay == k) & (rnd >= 2)).nonzero()[:, 0]
                inside = torch.tensor(shadow_k[cyy, cx], device=dev)[idx]
                dark = idx[inside & (lum[idx] < 0.85 * bl)]
                dark = dark[lum[dark].argsort()]
                rest = idx[~torch.isin(idx, dark)]
                rest = rest[prims.sigma[0, rest].max(-1).values.argsort()]
                keep[torch.cat([dark, rest])[:need]] = False
        else:
          for k in layers:
            need = sum(1 for g in regs if g[0] == k)
            if not need:
                continue
            idx = ((lay == k) & (rnd >= 2)).nonzero()[:, 0]
            if len(idx) == 0:
                continue
            med = lum[(lay == k).nonzero()[:, 0]].median()
            dark = idx[lum[idx] < 0.85 * med]
            dark = dark[lum[dark].argsort()]
            rest = idx[~torch.isin(idx, dark)]
            rest = rest[prims.sigma[0, rest].max(-1).values.argsort()]
            keep[torch.cat([dark, rest])[:need]] = False

        def live(g, sel_lay):
            """Trainable copy of Gaussians g with layer ids -> (params, fn() -> per-layer groups)."""
            p = {"xy": g.xy.clone().requires_grad_(True), "ls": g.sigma.clamp(min=1e-3).log().clone().requires_grad_(True),
                 "th": g.theta.clone().requires_grad_(True), "rgb": torch.logit(g.rgb.clamp(0.01, 0.99)).requires_grad_(True),
                 "al": g.alpha.clone().requires_grad_(True)}
            def fn():
                q = codec.snap(Primitives(xy=p["xy"], sigma=p["ls"].exp(), theta=p["th"], beta=g.beta,
                                          rgb=torch.sigmoid(p["rgb"]), alpha=p["al"].clamp(0, 0.99)))
                return {k: (ref_group(q[(sel_lay == k).nonzero()[:, 0]], 256, proto)
                            if (sel_lay == k).any() else empty()) for k in layers}
            return p, fn

        soft_now = [None]                                             # region edge softness while fitting

        def compose(groups, shade=None, shade_lay=None):
            imgt = base_c.expand(1, 3, 256, 256)
            if shade is None:
                a = None
            elif isinstance(shade, Regions):
                a = region_alpha_local(shade, 256, soft=soft_now[0])
            else:
                a = pixel_alpha(shade, 256, proto)
            for k in layers:
                P, T = groups[k]
                if a is not None and (shade_lay == k).any():
                    sel = (shade_lay == k).nonzero()[:, 0]
                    P = multiply(P, a[:, sel], shade.rgb[:, sel])
                imgt = P + T * imgt
            return imgt

        def fit(params, fn, extra=()):
            opt = torch.optim.Adam([{"params": [params["xy"]], "lr": 0.05}, {"params": [params["ls"], params["th"]], "lr": 0.01},
                                    {"params": [params["rgb"], params["al"]], "lr": 0.01}, *extra])
            for _ in range(args.steps):
                loss = ((fn() - target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            full = to_u8(compose({k: (ref_group(prims[(lay == k).nonzero()[:, 0]], 256, proto)) for k in layers}))
        pa, fa = live(prims, lay)                                         # control: all Gaussians refit
        fit(pa, lambda: compose(fa()))
        with torch.no_grad():
            refit = to_u8(compose(fa()))
        kidx = keep.nonzero()[:, 0]
        pk, fk = live(prims[kidx], lay[kidx])
        if M and args.regions:
            t = lambda v: torch.tensor(np.array(v), dtype=torch.float32, device=dev)[None]   # noqa: E731
            s_lay = torch.tensor([g[0] for g in regs], device=dev)
            rpts = t([g[1] for g in regs]).requires_grad_(True)
            rrgb = torch.logit(t([g[2] for g in regs]).clamp(0.02, 0.98)).requires_grad_(True)
            ral = torch.full((1, M), 1.5, device=dev, requires_grad=True)
            shade = lambda: Regions(pts=rpts, rgb=torch.sigmoid(rrgb), alpha=torch.sigmoid(ral) * 0.99)   # noqa: E731
            opt_extra = [{"params": [rpts], "lr": 0.15}, {"params": [rrgb, ral], "lr": 0.03}]
            step_box = [0]
            def fn_r():
                soft_now[0] = 0.7 + (0.2 - 0.7) * step_box[0] / max(args.steps - 1, 1)
                step_box[0] += 1
                return compose(fk(), shade(), s_lay)
            fit(pk, fn_r, extra=opt_extra)
            soft_now[0] = None
            with torch.no_grad():
                ax, ay, aa = proto.attr("x"), proto.attr("y"), proto.attr("alpha")
                rq = Regions(pts=torch.stack([ax.dequantize(ax.quantize(rpts[..., 0])), ay.dequantize(ay.quantize(rpts[..., 1]))], -1),
                             rgb=quantize_colours(torch.sigmoid(rrgb), codec), alpha=aa.dequantize(aa.quantize(torch.sigmoid(ral) * 0.99)))
                shaded = to_u8(compose(fk(), rq, s_lay))
        elif M:
            t = lambda v: torch.tensor(np.array(v), dtype=torch.float32, device=dev)[None]   # noqa: E731
            s_lay = torch.tensor([g[0] for g in regs], device=dev)
            sp = {"xy": t([[g[1], g[2]] for g in regs]).requires_grad_(True),
                  "ls": t([[g[3], g[4]] for g in regs]).log().requires_grad_(True),
                  "th": t([g[5] for g in regs]).requires_grad_(True),
                  "rgb": torch.logit(t([g[6] for g in regs]).clamp(0.02, 0.98)).requires_grad_(True),
                  "al": torch.full((1, M), 0.9, device=dev, requires_grad=True)}
            sbeta = torch.full((1, M), bmax, device=dev)
            shade = lambda: codec.snap(Primitives(xy=sp["xy"], sigma=sp["ls"].exp(), theta=sp["th"], beta=sbeta,  # noqa: E731
                                                  rgb=torch.sigmoid(sp["rgb"]), alpha=sp["al"].clamp(0, 0.99)))
            fit(pk, lambda: compose(fk(), shade(), s_lay),
                extra=[{"params": [sp["xy"]], "lr": 0.1}, {"params": [sp["ls"], sp["th"]], "lr": 0.02},
                       {"params": [sp["rgb"], sp["al"]], "lr": 0.02}])
            with torch.no_grad():
                shaded = to_u8(compose(fk(), shade(), s_lay))
        else:
            fit(pk, lambda: compose(fk()))
            with torch.no_grad():
                shaded = to_u8(compose(fk()))
        edge = lambda m: m ^ ndimage.binary_erosion(m)                          # noqa: E731
        gt = edge(shadow_mask(img, labels, base))
        se = lambda x: f1(edge(shadow_mask(x, labels, base)), gt)               # noqa: E731
        row = {"id": e.image_id, "gaussians": int(len(lay)), "shade_strokes": M, "removed": int((~keep).sum()),
               "psnr_full": psnr(full, img), "psnr_refit": psnr(refit, img), "psnr_shade": psnr(shaded, img),
               "edge_f1_full": se(full), "edge_f1_refit": se(refit), "edge_f1_shade": se(shaded)}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if len(sheet) < 6:
            sm = np.repeat(shadow_mask(img, labels, base)[..., None] * 255, 3, -1).astype(np.uint8)
            sheet.append(np.concatenate([img, full, refit, shaded, sm], 1))
    mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if k != "id"}
    print("mean", json.dumps(mean), flush=True)
    (args.out / "shade_probe.json").write_text(json.dumps({"rows": rows, "mean": mean, "args": vars(args)}, indent=1, default=str))
    Image.fromarray(np.concatenate(sheet, 0)).save(args.out / "sheet.png")   # original | all | refit | + shade | shadow mask


if __name__ == "__main__":
    main()
