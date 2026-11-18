"""Line strokes, step 1 (docs/line_strokes.md): replace the finest Gaussians of an anime decomposition by
quadratic Bezier line strokes at an equal stroke budget, and measure what that buys.

    python tools/line_probe.py --seq data/seq/anime_v2_l6_t0/decomp_003.npz \
        --images data/anime_portraits256_v2_l6view --n 16 --out out/line_probe

Per image:
  1. line art: thin dark structures (grayscale black-hat), cleaned, skeletonised, traced into chains and cut
     into pieces of at most --piece px; each piece becomes a Bezier through its ends and its middle, with
     width from the distance transform and the colour of the image under it;
  2. each piece is assigned to the topmost layer it touches (labels_*.npy, 1 px neighbourhood): an outline
     belongs to the part that occludes, and is composited right after that layer's Gaussians, so a face line
     is hidden by the hair in front of it;
  3. the K smallest Gaussians (larger radius) are removed, K = number of curves (equal budget);
  4. the curves (ends, control point, width, colour, opacity) are fitted with Adam to the image, soft edges
     annealed to hard; the Gaussians stay fixed.
Reports PSNR and line F1 (line art extracted from each reconstruction against the original's, 1 px tolerance)
for: all Gaussians, Gaussians without the K smallest, and those plus the K curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import load_shards  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.codebook import nearest  # noqa: E402
from lpp.color import oklab_to_srgb, srgb_to_oklab  # noqa: E402
from lpp.render.bezier import Curves, curve_alpha_local, points  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import Primitives  # noqa: E402
from lpp.render.reference import render_group as ref_group  # noqa: E402 (differentiable)

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def line_mask(img: np.ndarray, thr: float = 0.10, radius: int = 2, min_px: int = 12) -> np.ndarray:
    """Thin dark structures of an RGB uint8 image -> bool mask."""
    g = (img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255
    bh = closing(g, disk(radius)) - g                      # black-hat: dark lines narrower than the disk
    try:                       # scikit-image >= 0.26 (min_size deprecated there): removes objects of <= max_size px
        return remove_small_objects(bh > thr, max_size=min_px - 1)
    except TypeError:          # older scikit-image: removes objects of < min_size px (the same objects)
        return remove_small_objects(bh > thr, min_size=min_px)


def trace(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Skeleton -> chains of (y, x) pixels split at ends and junctions (8-connectivity)."""
    pix = set(zip(*np.nonzero(skel)))
    nb = {p: [(p[0] + dy, p[1] + dx) for dy, dx in NB8 if (p[0] + dy, p[1] + dx) in pix] for p in pix}
    used, chains = set(), []
    def walk(a, b):
        chain = [a, b]
        used.add((a, b)); used.add((b, a))
        while len(nb[chain[-1]]) == 2:
            nxt = [q for q in nb[chain[-1]] if (chain[-1], q) not in used]
            if not nxt:
                break
            used.add((chain[-1], nxt[0])); used.add((nxt[0], chain[-1]))
            chain.append(nxt[0])
        return chain
    for p in sorted(pix):                                  # from ends and junctions first
        if len(nb[p]) != 2:
            for q in nb[p]:
                if (p, q) not in used:
                    chains.append(walk(p, q))
    for p in sorted(pix):                                  # then closed loops
        for q in nb[p]:
            if (p, q) not in used:
                chains.append(walk(p, q))
    return chains


def init_curves(img, mask, labels, piece: int, max_curves: int, contour_first: bool = False):
    """Line pieces -> initial curve parameters (numpy), the layer of each, and whether it is a contour.
    A contour piece lies on a layer boundary (>= 60% of its pixels within 2 px of one). People draw lines
    mostly at occluding contours (Cole et al. 2008); 71% of the anime line art is within 2-3 px of a layer
    boundary, a place the model can read off its own canvas. contour_first spends the budget on those first."""
    skel = skeletonize(mask)
    dist = ndimage.distance_transform_edt(mask)
    top = ndimage.maximum_filter(labels, size=3)           # topmost layer within 1 px
    near = ndimage.binary_dilation(ndimage.maximum_filter(labels, 3) != ndimage.minimum_filter(labels, 3), iterations=2)
    out = []
    for ch in trace(skel):
        if len(ch) < 4:
            continue
        for s in range(0, len(ch) - 1, piece):
            seg = ch[s:s + piece + 1]
            if len(seg) < 4:
                continue
            yx = np.array(seg, np.float32) + 0.5
            p0, p2, m = yx[0, ::-1], yx[-1, ::-1], yx[len(yx) // 2, ::-1]
            p1 = 2 * m - (p0 + p2) / 2                      # B(0.5) passes through the middle pixel
            ys, xs = np.array(seg).T
            w = float(np.clip(2 * np.median(dist[ys, xs]), 1.0, 5.0))
            col = img[ys, xs, :3].astype(np.float32) / 255
            dark = col[col.sum(1) <= np.median(col.sum(1))]
            out.append((len(seg), p0, p1, p2, w, dark.mean(0), int(np.bincount(top[ys, xs]).argmax()),
                        bool(near[ys, xs].mean() >= 0.6)))
    out.sort(key=lambda r: ((not r[7]) if contour_first else 0, -r[0]))   # (contours first,) longest first
    return out[:max_curves]


def quantize_curves(c: Curves, codec: Codec, proto) -> Curves:
    """What a token would keep: ends and control point on the 512-bin x/y grid, colour from the codebook,
    opacity on the 16 alpha levels; width stays continuous (its bins are not designed yet)."""
    ax, ay, aa = proto.attr("x"), proto.attr("y"), proto.attr("alpha")
    q = lambda p: torch.stack([ax.dequantize(ax.quantize(p[..., 0])), ay.dequantize(ay.quantize(p[..., 1]))], -1)  # noqa: E731
    rgb = oklab_to_srgb(codec.color_codes[nearest(codec.color_codes, srgb_to_oklab(c.rgb.clamp(0, 1)))])
    return Curves(p0=q(c.p0), p1=q(c.p1), p2=q(c.p2), width=c.width, rgb=rgb,
                  alpha=aa.dequantize(aa.quantize(c.alpha)))


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    p = (pred & ndimage.binary_dilation(gt)).sum() / pred.sum()
    r = (gt & ndimage.binary_dilation(pred)).sum() / gt.sum()
    return float(2 * p * r / max(p + r, 1e-9))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1 / max(np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2), 1e-10)))


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="dir with index.json, images_*.npy, labels_*.npy")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--piece", type=int, default=14, help="max skeleton pixels per curve")
    ap.add_argument("--max-curves", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--contour-first", action="store_true", help="spend the curve budget on layer-boundary lines first")
    ap.add_argument("--udf-weight", type=float, default=0.0,
                    help="pull curve points onto the line-art skeleton (distance field, as LIVE's UDF loss)")
    ap.add_argument("--joint", action="store_true",
                    help="fit the remaining Gaussians too, snapped to the token grid each step (straight-through)")
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
    rows, sheet = [], []
    for e in exs:
        sh, r = where[e.image_id]
        img = np.array(np.load(args.images / f"images_{sh:03d}.npy", mmap_mode="r")[r])
        labels = np.array(np.load(args.images / f"labels_{sh:03d}.npy", mmap_mode="r")[r])
        target = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
        order = torch.sort(e.layer, stable=True).indices
        tok, lay = e.tokens[order].to(dev), e.layer[order].to(dev)
        prims = codec.decode({k: tok[:, j].unsqueeze(0) for j, k in enumerate(proto.attr_names)})
        layers = sorted(lay.unique().tolist())

        init = init_curves(img, line_mask(img), labels, args.piece, args.max_curves, args.contour_first)
        K = len(init)
        small = prims.sigma[0].max(-1).values.argsort()[:K]          # the K finest Gaussians
        keep = torch.ones(len(lay), dtype=torch.bool, device=dev)
        keep[small] = False

        def gauss(mask):
            """Per-layer fixed Gaussian groups (P, T) for the strokes in mask."""
            out = {}
            for k in layers:
                sel = ((lay == k) & mask).nonzero()[:, 0]
                if len(sel) == 0:
                    out[k] = (torch.zeros(1, 3, 256, 256, device=dev), torch.ones(1, 1, 256, 256, device=dev))
                    continue
                with torch.no_grad():
                    out[k] = render_group(prims[sel], 256, proto)
            return out

        def compose(groups, curves=None, cl=None, soft=None):
            imgt = base.expand(1, 3, 256, 256)
            a = curve_alpha_local(curves, 256, soft=soft) if curves is not None else None
            for k in layers:
                P, T = groups[k]
                if a is not None and (cl == k).any():
                    sel = (cl == k).nonzero()[:, 0]
                    ak, rgb = a[:, sel], curves.rgb[:, sel]
                    log_t = torch.log1p(-ak)
                    after = log_t.sum(1, keepdim=True) - torch.cumsum(log_t, 1)
                    Pc = torch.einsum("bnhw,bnc->bchw", ak * after.exp(), rgb)
                    Tc = log_t.sum(1, keepdim=True).exp()
                    P, T = Pc + Tc * P, Tc * T
                imgt = P + T * imgt
            return imgt

        full = to_u8(compose(gauss(torch.ones_like(keep))))
        g_keep = gauss(keep)
        trimmed = to_u8(compose(g_keep))
        if K:
            t = lambda v: torch.tensor(np.stack(v), dtype=torch.float32, device=dev)[None]   # noqa: E731
            p0, p1, p2 = (t([c[i] for c in init]).requires_grad_(True) for i in (1, 2, 3))
            w_raw = torch.log(torch.expm1(t([c[4] for c in init]))).requires_grad_(True)
            rgb_raw = torch.logit(t([c[5] for c in init]).clamp(0.02, 0.98)).requires_grad_(True)
            a_raw = torch.full((1, K), 2.0, device=dev, requires_grad=True)
            cl = torch.tensor([c[6] for c in init], device=dev)
            groups = [{"params": [p0, p1, p2], "lr": 0.15}, {"params": [w_raw, rgb_raw, a_raw], "lr": 0.05}]
            make = lambda: Curves(p0=p0, p1=p1, p2=p2, width=torch.nn.functional.softplus(w_raw),   # noqa: E731
                                  rgb=torch.sigmoid(rgb_raw), alpha=torch.sigmoid(a_raw) * 0.99)
            kidx = keep.nonzero()[:, 0]
            klay = lay[kidx]
            if args.joint:
                g0 = prims[kidx]
                gxy = g0.xy.clone().requires_grad_(True)
                gls = g0.sigma.clamp(min=1e-3).log().clone().requires_grad_(True)
                gth = g0.theta.clone().requires_grad_(True)
                grgb = torch.logit(g0.rgb.clamp(0.01, 0.99)).requires_grad_(True)
                gal = g0.alpha.clone().requires_grad_(True)
                groups += [{"params": [gxy], "lr": 0.05}, {"params": [gls, gth], "lr": 0.01},
                           {"params": [grgb, gal], "lr": 0.01}]
                def gauss_live():
                    q = codec.snap(Primitives(xy=gxy, sigma=gls.exp(), theta=gth, beta=g0.beta,
                                              rgb=torch.sigmoid(grgb), alpha=gal.clamp(0, 0.99)))
                    out = {}
                    for k in layers:
                        sel = (klay == k).nonzero()[:, 0]
                        out[k] = (ref_group(q[sel], 256, proto) if len(sel) else
                                  (torch.zeros(1, 3, 256, 256, device=dev), torch.ones(1, 1, 256, 256, device=dev)))
                    return out
            if args.udf_weight:
                sk = skeletonize(line_mask(img))
                udf = torch.from_numpy(ndimage.distance_transform_edt(~sk).astype(np.float32)).to(dev)[None, None]
            opt = torch.optim.Adam(groups)
            for step in range(args.steps):
                tau = 0.7 + (0.2 - 0.7) * step / max(args.steps - 1, 1)
                c = make()
                loss = ((compose(gauss_live() if args.joint else g_keep, c, cl, soft=tau) - target) ** 2).mean()
                if args.udf_weight:
                    pts = points(c)[0].reshape(1, 1, -1, 2) / 128 - 1           # grid_sample coordinates
                    loss = loss + args.udf_weight * F.grid_sample(udf, pts, align_corners=False).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                g_eval = gauss_live() if args.joint else g_keep               # snapped (forward values)
                lined = to_u8(compose(g_eval, quantize_curves(make(), codec, proto), cl))
        else:
            lined = trimmed
        n_contour = sum(c[7] for c in init)
        refit = None
        if args.joint:            # control: all Gaussians, no curves, the same joint fitting (same steps and rates)
            g0 = prims
            gxy = g0.xy.clone().requires_grad_(True)
            gls = g0.sigma.clamp(min=1e-3).log().clone().requires_grad_(True)
            gth = g0.theta.clone().requires_grad_(True)
            grgb = torch.logit(g0.rgb.clamp(0.01, 0.99)).requires_grad_(True)
            gal = g0.alpha.clone().requires_grad_(True)
            opt = torch.optim.Adam([{"params": [gxy], "lr": 0.05}, {"params": [gls, gth], "lr": 0.01},
                                    {"params": [grgb, gal], "lr": 0.01}])
            def all_live():
                q = codec.snap(Primitives(xy=gxy, sigma=gls.exp(), theta=gth, beta=g0.beta,
                                          rgb=torch.sigmoid(grgb), alpha=gal.clamp(0, 0.99)))
                return {k: ref_group(q[(lay == k).nonzero()[:, 0]], 256, proto) for k in layers}
            for step in range(args.steps):
                loss = ((compose(all_live()) - target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                refit = to_u8(compose(all_live()))
        gt = line_mask(img)
        row = {"id": e.image_id, "gaussians": int(len(lay)), "curves": K, "contour_curves": int(n_contour),
               "psnr_full": psnr(full, img), "psnr_trimmed": psnr(trimmed, img), "psnr_lines": psnr(lined, img),
               "f1_full": f1(line_mask(full), gt), "f1_trimmed": f1(line_mask(trimmed), gt),
               "f1_lines": f1(line_mask(lined), gt)}
        if refit is not None:
            row |= {"psnr_full_refit": psnr(refit, img), "f1_full_refit": f1(line_mask(refit), gt)}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if len(sheet) < 6:
            third = refit if refit is not None else trimmed     # with --joint: the same-steps Gaussian-only control
            sheet.append(np.concatenate([img, full, third, lined, np.repeat(gt[..., None] * 255, 3, -1).astype(np.uint8)], 1))
    mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if k != "id"}
    print("mean", json.dumps(mean), flush=True)
    (args.out / "line_probe.json").write_text(json.dumps({"rows": rows, "mean": mean, "args": vars(args)}, indent=1, default=str))
    Image.fromarray(np.concatenate(sheet, 0)).save(args.out / "sheet.png")   # original | all | trimmed (refit with --joint) | + lines | line art


if __name__ == "__main__":
    main()
