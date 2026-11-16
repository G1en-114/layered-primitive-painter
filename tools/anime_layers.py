"""Six-layer pseudo-labels for anime portraits (protocol v3 design, docs/protocol_v3_design.md section 2).

    python tools/anime_layers.py --data data/anime_portraits256_v2

Inputs per shard: images_XXX.npy, alpha_XXX.npy (anime-seg character matte, tools/segment_anime.py)
and the 28 face points in manifest.jsonl (256px crop coordinates). The points follow the
common 28-point anime-face layout, checked by plotting indices on samples:
  0-4 face contour (0/4 temples at eye level, 2 chin), 5-7 / 8-10 brows, 11-16 / 17-22 eyes,
  23 nose, 24-27 mouth.
Layers use the v2 face ids: 0 background, 1 body, 2 skin, 3 features, 4 hair, 5 accessories (unused).

Heuristics (pseudo-labels, recorded as such):
  features = dilated eye/mouth hulls, brow strokes, a nose dot, inside the character;
  skin     = face polygon (contour closed above the brows) pixels whose Oklab colour is close
             to the median cheek colour, minus features;
  hair     = non-skin character pixels above the eyes, plus pixels below whose colour is close
             to one of 3 k-means hair colours and that connect to the hair above;
  body     = the remaining character pixels (clothes, neck, hands).
Writes labels6_XXX.npy and layers6_quality.jsonl (fractions and review flags); nothing else changes.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SIZE = 256
BG, BODY, SKIN, FEATURES, HAIR = 0, 1, 2, 3, 4
CONTOUR, BROWS, EYE_L, EYE_R, NOSE, MOUTH = range(0, 5), (range(5, 8), range(8, 11)), range(11, 17), range(17, 23), 23, range(24, 28)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
    return np.cbrt(lin @ m1.T) @ m2.T


def kmeans(x: np.ndarray, k: int, iters: int = 10, seed: int = 0) -> np.ndarray:
    """Small deterministic k-means (k-means++ style init on a subsample) -> centres [k, D]."""
    rng = np.random.default_rng(seed)
    if len(x) > 4000:
        x = x[rng.choice(len(x), 4000, replace=False)]
    c = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None] - np.array(c)[None]) ** 2).sum(-1), 1)
        c.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(c)
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    return c


def polygon(points, size=SIZE, width: int = 0) -> np.ndarray:
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    pts = [tuple(map(float, p)) for p in points]
    if width:
        d.line(pts, fill=1, width=width, joint="curve")
    else:
        d.polygon(pts, fill=1)
    return np.asarray(im, bool)


def hull(points) -> list:
    from scipy.spatial import ConvexHull
    p = np.asarray(points, np.float64)
    return p[ConvexHull(p).vertices]


def label_one(img: np.ndarray, alpha: np.ndarray, lm: np.ndarray, skin_tau: float, hair_tau: float):
    char = alpha.astype(np.float32) > 0.5
    lab = srgb_to_oklab(img)
    eye_y = lm[list(EYE_L) + list(EYE_R), 1].mean()
    brow_y = lm[[i for r in BROWS for i in r], 1].mean()
    chin = lm[2]
    lift = max(eye_y - brow_y, 4.0) * 1.2

    # face polygon: contour, closed over the brows lifted towards the forehead
    brows_up = [(x, y - lift) for x, y in lm[[10, 9, 8, 7, 6, 5]]]
    face = polygon(list(map(tuple, lm[list(CONTOUR)])) + brows_up) & char

    feat = np.zeros_like(char)
    for eye in (EYE_L, EYE_R):
        feat |= ndimage.binary_dilation(polygon(hull(lm[list(eye)])), iterations=2)
    feat |= ndimage.binary_dilation(polygon(hull(lm[list(MOUTH)])), iterations=2)
    for brow in BROWS:
        feat |= polygon(lm[list(brow)], width=4)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    feat |= (yy - lm[NOSE, 1]) ** 2 + (xx - lm[NOSE, 0]) ** 2 <= 4
    feat &= char

    # cheek sample: face below the eyes and above the mouth, away from features
    mouth_y = lm[list(MOUTH), 1].min()
    cheek = face & ~ndimage.binary_dilation(feat, iterations=3) & (yy > eye_y + 2) & (yy < max(mouth_y, eye_y + 4))
    if cheek.sum() < 20:
        cheek = face & ~feat
    flags = []
    if cheek.sum() < 20:
        flags.append("no_cheek_sample")
        skin_ref = np.median(lab[face], 0) if face.any() else np.array([0.85, 0.02, 0.03], np.float32)
    else:
        skin_ref = np.median(lab[cheek], 0)
    skin = face & ~feat & (np.linalg.norm(lab - skin_ref, axis=-1) < skin_tau)
    skin = ndimage.binary_opening(skin, iterations=1)

    # hair. Above the eyes, character pixels that are not skin are hair (or hats); they also
    # give 3 k-means hair colours (anime hair is strongly shaded: one median colour misses it).
    # Below the eyes, a pixel is hair only if it is close to one of those colours and connected
    # to the hair above, so shoulders, hands and collars at face height stay body. Skin-coloured
    # pixels next to the face (ears) join skin.
    rest = char & ~skin & ~feat
    skin_like = np.linalg.norm(lab - skin_ref, axis=-1) < skin_tau * 1.2
    skin |= rest & skin_like & ndimage.binary_dilation(face, iterations=6) & (yy <= chin[1])
    rest &= ~skin
    upper = rest & (yy < eye_y)
    hair = upper.copy()
    if upper.sum() >= 30:
        centres = kmeans(lab[upper], 3)
        d = np.min(np.linalg.norm(lab[..., None, :] - centres, axis=-1), -1)
        cand = upper | (rest & (d < hair_tau))
        comp, _ = ndimage.label(cand, structure=np.ones((3, 3)))
        keep = np.unique(comp[upper])
        hair = np.isin(comp, keep[keep > 0])
    else:
        flags.append("no_hair_sample")
    body = char & ~skin & ~feat & ~hair

    out = np.full((SIZE, SIZE), BG, np.uint8)
    out[body], out[hair], out[skin], out[feat] = BODY, HAIR, SKIN, FEATURES
    frac = {name: round(float((out == k).mean()), 4) for k, name in enumerate(("background", "body", "skin",
                                                                               "features", "hair"))}
    face_area = max(int(face.sum()), 1)
    if skin.sum() / face_area < 0.25:
        flags.append("little_skin_in_face")
    if frac["hair"] < 0.03:
        flags.append("little_hair")
    if face.sum() < 0.5 * polygon(list(map(tuple, lm[list(CONTOUR)])) + brows_up).sum():
        flags.append("face_outside_character")
    return out, frac, flags


def work(task):
    data, shard, ids, rows, skin_tau, hair_tau = task
    imgs = np.load(data / f"images_{shard:03d}.npy", mmap_mode="r")
    alpha = np.load(data / f"alpha_{shard:03d}.npy", mmap_mode="r")
    labels, quality = [], []
    for i, image_id in enumerate(ids):
        lm = np.asarray(rows[image_id]["landmarks"], np.float32)
        lab, frac, flags = label_one(np.asarray(imgs[i]), np.asarray(alpha[i]), lm, skin_tau, hair_tau)
        labels.append(lab)
        quality.append({"id": image_id, "fractions": frac, "flags": flags})
    np.save(data / f"labels6_{shard:03d}.npy", np.stack(labels))
    return shard, quality


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--skin-tau", type=float, default=0.09, help="Oklab distance to the cheek colour")
    ap.add_argument("--hair-tau", type=float, default=0.08, help="Oklab distance to the hair colour")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()
    index = json.loads((args.data / "index.json").read_text())
    rows = {}
    for line in (args.data / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["id"]] = {"landmarks": r["landmarks"]}
    shards = index["shards"][: args.limit_shards]
    missing = [s["shard"] for s in shards if not (args.data / f"alpha_{s['shard']:03d}.npy").exists()]
    if missing:
        raise SystemExit(f"run tools/segment_anime.py first: no alpha for shards {missing[:5]}")
    tasks = [(args.data, s["shard"], s["ids"], {i: rows[i] for i in s["ids"]}, args.skin_tau, args.hair_tau)
             for s in shards]
    quality = []
    with ProcessPoolExecutor(args.workers) as pool:
        for shard, q in pool.map(work, tasks):
            quality += q
            print(f"shard {shard:03d}: {len(q)} images", flush=True)
    (args.data / "layers6_quality.jsonl").write_text("".join(json.dumps(q) + "\n" for q in quality))
    flagged = sum(bool(q["flags"]) for q in quality)
    mean = {k: round(float(np.mean([q["fractions"][k] for q in quality])), 4) for k in quality[0]["fractions"]}
    print(json.dumps({"images": len(quality), "flagged": flagged, "mean_fractions": mean,
                      "params": {"skin_tau": args.skin_tau, "hair_tau": args.hair_tau}}, indent=1))


if __name__ == "__main__":
    main()
