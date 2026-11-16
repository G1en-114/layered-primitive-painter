"""COCO Panoptic -> 256px images + per-image layer slots (protocol v3 design, docs/protocol_v3_design.md).

    python tools/prepare_coco.py --split val2017 --out data/coco256_val_v1

Each image becomes up to 8 slots, painted in slot order:
  stuff groups (<= 3, in the order sky -> other stuff -> ground/floor/water),
  one "misc things" slot (small things and crowds),
  the 4 largest thing instances (>= --min-thing area), far to near by bbox bottom edge.
Void pixels join the nearest labelled slot. Panoptic masks give visible pixels only: slot
order is an approximation of depth, recorded as such.

Outputs per shard: images_XXX.npy [N,256,256,3] uint8, labels_XXX.npy [N,256,256] uint8 slot ids,
slots_XXX.jsonl (slot kinds, COCO category ids, areas, pilot group). index.json lists shards/ids.
Image ids are ID_OFFSET + COCO image id so they never collide with other sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ID_OFFSET = 300_000_000
SIZE = 256
MAX_SLOTS = 8
MAX_BIG_THINGS = 4
SKY = {"sky", "ceiling"}
GROUND = {"ground", "floor", "water"}
OUTDOOR = {"sky", "ground", "water", "plant", "solid"}


def rgb2id(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.int64)
    return a[..., 0] + 256 * a[..., 1] + 256 * 256 * a[..., 2]


def choose_crop(seg: np.ndarray, target: np.ndarray | None) -> tuple[int, float]:
    """Offset along the long axis keeping most of the target mask (or centred). Returns (offset, kept)."""
    long = max(seg.shape)
    span = long - SIZE
    if span <= 0 or target is None:
        return span // 2, 1.0
    axis = 1 if seg.shape[1] > seg.shape[0] else 0
    profile = target.sum(axis=1 - axis).astype(np.float64)         # mass along the long axis
    window = np.convolve(profile, np.ones(SIZE), mode="valid")       # kept mass per offset
    best = window.max()
    candidates = np.flatnonzero(window >= best - 1e-9)
    offset = int(candidates[np.argmin(np.abs(candidates - span / 2))])
    return offset, float(best / max(profile.sum(), 1))


def process(task) -> dict | None:
    img_path, png_path, ann, cats, min_thing, min_kept = task
    rgb = Image.open(img_path).convert("RGB")
    pan = rgb2id(np.asarray(Image.open(png_path).convert("RGB")))
    w, h = rgb.size
    scale = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * scale)), max(SIZE, round(h * scale))
    rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    seg = np.asarray(Image.fromarray(pan.astype(np.int32)).resize((nw, nh), Image.Resampling.NEAREST))

    infos = {s["id"]: s for s in ann["segments_info"]}
    things = [s for s in ann["segments_info"] if cats[s["category_id"]]["isthing"] and not s["iscrowd"]]
    largest = max(things, key=lambda s: s["area"]) if things else None
    offset, kept = choose_crop(seg, seg == largest["id"] if largest else None)
    if kept < min_kept:
        return None
    if nw > nh:
        seg, rgb = seg[:, offset:offset + SIZE], rgb.crop((offset, 0, offset + SIZE, SIZE))
    else:
        seg, rgb = seg[offset:offset + SIZE, :], rgb.crop((0, offset, SIZE, offset + SIZE))

    ids, counts = np.unique(seg, return_counts=True)
    area = {int(i): int(c) for i, c in zip(ids, counts) if int(i) in infos}
    total = SIZE * SIZE
    groups: dict[str, list[int]] = {"sky": [], "stuff": [], "ground": []}
    big, misc = [], []
    for sid, a in area.items():
        info, cat = infos[sid], cats[infos[sid]["category_id"]]
        if not cat["isthing"]:
            sc = cat["supercategory"]
            groups["sky" if sc in SKY else "ground" if sc in GROUND else "stuff"].append(sid)
        elif not info["iscrowd"] and a >= min_thing * total:
            big.append(sid)
        else:
            misc.append(sid)
    ys = np.arange(SIZE)[:, None]
    bottom = {sid: int((ys * (seg == sid)).max()) for sid in big}
    big = sorted(sorted(big, key=lambda s: -area[s])[:MAX_BIG_THINGS], key=lambda s: bottom[s])
    misc += [s for s in area if s not in misc and s not in big and cats[infos[s]["category_id"]]["isthing"]]

    slots = []
    for kind in ("sky", "stuff", "ground"):
        if groups[kind]:
            slots.append({"kind": kind, "segments": groups[kind]})
    if misc:
        slots.append({"kind": "misc_things", "segments": misc})
    for sid in big:
        slots.append({"kind": "thing", "segments": [sid]})
    if not slots or len(slots) > MAX_SLOTS:
        return None

    label = np.full((SIZE, SIZE), 255, np.uint8)
    for k, slot in enumerate(slots):
        label[np.isin(seg, slot["segments"])] = k
    void = label == 255
    if void.all():
        return None
    if void.any():                                   # void pixels join the nearest labelled slot
        _, (iy, ix) = ndimage.distance_transform_edt(void, return_indices=True)
        label = label[iy, ix]
    for k, slot in enumerate(slots):
        slot["area"] = round(float((label == k).mean()), 5)
        slot["categories"] = sorted({infos[s]["category_id"] for s in slot["segments"]})
        del slot["segments"]

    stuff_area = sum(s["area"] for s in slots if s["kind"] in ("sky", "stuff", "ground"))
    outdoor = sum(area[s] for s in area if not cats[infos[s]["category_id"]]["isthing"]
                  and cats[infos[s]["category_id"]]["supercategory"] in OUTDOOR) / total
    top = max((s for s in slots if s["kind"] == "thing"), key=lambda s: s["area"], default=None)
    top_sc = cats[top["categories"][0]]["supercategory"] if top else None
    if top and top["area"] >= 0.08 and top_sc == "animal":
        group = "animal"
    elif top and top["area"] >= 0.08 and top_sc not in ("animal", "person"):
        group = "object"
    elif stuff_area >= 0.85 and outdoor >= 0.6:
        group = "landscape"
    elif top_sc == "person":
        group = "person"
    else:
        group = "other"
    return {"id": ID_OFFSET + ann["image_id"], "coco_id": ann["image_id"], "file": img_path.name,
            "kept_fraction": round(kept, 4), "group": group, "slots": slots,
            "rgb": np.asarray(rgb, np.uint8), "label": label}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/coco2017"))
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--min-thing", type=float, default=0.015, help="area fraction for an own slot")
    ap.add_argument("--min-kept", type=float, default=0.6, help="reject crops keeping less of the main thing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; use a new versioned output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    ann_file = args.root / "annotations" / f"panoptic_{args.split}.json"
    meta = json.loads(ann_file.read_text())
    cats = {c["id"]: c for c in meta["categories"]}
    anns = sorted(meta["annotations"], key=lambda a: a["image_id"])[: args.limit or None]
    tasks = [(args.root / args.split / a["file_name"].replace(".png", ".jpg"),
              args.root / "annotations" / f"panoptic_{args.split}" / a["file_name"], a, cats,
              args.min_thing, args.min_kept) for a in anns]

    shards, buf, groups, slot_counts, rejected = [], [], Counter(), Counter(), 0

    def flush():
        s = len(shards)
        np.save(args.out / f"images_{s:03d}.npy", np.stack([r["rgb"] for r in buf]))
        np.save(args.out / f"labels_{s:03d}.npy", np.stack([r["label"] for r in buf]))
        with (args.out / f"slots_{s:03d}.jsonl").open("w") as f:
            for r in buf:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("rgb", "label")}) + "\n")
        shards.append({"shard": s, "ids": [r["id"] for r in buf]})
        buf.clear()

    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(process, tasks, chunksize=16):
            if r is None:
                rejected += 1
                continue
            buf.append(r)
            groups[r["group"]] += 1
            slot_counts[len(r["slots"])] += 1
            if len(buf) == args.shard_size:
                flush()
                print(f"{sum(len(s['ids']) for s in shards)} images, rejected {rejected}", flush=True)
    if buf:
        flush()
    stats = {"size": SIZE, "count": sum(len(s["ids"]) for s in shards), "rejected": rejected,
             "groups": dict(groups), "slots_per_image": dict(sorted(slot_counts.items())),
             "source": f"COCO panoptic {args.split}", "id_offset": ID_OFFSET,
             "annotation_sha256": hashlib.sha256(ann_file.read_bytes()).hexdigest(),
             "params": {"min_thing": args.min_thing, "min_kept": args.min_kept, "max_slots": MAX_SLOTS},
             "categories": {c["id"]: {"name": c["name"], "supercategory": c["supercategory"],
                                      "isthing": c["isthing"]} for c in meta["categories"]},
             "note": "slot order approximates depth; panoptic masks are visible regions only"}
    (args.out / "index.json").write_text(json.dumps({"shards": shards, "stats": stats}, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "categories"}, indent=1))


if __name__ == "__main__":
    main()
