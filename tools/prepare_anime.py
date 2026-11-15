"""Screen the actual target illustrations in downloaded anime-face-control Parquet shards.

Preserves supplied image bytes, creates 256px portraits, records crop-aware template
captions and fixed splits. Does not claim image-only outputs already contain layer masks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.data.anime import (  # noqa: E402
    ID_OFFSET, SOURCE, DuplicateIndex, face_crop_box, screen_tags, split_for_id, tags_from_prompt,
    template_captions,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/anime_face_control_60k"))
    ap.add_argument("--out", type=Path, default=Path("data/anime_portraits256_v1"))
    ap.add_argument("--limit", type=int, default=1000, help="accepted portraits; 0 scans all available shards")
    ap.add_argument("--shard-size", type=int, default=500)
    args = ap.parse_args()
    if args.limit < 0 or args.shard_size < 1:
        ap.error("invalid limit/shard size")
    if args.out.exists():
        ap.error("output exists; use a new version directory to preserve its data and split")
    files = sorted((args.raw / "data").glob("*.parquet"))
    if not files:
        ap.error("no downloaded Parquet shards")
    info = json.loads((args.raw / "repo_info.json").read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = args.out.with_name(args.out.name + ".building-" + uuid.uuid4().hex[:8])
    (work / "originals").mkdir(parents=True)
    counts, splits, dedup = Counter(), Counter(), DuplicateIndex()
    image_ids, pixels, shards, previews = [], [], [], []
    input_files, original_hashes, seen_ids = [], {}, set()
    columns = ["item_id", "prompt", "blip_caption", "landmarks", "target", "origin_path"]

    def flush():
        if not image_ids:
            return
        shard = len(shards)
        np.save(work / f"images_{shard:03d}.npy", np.stack(pixels))
        shards.append({"shard": shard, "ids": list(image_ids)})
        image_ids.clear()
        pixels.clear()

    with (work / "manifest.jsonl").open("w") as manifest, (work / "rejected.jsonl").open("w") as rejected:
        for source_file in files:
            if args.limit and counts["accepted"] >= args.limit:
                break
            pf = pq.ParquetFile(source_file)
            source_hash = hashlib.file_digest(source_file.open("rb"), "sha256").hexdigest()
            input_files.append({"file": source_file.name, "sha256": source_hash, "rows": pf.metadata.num_rows})
            for batch in pf.iter_batches(batch_size=32, columns=columns):
                for row in batch.to_pylist():
                    if args.limit and counts["accepted"] >= args.limit:
                        break
                    counts["scanned"] += 1
                    source_id = row["item_id"]
                    image_id = ID_OFFSET + int(source_id)
                    if image_id in seen_ids:
                        raise ValueError(f"repeated source ID {source_id}")
                    seen_ids.add(image_id)
                    tags = tags_from_prompt(row["prompt"] or "")
                    reason, duplicate_of = screen_tags(tags), None
                    if reason is None:
                        try:
                            blob = row["target"]["bytes"]
                            if not blob:
                                raise ValueError("missing_target_bytes")
                            original = Image.open(io.BytesIO(blob))
                            original.load()
                            if original.width < 256 or original.height < 256:
                                raise ValueError("low_resolution")
                            box = face_crop_box(row["landmarks"], original.size)
                            crop = original.convert("RGB").crop(box).resize((256, 256), Image.Resampling.LANCZOS)
                            digest = hashlib.sha256(blob).hexdigest()
                            duplicate_of = original_hashes.get(digest)
                            if duplicate_of is None:
                                duplicate_of = dedup.add(crop, image_id)
                            if duplicate_of is not None:
                                reason = "duplicate"
                        except (ValueError, OSError, TypeError, KeyError) as exc:
                            reason = str(exc) or type(exc).__name__
                    if reason is not None:
                        counts[reason] += 1
                        rejected.write(json.dumps({"source_id": source_id, "reason": reason,
                                                   "duplicate_of": duplicate_of}) + "\n")
                        continue
                    original_hashes[digest] = image_id
                    ext = "png" if original.format == "PNG" else "jpg" if original.format == "JPEG" else "img"
                    original_path = f"originals/{image_id}.{ext}"
                    (work / original_path).write_bytes(blob)
                    points = np.asarray(row["landmarks"], dtype=np.float64)
                    transformed = (points - np.array(box[:2])) * (256 / (box[2] - box[0]))
                    split = split_for_id(image_id)
                    record = {"id": image_id, "source": SOURCE, "source_id": source_id, "revision": info["sha"],
                              "split": split, "original": original_path, "original_sha256": digest,
                              "original_size": list(original.size), "original_dataset_path": row["origin_path"],
                              "image_field": "target", "crop_box": list(box), "landmarks": transformed.tolist(),
                              "shard": len(shards), "row": len(image_ids),
                              "captions": template_captions(tags), "caption_method": "face_tag_template",
                              "original_prompt": row["prompt"], "original_caption": row["blip_caption"],
                              "license_status": "unspecified in upstream dataset card; not cleared for redistribution"}
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    image_ids.append(image_id)
                    pixels.append(np.asarray(crop))
                    if len(previews) < 32:
                        previews.append((image_id, split, crop.copy()))
                    counts["accepted"] += 1
                    splits[split] += 1
                    if len(image_ids) == args.shard_size:
                        flush()
                        print(f"accepted {counts['accepted']} / scanned {counts['scanned']}", flush=True)
                if args.limit and counts["accepted"] >= args.limit:
                    break
        flush()
    if not counts["accepted"]:
        raise RuntimeError(f"no accepted portraits; inspect {work / 'rejected.jsonl'}")
    stats = {"count": counts["accepted"], "size": 256, "counts": dict(counts), "splits": dict(splits),
             "source": SOURCE, "revision": info["sha"], "input_files": input_files,
             "status": "image_only; layer masks not generated yet", "image_field": "target",
             "selection": {"limit": args.limit, "min_face_fraction": 0.18, "crop_expansion": 2.15,
                           "phash_radius": 4, "near_duplicate_rgb_error": 0.035}}
    (work / "index.json").write_text(json.dumps({"shards": shards, "stats": stats}, indent=2) + "\n")
    (work / "provenance.upstream.md").write_text((args.raw / "README.upstream.md").read_text())
    sheet = Image.new("RGB", (8 * 128, ((len(previews) + 7) // 8) * 150), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (image_id, split, crop) in enumerate(previews):
        x, y = n % 8 * 128, n // 8 * 150
        sheet.paste(crop.resize((128, 128), Image.Resampling.LANCZOS), (x, y + 22))
        draw.text((x + 3, y + 4), f"{image_id} {split}", fill="black")
    sheet.save(work / "contact_sheet.jpg", quality=92)
    os.rename(work, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
