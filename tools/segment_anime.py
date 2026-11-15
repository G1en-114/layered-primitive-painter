"""Generate resumable character/background pseudo-labels for prepared anime portraits."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from lpp.protocol import load_protocol  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/anime_portraits256_v1"))
    ap.add_argument("--model", type=Path, default=Path("data/models/anime-seg/isnetis.onnx"))
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-mb", type=int, default=2560)
    args = ap.parse_args()
    import onnxruntime as ort
    if args.device == "cuda":
        import torch  # noqa: F401
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
    providers = ["CPUExecutionProvider"]
    if args.device == "cuda":
        providers.insert(0, ("CUDAExecutionProvider", {"gpu_mem_limit": args.memory_mb * 1024**2,
                                                      "arena_extend_strategy": "kSameAsRequested"}))
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=providers)
    if args.device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("requested CUDA provider could not be loaded")
    inp = session.get_inputs()[0]
    if inp.shape != [1, 3, 1024, 1024]:
        raise ValueError(f"unexpected model input {inp.shape}")
    index_path = args.data / "index.json"
    index = json.loads(index_path.read_text())
    records = {r["id"]: r for r in map(json.loads, (args.data / "manifest.jsonl").read_text().splitlines())}
    with args.model.open("rb") as source:
        model_sha = hashlib.file_digest(source, "sha256").hexdigest()
    config = {"model_sha256": model_sha, "model_repo": "skytnt/anime-seg",
              "revision": json.loads((args.model.parent / "repo_info.json").read_text())["sha"],
              "manifest_sha256": hashlib.sha256((args.data / "manifest.jsonl").read_bytes()).hexdigest(),
              "input_size": 1024, "normalization": "RGB / 255, bilinear resize", "threshold": 0.5,
              "labels": {"0": "background", "1": "character"},
              "label_kind": "model-generated visible foreground; not facial-part or amodal ground truth"}
    config_path = args.data / "segmentation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("cache has a different model/input; use a new dataset version")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    cache = args.data / "mask_cache"
    cache.mkdir(exist_ok=True)
    quality, previews = [], []
    t0, inferred = time.monotonic(), 0
    for sh in index["shards"]:
        sid = sh["shard"]
        images = np.load(args.data / f"images_{sid:03d}.npy", mmap_mode="r")
        if len(images) != len(sh["ids"]):
            raise ValueError("image shard / ID count mismatch")
        alphas, labels = [], []
        for row, image_id in enumerate(sh["ids"]):
            record = records[image_id]
            if (record["shard"], record["row"]) != (sid, row):
                raise ValueError("manifest / shard ordering mismatch")
            target = cache / f"{image_id}.npz"
            if target.exists():
                with np.load(target) as saved:
                    alpha = saved["alpha"]
            else:
                resized = Image.fromarray(np.array(images[row])).resize((1024, 1024), Image.Resampling.BILINEAR)
                inputs = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255
                prediction = session.run(None, {inp.name: inputs})[0][0, 0]
                if not np.isfinite(prediction).all() or prediction.min() < -1e-4 or prediction.max() > 1.0001:
                    raise ValueError(f"invalid model probability for {image_id}")
                alpha = np.asarray(Image.fromarray(prediction).resize((256, 256), Image.Resampling.BILINEAR))
                alpha = alpha.clip(0, 1).astype(np.float16)
                tmp = target.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, alpha=alpha)
                os.replace(tmp, target)
                inferred += 1
            if alpha.shape != (256, 256) or not np.isfinite(alpha).all():
                raise ValueError(f"invalid cached mask for {image_id}")
            label = (alpha >= 0.5).astype(np.uint8)
            points = np.asarray(record["landmarks"]).round().astype(int).clip(0, 255)
            score, foreground = float(alpha[points[:, 1], points[:, 0]].mean()), float(label.mean())
            flags = []
            if foreground < 0.05 or foreground > 0.999:
                flags.append("extreme_foreground_area")
            if score < 0.8:
                flags.append("low_face_foreground_score")
            quality.append({"id": image_id, "split": record["split"], "foreground_fraction": foreground,
                            "uncertain_fraction": float(((alpha > 0.1) & (alpha < 0.9)).mean()),
                            "face_foreground_score": score, "review_flags": flags})
            alphas.append(alpha)
            labels.append(label)
            if len(previews) < 16:
                previews.append((image_id, np.array(images[row]), label))
            if len(quality) % 50 == 0 or len(quality) == index["stats"]["count"]:
                rate = inferred / max(time.monotonic() - t0, 1e-6)
                print(f"{len(quality)}/{index['stats']['count']} masks, {rate:.2f} new images/s, "
                      f"review flags {sum(bool(q['review_flags']) for q in quality)}", flush=True)
        for prefix, values in (("labels", labels), ("alpha", alphas)):
            final = args.data / f"{prefix}_{sid:03d}.npy"
            tmp = final.with_suffix(".tmp.npy")
            np.save(tmp, np.stack(values))
            os.replace(tmp, final)
    (args.data / "mask_quality.jsonl").write_text("".join(json.dumps(q) + "\n" for q in quality))
    eligible = [q for q in quality if not q["review_flags"]]
    for split in ("train", "val"):
        (args.data / f"{split}_ids.json").write_text(json.dumps(
            [q["id"] for q in eligible if q["split"] == split]) + "\n")
    (args.data / "eligible_ids.json").write_text(json.dumps([q["id"] for q in eligible]) + "\n")
    (args.data / "review_ids.json").write_text(json.dumps(
        [q["id"] for q in quality if q["review_flags"]]) + "\n")
    profile = copy.deepcopy(load_protocol().raw)
    profile["version"] = 2
    profile["profile"] = "anime_fg_bg_pilot_v1; decomposition only, not a checkpoint migration"
    profile["layers"] = [{"id": 0, "name": "background", "max_primitives": 96, "celebamask": []},
                         {"id": 1, "name": "character", "max_primitives": 768, "celebamask": []}]
    profile["events"] = ["PRIMITIVE", "LAYER_0", "LAYER_1", "REDRAW_0", "REDRAW_1",
                         "REFINE_0", "REFINE_1", "EOS"]
    profile_path = args.data / "decomposition_protocol.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    load_protocol(profile_path)
    index["stats"].update(status="two_layer_pseudo_labels_ready; review flags recorded separately",
                          layers=["background", "character"], label_kind="visible_region_pseudo_labels",
                          segmentation=config, review_flag_count=sum(bool(q["review_flags"]) for q in quality),
                          eligible_count=len(eligible), eligible_splits={
                              split: sum(q["split"] == split for q in eligible) for split in ("train", "val")})
    tmp_index = index_path.with_suffix(".tmp.json")
    tmp_index.write_text(json.dumps(index, indent=2) + "\n")
    os.replace(tmp_index, index_path)
    sheet = Image.new("RGB", (1024, ((len(previews) + 1) // 2) * 280), "white")
    draw = ImageDraw.Draw(sheet)
    palette = np.array([[45, 55, 75], [230, 170, 120]], np.uint8)
    for n, (image_id, image, label) in enumerate(previews):
        x, y = n % 2 * 512, n // 2 * 280
        sheet.paste(Image.fromarray(image), (x, y + 24))
        overlay = (0.55 * image + 0.45 * palette[label]).astype(np.uint8)
        sheet.paste(Image.fromarray(overlay), (x + 256, y + 24))
        draw.text((x + 4, y + 6), f"{image_id}: RGB | character/background", fill="black")
    sheet.save(args.data / "mask_contact_sheet.jpg", quality=92)
    print(json.dumps({"masks": len(quality), "review_flags": index["stats"]["review_flag_count"],
                      "providers": session.get_providers(), "seconds": time.monotonic() - t0}), flush=True)


if __name__ == "__main__":
    main()
