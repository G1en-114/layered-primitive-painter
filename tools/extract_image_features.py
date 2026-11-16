"""Frozen Chinese-CLIP image tokens for image-conditioned painting (image -> layered primitives).

    python tools/extract_image_features.py --data data/celebamask256 --jpeg data/celebamask256_jpeg \
        --text-encoder data/models/chinese-clip-vit-base-patch16 --out data/clip_image_tokens_v1

Per image, 50 tokens of width 768: the pooled [CLS] output plus the 14x14 patch grid (after the
vision tower's final LayerNorm) average-pooled to 7x7. Computed for the image and its mirror
(the flipped decomposition trajectories need mirrored conditions). Every image goes through the
same JPEG q95 round trip first, whether it is read from images_XXX.npy or a JPEG pack, so all
tokens come from identical preprocessing.

Output: tokens_orig.npy / tokens_flip.npy [N, 50, 768] float16 (memory-mappable), ids.json.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pack_jpeg import decode_pack  # noqa: E402

GRID = 7


def jpeg_roundtrip(imgs: np.ndarray, quality: int = 95) -> np.ndarray:
    out = []
    for im in imgs:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(im)).save(buf, "JPEG", quality=quality, subsampling=0)
        out.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    return np.stack(out)


@torch.no_grad()
def tokens(model, proc, imgs: np.ndarray, device: str) -> torch.Tensor:
    px = proc(images=[Image.fromarray(a) for a in imgs], return_tensors="pt")["pixel_values"].to(device)
    vm = model.vision_model
    out = vm(pixel_values=px)
    cls = out.pooler_output                                        # post-LN [CLS]
    patches = vm.post_layernorm(out.last_hidden_state[:, 1:])      # [B, 196, D]
    B, n, D = patches.shape
    side = int(round(n ** 0.5))
    grid = patches.transpose(1, 2).reshape(B, D, side, side)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, GRID).flatten(2).transpose(1, 2)
    return torch.cat([cls[:, None], pooled], 1)                    # [B, 1 + GRID^2, D]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dir with index.json (+ images_XXX.npy where present)")
    ap.add_argument("--jpeg", type=Path, default=None, help="dir with images_XXX.jpg.npz for shards lacking .npy")
    ap.add_argument("--text-encoder", type=Path, required=True, help="Chinese-CLIP model directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    from transformers import ChineseCLIPImageProcessor, ChineseCLIPModel
    dev = "cuda"
    model = ChineseCLIPModel.from_pretrained(str(args.text_encoder)).to(dev).eval().half()
    proc = ChineseCLIPImageProcessor.from_pretrained(str(args.text_encoder))
    index = json.loads((args.data / "index.json").read_text())
    ids = [i for s in index["shards"] for i in s["ids"]]
    args.out.mkdir(parents=True, exist_ok=True)
    D, T = model.config.vision_config.hidden_size, 1 + GRID * GRID
    mm = {k: np.lib.format.open_memmap(args.out / f"tokens_{k}.npy", "w+", np.float16, (len(ids), T, D))
          for k in ("orig", "flip")}
    row = 0
    for s in index["shards"]:
        npy = args.data / f"images_{s['shard']:03d}.npy"
        if npy.exists():
            imgs, src = jpeg_roundtrip(np.load(npy)), "npy+jpeg"
        else:
            if args.jpeg is None:
                raise SystemExit(f"shard {s['shard']} has no images_*.npy and no --jpeg pack")
            imgs, pack_ids = decode_pack(args.jpeg / f"images_{s['shard']:03d}.jpg.npz")
            if pack_ids.tolist() != s["ids"]:
                raise ValueError(f"JPEG pack ids differ from index for shard {s['shard']}")
            src = "jpeg"
        for b0 in range(0, len(imgs), args.batch):
            chunk = imgs[b0:b0 + args.batch]
            mm["orig"][row + b0: row + b0 + len(chunk)] = tokens(model, proc, chunk, dev).cpu().numpy()
            mm["flip"][row + b0: row + b0 + len(chunk)] = tokens(model, proc, chunk[:, :, ::-1], dev).cpu().numpy()
        row += len(imgs)
        print(f"shard {s['shard']:03d} ({src}): {row}/{len(ids)}", flush=True)
    for m in mm.values():
        m.flush()
    (args.out / "ids.json").write_text(json.dumps(ids))
    (args.out / "meta.json").write_text(json.dumps({"tokens": T, "dim": D, "grid": GRID, "dtype": "float16",
                                                     "encoder": str(args.text_encoder), "jpeg_quality": 95,
                                                     "layout": "[CLS pooled] + 7x7 avg-pooled post-LN patches"}))


if __name__ == "__main__":
    main()
