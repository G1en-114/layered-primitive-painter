"""Decompose images into layered primitives and save a visual report.

Synthetic scene (no dataset needed):
    python tools/decompose_demo.py --out out/decompose
Real image + layer label map (a PNG whose pixel values are layer ids 0..5):
    python tools/decompose_demo.py --image face.jpg --label face_layers.png --out out/face
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.decompose import DecomposeConfig, LayeredDecomposer  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.render.reference import render_layers, stack_layers, to_straight_rgba  # noqa: E402
from lpp.synthetic import synthetic_scene  # noqa: E402

PALETTE = np.array([[40, 40, 40], [80, 120, 200], [230, 180, 150], [220, 60, 60],
                    [120, 80, 40], [240, 220, 60]], dtype=np.uint8)


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def on_checker(rgba: torch.Tensor) -> np.ndarray:
    _, H, W = rgba.shape
    yy, xx = np.mgrid[:H, :W]
    checker = np.where(((yy // 8) + (xx // 8)) % 2 == 0, 0.85, 0.65)[..., None].repeat(3, -1)
    a = rgba[3:].permute(1, 2, 0).cpu().numpy()
    rgb = rgba[:3].permute(1, 2, 0).cpu().numpy()
    return ((rgb * a + checker * (1 - a)) * 255 + 0.5).astype(np.uint8)


def load_pair(image: Path, label: Path, size: int, dev):
    img = Image.open(image).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    lab = Image.open(label).resize((size, size), Image.Resampling.NEAREST)
    img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).unsqueeze(0)
    lab_t = torch.from_numpy(np.asarray(lab).astype(np.int64)).unsqueeze(0)
    return img_t.to(dev), lab_t.to(dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/decompose"))
    ap.add_argument("--image", type=Path)
    ap.add_argument("--label", type=Path)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    proto = load_protocol()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.image:
        img, label = load_pair(args.image, args.label, args.size, dev)
    else:
        img, label, _, _ = synthetic_scene(args.batch, args.size, seed=args.seed, device=dev)

    codec = Codec.load(device=dev)
    dec = LayeredDecomposer(codec, DecomposeConfig(size=args.size, steps_per_round=args.steps))
    if dev == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    results = dec(img, label)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    print(f"{img.shape[0]} image(s) at {args.size}px in {dt:.1f}s "
          f"({dt / img.shape[0]:.1f}s/image), peak GPU memory {peak:.2f} GiB")

    for b, d in enumerate(results):
        counts = [int((d.layer == k).sum()) for k in range(proto.num_layers)]
        print(f"[{b}] PSNR {d.psnr:.2f} dB  primitives {d.layer.numel()}  per layer {counts}  "
              f"leak {[round(x, 3) for x in d.layer_leak]}")
        prims = codec.decode({n: t.unsqueeze(0).to(dev) for n, t in d.tokens.items()})
        groups = render_layers(prims, d.layer.unsqueeze(0).to(dev), args.size)
        full = stack_layers(groups)[0]
        top = max(k for k in range(proto.num_layers) if counts[k] > 0)
        hidden = stack_layers(groups, visible=[k != top for k in range(proto.num_layers)])[0]
        row = [to_u8(img[b]), PALETTE[label[b].cpu().numpy()], to_u8(full), to_u8(hidden)]
        row += [on_checker(to_straight_rgba(g)[0]) for g in groups]
        Image.fromarray(np.concatenate(row, axis=1)).save(args.out / f"report_{b}.png")
    print(f"columns: input | labels | decomposed | top layer hidden | layers 0..{proto.num_layers - 1}")
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
