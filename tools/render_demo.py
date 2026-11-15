"""Visual smoke test: random layered primitives -> composite, per-layer PNGs, snapped version.

    python tools/render_demo.py --out out/demo
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.render import Primitives, render, render_layers, stack_layers, to_straight_rgba  # noqa: E402


def save(t: torch.Tensor, path: Path) -> None:
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/demo"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    proto = load_protocol()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(args.seed)
    n, L = args.n, proto.num_layers
    layer_id = torch.sort(torch.randint(L, (1, n), generator=g)).values
    # coarse-to-fine inside the sequence: earlier primitives are bigger
    scale = torch.linspace(1.0, 0.15, n).view(1, n, 1)
    r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    prims = Primitives(
        xy=r(1, n, 2) * 256,
        sigma=(4 + r(1, n, 2) * 60) * scale,
        theta=r(1, n) * math.pi,
        beta=torch.tensor(proto.beta_levels)[torch.randint(len(proto.beta_levels), (1, n), generator=g)],
        rgb=r(1, n, 3),
        alpha=0.4 + 0.59 * r(1, n),
    )
    prims = Primitives(**{k: getattr(prims, k).to(dev) for k in prims.__dataclass_fields__})
    layer_id = layer_id.to(dev)

    save(render(prims, args.size, chunk=64)[0], args.out / "composite.png")
    layers = render_layers(prims, layer_id, args.size)
    for k, grp in enumerate(layers):
        rgba = to_straight_rgba(grp)[0]
        arr = (rgba.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        Image.fromarray(arr, "RGBA").save(args.out / f"layer{k}_{proto.layers[k].name}.png")
    hidden = [k != L - 1 for k in range(L)]
    save(stack_layers(layers, visible=hidden)[0], args.out / "composite_without_top_layer.png")

    codec = Codec.load(device=dev)
    snapped = codec.snap(prims, straight_through=False)
    a, b = render(prims, args.size, chunk=64), render(snapped, args.size, chunk=64)
    save(b[0], args.out / "composite_snapped.png")
    print(f"tokens -> image PSNR vs continuous: {-10 * torch.log10(((a - b) ** 2).mean()):.2f} dB")
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
