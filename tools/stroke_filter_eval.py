"""Do the finest strokes help or hurt? Paint once, render with some strokes removed, score every version.

    python tools/stroke_filter_eval.py --ckpt runs/base_v3_snap/model_final.pt --out out/stroke_filter_face \
        --text-encoder data/models/chinese-clip-vit-base-patch16 \
        --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16 \
        --data data/celebamask256_val --captions data/captions/captions_v3.jsonl \
        --split-manifest data/celebamask256_split_v1.jsonl --cfg-scale 5.0 --top-p 0.6

The decomposer's last rounds fix pixel-level differences to one particular photo. From a caption alone those
strokes are close to unpredictable, so the model may paint them as noise (the specks in the samples). Each
generated painting is rendered again:
  - without strokes whose larger radius is under 1 / 2 / 3 px;
  - keeping only the first 60% / 80% of each layer's strokes (a layer is painted coarse to fine, so this
    drops its last rounds).
Every version of a painting comes from the same sampled sequence, so the differences are the removed
strokes alone. Scored like eval_text.py: the independent judge's R@1 / R@5 and FD to the real images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_text import frechet, load_val, retrieval  # noqa: E402
from lpp.clip_score import ClipScorer  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import paint_order  # noqa: E402
from lpp.text import TextEncoder  # noqa: E402

SIZE_CUTS = (1.0, 2.0, 3.0)        # drop strokes whose larger radius (sigma, px) is below this
KEEP_FRACS = (0.6, 0.8)            # keep this share of each layer's strokes, in paint order


def variants(lay: torch.Tensor, sigma: torch.Tensor) -> dict[str, torch.Tensor]:
    """Boolean keep-masks [N] for every version of one painting."""
    big = sigma.max(-1).values
    out = {"full": torch.ones_like(big, dtype=torch.bool)}
    for c in SIZE_CUTS:
        out[f"no_under_{c:g}px"] = big >= c
    rank = torch.zeros_like(big)
    for k in lay.unique():
        idx = (lay == k).nonzero()[:, 0]
        rank[idx] = torch.arange(len(idx), device=big.device, dtype=big.dtype) / max(len(idx), 1)
    for f in KEEP_FRACS:
        out[f"first_{int(f * 100)}pct"] = rank < f
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-encoder", type=Path, required=True)
    ap.add_argument("--judge-encoder", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    proto = _parse(state["protocol"])
    codec = Codec.load(proto=proto, device=dev)
    model = Painter(ModelConfig(**state["config"]), proto, codec).to(dev).eval()
    model.load_state_dict(state["model"])
    encoder = TextEncoder(args.text_encoder).to(dev)
    ids, prompts, real = load_val(args)
    sc = SampleConfig(temperature=1.0, top_p=args.top_p, cfg_scale=args.cfg_scale)

    imgs: dict[str, list[np.ndarray]] = {}
    kept: dict[str, list[float]] = {}
    for b0 in range(0, args.n, args.batch):
        text, text_valid = encoder(prompts[b0:b0 + args.batch])
        g = torch.Generator(device=dev).manual_seed(args.seed * 100003 + b0)     # as eval_text.py
        with torch.no_grad():
            for s in generate(model, batch=text.shape[0], cfg=sc, text=text, text_valid=text_valid,
                              generator=g, codec=codec):
                attrs, lay = paint_order(s)
                p = codec.decode({k: attrs[:, i].unsqueeze(0).to(dev) for i, k in enumerate(proto.attr_names)})
                for name, keep in variants(lay.to(dev), p.sigma[0]).items():
                    q = p.with_alpha(p.alpha * keep[None].to(p.alpha.dtype))
                    im = flatten_on_base(render_group(q, 256, proto), proto)[0]
                    imgs.setdefault(name, []).append((im.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                    kept.setdefault(name, []).append(float(keep.float().mean()))
        print(f"  generated {len(imgs['full'])}/{args.n}", flush=True)
    del model
    torch.cuda.empty_cache()

    judge = ClipScorer(args.judge_encoder, dev)
    t = judge.texts(prompts)
    r = judge.images(real).cpu().double().numpy()
    result = {"checkpoint": str(args.ckpt), "n": args.n, "cfg_scale": args.cfg_scale, "top_p": args.top_p, "versions": {}}
    for name, ims in imgs.items():
        e = judge.images(ims)
        rr = retrieval(e, t)
        result["versions"][name] = {"strokes_kept": round(float(np.mean(kept[name])), 3),
                                    "judge_r@1": rr["r_precision@1"], "judge_r@5": rr["r_precision@5"],
                                    "judge_clip": round(100 * float((e * t).sum(1).mean()), 3),
                                    "fd": round(frechet(e.cpu().double().numpy(), r), 4)}
        print(f"{name:16s} kept {result['versions'][name]['strokes_kept']:.3f}  "
              f"R@1 {rr['r_precision@1']:.3f}  R@5 {rr['r_precision@5']:.3f}  FD {result['versions'][name]['fd']:.4f}",
              flush=True)
    (args.out / "stroke_filter.json").write_text(json.dumps(result, indent=1))
    names = list(imgs)
    sheet = np.concatenate([np.concatenate([imgs[n][k] for k in range(8)], 1) for n in names] +
                           [np.concatenate([real[k][..., :3] for k in range(8)], 1)], 0)
    Image.fromarray(sheet).save(args.out / "versions.png")
    (args.out / "versions_rows.txt").write_text("\n".join(names + ["real"]) + "\n")


if __name__ == "__main__":
    main()
