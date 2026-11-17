"""How early does a painting show its subject? CLIP scores of partial paintings.

    python tools/prefix_quality.py --ckpt runs/x/model_final.pt --out out/prefix_x \
        --text-encoder data/models/chinese-clip-vit-base-patch16 \
        --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16 \
        --data data/celebamask256 --captions data/captions/captions_clean_v1.jsonl \
        --split-manifest data/celebamask256_split_v1.jsonl --n 96

Generates one painting per validation prompt (same prompts and seeds as eval_text.py), cuts each
after the first 25 / 50 / 100 / 200 / 400 strokes in the order they were painted (composited by
layer, as the canvas does) and scores every cut with the conditioning CLIP and the independent
judge: 100 * cosine(image, prompt) and retrieval R@5 of the prompt among all n prompts.

Built to compare painting orders (layer by layer vs draft, docs/draft_order.md): with a draft
order the subject should be recognisable after far fewer strokes, at an unchanged final score.
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

from eval_text import load_val, retrieval  # noqa: E402
from lpp.clip_score import ClipScorer  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import KIND_PRIMITIVE  # noqa: E402
from lpp.text import TextEncoder  # noqa: E402

CUTS = (25, 50, 100, 200, 400)


@torch.no_grad()
def render_prefix(seq, n: int, codec: Codec) -> np.ndarray:
    """The first n painted primitives of seq (sequence order), composited by layer."""
    proto = codec.proto
    is_prim = seq.kind == KIND_PRIMITIVE
    attrs, lay = seq.attrs[is_prim][:n], seq.layer[is_prim][:n]
    if attrs.shape[0] == 0:
        return np.full((256, 256, 3), 255, np.uint8)
    order = torch.sort(lay, stable=True).indices
    attrs = attrs[order]
    p = codec.decode({k: attrs[:, i].unsqueeze(0).cuda() for i, k in enumerate(proto.attr_names)})
    img = flatten_on_base(render_group(p, 256, proto), proto)[0]
    return (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-encoder", type=Path, required=True)
    ap.add_argument("--judge-encoder", type=Path, default=None)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-strips", type=int, default=6)
    ap.add_argument("--sdpa-backend", choices=["auto", "math"], default=None,
                    help="default: math on ROCm/HIP (the DCU, as every eval there), auto elsewhere")
    args = ap.parse_args()
    if args.sdpa_backend is None:
        args.sdpa_backend = "math" if torch.version.hip else "auto"
    if args.sdpa_backend == "math":                      # as eval_text.py (DCU)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    proto = _parse(state["protocol"])
    codec = Codec.load(proto=proto, device=dev)
    model = Painter(ModelConfig(**state["config"]), proto, codec).to(dev).eval()
    model.load_state_dict(state["model"])
    order = getattr(model.cfg, "order", "layer")
    encoder = TextEncoder(args.text_encoder).to(dev)
    _, prompts, _ = load_val(args)
    print(f"checkpoint step {state['step']}, order {order}, {len(prompts)} prompts", flush=True)

    sc = SampleConfig(temperature=1.0, top_p=args.top_p, cfg_scale=args.cfg_scale)
    cuts = {c: [] for c in CUTS}
    final, lengths = [], []
    for b0 in range(0, args.n, args.batch):
        text, text_valid = encoder(prompts[b0:b0 + args.batch])
        g = torch.Generator(device=dev).manual_seed(args.seed * 100003 + b0)   # as eval_text.py
        for s in generate(model, batch=text.shape[0], cfg=sc, text=text, text_valid=text_valid,
                          generator=g, codec=codec):
            n_prims = int((s.kind == KIND_PRIMITIVE).sum())
            lengths.append(n_prims)
            for c in CUTS:
                cuts[c].append(render_prefix(s, min(c, n_prims), codec))
            final.append(render_prefix(s, n_prims, codec))
        print(f"  generated {len(final)}/{args.n}", flush=True)
    del model
    torch.cuda.empty_cache()

    result = {"step": state["step"], "checkpoint": str(args.ckpt), "order": order, "n": args.n,
              "mean_length": round(float(np.mean(lengths)), 1), "cuts": {}}
    scorers = [("b16", args.text_encoder)] + ([("judge", args.judge_encoder)] if args.judge_encoder else [])
    for name, path in scorers:
        clip = ClipScorer(path, dev)
        t = clip.texts(prompts)
        for c, imgs in [*cuts.items(), ("final", final)]:
            i = clip.images(imgs)
            row = result["cuts"].setdefault(str(c), {})
            row[f"{name}_clip"] = round(100 * float((i * t).sum(1).mean()), 3)
            row[f"{name}_r@5"] = retrieval(i, t)["r_precision@5"]
        del clip
        torch.cuda.empty_cache()
    (args.out / "prefix_quality.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    rows = [np.concatenate([cuts[c][k] for c in CUTS] + [final[k]], 1) for k in range(min(args.save_strips, len(final)))]
    if rows:
        Image.fromarray(np.concatenate(rows, 0)).save(args.out / "prefix_strips.png")
    print(json.dumps(result["cuts"], indent=1), flush=True)


if __name__ == "__main__":
    main()
