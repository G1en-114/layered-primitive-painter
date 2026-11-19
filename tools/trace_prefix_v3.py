"""Tracing by continuation, protocol v3 / draft-order models: how close can the model get to a GIVEN image?

    python tools/trace_prefix_v3.py --ckpt runs/v3_formal/B_ft/model_step012000.pt --out out/trace_v3 --n 48

For validation images that have decomposer sequences (faces: Gaussian-only v2 trajectory t0, widened; anime: the v3
line decomposition), the final painting's PSNR against the real image when the model gets:
  none       the caption only (what text-to-image gives; the floor)
  sketch     the decomposer's sketch (every layer's rounds 0-1, the draft order's first part); the model refines
  half       the decomposer's sequence up to half of its strokes (sketch + part of the refinement); the model ends it
  teacher    the decomposer's whole sequence (the ceiling, no model)
The draft-order version of tools/trace_prefix_eval.py (layer order, Gaussians only). Writes summary.json and
strips_<domain>.jpg (real | none | sketch | half | teacher).
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

from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import load_shards, widen  # noqa: E402
from lpp.data.splits import load_split_manifest  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.render.strokes import render_strokes  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import KIND_PRIMITIVE, DrawSeq, build, strokes  # noqa: E402
from lpp.text import TextEncoder  # noqa: E402

DOMAINS = {   # name: (decomposer sequences, real images)
    "face": ("data/seq/v2", "data/celebamask256"),
    "anime": ("data/seq/anime_v3_lines_full_t0", "data/anime_portraits256_v2_l6view"),
}


def cut(s: DrawSeq, n: int) -> DrawSeq:
    return DrawSeq(s.kind[:n], s.event[:n], s.attrs[:n], s.layer[:n])


def prefixes(ex, proto) -> dict[str, DrawSeq | None]:
    full = build(widen(ex.tokens, proto), ex.layer, ex.round, "draft", proto=proto, is_line=ex.is_line)
    refine = [i for i, e in enumerate(full.event.tolist()) if e in {proto.refine_event(k) for k in range(proto.num_layers)}]
    prim = (full.kind == KIND_PRIMITIVE).nonzero()[:, 0]
    out = {"none": None, "teacher": full}
    if refine:
        out["sketch"] = cut(full, refine[0])
    out["half"] = cut(full, int(prim[len(prim) // 2]))
    return out


@torch.no_grad()
def render(seq: DrawSeq, codec: Codec) -> np.ndarray:
    a, lay, ln = strokes(seq, codec.proto)
    img = render_strokes(a, lay, ln, codec, 256)[0]
    return (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=48, help="validation images per domain")
    ap.add_argument("--domains", default="face,anime")
    ap.add_argument("--split-manifest", type=Path, default=Path("data/split_v3_faces_anime.jsonl"))
    ap.add_argument("--captions", type=Path, default=Path("data/captions/captions_v3.jsonl"))
    ap.add_argument("--text-encoder", type=Path, default=Path("data/models/chinese-clip-vit-base-patch16"))
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--top-p", type=float, default=0.6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    proto = _parse(s["protocol"])
    codec = Codec.load(proto=proto, device="cuda")
    model = Painter(ModelConfig(**s["config"]), proto, codec).cuda().eval()
    model.load_state_dict(s["model"])
    enc = TextEncoder(args.text_encoder, max_positions=proto.max_text_positions).cuda()
    splits = load_split_manifest(args.split_manifest)
    caps = {int(r["id"]): r["long"] for r in (json.loads(x) for x in args.captions.read_text(encoding="utf-8").splitlines() if x.strip())}
    sc = SampleConfig(temperature=1.0, top_p=args.top_p, cfg_scale=args.cfg_scale)
    summary = {}
    for dom in args.domains.split(","):
        seq_dir, img_dir = map(Path, DOMAINS[dom])
        examples = []
        for f in sorted(seq_dir.glob("decomp_*.npz")):
            examples += [e for e in load_shards([f]) if splits.get(e.image_id) == "val" and e.image_id in caps]
            if len(examples) >= args.n:
                break
        examples = examples[: args.n]
        index = json.loads((img_dir / "index.json").read_text())
        where = {int(i): (sh["shard"], r) for sh in index["shards"] for r, i in enumerate(sh["ids"])}
        cache: dict[int, np.ndarray] = {}

        def real(image_id: int) -> np.ndarray:
            sh, r = where[image_id]
            if sh not in cache:
                cache[sh] = np.load(img_dir / f"images_{sh:03d}.npy", mmap_mode="r")
            return np.array(cache[sh][r])

        res: dict[str, list[float]] = {}
        added: dict[str, list[int]] = {}
        strips = []
        for k, ex in enumerate(examples):
            pre, target = prefixes(ex, proto), real(ex.image_id)
            t, tv = enc([caps[ex.image_id]])
            row = [target]
            for name in ("none", "sketch", "half", "teacher"):
                if name not in pre:
                    row.append(np.full_like(target, 255))
                    continue
                if name == "teacher":
                    seq = pre[name]
                else:
                    g = torch.Generator(device="cuda").manual_seed(1000 + k)
                    forced = None if pre[name] is None else [pre[name]]
                    seq = generate(model, batch=1, cfg=sc, text=t, text_valid=tv, generator=g, codec=codec, forced=forced)[0]
                    n_pre = 0 if pre[name] is None else int((pre[name].kind == KIND_PRIMITIVE).sum())
                    added.setdefault(name, []).append(int((seq.kind == KIND_PRIMITIVE).sum()) - n_pre)
                img = render(seq, codec)
                res.setdefault(name, []).append(psnr(img, target))
                row.append(img)
            if k < 8:
                strips.append(np.concatenate(row, 1))
            print(f"{dom} {k + 1}/{len(examples)}", {n: round(v[-1], 2) for n, v in res.items()}, flush=True)
        summary[dom] = {n: {"psnr": round(float(np.mean(v)), 2), "n": len(v),
                            "model_added_strokes": round(float(np.mean(added[n])), 1) if n in added else 0}
                        for n, v in res.items()}
        Image.fromarray(np.concatenate(strips, 0)).save(args.out / f"strips_{dom}.jpg", quality=90)
    (args.out / "summary.json").write_text(json.dumps({"checkpoint": str(args.ckpt), "cfg_scale": args.cfg_scale,
                                                       "top_p": args.top_p, "results": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
