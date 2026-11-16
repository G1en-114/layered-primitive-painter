"""Tracing by continuation: the decomposer supplies part of a painting, the model continues it.

    python tools/trace_prefix_eval.py --ckpt out/try/s2_step12000.pt --seq data/seq/v2_clean/t0 \
        --data data/celebamask256 --split-manifest data/celebamask256_split_v1.jsonl \
        --captions data/captions/captions_clean_v1.jsonl --text-encoder data/models/chinese-clip-vit-base-patch16 \
        --n 48 --out out/trace_prefix

For validation faces with decomposer sequences, compare the final painting's PSNR to the real
image when the model gets: no prefix (caption only); the decomposer's background+body+skin
layers; every layer but hair then REDRAW_4 (model paints the hair); hair's first rounds then
REFINE_4 (model finishes the hair); and the decomposer's full sequence (upper bound, no model).
This is how POP 2 does super-resolution and local redraw (vectorize, then continue drawing).
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
from lpp.data.sequences import load_shards  # noqa: E402
from lpp.data.splits import load_split_manifest  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import DrawSeq, build, paint_order  # noqa: E402
from lpp.text import TextEncoder  # noqa: E402

HAIR = 4


def cut(s: DrawSeq, n: int) -> DrawSeq:
    return DrawSeq(s.kind[:n], s.event[:n], s.attrs[:n], s.layer[:n])


def prefixes(ex, proto) -> dict[str, DrawSeq | None]:
    full = build(ex.tokens, ex.layer, ex.round, "full", proto=proto)
    out = {"none": None, "teacher_full": full}
    ev = full.event.tolist()
    marker_hair = [i for i, e in enumerate(ev) if e == proto.layer_event(HAIR)]
    marker_feat = [i for i, e in enumerate(ev) if e == proto.layer_event(3)]
    if marker_feat:                                  # background + body + skin complete
        out["first_3_layers"] = cut(full, marker_feat[0])
    if (ex.layer == HAIR).any():
        rd = build(ex.tokens, ex.layer, ex.round, "redraw", target=HAIR, proto=proto)
        i = rd.event.tolist().index(proto.redraw_event(HAIR))
        out["redraw_hair"] = cut(rd, i + 1)          # everything but hair, then REDRAW_4
        rounds = ex.round[ex.layer == HAIR]
        if int(rounds.max()) >= 2:
            rf = build(ex.tokens, ex.layer, ex.round, "refine", target=HAIR, split_round=2, proto=proto)
            j = rf.event.tolist().index(proto.refine_event(HAIR))
            out["refine_hair"] = cut(rf, j + 1)      # hair rounds 0-1 in the prefix, then REFINE_4
    del marker_hair
    return out


@torch.no_grad()
def render(seq: DrawSeq, codec: Codec) -> np.ndarray:
    proto = codec.proto
    attrs, lay = paint_order(seq)
    p = codec.decode({k: attrs[:, i].unsqueeze(0).cuda() for i, k in enumerate(proto.attr_names)})
    img = flatten_on_base(render_group(p, 256, proto), proto)[0]
    return (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seq", type=Path, required=True, help="decomposer sequences (one trajectory)")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--text-encoder", type=Path, required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    proto = _parse(s["protocol"])
    codec = Codec.load(proto=proto, device="cuda")
    model = Painter(ModelConfig(**s["config"]), proto, codec).cuda().eval()
    model.load_state_dict(s["model"])
    enc = TextEncoder(args.text_encoder, max_positions=proto.max_text_positions).cuda()
    splits = load_split_manifest(args.split_manifest)
    caps = {int(r["id"]): r["long"] for r in (json.loads(line) for line in args.captions.read_text(encoding="utf-8").splitlines() if line.strip())}
    examples = [e for e in load_shards(sorted(args.seq.glob("decomp_*.npz"))) if splits.get(e.image_id) == "val"]
    examples = [e for e in examples if e.image_id in caps][: args.n]
    index = json.loads((args.data / "index.json").read_text())
    where = {i: (sh["shard"], r) for sh in index["shards"] for r, i in enumerate(sh["ids"])}
    cache = {}

    def real(image_id):
        sh, r = where[image_id]
        if sh not in cache:
            cache[sh] = np.load(args.data / f"images_{sh:03d}.npy", mmap_mode="r")
        return np.array(cache[sh][r])

    sc = SampleConfig(temperature=1.0, top_p=0.9, cfg_scale=args.cfg_scale, require_before_eos=(0, 2))
    results: dict[str, list[float]] = {}
    added: dict[str, list[int]] = {}
    strips = []
    for k, ex in enumerate(examples):
        pre = prefixes(ex, proto)
        target = real(ex.image_id)
        t, tv = enc([caps[ex.image_id]])
        row = [target]
        for name in ("none", "first_3_layers", "redraw_hair", "refine_hair", "teacher_full"):
            if name not in pre:
                row.append(np.full_like(target, 255))
                continue
            if name == "teacher_full":
                seq = pre[name]
            else:
                g = torch.Generator(device="cuda").manual_seed(1000 + k)
                forced = None if pre[name] is None else [pre[name]]
                seq = generate(model, batch=1, cfg=sc, text=t, text_valid=tv, generator=g, codec=codec, forced=forced)[0]
                n_pre = 0 if pre[name] is None else int((pre[name].kind == 3).sum())
                added.setdefault(name, []).append(int((seq.kind == 3).sum()) - n_pre)
            img = render(seq, codec)
            results.setdefault(name, []).append(psnr(img, target))
            row.append(img)
        if k < 8:
            strips.append(np.concatenate(row, 1))
        print(f"{k + 1}/{len(examples)}", {n: round(v[-1], 2) for n, v in results.items()}, flush=True)
    summary = {name: {"input_psnr": round(float(np.mean(v)), 3), "n": len(v),
                      "model_added_primitives": round(float(np.mean(added[name])), 1) if name in added else 0}
               for name, v in results.items()}
    (args.out / "summary.json").write_text(json.dumps({"checkpoint": str(args.ckpt), "cfg_scale": args.cfg_scale,
                                                       "results": summary}, indent=1))
    Image.fromarray(np.concatenate(strips, 0)).save(args.out / "strips.jpg", quality=88)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
