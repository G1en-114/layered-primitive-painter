"""Compare quantized ONNX variants with the float32 export: speed and distribution-level quality.

    python tools/quant_eval.py --text-encoder data/models/chinese-clip-vit-base-patch16 \
        --captions data/captions/captions_clean_v1.jsonl --split-manifest data/celebamask256_split_v1.jsonl \
        --variant fp32-cpu=out/onnx/m:CPU --variant int8-cpu=out/onnx/m_int8:CPU \
        --variant fp32-cuda=out/onnx/m:CUDA --variant fp16-cuda=out/onnx/m_fp16:CUDA --out out/quant_eval

Every variant decodes the same validation prompts with the same uniform table (tools/ort_generate.py).
The first variant is the reference. Reported per variant:
  records/s (whole batch, Python decode loop), token agreement with the reference and the first divergence,
  PSNR of each rendered image against the reference image (same seed), CLIP score of image vs prompt
  (distribution-level: quantization changes which samples are drawn, not necessarily how good they are),
  mean length and early-EOS count. CUDA runs with TF32 off (TF32 alone makes float32 diverge).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from lpp.clip_score import ClipScorer  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.sample import SampleConfig  # noqa: E402
from lpp.sequence import paint_order  # noqa: E402
from lpp.text import TextEncoder  # noqa: E402
from ort_generate import OrtPainter, ort_generate  # noqa: E402

PROVIDERS = {"CPU": ["CPUExecutionProvider"],
             "CUDA": [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]}


def render(seq, codec) -> np.ndarray:
    proto = codec.proto
    attrs, _ = paint_order(seq)
    if attrs.shape[0] == 0:
        return np.full((256, 256, 3), 255, np.uint8)
    p = codec.decode({k: attrs[:, i].unsqueeze(0) for i, k in enumerate(proto.attr_names)})
    img = flatten_on_base(render_group(p, 256, proto, backend="torch"), proto)[0]
    return (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


class Recorder:
    """Wraps OrtPainter.run on the reference variant: keeps the first `limit` input feeds per graph, so
    each quantized graph can be judged on real activations (one step, no error accumulation)."""

    def __init__(self, painter, limit: int):
        self.p, self.limit, self.feeds = painter, limit, {}
        self._run = painter.run

    def run(self, name, **feeds):
        bucket = self.feeds.setdefault(name, [])
        if len(bucket) < self.limit:
            bucket.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        return self._run(name, **feeds)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def fidelity(ref, var, feeds_by_graph) -> dict:
    """Per graph, on the recorded inputs: logits -> mean KL(ref || variant) of the softmax and top-1
    agreement; other outputs -> relative L2 error."""
    out = {}
    for name, feeds in feeds_by_graph.items():
        kls, top1, rel = [], [], []
        names = [o.name for o in ref.s[name].get_outputs()]
        for f in feeds:
            a, b = ref.s[name].run(None, f), var.s[name].run(None, f)
            for oname, x, y in zip(names, a, b):
                x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
                if oname.endswith("_logits"):
                    lx, ly = log_softmax(x), log_softmax(y)
                    kls.append(float((np.exp(lx) * (lx - ly)).sum(-1).mean()))
                    top1.append(float((x.argmax(-1) == y.argmax(-1)).mean()))
                elif x.size and np.isfinite(x).all():
                    rel.append(float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-12)))
        out[name] = {k: round(float(np.mean(v)), 6) for k, v in
                     (("kl", kls), ("top1_agree", top1), ("rel_l2", rel)) if v}
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True, help="name=onnx_dir:CPU|CUDA (first = reference)")
    ap.add_argument("--text-encoder", type=Path, required=True)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-records", type=int, default=1024)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--capture", type=int, default=40,
                    help="per graph: replay this many real input sets from the reference run through every variant")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    caps = {}
    for line in args.captions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            caps[int(r["id"])] = r["long"]
    val = [int(r["id"]) for r in map(json.loads, args.split_manifest.read_text().splitlines())
           if r["split"] == "val" and int(r["id"]) in caps][: args.n]
    prompts = [caps[i] for i in val]

    variants = []
    for v in args.variant:
        name, rest = v.split("=", 1)
        d, prov = rest.rsplit(":", 1)
        variants.append((name, Path(d), prov))
    first = OrtPainter(variants[0][1], PROVIDERS["CPU"])
    codec = Codec.load(proto=first.proto, device="cpu")
    enc = TextEncoder(args.text_encoder, max_positions=first.proto.max_text_positions)
    text, tv = enc(prompts)
    del first
    sc = SampleConfig(temperature=1.0, top_p=0.9, cfg_scale=args.cfg_scale, max_records=args.max_records,
                      require_before_eos=(0, 2))
    g = torch.Generator().manual_seed(0)
    fields = None
    results, ref_seqs, ref_imgs, ref_painter, recorder = {}, None, None, None, None
    clip = ClipScorer(args.text_encoder, "cpu")
    tfeat = clip.texts(prompts)
    for name, d, prov in variants:
        m = OrtPainter(d, PROVIDERS[prov])
        if ref_painter is None:
            ref_painter, recorder = m, Recorder(m, args.capture)
            m.run = recorder.run
        if fields is None:
            fields = ["event"] + list(m.attr_names)
            table = torch.rand(args.max_records, len(fields), len(prompts), generator=g)

        def uniforms(step, field):
            return table[step, fields.index(field)]

        t0 = time.perf_counter()
        seqs = ort_generate(m, codec, len(prompts), sc, text.numpy(), tv.numpy(), uniforms)
        dt = time.perf_counter() - t0
        n_rec = sum(len(s) for s in seqs)
        imgs = [render(s, codec) for s in seqs]
        ifeat = clip.images(imgs)
        r = {"provider": prov, "onnx": str(d), "records_per_s": round(n_rec / dt, 1), "seconds": round(dt, 1),
             "mean_length": round(n_rec / len(seqs), 1),
             "early_eos": int(sum(2 not in paint_order(s)[1].tolist() for s in seqs)),
             "clip_score": round(100 * float((ifeat * tfeat).sum(1).mean()), 3)}
        if ref_seqs is None:
            ref_seqs, ref_imgs = seqs, imgs
        else:
            agree, total, first_div = 0, 0, []
            for a, b in zip(seqs, ref_seqs):
                n = min(len(a), len(b))
                same = (a.event[:n] == b.event[:n]) & (a.attrs[:n] == b.attrs[:n]).all(1)
                agree += int(same.sum())
                total += max(len(a), len(b))
                bad = torch.nonzero(~same)
                first_div.append(int(bad[0]) if len(bad) else (n if len(a) != len(b) else -1))
            p = [psnr(a, b) for a, b in zip(imgs, ref_imgs)]
            r["one_step_fidelity"] = fidelity(ref_painter, m, recorder.feeds)
            r |= {"token_agreement": round(agree / total, 4), "identical_sequences": sum(f == -1 for f in first_div),
                  "median_first_divergence": int(np.median([f for f in first_div if f >= 0])) if any(f >= 0 for f in first_div) else None,
                  "psnr_vs_reference_mean": round(float(np.mean([x for x in p if x < 99])) if any(x < 99 for x in p) else 99.0, 2)}
        results[name] = r
        print(name, json.dumps(r, ensure_ascii=False), flush=True)
        strip = np.concatenate(imgs[:8], 1)
        from PIL import Image
        Image.fromarray(strip).save(args.out / f"{name}.jpg", quality=90)
    (args.out / "results.json").write_text(json.dumps({"n": len(prompts), "cfg_scale": args.cfg_scale,
                                                       "variants": results}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
