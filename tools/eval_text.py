"""Text-conditioned evaluation: CLIP score, retrieval precision, CLIP feature distance, orphans.

    python tools/eval_text.py --ckpt runs/x/last.pt --out out/eval_x \
        --text-encoder data/models/chinese-clip-vit-base-patch16 \
        --data data/celebamask256 --captions data/captions/captions_clean_v1.jsonl \
        --split-manifest data/celebamask256_split_v1.jsonl --n 192

Prompts are the VLM captions of the first n validation images (fixed order), so every model
is scored on the same prompts. Metrics, all from the frozen Chinese-CLIP used for conditioning:

- clip_score: 100 * cosine(image, its prompt), averaged. Reported for the real validation
  images too (an upper reference for this caption set).
- r_precision@1/@5: does an image retrieve its own prompt among all n prompts?
- clip_fd: Frechet distance between CLIP image embeddings of generated and real validation
  images. "real_split_fd" (two disjoint halves of the real set) is the noise floor at this n.

Also orphan rate, early stops (no skin layer) and sequence length, as in sample_eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.clip_score import ClipScorer  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import ImageTokenStore  # noqa: E402
from lpp.metrics import orphan_stats  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import paint_order  # noqa: E402
from lpp.text import TextEncoder  # noqa: E402

def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (float64, eigen sqrtm)."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # sqrt(ca) cb sqrt(ca) is symmetric PSD, so its trace-sqrt equals tr sqrtm(ca cb)
    w, v = np.linalg.eigh(ca)
    sa = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = sa @ cb @ sa
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * tr_sqrt)


def retrieval(img: torch.Tensor, txt: torch.Tensor) -> dict[str, float]:
    sim = img @ txt.T                                   # [n images, n prompts]
    rank = (sim > sim.diag().unsqueeze(1)).sum(1)       # prompts scoring above the true one
    return {"r_precision@1": round(float((rank < 1).float().mean()), 4),
            "r_precision@5": round(float((rank < 5).float().mean()), 4)}


def load_val(args) -> tuple[list[int], list[str], list[np.ndarray]]:
    """First n validation ids (manifest order) that have a caption, with their RGB images."""
    caps = {}
    for line in args.captions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("long"):
                caps[int(r["id"])] = r["long"]
    val = [int(r["id"]) for r in map(json.loads, args.split_manifest.read_text().splitlines())
           if r["split"] == "val" and int(r["id"]) in caps]
    if len(val) < args.n:
        raise ValueError(f"only {len(val)} captioned validation images, asked for {args.n}")
    index = json.loads((args.data / "index.json").read_text())
    where = {image_id: (s["shard"], row) for s in index["shards"] for row, image_id in enumerate(s["ids"])}
    ids = val[:args.n]
    images, cache = [], {}
    for image_id in ids:
        shard, row = where[image_id]
        if shard not in cache:
            cache[shard] = np.load(args.data / f"images_{shard:03d}.npy", mmap_mode="r")
        images.append(np.array(cache[shard][row]))
    return ids, [caps[i] for i in ids], images


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-encoder", type=Path, required=True, help="Chinese-CLIP model directory")
    ap.add_argument("--data", type=Path, required=True, help="prepared RGB shards (index.json, images_*.npy)")
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-grid", type=int, default=16, help="samples (with their real images) in the grid")
    ap.add_argument("--judge-encoder", type=Path, default=None,
                    help="second, independent Chinese-CLIP (e.g. ViT-L/14) to re-score: guards against reward hacking "
                         "when the RL reward uses the conditioning encoder")
    ap.add_argument("--image-tokens", type=Path, default=None,
                    help="image-conditioned models: condition each sample on its validation image's CLIP tokens")
    ap.add_argument("--no-text", action="store_true", help="with --image-tokens: image only, no prompt")
    ap.add_argument("--sdpa-backend", choices=["auto", "math"], default="auto",
                    help="math avoids unavailable FlashAttention libraries on some DTK builds")
    args = ap.parse_args()
    if args.n < 10:
        ap.error("--n must be at least 10 for retrieval and distance metrics")
    if args.sdpa_backend == "math":
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
    encoder = TextEncoder(args.text_encoder).to(dev)
    ids, prompts, real = load_val(args)
    store = None
    if args.image_tokens is not None:
        if not model.cfg.image:
            raise SystemExit("--image-tokens needs an image-conditioned checkpoint")
        store = ImageTokenStore(args.image_tokens)
    elif args.no_text:
        raise SystemExit("--no-text only makes sense with --image-tokens")
    print(f"checkpoint step {state['step']}, {model.num_params() / 1e6:.1f}M; {len(ids)} prompts", flush=True)

    sc = SampleConfig(temperature=1.0, top_p=args.top_p, cfg_scale=args.cfg_scale)
    gen, orphan, lengths, early = [], [], [], 0
    cache = args.out / "generated.npz"
    if cache.exists():               # scoring can be rerun without regenerating
        z = np.load(cache)
        if list(z["ids"]) != ids:
            raise ValueError(f"{cache} was generated for other prompts; delete it to regenerate")
        gen, orphan, lengths = list(z["images"]), list(z["orphan"]), list(z["lengths"])
        early, gen_seconds = int(z["early"]), float(z["seconds"])
        print(f"reusing {len(gen)} generated images from {cache}", flush=True)
    started = time.perf_counter()
    for b0 in range(len(gen), args.n, args.batch):
        batch_prompts = prompts[b0:b0 + args.batch]
        text, text_valid = (None, None) if args.no_text else encoder(batch_prompts)
        image = image_valid = None
        if store is not None:
            image = torch.from_numpy(np.stack([store.get(i, False) for i in ids[b0:b0 + args.batch]])).float().to(dev)
            image_valid = torch.ones(image.shape[:2], dtype=torch.bool, device=dev)
        g = torch.Generator(device=dev).manual_seed(args.seed * 100003 + b0)
        for s in generate(model, batch=len(batch_prompts), cfg=sc, text=text, text_valid=text_valid,
                          generator=g, codec=codec, image=image, image_valid=image_valid):
            attrs, lay = paint_order(s)
            lengths.append(len(s))
            early += int(2 not in lay.tolist())
            if attrs.shape[0] == 0:
                gen.append(np.full((256, 256, 3), 255, np.uint8))
                continue
            prims = codec.decode({k: attrs[:, i].unsqueeze(0).to(dev) for i, k in enumerate(proto.attr_names)})
            orphan.append(orphan_stats(prims, lay.to(dev), 128, proto=proto)["orphan"])
            img = flatten_on_base(render_group(prims, 256, proto), proto)[0]
            gen.append((img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        print(f"  generated {len(gen)}/{args.n}  {len(gen) / (time.perf_counter() - started):.2f} img/s",
              flush=True)
    if not cache.exists():
        gen_seconds = time.perf_counter() - started
        np.savez(cache, ids=np.array(ids), images=np.stack(gen), orphan=np.array(orphan),
                 lengths=np.array(lengths), early=early, seconds=gen_seconds)
    del model
    torch.cuda.empty_cache()

    clip = ClipScorer(args.text_encoder, dev)
    t, gi, ri = clip.texts(prompts), clip.images(gen), clip.images(real)
    g_np, r_np = gi.cpu().double().numpy(), ri.cpu().double().numpy()
    half = len(r_np) // 2
    result = {
        "step": state["step"], "checkpoint": str(args.ckpt), "n": args.n, "cfg_scale": args.cfg_scale,
        "top_p": args.top_p, "seed": args.seed,
        "generated": {"clip_score": round(100 * float((gi * t).sum(1).mean()), 3), **retrieval(gi, t),
                      "clip_fd": round(frechet(g_np, r_np), 4),
                      "orphan_rate": round(float(np.mean(orphan)), 4) if orphan else None,
                      "early_stop": round(early / args.n, 4), "mean_length": round(float(np.mean(lengths)), 1),
                      "images_per_s": round(args.n / gen_seconds, 3)},
        "condition": ("image" if args.no_text else "image+text") if store is not None else "text",
        "real_reference": {"clip_score": round(100 * float((ri * t).sum(1).mean()), 3), **retrieval(ri, t),
                           "real_split_fd": round(frechet(r_np[:half], r_np[half:]), 4)},
    }
    if args.judge_encoder is not None:
        del clip
        torch.cuda.empty_cache()
        judge = ClipScorer(args.judge_encoder, dev)
        jt, jg, jr = judge.texts(prompts), judge.images(gen), judge.images(real)
        jg_np, jr_np = jg.cpu().double().numpy(), jr.cpu().double().numpy()
        result["judge"] = {"encoder": str(args.judge_encoder),
                           "generated": {"clip_score": round(100 * float((jg * jt).sum(1).mean()), 3),
                                         **retrieval(jg, jt), "clip_fd": round(frechet(jg_np, jr_np), 4)},
                           "real_reference": {"clip_score": round(100 * float((jr * jt).sum(1).mean()), 3),
                                              **retrieval(jr, jt),
                                              "real_split_fd": round(frechet(jr_np[:half], jr_np[half:]), 4)}}
    if store is not None:      # image -> painting: how faithful is the result to its input image
        g_img, r_img = np.stack(gen).astype(np.float64), np.stack(real).astype(np.float64)
        mse = ((g_img - r_img) ** 2).mean((1, 2, 3))
        result["generated"]["input_psnr"] = round(float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-9)))), 3)
        result["generated"]["clip_image_to_input"] = round(float((gi * ri).sum(1).mean()), 4)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (args.out / "prompts.json").write_text(json.dumps(dict(zip(map(str, ids), prompts)), indent=1,
                                                      ensure_ascii=False))
    k = min(args.save_grid, args.n)
    if k:
        grid = np.concatenate([np.concatenate(gen[:k], 1), np.concatenate(real[:k], 1)], 0)
        Image.fromarray(grid).save(args.out / "grid_generated_over_real.png")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
