"""Train the painter on decomposition shards.

    python tools/train.py --data "data/seq/v*/decomp_*.npz" --config small --out work/run1
    python tools/train.py ... --resume            # continue from work/run1/last.pt

Logs JSON lines to <out>/log.jsonl; saves last.pt every --save-every steps and renders
a few unconditional samples at every evaluation (<out>/samples_<step>.png).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.canvas import build_training_canvases  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import ImageTokenStore, SequenceDataset, collate, load_shards  # noqa: E402
from lpp.data.splits import load_split_manifest  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.metrics import orphan_stats  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.text import CachedTextEncoder, CaptionStore, TextEncoder  # noqa: E402
from lpp.runtime_env import available_cpus, configure_threads, default_workers  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import paint_order  # noqa: E402


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def to_gpu(batch: dict, encoder, codec=None, model_cfg=None) -> dict:
    """Tensors to the GPU; caption strings through the frozen text encoder; checkpoint
    canvases rendered for canvas-feedback models."""
    captions = batch.pop("captions", None)
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    if captions is not None and encoder is not None:
        batch["text"], batch["text_valid"] = encoder(captions)
    if model_cfg is not None and model_cfg.canvas:
        batch["canvas"], batch["seg"] = build_training_canvases(batch, codec, model_cfg.canvas_k,
                                                                model_cfg.canvas_size)
    return batch


SAMPLE_PROMPTS = [
    "金发碧眼的女性，微笑，浅色背景",
    "黑色短发的男性，戴眼镜，深色背景",
    "棕色长卷发的女性，涂红色口红，戴耳环",
    "灰白头发的男性，留着胡子，表情严肃",
    "黑色长直发的女性，有刘海，白色上衣，绿色背景",
    "光头男性，微笑，穿深色西装",
]


def loader(ds, args, shuffle=True):
    """Endless batches. Training drops the ragged last batch; a dataset smaller than one
    batch (a tiny validation split) keeps it, otherwise the epoch would be empty forever."""
    if len(ds) == 0:
        raise ValueError("empty dataset")
    drop_last = shuffle and len(ds) >= args.batch
    dl = DataLoader(ds, batch_size=args.batch, shuffle=shuffle, collate_fn=collate,
                    num_workers=args.workers, drop_last=drop_last, persistent_workers=args.workers > 0)
    while True:
        yield from dl


@torch.no_grad()
def render_samples(model, codec, n, path, seed, encoder=None, cfg_scale=2.0):
    """n samples; with a text encoder, one per fixed prompt (guided), else unconditional."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    text = text_valid = None
    sc = SampleConfig(temperature=1.0, top_p=0.9)
    if encoder is not None:
        text, text_valid = encoder(SAMPLE_PROMPTS[:n])
        sc.cfg_scale = cfg_scale
    seqs = generate(model, batch=n, cfg=sc, text=text, text_valid=text_valid, generator=g, codec=codec)
    tiles, orphan = [], []
    for s in seqs:
        attrs, lay = paint_order(s)
        if attrs.shape[0] == 0:
            tiles.append(np.full((256, 256, 3), 255, np.uint8))
            continue
        prims = codec.decode({k: attrs[:, i].unsqueeze(0).cuda() for i, k in enumerate(codec.attr_names)})
        img = flatten_on_base(render_group(prims, 256))
        tiles.append((img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        orphan.append(orphan_stats(prims, lay.cuda(), 128, proto=codec.proto)["orphan"])
    Image.fromarray(np.concatenate(tiles, 1)).save(path)
    return [len(s) for s in seqs], round(float(np.mean(orphan)) if orphan else 0.0, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob of decomp_*.npz shards")
    ap.add_argument("--config", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--split-manifest", type=Path, default=None,
                    help="JSONL id/split records; overrides --val-frac")
    ap.add_argument("--protocol", type=Path, default=None, help="explicit versioned protocol/profile")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None, help="default: half the CPU quota, max 8")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--sdpa-backend", choices=["auto", "math"], default="auto",
                    help="math avoids unavailable FlashAttention libraries on some DTK builds")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--captions", type=Path, default=None, help="captions.jsonl from tools/caption_vlm.py")
    ap.add_argument("--attributes", type=Path, default=None, help="attributes.json (template captions)")
    ap.add_argument("--text-encoder", type=Path, default=None, help="Chinese-CLIP model directory")
    ap.add_argument("--text-drop", type=float, default=0.1, help="caption dropout (for CFG)")
    ap.add_argument("--no-text-cache", action="store_true",
                    help="run the frozen text encoder every step instead of memoizing caption features")
    ap.add_argument("--cfg-scale", type=float, default=2.0, help="guidance for text-conditioned samples")
    ap.add_argument("--canvas", action="store_true", help="canvas feedback (docs/canvas_feedback_design.md)")
    ap.add_argument("--canvas-k", type=int, default=32)
    ap.add_argument("--canvas-size", type=int, default=64)
    ap.add_argument("--perturb", type=float, default=0.0, help="share of training sequences with a corrupted history")
    ap.add_argument("--image-tokens", type=Path, default=None,
                    help="CLIP image tokens (tools/extract_image_features.py): enables image conditioning")
    ap.add_argument("--p-image", type=float, default=0.5, help="share of sequences conditioned on their image")
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="start a NEW run from these weights (fresh optimizer and step; missing modules such "
                         "as canvas layers start at their init). Not the same as --resume.")
    args = ap.parse_args()
    if args.sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.workers is None:
        args.workers = default_workers()
    threads = configure_threads(reserve_for_workers=args.workers)
    print(f"CPU budget {available_cpus()} -> {threads} torch threads, {args.workers} data workers")
    torch.manual_seed(args.seed)

    proto = load_protocol(args.protocol) if args.protocol else load_protocol()
    codec = Codec.load(proto=proto, device="cuda")
    files = sorted(Path(p) for p in glob.glob(args.data))
    if not files:
        raise SystemExit(f"no shards match {args.data}")
    for d in sorted({f.parent for f in files}):        # data and model must use the same protocol
        cfg_file = d / "config.json"
        data_version = json.loads(cfg_file.read_text()).get("protocol_version", 1) if cfg_file.exists() else 1
        if data_version != proto.version:
            raise SystemExit(f"{d} was made with protocol v{data_version}, training uses v{proto.version}; "
                             f"pass --protocol for that version")
    examples = load_shards(files)
    is_val = lambda e: (e.image_id * 2654435761 % 2**32) / 2**32 < args.val_frac  # noqa: E731  same split for flips
    if args.split_manifest:
        splits = load_split_manifest(args.split_manifest)
        missing = {e.image_id for e in examples} - splits.keys()
        if missing:
            raise ValueError(f"{len(missing)} image IDs missing from the split manifest")
        is_val = lambda e: splits[e.image_id] == "val"  # noqa: E731
    store = encoder = sample_encoder = None
    if args.text_encoder:
        store = CaptionStore(args.captions, args.attributes)
        encoder = TextEncoder(args.text_encoder, max_positions=proto.max_text_positions).cuda()
        sample_encoder = encoder
        if not args.no_text_cache:
            encoder = CachedTextEncoder(encoder)
        print(f"text conditioning: {len(store.vlm)} VLM-captioned, {len(store.attrs)} with attributes, "
              f"caption dropout {args.text_drop}")
    image_tokens = ImageTokenStore(args.image_tokens) if args.image_tokens else None
    if image_tokens is not None:
        missing = {e.image_id for e in examples} - set(image_tokens.row)
        if missing:
            raise SystemExit(f"{len(missing)} image IDs have no image tokens in {args.image_tokens}")
        print(f"image conditioning: {len(image_tokens.row)} images, p_image {args.p_image}")
    train_ds = SequenceDataset([e for e in examples if not is_val(e)], proto, seed=args.seed, captions=store,
                               p_text_drop=args.text_drop, p_perturb=args.perturb, image_tokens=image_tokens,
                               p_image=args.p_image)
    val_ds = SequenceDataset([e for e in examples if is_val(e)], proto, seed=1234, p_redraw=0, p_refine=0,
                             captions=store, p_text_drop=0.0, image_tokens=image_tokens, p_image=args.p_image)
    print(f"{len(files)} shards, {len(train_ds)} train / {len(val_ds)} val sequences")

    cfg = {"tiny": ModelConfig.tiny, "small": ModelConfig.small, "base": ModelConfig}[args.config]()
    if args.canvas:
        cfg.canvas, cfg.canvas_k, cfg.canvas_size = True, args.canvas_k, args.canvas_size
    if image_tokens is not None:
        n_img = json.loads((args.image_tokens / "meta.json").read_text())["tokens"]
        # the image prefix comes on top of the protocol's worst case (rotary tables are not saved,
        # so a larger context still loads checkpoints trained without images)
        cfg.image, cfg.max_positions = True, max(cfg.max_positions, proto.worst_case_positions() + n_img)
    if encoder is not None and cfg.text_dim != encoder.dim:
        raise SystemExit(f"model text_dim {cfg.text_dim} != text encoder width {encoder.dim}")
    model = Painter(cfg, proto, codec).cuda()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    if args.init_weights and not (args.resume and (args.out / "last.pt").exists()):
        src = torch.load(args.init_weights, map_location="cuda", weights_only=False)
        missing, unexpected = model.load_state_dict(src["model"], strict=False)
        if unexpected:
            raise SystemExit(f"--init-weights has parameters this model lacks: {unexpected[:5]}")
        print(f"initialised from {args.init_weights} (step {src.get('step')}); "
              f"{len(missing)} new tensors start fresh: {sorted({k.split('.')[0] for k in missing})}")
    step = 0
    ckpt = args.out / "last.pt"
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"resumed at step {step}")
    print(f"model {args.config}: {model.num_params() / 1e6:.1f}M parameters")
    fwd = torch.compile(model) if args.compile else model

    train_it, val_it = loader(train_ds, args), loader(val_ds, args, shuffle=False)
    log = open(args.out / "log.jsonl", "a")
    t0, seen = time.perf_counter(), 0
    while step < args.steps:
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        totals = {}
        for _ in range(args.accum):
            batch = to_gpu(next(train_it), encoder, codec, cfg)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = fwd(batch)
            (losses["loss"] / args.accum).backward()
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item() / args.accum
            seen += int(batch["valid"].sum())
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            dt = time.perf_counter() - t0
            gates = ([round(float(torch.tanh(c.gate.detach())), 4) for c in model.cross.values()]
                     if model.cfg.canvas else None)
            rec = {"step": step, "lr": lr_at(step, args), "gnorm": round(gnorm, 3), "canvas_gates": gates,
                   "records_per_s": round(seen / dt), **{k: round(v, 4) for k, v in totals.items()}}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0, seen = time.perf_counter(), 0

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            vals = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(10):
                    batch = to_gpu(next(val_it), encoder, codec, cfg)
                    for k, v in model(batch).items():
                        vals[k] = vals.get(k, 0.0) + v.item() / 10
            lens = render_samples(model, codec, 6, args.out / f"samples_{step:06d}.png", seed=step,
                                  encoder=sample_encoder, cfg_scale=args.cfg_scale)
            lens, orphan = lens
            rec = {"step": step, "val": {k: round(v, 4) for k, v in vals.items()}, "sample_lengths": lens,
                   "sample_orphan_rate": orphan}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        if step % args.save_every == 0 or step == args.steps:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "config": cfg.to_dict(), "protocol": proto.raw,
                        "args": vars(args) | {"out": str(args.out)}}, tmp)
            tmp.replace(ckpt)


if __name__ == "__main__":
    main()
