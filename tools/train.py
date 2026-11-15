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

from lpp.codec import ATTR_ORDER, Codec  # noqa: E402
from lpp.data.sequences import SequenceDataset, collate, load_shards  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
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
def render_samples(model, codec, n, path, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seqs = generate(model, batch=n, cfg=SampleConfig(temperature=1.0, top_p=0.9), generator=g)
    tiles = []
    for s in seqs:
        attrs, lay = paint_order(s)
        if attrs.shape[0] == 0:
            tiles.append(np.full((256, 256, 3), 255, np.uint8))
            continue
        prims = codec.decode({k: attrs[:, i].unsqueeze(0).cuda() for i, k in enumerate(ATTR_ORDER)})
        img = flatten_on_base(render_group(prims, 256))
        tiles.append((img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(tiles, 1)).save(path)
    return [len(s) for s in seqs]


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
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None, help="default: half the CPU quota, max 8")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.workers is None:
        args.workers = default_workers()
    threads = configure_threads(reserve_for_workers=args.workers)
    print(f"CPU budget {available_cpus()} -> {threads} torch threads, {args.workers} data workers")
    torch.manual_seed(args.seed)

    proto, codec = load_protocol(), Codec.load(device="cuda")
    files = sorted(Path(p) for p in glob.glob(args.data))
    if not files:
        raise SystemExit(f"no shards match {args.data}")
    examples = load_shards(files)
    is_val = lambda e: (e.image_id * 2654435761 % 2**32) / 2**32 < args.val_frac  # noqa: E731  same split for flips
    train_ds = SequenceDataset([e for e in examples if not is_val(e)], proto, seed=args.seed)
    val_ds = SequenceDataset([e for e in examples if is_val(e)], proto, seed=1234, p_redraw=0, p_refine=0)
    print(f"{len(files)} shards, {len(train_ds)} train / {len(val_ds)} val sequences")

    cfg = {"tiny": ModelConfig.tiny, "small": ModelConfig.small, "base": ModelConfig}[args.config]()
    model = Painter(cfg, proto, codec).cuda()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
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
            batch = {k: v.cuda(non_blocking=True) for k, v in next(train_it).items()}
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
            rec = {"step": step, "lr": lr_at(step, args), "gnorm": round(gnorm, 3),
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
                    batch = {k: v.cuda() for k, v in next(val_it).items()}
                    for k, v in model(batch).items():
                        vals[k] = vals.get(k, 0.0) + v.item() / 10
            lens = render_samples(model, codec, 6, args.out / f"samples_{step:06d}.png", seed=step)
            rec = {"step": step, "val": {k: round(v, 4) for k, v in vals.items()}, "sample_lengths": lens}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        if step % args.save_every == 0 or step == args.steps:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "config": cfg.to_dict(), "args": vars(args) | {"out": str(args.out)}}, tmp)
            tmp.replace(ckpt)


if __name__ == "__main__":
    main()
