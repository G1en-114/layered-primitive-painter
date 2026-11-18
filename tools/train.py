"""Train the painter on decomposition shards.

    python tools/train.py --data "data/seq/v*/decomp_*.npz" --config small --out work/run1
    python tools/train.py ... --resume            # continue from work/run1/last.pt

Logs JSON lines to <out>/log.jsonl; saves last.pt every --save-every steps and renders
a few unconditional samples at every evaluation (<out>/samples_<step>.png).
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import shutil
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
from lpp.data.sequences import ImageTokenStore, SequenceDataset, TargetImageStore, collate, load_shards  # noqa: E402
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


def _log_line(rec: dict, log) -> None:
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    log.flush()


def save_checkpoint(out: Path, state: dict, log) -> bool:
    """Write last.pt without ever destroying the only good checkpoint: write a temp file, verify its
    size, then rotate last.pt -> prev.pt and move the temp file in. A full disk (2026-11-16: a core dump
    filled the quota and a truncated last.pt made every resume fail) skips this save with a log line
    instead of killing the run."""
    ckpt, prev, tmp = out / "last.pt", out / "prev.pt", out / "last.tmp"
    need = 2 * (ckpt.stat().st_size if ckpt.exists() else 0)
    free = shutil.disk_usage(out).free
    if need and free < need:
        msg = {"step": state["step"], "checkpoint": "skipped", "free_bytes": free, "needed_bytes": need}
        _log_line(msg, log)
        return False
    try:
        torch.save(state, tmp)
        if tmp.stat().st_size < 1024:
            raise OSError(f"checkpoint {tmp} is only {tmp.stat().st_size} bytes")
    except (OSError, RuntimeError) as e:
        tmp.unlink(missing_ok=True)
        msg = {"step": state["step"], "checkpoint": "failed", "error": str(e)[:200]}
        _log_line(msg, log)
        return False
    if ckpt.exists():
        ckpt.replace(prev)
    tmp.replace(ckpt)
    return True


def load_checkpoint(out: Path) -> dict | None:
    """last.pt, or prev.pt when last.pt is missing or unreadable (truncated by a failed save)."""
    for name in ("last.pt", "prev.pt"):
        f = out / name
        if not f.exists():
            continue
        try:
            return torch.load(f, map_location="cuda", weights_only=False)
        except (EOFError, RuntimeError, OSError) as e:
            print(f"{f} unreadable ({type(e).__name__}: {e}); trying the previous checkpoint", flush=True)
    return None


def loader(ds, args, shuffle=True, rank: int = 0, world: int = 1):
    """Endless batches. Training drops the ragged last batch; a dataset smaller than one
    batch (a tiny validation split) keeps it, otherwise the epoch would be empty forever.
    With world > 1 each rank reads a disjoint shard of every epoch (DistributedSampler,
    reshuffled per epoch with the same seed on all ranks)."""
    if len(ds) == 0:
        raise ValueError("empty dataset")
    drop_last = shuffle and len(ds) >= args.batch * world
    sampler = None
    if world > 1 and shuffle:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    # fork on Linux: Python 3.14 defaults to forkserver, which pickles the whole dataset into /dev/shm
    # (quota exceeded on the laptop); fork shares it copy-on-write, as Python <= 3.13 did
    ctx = "fork" if args.workers > 0 and sys.platform == "linux" else None
    dl = DataLoader(ds, batch_size=args.batch, shuffle=shuffle and sampler is None, sampler=sampler, collate_fn=collate,
                    num_workers=args.workers, drop_last=drop_last, persistent_workers=args.workers > 0,
                    multiprocessing_context=ctx)
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from dl
        epoch += 1


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
    ap.add_argument("--sdpa-backend", choices=["auto", "math"], default=None,
                    help="default: math on ROCm/HIP (no FlashAttention libraries in the DTK builds), auto elsewhere")
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
    ap.add_argument("--order", choices=["layer", "draft"], default="layer",
                    help="painting order of full sequences: layer by layer, or draft (all layers' coarse rounds first)")
    ap.add_argument("--draft-rounds", type=int, default=2, help="--order draft: rounds per layer in the sketch")
    ap.add_argument("--line-heads", action="store_true", help="protocol v3: separate attribute heads for LINE strokes")
    ap.add_argument("--rel-pos", action="store_true",
                    help="relative position heads for x/y (offset from the previous primitive, model.py)")
    ap.add_argument("--canvas-k", type=int, default=32)
    ap.add_argument("--canvas-size", type=int, default=64)
    ap.add_argument("--perturb", type=float, default=0.0, help="share of training sequences with a corrupted history")
    ap.add_argument("--image-tokens", type=Path, default=None,
                    help="CLIP image tokens (tools/extract_image_features.py): enables image conditioning")
    ap.add_argument("--p-image", type=float, default=0.5, help="share of sequences conditioned on their image")
    ap.add_argument("--target-images", type=Path, default=None,
                    help="pixel target store (tools/build_target_store.py): enables target-image cross-attention")
    ap.add_argument("--p-target", type=float, default=0.5, help="share of sequences given their pixel target")
    ap.add_argument("--target-gate-init", type=float, default=0.5,
                    help="raw value of the target cross-attention gates at init (0 = exact no-op, but it stalls)")
    ap.add_argument("--new-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for tensors missing from --init-weights (new modules)")
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="start a NEW run from these weights (fresh optimizer and step; missing modules such "
                         "as canvas layers start at their init). Not the same as --resume.")
    args = ap.parse_args()
    # data parallelism: launched by torchrun (RANK / LOCAL_RANK / WORLD_SIZE set) -> one process per GPU,
    # gradients all-reduced by DistributedDataParallel (NCCL on NVIDIA, RCCL under the same name on ROCm/DTK)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world > 1
    if ddp:
        dev = local_rank % torch.cuda.device_count()     # several ranks may share a GPU in smoke tests
        torch.cuda.set_device(dev)
        # LPP_DDP_BACKEND=gloo lets several ranks share one GPU (smoke tests on a single-GPU machine)
        torch.distributed.init_process_group(backend=os.environ.get("LPP_DDP_BACKEND", "nccl"))
    is_main = rank == 0
    if not is_main:                                   # one copy of the console output
        sys.stdout = open(os.devnull, "w")
    if args.sdpa_backend is None:
        args.sdpa_backend = "math" if torch.version.hip else "auto"
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
    targets = TargetImageStore(args.target_images) if args.target_images else None
    if targets is not None:
        missing = {e.image_id for e in examples} - set(targets.row)
        if missing:
            raise SystemExit(f"{len(missing)} image IDs have no target image in {args.target_images}")
        print(f"target-image conditioning: {len(targets.row)} images, p_target {args.p_target}")
    train_ds = SequenceDataset([e for e in examples if not is_val(e)], proto, seed=args.seed + 1000 * rank, captions=store,
                               p_text_drop=args.text_drop, p_perturb=args.perturb, image_tokens=image_tokens,
                               p_image=args.p_image, targets=targets, p_target=args.p_target,
                               order=args.order, draft_rounds=args.draft_rounds)
    val_ds = SequenceDataset([e for e in examples if is_val(e)], proto, seed=1234, p_redraw=0, p_refine=0,
                             captions=store, p_text_drop=0.0, image_tokens=image_tokens, p_image=args.p_image,
                             targets=targets, p_target=args.p_target, order=args.order,
                             draft_rounds=args.draft_rounds)
    print(f"{len(files)} shards, {len(train_ds)} train / {len(val_ds)} val sequences")

    cfg = {"tiny": ModelConfig.tiny, "small": ModelConfig.small, "base": ModelConfig}[args.config]()
    if args.canvas:
        cfg.canvas, cfg.canvas_k, cfg.canvas_size = True, args.canvas_k, args.canvas_size
    cfg.order, cfg.draft_rounds = args.order, args.draft_rounds
    cfg.rel_pos = args.rel_pos
    cfg.line_heads = args.line_heads
    if image_tokens is not None:
        n_img = json.loads((args.image_tokens / "meta.json").read_text())["tokens"]
        # the image prefix comes on top of the protocol's worst case (rotary tables are not saved,
        # so a larger context still loads checkpoints trained without images)
        cfg.image, cfg.max_positions = True, max(cfg.max_positions, proto.worst_case_positions() + n_img)
    if targets is not None:
        cfg.target, cfg.target_gate_init = True, args.target_gate_init
        cfg.target_size = int(json.loads((args.target_images / "meta.json").read_text())["size"])
    if encoder is not None and cfg.text_dim != encoder.dim:
        raise SystemExit(f"model text_dim {cfg.text_dim} != text encoder width {encoder.dim}")
    model = Painter(cfg, proto, codec).cuda()
    step = 0
    state = load_checkpoint(args.out) if args.resume else None
    new_params: set[str] = set(state.get("new_params", [])) if state else set()
    if args.init_weights and state is None:
        src = torch.load(args.init_weights, map_location="cuda", weights_only=False)
        missing, unexpected = model.load_state_dict(src["model"], strict=False)
        if unexpected:
            raise SystemExit(f"--init-weights has parameters this model lacks: {unexpected[:5]}")
        new_params = set(missing)
        print(f"initialised from {args.init_weights} (step {src.get('step')}); "
              f"{len(missing)} new tensors start fresh: {sorted({k.split('.')[0] for k in missing})}")
    # parameter groups: weight decay or not, and new (fresh) tensors at lr * --new-lr-mult, so modules
    # added to a trained model are not starved by the small fine-tuning learning rate
    groups = []
    for is_new in (False, True):
        for wd in (True, False):
            ps = [p for n, p in model.named_parameters()
                  if (n in new_params) == is_new and (p.dim() >= 2 and "emb" not in n) == wd]
            if ps:
                groups.append({"params": ps, "weight_decay": args.wd if wd else 0.0,
                               "lr_mult": args.new_lr_mult if is_new else 1.0})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
    if new_params:
        print(f"{len(new_params)} new tensors train at lr x {args.new_lr_mult}")
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"resumed at step {step}")
    print(f"model {args.config}: {model.num_params() / 1e6:.1f}M parameters")
    net = model
    if ddp:
        # parameters that can never receive a gradient would make DDP wait forever for their all-reduce:
        # the last attribute's embedding (nothing in the decoder comes after it) and the text projection
        # when training without text. They never change in single-GPU training either; freezing them
        # beats find_unused_parameters=True, which walks the autograd graph every step.
        dead = [f"dec_attr_emb.mods.{proto.attr_names[-1]}."] + ([] if encoder is not None else ["text_proj."])
        for n, prm in model.named_parameters():
            if any(n.startswith(d) for d in dead):
                prm.requires_grad_(False)
        # conditional modules (image prefix, target cross-attention) are skipped by batches without
        # those inputs; DDP must then tolerate parameters without gradients in a step
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dev], find_unused_parameters=bool(cfg.image or cfg.target))
        print(f"DDP: {world} processes, per-GPU batch {args.batch} x accum {args.accum} "
              f"= global batch {args.batch * args.accum * world}")
    fwd = torch.compile(net) if args.compile else net

    train_it = loader(train_ds, args, rank=rank, world=world)
    val_it = loader(val_ds, args, shuffle=False)
    log = open(args.out / "log.jsonl", "a") if is_main else open(os.devnull, "w")
    t0, seen = time.perf_counter(), 0
    while step < args.steps:
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args) * g.get("lr_mult", 1.0)
        totals = {}
        for micro in range(args.accum):
            batch = to_gpu(next(train_it), encoder, codec, cfg)
            sync = not ddp or micro == args.accum - 1      # all-reduce gradients once per optimiser step
            with (contextlib.nullcontext() if sync else net.no_sync()):
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
            if ddp:                                    # records/s of the whole job, not one rank
                t = torch.tensor([seen], device="cuda", dtype=torch.float64)
                torch.distributed.all_reduce(t)
                seen = int(t.item())
            dt = time.perf_counter() - t0
            gates = ([round(float(torch.tanh(c.gate.detach())), 4) for c in model.cross.values()]
                     if model.cfg.canvas else None)
            rec = {"step": step, "lr": lr_at(step, args), "gnorm": round(gnorm, 3), "canvas_gates": gates,
                   "records_per_s": round(seen / dt), **{k: round(v, 4) for k, v in totals.items()}}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0, seen = time.perf_counter(), 0

        if is_main and (step % args.eval_every == 0 or step == args.steps):
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

        if is_main and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(args.out, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                       "new_params": sorted(new_params), "config": cfg.to_dict(), "protocol": proto.raw,
                                       "args": vars(args) | {"out": str(args.out)}}, log)
    if ddp:
        # DDP keeps replicas identical by construction; check it (cheap: one checksum per tensor)
        sums = torch.stack([p.detach().float().sum() for p in model.parameters()])
        ref = sums.clone()
        torch.distributed.broadcast(ref, 0)
        drift = (sums - ref).abs().max().item()
        sys.__stdout__.write(f"rank {rank}: max parameter-checksum drift vs rank 0 = {drift:.3g}\n")
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
