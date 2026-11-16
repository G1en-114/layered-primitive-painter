"""Where does a training step spend its time? Times each stage of tools/train.py's step.

    python bench/profile_train_step.py --data "data/seq/v2_clean/t0/decomp_00[0-3].npz" \
        --config small --batch 32 --steps 30 --text-encoder data/models/chinese-clip-vit-base-patch16 \
        --captions data/captions/captions_clean_v1.jsonl --attributes data/celebamask256/attributes.json \
        --canvas --perturb 0.3 --out bench/results/step_profile_small.json

Every stage is bracketed by torch.cuda.synchronize(), so the numbers are exclusive wall times
(the real loop overlaps the data loader with GPU work; synchronizing removes that overlap,
which is the point: it shows what each stage costs on its own). The first --warmup steps
are dropped. Reports median ms per stage and per record, plus records/s of the whole step.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from lpp.canvas import build_training_canvases  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import SequenceDataset, load_shards  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.text import CachedTextEncoder, CaptionStore, TextEncoder  # noqa: E402
from train import loader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--config", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--text-encoder", type=Path, default=None)
    ap.add_argument("--captions", type=Path, default=None)
    ap.add_argument("--attributes", type=Path, default=None)
    ap.add_argument("--canvas", action="store_true")
    ap.add_argument("--text-cache", action="store_true", help="memoize caption features as train.py does")
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--sdpa-backend", choices=["auto", "math"], default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    proto = load_protocol(args.protocol) if args.protocol else load_protocol()
    codec = Codec.load(proto=proto, device="cuda")
    examples = load_shards(sorted(Path(p) for p in glob.glob(args.data)))
    store = encoder = None
    if args.text_encoder:
        store = CaptionStore(args.captions, args.attributes)
        encoder = TextEncoder(args.text_encoder, max_positions=proto.max_text_positions).cuda()
        if args.text_cache:
            encoder = CachedTextEncoder(encoder)
    ds = SequenceDataset(examples, proto, seed=0, captions=store, p_text_drop=0.1, p_perturb=args.perturb)
    cfg = {"tiny": ModelConfig.tiny, "small": ModelConfig.small, "base": ModelConfig}[args.config]()
    if args.canvas:
        cfg.canvas = True
    model = Painter(cfg, proto, codec).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    it = loader(ds, argparse.Namespace(batch=args.batch, workers=args.workers))

    stages = ["load", "h2d", "text", "canvas", "forward", "backward", "optimizer", "loss_items"]
    times = {s: [] for s in stages}
    records = []

    def mark():
        torch.cuda.synchronize()
        return time.perf_counter()

    for step in range(args.warmup + args.steps):
        t = {"start": mark()}
        batch = next(it)
        t["load"] = mark()
        captions = batch.pop("captions", None)
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        t["h2d"] = mark()
        if encoder is not None and captions is not None:
            batch["text"], batch["text_valid"] = encoder(captions)
        t["text"] = mark()
        if cfg.canvas:
            batch["canvas"], batch["seg"] = build_training_canvases(batch, codec, cfg.canvas_k, cfg.canvas_size)
        t["canvas"] = mark()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model(batch)
        t["forward"] = mark()
        losses["loss"].backward()
        t["backward"] = mark()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        t["optimizer"] = mark()
        _ = {k: v.item() for k, v in losses.items()}
        t["loss_items"] = mark()
        if step >= args.warmup:
            prev = t["start"]
            for s in stages:
                times[s].append((t[s] - prev) * 1e3)
                prev = t[s]
            records.append(int(batch["valid"].sum()))

    med = {s: statistics.median(v) for s, v in times.items()}
    total = sum(med.values())
    rec = statistics.median(records)
    result = {"config": args.config, "params_M": round(model.num_params() / 1e6, 2), "batch": args.batch,
              "canvas": args.canvas, "text": encoder is not None, "device": torch.cuda.get_device_name(),
              "records_per_step": rec, "step_ms": round(total, 2), "records_per_s": round(rec / total * 1e3),
              "stages_ms": {s: round(v, 2) for s, v in med.items()},
              "stages_share": {s: round(v / total, 3) for s, v in med.items()},
              "us_per_record": {s: round(v * 1e3 / rec, 2) for s, v in med.items()},
              "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
