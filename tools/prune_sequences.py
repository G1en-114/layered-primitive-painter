"""Prune nearly invisible primitives and remeasure PSNR against the original RGB.

    python tools/prune_sequences.py --src data/seq/v2 --out data/seq/v2_clean/t0 \
        --data data/celebamask256 --min-visible 2

Preserves raw inputs, IDs, order, layer and round labels. Chunked visibility bounds
GPU memory. A candidate losing more than --max-psnr-loss dB keeps its original.
Output psnr is measured after pruning; source metrics are retained separately.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.protocol import PROTOCOL_PATH, load_protocol  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import flatten_on_base, pixel_alpha  # noqa: E402


@torch.no_grad()
def visible_masses(prims, proto, chunk=64):
    """Exact suffix visibility, using at most chunk opacity maps at once."""
    front = prims.alpha.new_ones(prims.alpha.shape[0], 1, proto.canvas_size, proto.canvas_size)
    masses = torch.empty_like(prims.alpha)
    for end in range(prims.num, 0, -chunk):
        start = max(0, end - chunk)
        alpha = pixel_alpha(prims[start:end], proto.canvas_size, proto)
        logs = torch.log1p(-alpha)
        total = logs.sum(1, keepdim=True)
        after = total - logs.cumsum(1)
        masses[:, start:end] = (alpha * after.exp() * front).sum((2, 3))
        front *= total.exp()
    return masses


def select_rows(mass, threshold):
    if mass.numel() == 0 or not torch.isfinite(mass).all():
        raise ValueError("empty or non-finite primitive visibility")
    rows = torch.nonzero(mass >= threshold).flatten()
    return rows if rows.numel() else mass.argmax().view(1)


@torch.no_grad()
def reconstruction_psnr(prims, target, proto):
    image = flatten_on_base(render_group(prims, proto.canvas_size, proto), proto)
    return float(-10 * (image - target).square().mean().clamp(min=1e-10).log10())


def write_config(path, config):
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("output config differs; use a new output directory")
        return
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(config, indent=1))
    os.replace(temporary, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="prepared RGB shards; defaults to source config")
    ap.add_argument("--min-visible", type=float, default=2.0, help="pixels at the protocol canvas size")
    ap.add_argument("--max-psnr-loss", type=float, default=0.1, help="retain original above this dB loss")
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-guard", action="store_true",
                    help="no RGB targets available (e.g. sequences moved to a machine without the images): drop "
                         "invisible primitives without the PSNR loss guard; psnr fields keep the source values")
    args = ap.parse_args()
    if not np.isfinite([args.min_visible, args.max_psnr_loss]).all() or min(args.min_visible, args.max_psnr_loss) < 0:
        ap.error("visibility and PSNR loss limits must be finite and nonnegative")
    src, dst = args.src.resolve(), args.out.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        ap.error("source and output directories must be separate, without nesting")
    proto_file = args.protocol or PROTOCOL_PATH
    proto = load_protocol(proto_file)
    src_cfg = json.loads((args.src / "config.json").read_text())
    if (src_cfg.get("protocol_version", 1) != proto.version or
            src_cfg.get("protocol_sha256") != hashlib.sha256(proto_file.read_bytes()).hexdigest()):
        raise ValueError("source protocol version/hash does not match the selected protocol")
    if not args.no_guard:
        data_root = args.data or Path(src_cfg["data"])
        if data_root.resolve() != Path(src_cfg["data"]).resolve():
            raise ValueError("RGB targets must match the decomposition source data")
        index_file = data_root / "index.json"
        index = {s["shard"]: s for s in json.loads(index_file.read_text())["shards"]}
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".prune.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.no_guard:
        cfg = dict(src_cfg, pruned_from=str(args.src), min_visible_px=args.min_visible, pruning_version=2,
                   psnr_guard=False, psnr_metric="source_psnr_unguarded")
    else:
        cfg = dict(src_cfg, pruned_from=str(args.src), min_visible_px=args.min_visible,
                   pruning_version=2, max_psnr_loss_db=args.max_psnr_loss,
                   psnr_metric="post_prune_rgb_mse", target_data=str(data_root),
                   target_index_sha256=hashlib.sha256(index_file.read_bytes()).hexdigest())
    write_config(args.out / "config.json", cfg)
    codec = Codec.load(proto=proto, device=args.device)

    for f in sorted(args.src.glob("decomp_[0-9][0-9][0-9].npz")):
        destination = args.out / f.name
        with f.open("rb") as source:
            source_sha = hashlib.file_digest(source, "sha256").hexdigest()
        if destination.exists():
            with np.load(destination) as z:
                if str(z["source_sha256"]) != source_sha or not np.isfinite(z["psnr"]).all():
                    raise ValueError(f"{destination}: source changed or output metric invalid")
            continue
        started = time.perf_counter()
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        shard = int(f.stem.split("_")[-1])
        if not args.no_guard:
            image_rows = {image_id: row for row, image_id in enumerate(index[shard]["ids"])}
            images = np.load(data_root / f"images_{shard:03d}.npy", mmap_mode="r")
        off = data["offsets"]
        keep_rows, psnrs, original_psnrs, retained = [], [], [], 0
        for i, image_id in enumerate(data["ids"]):
            tok = torch.from_numpy(data["tokens"][off[i]:off[i + 1]].astype(np.int64)).to(args.device)

            def decode(tokens):
                return codec.decode({n: tokens[:, k].unsqueeze(0) for k, n in enumerate(proto.attr_names)})

            if args.no_guard:
                with torch.no_grad():
                    rows = select_rows(visible_masses(decode(tok), proto)[0], args.min_visible)
                keep_rows.append(int(off[i]) + rows.cpu().numpy())
                psnrs.append(float(data["psnr"][i]))
                original_psnrs.append(float(data["psnr"][i]))
                continue
            target = torch.from_numpy(np.array(images[image_rows[int(image_id)]], copy=True)).to(args.device)
            target = target.permute(2, 0, 1).unsqueeze(0).float().div_(255)
            if src_cfg.get("flip", False):
                target = target.flip(-1)
            with torch.no_grad():
                prims = decode(tok)
                rows = select_rows(visible_masses(prims, proto)[0], args.min_visible)
                before = reconstruction_psnr(prims, target, proto)
                after = reconstruction_psnr(decode(tok[rows]), target, proto)
                if before - after > args.max_psnr_loss:
                    rows = torch.arange(len(tok), device=tok.device)
                    after = before
                    retained += 1
            if not np.isfinite([before, after]).all():
                raise ValueError(f"{f}: non-finite reconstruction for image {image_id}")
            keep_rows.append(int(off[i]) + rows.cpu().numpy())
            psnrs.append(after)
            original_psnrs.append(before)
        rows = np.concatenate(keep_rows)
        new_off = np.concatenate([[0], np.cumsum([len(r) for r in keep_rows])]).astype(np.int64)
        out = dict(data, tokens=data["tokens"][rows], layer=data["layer"][rows], round=data["round"][rows],
                   offsets=new_off, source_psnr=data["psnr"], psnr=np.array(psnrs, dtype=np.float32),
                   unpruned_psnr=np.array(original_psnrs, dtype=np.float32), source_sha256=np.array(source_sha))
        tmp = destination.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        os.replace(tmp, destination)
        print(f"{f.name}: kept {len(rows)}/{off[-1]} primitives ({1 - len(rows) / off[-1]:.1%} removed); "
              f"PSNR {np.mean(original_psnrs):.4f} -> {np.mean(psnrs):.4f} dB; "
              f"retained {retained} originals at loss guard; {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
