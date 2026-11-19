"""Remove the line strokes that do not earn their place from v3 decompositions (plan v4 A1: stray lines).

    python tools/prune_lines.py --seq "data/seq/anime_v3_lines_full_t0/decomp_*.npz" \
        --images data/anime_portraits256_v2_l6view --out data/seq/anime_v3_lines_pruned_t0

The model learned stray lines (horizontal streaks under the figure, hatching over hair) from the data: the line
decomposer places lines on the residual, and some of them barely help. For every line of an image, the painting is
rendered once without that line (all lines of a layer in one batched Triton call; the layers below and above are
composited once), and its gain is measured on the pixels it touches:
    gain = (error without the line - error with it) / error without it        (squared error vs the real image)
A line is removed when gain < --min-gain (default 0.1: it fixes less than 10% of the error where it sits, or makes
it worse). Removal is decided per line on the full painting, so the joint effect is measured again at the end
(PSNR before / after, written to the shard and printed). Gaussians are kept. Output shards have the input format.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import load_shards  # noqa: E402
from lpp.protocol import PROTOCOL_V3_PATH, load_protocol  # noqa: E402
from lpp.render.bezier import Curves  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.strokes import render_strokes  # noqa: E402
from lpp.render.triton_lines import render_lines_triton  # noqa: E402


def groups(tok, layer, is_line, codec, size):
    """Per layer, in compositing order: [(kind, layer, idx, P, T)], kind 'g' (Gaussians) or 'l' (lines)."""
    proto = codec.proto
    t = lambda sel: {k: tok[sel, j].unsqueeze(0) for j, k in enumerate(proto.attr_names)}   # noqa: E731
    out = []
    for k in layer.unique().sort().values.tolist():
        g = ((layer == k) & ~is_line).nonzero()[:, 0]
        ln = ((layer == k) & is_line).nonzero()[:, 0]
        if len(g):
            P, T = render_group(codec.decode(t(g)), size, proto)
            out.append(("g", k, g, P, T))
        if len(ln):
            c = codec.decode_lines(t(ln))
            P, T = render_lines_triton(c, size, canvas=proto.canvas_size, max_alpha=proto.max_alpha)
            out.append(("l", k, ln, P, T, c))
    return out


@torch.no_grad()
def line_gains(tok, layer, is_line, real, codec, size=256):
    """gain [N_lines] for every line stroke (index into the image's strokes), see the module docstring."""
    proto = codec.proto
    gs = groups(tok, layer, is_line, codec, size)
    base = torch.tensor(proto.base_color, device=real.device).view(1, 3, 1, 1).expand(1, 3, size, size)
    below = []                                         # image under each group
    img = base
    for gr in gs:
        below.append(img)
        img = gr[3] + gr[4] * img
    full = img
    # what lies above each group, as one premultiplied group (P, T): out = P_ab + T_ab * (group output)
    above = [None] * len(gs)
    P_ab, T_ab = torch.zeros_like(full), torch.ones_like(full[:, :1])
    for i in range(len(gs) - 1, -1, -1):
        above[i] = (P_ab, T_ab)
        P, T = gs[i][3], gs[i][4]
        P_ab, T_ab = P_ab + T_ab * P, T_ab * T
    err_full = ((full - real) ** 2).sum(1)              # [1, H, W]
    gains, index = [], []
    for i, gr in enumerate(gs):
        if gr[0] != "l":
            continue
        idx, c = gr[2], gr[5]
        n = len(idx)
        for s in range(0, n, 64):                       # 64 variants per Triton call
            m = min(64, n - s)
            alpha = c.alpha.expand(m, -1).clone()
            alpha[torch.arange(m), torch.arange(s, s + m)] = 0
            cv = Curves(p0=c.p0.expand(m, -1, -1).contiguous(), p1=c.p1.expand(m, -1, -1).contiguous(), p2=c.p2.expand(m, -1, -1).contiguous(),
                        width=c.width.expand(m, -1).contiguous(), rgb=c.rgb.expand(m, -1, -1).contiguous(), alpha=alpha)
            P, T = render_lines_triton(cv, size, canvas=proto.canvas_size, max_alpha=proto.max_alpha)
            out = above[i][0] + above[i][1] * (P + T * below[i])
            err = ((out - real) ** 2).sum(1)            # [m, H, W] without line s+j
            touched = (out - full).abs().amax(1) > 1e-4
            e_without = (err * touched).flatten(1).sum(1)
            e_with = (err_full * touched).flatten(1).sum(1)
            gains.append((e_without - e_with) / e_without.clamp(min=1e-8))
            index.append(idx[s:s + m])
    if not gains:
        return torch.zeros(0, device=real.device), torch.zeros(0, dtype=torch.long, device=real.device), full
    return torch.cat(gains), torch.cat(index), full


def psnr(a, b):
    return float(10 * torch.log10(1.0 / ((a - b) ** 2).mean().clamp(min=1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="glob of v3 decomposition shards")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-gain", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="images per shard (trial runs)")
    args = ap.parse_args()
    dev = "cuda"
    proto = load_protocol(PROTOCOL_V3_PATH)
    codec = Codec.load(proto=proto, device=dev)
    index = json.loads((args.images / "index.json").read_text())
    where = {int(i): (s["shard"], r) for s in index["shards"] for r, i in enumerate(s["ids"])}
    cache: dict[int, np.ndarray] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(p) for p in glob.glob(args.seq))
    src_cfg = files[0].parent / "config.json"
    cfg = json.loads(src_cfg.read_text()) if src_cfg.exists() else {}
    cfg.update({"pruned_from": str(files[0].parent), "prune_min_gain": args.min_gain})
    (args.out / "config.json").write_text(json.dumps(cfg, indent=1))
    totals = {"lines": 0, "removed": 0, "psnr_before": [], "psnr_after": []}
    for f in files:
        final = args.out / f.name
        if final.exists():
            continue
        t0 = time.time()
        exs = load_shards([f])[: args.limit]
        toks, lays, rnds, lines, offsets, ids, ps = [], [], [], [], [0], [], []
        for e in exs:
            sh, r = where[e.image_id]
            if sh not in cache:
                cache.clear()
                cache[sh] = np.load(args.images / f"images_{sh:03d}.npy", mmap_mode="r")
            img = np.array(cache[sh][r])
            if e.flip:
                img = img[:, ::-1].copy()
            real = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
            tok, lay = e.tokens.to(dev), e.layer.to(dev)
            isl = e.is_line.to(dev) if e.is_line is not None else torch.zeros(len(tok), dtype=torch.bool, device=dev)
            gain, idx, full = line_gains(tok, lay, isl, real, codec)
            keep = torch.ones(len(tok), dtype=torch.bool, device=dev)
            keep[idx[gain < args.min_gain]] = False
            after = render_strokes(tok[keep], lay[keep], isl[keep], codec, 256)
            totals["lines"] += int(isl.sum()); totals["removed"] += int((~keep).sum())
            totals["psnr_before"].append(psnr(full, real)); totals["psnr_after"].append(psnr(after, real))
            k = keep.cpu()
            toks.append(e.tokens[k].numpy()); lays.append(e.layer[k].numpy()); rnds.append(e.round[k].numpy())
            lines.append((e.is_line[k] if e.is_line is not None else torch.zeros(int(k.sum()), dtype=torch.bool)).numpy())
            offsets.append(offsets[-1] + int(k.sum())); ids.append(e.image_id); ps.append(totals["psnr_after"][-1])
        tmp = args.out / (f.stem + ".tmp.npz")
        np.savez(tmp, tokens=np.concatenate(toks), layer=np.concatenate(lays), round=np.concatenate(rnds),
                 is_line=np.concatenate(lines), offsets=np.array(offsets, dtype=np.int64),
                 ids=np.array(ids, dtype=np.int32), psnr=np.array(ps, dtype=np.float32))
        os.replace(tmp, final)
        n = len(exs)
        print(f"{final}: {n} images, {time.time() - t0:.0f}s | lines removed so far {totals['removed']}/{totals['lines']} "
              f"({100 * totals['removed'] / max(1, totals['lines']):.1f}%) | PSNR {np.mean(totals['psnr_before']):.2f} -> "
              f"{np.mean(totals['psnr_after']):.2f} dB", flush=True)


if __name__ == "__main__":
    main()
