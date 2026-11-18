"""Protocol v3 data: turn Gaussian decompositions (v2 shards) into Gaussian + Bezier line decompositions.

    python tools/decompose_lines.py --seq data/seq/anime_v2_l6_t0/decomp_000.npz \
        --images data/anime_portraits256_v2_l6view --out data/seq/anime_v3_lines_t0 [--limit 100]

Per image (the line-probe recipe that won, docs/line_strokes.md):
  1. line art -> skeleton -> pieces of <= 20 px, contour pieces first, at most 200 lines; each line goes to the
     topmost layer it touches;
  2. the K smallest Gaussians are dropped (equal stroke count);
  3. lines and the remaining Gaussians are fitted jointly to the image, both snapped to their tokens every step
     (straight-through). Lines are fitted in the token parametrisation (chord midpoint, direction, half
     length, half width, bend), so encoding loses nothing;
  4. order inside each layer: sketch rounds 0-1, then the lines (round 2, Z-order), then the remaining
     Gaussian rounds (shifted by one): colour sketch -> lines -> refinement, as designed.
The PSNR written to the shard is measured by rendering the written tokens with lpp.render.strokes, the
protocol's own compositing, so the data is checked end to end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from line_probe import init_curves, line_mask  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.data.sequences import load_shards, widen  # noqa: E402
from lpp.decompose.fill import morton_code  # noqa: E402
from lpp.protocol import PROTOCOL_V3_PATH, load_protocol  # noqa: E402
from lpp.render.bezier import Curves  # noqa: E402
from lpp.render.triton_lines import render_lines_triton  # noqa: E402
from lpp.render.reference import Primitives  # noqa: E402
from lpp.render.dispatch import render_group as ref_group  # noqa: E402  (Triton/native tile rasterizer, differentiable: 60x the reference)
from lpp.render.strokes import render_strokes  # noqa: E402

LINE_ROUND = 2          # lines come right after the two sketch rounds of their layer
MAX_HALF_LEN, MAX_HALF_WIDTH = 16.0, 4.0
WINDOW = 64             # 2 * 16 + 2 * 4 + soft margin 2 + 8 * 0.7 < 64


def st(cont: torch.Tensor, disc: torch.Tensor) -> torch.Tensor:
    return cont + (disc - cont).detach()


def global_swap(g0, lay, init, proto):
    """The K finest Gaussians of the whole image make way for the K lines (the line probe's +1.1 dB recipe):
    those are the strokes that tried to draw the lines. Each layer must stay within its cap: where a layer would
    overflow (it gains more lines than it loses Gaussians), its last (shortest) lines are dropped and as many of
    the largest removed Gaussians come back, so the total stays equal."""
    dev = lay.device
    big = g0.sigma[0].max(-1).values
    order = big.argsort()                                      # finest first
    removed = order[: len(init)].tolist()
    cap = {l.id: l.max_primitives for l in proto.layers}
    count = {k: int((lay == k).sum()) for k in lay.unique().tolist()}
    lines_by = {}
    for c in init:
        lines_by.setdefault(c[6], []).append(c)
    rem_by = {}
    for i in removed:
        rem_by.setdefault(int(lay[i]), []).append(i)
    restore = []
    for k, mine in lines_by.items():
        over = count.get(k, 0) - len(rem_by.get(k, [])) + len(mine) - cap[k]
        if over > 0:
            del mine[len(mine) - over:]                        # init is longest-first: drop the shortest
            restore.append(over)
    n_restore = sum(restore)
    if n_restore:                                              # give back the largest removed Gaussians
        removed = sorted(removed, key=lambda i: float(big[i]))[: len(removed) - n_restore]
    keep = torch.ones(len(lay), dtype=torch.bool, device=dev)
    keep[torch.tensor(removed, dtype=torch.long, device=dev)] = False
    lines = [c for k in sorted(lines_by) for c in lines_by[k]]
    for k in count:                                            # a restored Gaussian may land in a full layer
        if int(((lay == k) & keep).sum()) + len(lines_by.get(k, [])) > cap[k]:
            return None
    return keep, lines


def decompose_one(e, img, labels, codec, proto, args, dev):
    target = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
    base = torch.tensor(proto.base_color, device=dev).view(1, 3, 1, 1)
    tok_all = widen(e.tokens, proto).to(dev)
    lay, rnd = e.layer.to(dev), e.round.to(dev)
    g0 = codec.decode({k: tok_all[:, j].unsqueeze(0) for j, k in enumerate(proto.attr_names)})
    init = init_curves(img, line_mask(img), labels, args.piece, args.max_curves, contour_first=True)
    swapped = global_swap(g0, lay, init, proto) if args.swap == "global" else None
    if swapped is not None:
        keep, init = swapped
    else:
        # equal swap per layer: a layer drops as many of its smallest Gaussians as it gains lines, so every
        # layer keeps its stroke count (and its cap); lines beyond what a layer can swap are left out
        big = g0.sigma[0].max(-1).values
        keep = torch.ones(len(lay), dtype=torch.bool, device=dev)
        kept_init = []
        for k in sorted({c[6] for c in init}):
            mine = [c for c in init if c[6] == k]
            idx = (lay == k).nonzero()[:, 0]
            n = min(len(mine), max(len(idx) - 4, 0))        # leave a layer a few Gaussians of its own
            keep[idx[big[idx].argsort()[:n]]] = False
            kept_init += mine[:n]                          # init is longest-first (contours first)
        init = kept_init
    K = len(init)
    kidx = keep.nonzero()[:, 0]
    gk, klay, krnd = g0[kidx], lay[kidx], rnd[kidx]
    layers = sorted(lay.unique().tolist())

    gxy = gk.xy.clone().requires_grad_(True)
    gls = gk.sigma.clamp(min=1e-3).log().clone().requires_grad_(True)
    gth = gk.theta.clone().requires_grad_(True)
    grgb = torch.logit(gk.rgb.clamp(0.01, 0.99)).requires_grad_(True)
    gal = gk.alpha.clone().requires_grad_(True)
    groups = [{"params": [gxy], "lr": 0.05}, {"params": [gls, gth], "lr": 0.01}, {"params": [grgb, gal], "lr": 0.01}]

    def gauss():
        return codec.snap(Primitives(xy=gxy, sigma=gls.exp(), theta=gth, beta=gk.beta,
                                     rgb=torch.sigmoid(grgb), alpha=gal.clamp(0, 0.99)))

    if K:
        t = lambda v: torch.tensor(np.array(v), dtype=torch.float32, device=dev)[None]   # noqa: E731
        p0, p1, p2 = (t([c[i] for c in init]) for i in (1, 2, 3))
        mid, chord = (p0 + p2) / 2, p2 - p0
        th0 = torch.atan2(chord[..., 1], chord[..., 0])
        h0 = (chord.norm(dim=-1) / 2).clamp(min=0.75)
        n0 = torch.stack([-th0.sin(), th0.cos()], -1)
        bend0 = (((p1 - mid) * n0).sum(-1) / h0).clamp(-0.95, 0.95)
        lc = mid.clone().requires_grad_(True)
        lth = th0.clone().requires_grad_(True)
        llh = h0.log().clone().requires_grad_(True)
        llw = (t([c[4] for c in init]) / 2).clamp(min=0.5).log().requires_grad_(True)
        lbend = torch.atanh(bend0).requires_grad_(True)
        lrgb = torch.logit(t([c[5] for c in init]).clamp(0.02, 0.98)).requires_grad_(True)
        lal = torch.full((1, K), 2.0, device=dev, requires_grad=True)
        lline = torch.tensor([c[6] for c in init], device=dev)
        groups += [{"params": [lc], "lr": 0.15}, {"params": [lth, llh, llw, lbend], "lr": 0.03},
                   {"params": [lrgb, lal], "lr": 0.05}]

        def lines():
            # pieces are <= 20 px: keep them short and thin while fitting (bounded windows, no runaway)
            h, th, b = llh.exp().clamp(max=MAX_HALF_LEN), lth, torch.tanh(lbend)
            d = torch.stack([th.cos(), th.sin()], -1)
            n = torch.stack([-th.sin(), th.cos()], -1)
            cv = Curves(p0=lc - h[..., None] * d, p1=lc + (b * h)[..., None] * n, p2=lc + h[..., None] * d,
                        width=2 * llw.exp().clamp(max=MAX_HALF_WIDTH), rgb=torch.sigmoid(lrgb), alpha=torch.sigmoid(lal) * 0.99)
            if not snap_lines[0]:
                return cv
            q = codec.decode_lines(codec.encode_lines(cv))
            return Curves(p0=st(cv.p0, q.p0), p1=st(cv.p1, q.p1), p2=st(cv.p2, q.p2), width=st(cv.width, q.width),
                          rgb=st(cv.rgb, q.rgb), alpha=st(cv.alpha, q.alpha))

    def compose(soft):
        g = gauss()
        cv = lines() if K else None
        a = None
        out = base.expand(1, 3, 256, 256)
        for k in layers:
            P, T = ref_group(g[(klay == k).nonzero()[:, 0]], 256, proto)
            out = P + T * out
            if K and (lline == k).any():
                sel = (lline == k).nonzero()[:, 0]
                # lines of layer k over everything so far equals lines over layer k's group, then the group
                # over the layers below (curves sit at the top of layer k)
                if os.environ.get("LPP_LINES_TORCH"):            # dense reference path (A/B checks)
                    from lpp.render.bezier import composite, curve_alpha_local
                    cs = cv[sel]
                    out = composite(curve_alpha_local(cs, 256, window=WINDOW, soft=soft), cs.rgb, out)
                else:
                    Pl, Tl = render_lines_triton(cv[sel], 256, soft=soft)   # tile kernel, 45x the dense path
                    out = Pl + Tl * out
        return out

    snap_lines = [args.line_qat == "step"]
    opt = torch.optim.Adam(groups)
    for step in range(args.steps):
        if args.line_qat == "late" and step == int(0.8 * args.steps):
            snap_lines[0] = True                       # last 20%: settle onto the token grid
        if args.line_qat == "freeze" and K and step == int(0.8 * args.steps):
            # snap position and angle (the two grids that cost: 0.33 and 0.12 dB) and keep them; the rest
            # (width, length, bend, colour, opacity, the Gaussians) adapts for the remaining steps
            with torch.no_grad():
                ax = proto.attr
                wrap = torch.remainder(torch.floor(lth / torch.pi), 2) == 1
                lbend.copy_(torch.where(wrap, -lbend, lbend))
                lth.copy_(ax("theta").dequantize(ax("theta").quantize(torch.remainder(lth, torch.pi))))
                lc.copy_(torch.stack([ax("x").dequantize(ax("x").quantize(lc[..., 0])),
                                      ax("y").dequantize(ax("y").quantize(lc[..., 1]))], -1))
            lc.requires_grad_(False), lth.requires_grad_(False)
        tau = 0.7 + (0.2 - 0.7) * step / max(args.steps - 1, 1)
        loss = ((compose(tau) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    if os.environ.get("LPP_DEBUG_PREQ"):
        with torch.no_grad():
            pre = float(10 * torch.log10(1 / ((compose(None).clamp(0, 1) - target) ** 2).mean()))
        print(f"  pre-quantisation PSNR {pre:.2f} dB", flush=True)
        if K:                                          # which line attribute's grid costs the most
            ax = proto.attr
            qd = lambda nm, v: ax(nm).dequantize(ax(nm).quantize(v))   # noqa: E731
            base_params = dict(c=lc.detach(), th=lth.detach(), h=llh.exp().clamp(max=MAX_HALF_LEN).detach(),
                               w=llw.exp().clamp(max=MAX_HALF_WIDTH).detach(), b=torch.tanh(lbend).detach(),
                               rgb=torch.sigmoid(lrgb).detach(), al=(torch.sigmoid(lal) * 0.99).detach())
            def curves_of(pp):
                d = torch.stack([pp["th"].cos(), pp["th"].sin()], -1)
                n = torch.stack([-pp["th"].sin(), pp["th"].cos()], -1)
                return Curves(p0=pp["c"] - pp["h"][..., None] * d, p1=pp["c"] + (pp["b"] * pp["h"])[..., None] * n,
                              p2=pp["c"] + pp["h"][..., None] * d, width=2 * pp["w"], rgb=pp["rgb"], alpha=pp["al"])
            quant = {"xy": lambda pp: pp | {"c": torch.stack([qd("x", pp["c"][..., 0]), qd("y", pp["c"][..., 1])], -1)},
                     # wrapping theta by an odd multiple of pi flips the normal, so bend changes sign (as encode_lines)
                     "theta": lambda pp: pp | {"th": qd("theta", torch.remainder(pp["th"], torch.pi)),
                                               "b": torch.where(torch.remainder(torch.floor(pp["th"] / torch.pi), 2) == 1, -pp["b"], pp["b"])},
                     "length": lambda pp: pp | {"h": qd("sigma_x", pp["h"].clamp(proto.sigma_min, proto.sigma_max))},
                     "width": lambda pp: pp | {"w": qd("sigma_y", pp["w"].clamp(proto.sigma_min, proto.sigma_max))},
                     "bend": lambda pp: pp | {"b": qd("bend", pp["b"])},
                     "alpha": lambda pp: pp | {"al": qd("alpha", pp["al"])}}
            parts = []
            for nm, f in quant.items():
                cvq = curves_of(f(dict(base_params)))
                with torch.no_grad():
                    out = base.expand(1, 3, 256, 256)
                    g = gauss()
                    for k in layers:
                        P, T = ref_group(g[(klay == k).nonzero()[:, 0]], 256, proto)
                        out = P + T * out
                        if (lline == k).any():
                            sel = (lline == k).nonzero()[:, 0]
                            Pl, Tl = render_lines_triton(cvq[sel], 256)
                            out = Pl + Tl * out
                    parts.append(f"{nm} {float(10 * torch.log10(1 / ((out.clamp(0, 1) - target) ** 2).mean())):.2f}")
            print("  one attribute quantised: " + ", ".join(parts), flush=True)
    snap_lines[0] = True                               # the written tokens are always the snapped ones
    with torch.no_grad():
        gt = codec.encode(gauss())
        gtok = torch.stack([gt[n][0] if n in gt else torch.full_like(gt["x"][0], proto.fixed_tokens(proto.event_id("PRIMITIVE"))[n])
                            for n in proto.attr_names], -1)
        grnd = torch.where(krnd >= LINE_ROUND, krnd + 1, krnd)
        if K:
            lt = codec.encode_lines(lines())
            ltok = torch.stack([lt[n][0] for n in proto.attr_names], -1)
            tok = torch.cat([gtok, ltok])
            lay_o = torch.cat([klay, lline[0] if lline.dim() > 1 else lline])
            rnd_o = torch.cat([grnd, torch.full((K,), LINE_ROUND, device=dev)])
            is_line = torch.cat([torch.zeros(len(gtok), dtype=torch.bool, device=dev), torch.ones(K, dtype=torch.bool, device=dev)])
        else:
            tok, lay_o, rnd_o = gtok, klay, grnd
            is_line = torch.zeros(len(gtok), dtype=torch.bool, device=dev)
        # order: layer, round, then sequence position for Gaussians (their Morton order) / Morton for lines
        xi = proto.attr("x").quantize(proto.attr("x").dequantize(tok[:, 0])) // 2
        yi = proto.attr("y").quantize(proto.attr("y").dequantize(tok[:, 1])) // 2
        pos = torch.where(is_line, morton_code(xi, yi) + (1 << 20), torch.arange(len(tok), device=dev))
        key = (lay_o * 16 + rnd_o) * (1 << 22) + pos
        order = key.argsort()
        tok, lay_o, rnd_o, is_line = tok[order], lay_o[order], rnd_o[order], is_line[order]
        img_out = render_strokes(tok, lay_o, is_line, codec)
        psnr = float(10 * torch.log10(1 / ((img_out.clamp(0, 1) - target) ** 2).mean()))
    return tok.cpu(), lay_o.cpu(), rnd_o.cpu(), is_line.cpu(), psnr, K


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, nargs="+", required=True, help="v2 decomp_*.npz shards")
    ap.add_argument("--images", type=Path, required=True, help="dir with index.json, images_*.npy, labels_*.npy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--piece", type=int, default=20)
    ap.add_argument("--max-curves", type=int, default=200)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="images per shard (smoke runs)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip each shard's first N images (finish a shard an earlier --limit run cut short); the output "
                         "is written as <shard>_rest.npz next to it, so both parts load together")
    ap.add_argument("--line-qat", choices=["step", "late", "end", "freeze"], default="freeze",
                    help="lines on their token grid: every step (straight-through), the last 20%% of the steps, only "
                         "when writing, or freeze: position and angle snapped and fixed for the last 20%%, the rest adapts")
    ap.add_argument("--swap", choices=["global", "layer"], default="global",
                    help="which Gaussians make way for the lines: the finest of the image (capped per layer) or per layer")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    proto = load_protocol(PROTOCOL_V3_PATH)
    codec = Codec.load(proto=proto, device=dev)
    index = json.loads((args.images / "index.json").read_text())
    where = {i: (s["shard"], r) for s in index["shards"] for r, i in enumerate(s["ids"])}
    (args.out / "config.json").write_text(json.dumps({"flip": False, "protocol_version": 3, "source": [str(s) for s in args.seq],
                                                      "piece": args.piece, "max_curves": args.max_curves, "steps": args.steps}, indent=1))
    for shard in args.seq:
        final = args.out / (shard.name if not args.skip else shard.name.replace(".npz", "_rest.npz"))
        if final.exists():
            print(f"{final} exists, skipped", flush=True)
            continue
        exs = [e for e in load_shards([shard]) if not e.flip and e.image_id in where][args.skip:][: args.limit]
        toks, lays, rnds, lines, offsets, ids, psnrs, t0 = [], [], [], [], [0], [], [], time.time()
        for i, e in enumerate(exs):
            sh, r = where[e.image_id]
            img = np.array(np.load(args.images / f"images_{sh:03d}.npy", mmap_mode="r")[r])
            labels = np.array(np.load(args.images / f"labels_{sh:03d}.npy", mmap_mode="r")[r])
            tok, lay, rnd, isl, psnr, K = decompose_one(e, img, labels, codec, proto, args, dev)
            toks.append(tok.numpy().astype(np.int16)), lays.append(lay.numpy().astype(np.int8))
            rnds.append(rnd.numpy().astype(np.int8)), lines.append(isl.numpy())
            offsets.append(offsets[-1] + len(tok)), ids.append(e.image_id), psnrs.append(psnr)
            if (i + 1) % 10 == 0 or i + 1 == len(exs):
                print(f"{shard.name}: {i + 1}/{len(exs)}  PSNR {np.mean(psnrs):.2f} dB  last {K} lines  "
                      f"{(i + 1) / (time.time() - t0):.2f} img/s", flush=True)
        tmp = final.with_suffix(".tmp.npz")
        np.savez(tmp, tokens=np.concatenate(toks), layer=np.concatenate(lays), round=np.concatenate(rnds),
                 is_line=np.concatenate(lines), offsets=np.array(offsets, dtype=np.int64),
                 ids=np.array(ids, dtype=np.int32), psnr=np.array(psnrs, dtype=np.float32))
        os.replace(tmp, final)
        print(f"{final}: written, {offsets[-1] / len(ids):.0f} strokes/image, mean PSNR {np.mean(psnrs):.2f} dB", flush=True)


if __name__ == "__main__":
    main()
