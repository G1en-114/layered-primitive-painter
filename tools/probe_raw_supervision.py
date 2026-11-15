"""Probe bounded continuous corrections using paired RGB targets, without training a model.

This is per-image optimization with access to the original image: an oracle diagnostic,
not a measurement of generation quality. Primitive count, layer assignment, paint order,
and beta stay fixed. The output compares the existing target, bounded refinement, and
naively re-quantizing that refinement (not quantization-aware optimization).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.render.dispatch import render_group  # noqa: E402
from lpp.render.reference import Primitives, flatten_on_base, over  # noqa: E402
from lpp.runtime_env import configure_threads  # noqa: E402


def psnr(image, target):
    return -10 * ((image - target).square().mean((1, 2, 3)).clamp_min(1e-10)).log10()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=Path("data/celebamask256"))
    ap.add_argument("--sequences", type=Path, default=Path("data/seq/v0"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="native", choices=["native", "triton", "torch"])
    ap.add_argument("--quantized-control", action="store_true",
                    help="also optimize with straight-through snapping at every step")
    ap.add_argument("--out", type=Path, default=Path("out/raw_supervision_probe"))
    args = ap.parse_args()
    if args.count < 1 or args.steps < 1:
        ap.error("--count and --steps must be positive")
    configure_threads()
    proto = load_protocol(args.protocol) if args.protocol else load_protocol()
    codec = Codec.load(proto=proto, device=args.device)
    z = np.load(args.sequences / f"decomp_{args.shard:03d}.npz")
    if args.count > len(z["ids"]):
        ap.error("--count exceeds shard size")
    # Systematically cover reconstruction quality deciles; no visual cherry-picking.
    quantiles = np.linspace(0.1, 0.9, args.count)
    chosen = np.argsort(z["psnr"])[np.rint(quantiles * (len(z["ids"]) - 1)).astype(int)]
    index = json.loads((args.prepared / "index.json").read_text())
    source_ids = next(s["ids"] for s in index["shards"] if s["shard"] == args.shard)
    lookup = {image_id: row for row, image_id in enumerate(source_ids)}
    rows = [lookup[int(z["ids"][i])] for i in chosen]
    images = np.load(args.prepared / f"images_{args.shard:03d}.npy", mmap_mode="r")
    labels = np.load(args.prepared / f"labels_{args.shard:03d}.npy", mmap_mode="r")
    target = torch.tensor(np.array(images[rows]), device=args.device).permute(0, 3, 1, 2).float() / 255
    label = torch.tensor(np.array(labels[rows]), device=args.device)
    size = target.shape[-1]
    # Optimize within small boxes around the decoded target, in canvas-pixel units.
    ranges = {"xy": 2.0, "log_sigma": 0.25, "theta": 0.25, "rgb": 0.10, "alpha": 0.10}
    rates = {"xy": 0.10, "log_sigma": 0.01, "theta": 0.01, "rgb": 0.01, "alpha": 0.01}
    groups, optim_groups = [], []
    for k in range(proto.num_layers):
        tokens = []
        for i in chosen:
            a, b = z["offsets"][i:i + 2]
            tokens.append(z["tokens"][a:b][z["layer"][a:b] == k])
        width = max(map(len, tokens))
        if width == 0:
            continue
        packed = torch.zeros(args.count, width, len(codec.attr_names), dtype=torch.long, device=args.device)
        active = torch.zeros(args.count, width, device=args.device)
        for b, tok in enumerate(tokens):
            packed[b, :len(tok)] = torch.tensor(tok.astype(np.int64), device=args.device)
            active[b, :len(tok)] = 1
        p = codec.decode({name: packed[..., j] for j, name in enumerate(codec.attr_names)})
        base = dict(xy=p.xy, log_sigma=p.sigma.log(), theta=p.theta, rgb=p.rgb, alpha=p.alpha)
        leaves = {name: value.detach().clone().requires_grad_() for name, value in base.items()}
        for name, leaf in leaves.items():
            optim_groups.append({"params": [leaf], "lr": rates[name]})
        groups.append((k, base, leaves, p.beta, active))

    def draw(snap=False, straight_through=False):
        composite = None
        leaks = []
        for k, _, leaves, beta, active in groups:
            p = Primitives(xy=leaves["xy"], sigma=leaves["log_sigma"].exp().clamp(proto.sigma_min,
                           proto.sigma_max), theta=leaves["theta"], beta=beta,
                           rgb=leaves["rgb"].clamp(0, 1), alpha=leaves["alpha"].clamp(0, proto.max_alpha))
            if snap:
                p = codec.snap(p, straight_through=straight_through)
            p = p.with_alpha(p.alpha * active)
            group = render_group(p, size, proto, backend=args.backend)
            composite = group if composite is None else over(composite, group)
            behind = (label < k).unsqueeze(1)
            leaks.append(((1 - group[1]) * behind).sum((1, 2, 3)) /
                         behind.sum((1, 2, 3)).clamp_min(1))
        return flatten_on_base(composite, proto), torch.stack(leaks, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        before, initial_leak = draw()
        before_psnr = psnr(before, target)
    t0 = time.perf_counter()

    def fit(quantized=False):
        opt = torch.optim.Adam(optim_groups)
        for step in range(args.steps):
            image, leak = draw(snap=quantized, straight_through=quantized)
            # RGB supervision plus a penalty for increasing leakage behind another layer.
            loss = F.mse_loss(image, target) + 0.1 * (leak - initial_leak).clamp_min(0).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for _, base, leaves, _, _ in groups:
                    for name, leaf in leaves.items():
                        leaf.clamp_(base[name] - ranges[name], base[name] + ranges[name])
            if (step + 1) % 20 == 0 or step + 1 == args.steps:
                mode = "quantized" if quantized else "continuous"
                print(f"{mode} {step + 1}/{args.steps}: loss {loss.item():.6f}", flush=True)

    fit()
    with torch.no_grad():
        after, final_leak = draw()
        snapped, snapped_leak = draw(snap=True)
        continuous_psnr, snapped_psnr = psnr(after, target), psnr(snapped, target)
        continuous_parameters = {"ids": z["ids"][chosen]}
        for k, _, leaves, beta, active in groups:
            fields = {"xy": leaves["xy"],
                      "sigma": leaves["log_sigma"].exp().clamp(proto.sigma_min, proto.sigma_max),
                      "theta": leaves["theta"], "beta": beta, "rgb": leaves["rgb"].clamp(0, 1),
                      "alpha": leaves["alpha"].clamp(0, proto.max_alpha) * active, "active": active}
            for name, value in fields.items():
                continuous_parameters[f"layer_{k}_{name}"] = value.detach().cpu().numpy()
        np.savez_compressed(args.out / "continuous_parameters.npz", **continuous_parameters)
    quantized = None
    if args.quantized_control:
        with torch.no_grad():
            for _, base, leaves, _, _ in groups:
                for name, leaf in leaves.items():
                    leaf.copy_(base[name])
                    leaf.grad = None
        fit(quantized=True)
        with torch.no_grad():
            quantized, quantized_leak = draw(snap=True)
            quantized_psnr = psnr(quantized, target)
    records = []
    for b, i in enumerate(chosen):
        records.append(dict(image_id=int(z["ids"][i]), selection_quantile=float(quantiles[b]),
                            primitives=int(z["offsets"][i + 1] - z["offsets"][i]),
                            before_psnr=before_psnr[b].item(), continuous_psnr=continuous_psnr[b].item(),
                            naively_resnapped_psnr=snapped_psnr[b].item(),
                            before_leak=initial_leak[b].cpu().tolist(),
                            continuous_leak=final_leak[b].cpu().tolist(),
                            naively_resnapped_leak=snapped_leak[b].cpu().tolist()))
        if quantized is not None:
            records[-1].update(quantization_aware_psnr=quantized_psnr[b].item(),
                               quantization_aware_leak=quantized_leak[b].cpu().tolist())
    result = dict(experiment="per-image bounded oracle refinement; not model training", args=vars(args),
                  bounds=ranges, learning_rates=rates,
                  mean_before_psnr=before_psnr.mean().item(),
                  mean_continuous_psnr=continuous_psnr.mean().item(),
                  mean_naively_resnapped_psnr=snapped_psnr.mean().item(),
                  seconds=time.perf_counter() - t0, torch_version=torch.__version__,
                  device_name=torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
                  records=records)
    if quantized is not None:
        result["mean_quantization_aware_psnr"] = quantized_psnr.mean().item()
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    sheet = Image.new("RGB", ((5 if quantized is not None else 4) * size, args.count * (size + 28)), "white")
    pen = ImageDraw.Draw(sheet)
    for b, rec in enumerate(records):
        names = [f'Original {rec["image_id"]}', f'Existing {rec["before_psnr"]:.2f} dB',
                 f'Bounded oracle {rec["continuous_psnr"]:.2f} dB',
                 f'Re-snap {rec["naively_resnapped_psnr"]:.2f} dB']
        variants = [target, before, after, snapped]
        if quantized is not None:
            variants.append(quantized)
            names.append(f'QAT oracle {rec["quantization_aware_psnr"]:.2f} dB')
        for col, image in enumerate(variants):
            pixels = (image[b].detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            sheet.paste(Image.fromarray(pixels), (col * size, b * (size + 28) + 28))
            pen.text((col * size + 4, b * (size + 28) + 7), names[col], fill="black")
    sheet.save(args.out / "comparison.png")
    print(json.dumps({k: v for k, v in result.items() if k.startswith("mean_")}), flush=True)
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
