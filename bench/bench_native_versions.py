"""Interleaved native raster backward modes on one compiled extension.

Measures the complete decompose step with identical tensors. Run on an idle GPU;
the caller is responsible for pausing/resuming any existing workloads safely.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bench_raster import scene, step_fn  # noqa: E402
from lpp.render import native_raster  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--primitives", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error("use at least 10 measurements per mode")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    p = scene(args.batch, args.primitives, 1.5, 48.0)
    target = torch.rand(args.batch, 3, 256, 256, device="cuda")
    run = step_fn(native_raster.render_group_native, p, target, 256)
    original = native_raster.SKIP_IDLE_WAVES
    times = {"v1": [], "v2_skip_idle": []}
    try:
        for mode in (False, True):
            native_raster.SKIP_IDLE_WAVES = mode
            for _ in range(5):
                run()
        torch.cuda.synchronize()
        for pair in range(args.repeats):
            # Alternate AB / BA order to reduce temperature/clock drift bias.
            for mode in ((False, True) if pair % 2 == 0 else (True, False)):
                native_raster.SKIP_IDLE_WAVES = mode
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                end.synchronize()
                times["v2_skip_idle" if mode else "v1"].append(start.elapsed_time(end))
    finally:
        native_raster.SKIP_IDLE_WAVES = original
    result = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "hip": torch.version.hip, "batch": args.batch, "primitives": args.primitives,
              "size": 256, "measurement": "complete forward + backward MSE step, GPU events, AB/BA",
              "warmup_each": 5, "repeats_each": args.repeats, "modes": {}}
    for mode, values in times.items():
        values.sort()
        result["modes"][mode] = {"median_ms": statistics.median(values),
                                 "p10_ms": values[int(len(values) * 0.1)],
                                 "p90_ms": values[min(len(values) - 1, int(len(values) * 0.9))]}
    old, new = (result["modes"][mode]["median_ms"] for mode in ("v1", "v2_skip_idle"))
    result.update(speedup=old / new, latency_reduction=1 - new / old)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
