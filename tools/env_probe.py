"""Probe an accelerator machine before spending hours on it (NVIDIA CUDA or Hygon DCU / DTK).

    python tools/env_probe.py            # full probe, ~2-3 minutes
    python tools/env_probe.py --quick    # skip model benchmarks

Prints one JSON report: software stack, devices, which features work (bf16, SDPA with
masks + GQA, fused AdamW, torch.compile, Triton, the project's Triton rasterizer), matmul
throughput, and training-step time of the painter at each model size. Paste the output back.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

report: dict = {}


def attempt(name):
    """Decorator: run a probe, record result or the error, never abort the whole report."""
    def wrap(fn):
        try:
            t = time.perf_counter()
            report[name] = fn()
            print(f"[ok]   {name}  ({time.perf_counter() - t:.1f}s)", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            report[name] = {"error": f"{type(e).__name__}: {e}"[:400],
                            "trace": traceback.format_exc().splitlines()[-3:]}
            print(f"[fail] {name}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return fn
    return wrap


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()[:2000]
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def timeit(fn, reps=20):
    import torch
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    @attempt("system")
    def _():
        from lpp.runtime_env import available_cpus
        return {"python": sys.version.split()[0], "platform": platform.platform(),
                "cpus_host": os.cpu_count(), "cpus_quota": available_cpus(), "hostname": platform.node(),
                "hy_smi": sh("hy-smi 2>/dev/null | head -30"), "rocm_smi": sh("rocm-smi 2>/dev/null | head -30"),
                "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null"),
                "dtk": sh("ls -d /opt/dtk* 2>/dev/null; cat /opt/dtk/.info/version 2>/dev/null"),
                "slurm": sh("sinfo --version 2>/dev/null"),
                "internet": sh("curl -s -m 5 -o /dev/null -w '%{http_code}' https://huggingface.co"),
                "hf_mirror": sh("curl -s -m 5 -o /dev/null -w '%{http_code}' https://hf-mirror.com"),
                "pip_index": sh("pip config list 2>/dev/null")}

    import torch

    @attempt("torch")
    def _():
        info = {"version": torch.__version__, "cuda_api_available": torch.cuda.is_available(),
                "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda,
                "devices": []}
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({"name": p.name, "mem_gib": round(p.total_memory / 2**30, 1),
                                    "multiprocessors": p.multi_processor_count,
                                    "arch": getattr(p, "gcnArchName", None)})
        return info

    if not torch.cuda.is_available():
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return
    dev = "cuda"

    @attempt("matmul_tflops")
    def _():
        out = {}
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            n = 4096
            a, b = torch.randn(n, n, device=dev, dtype=dt), torch.randn(n, n, device=dev, dtype=dt)
            out[str(dt).split(".")[-1]] = round(2 * n**3 / timeit(lambda: a @ b) / 1e12, 1)
        return out

    @attempt("bandwidth_gbs")
    def _():
        x = torch.empty(256 * 2**20, device=dev, dtype=torch.float32)
        return round(2 * x.numel() * 4 / timeit(lambda: x.mul_(1.0)) / 1e9)

    @attempt("sdpa_bool_mask_gqa")
    def _():
        import torch.nn.functional as F
        q = torch.randn(4, 12, 512, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(4, 4, 512, 64, device=dev, dtype=torch.bfloat16)
        m = torch.ones(512, 512, dtype=torch.bool, device=dev).tril()
        o = F.scaled_dot_product_attention(q, k, k, attn_mask=m, enable_gqa=True)
        return {"ok": bool(torch.isfinite(o).all()), "ms": round(1e3 * timeit(
            lambda: F.scaled_dot_product_attention(q, k, k, attn_mask=m, enable_gqa=True)), 3)}

    @attempt("fused_adamw")
    def _():
        p = torch.nn.Parameter(torch.randn(1000, device=dev))
        opt = torch.optim.AdamW([p], fused=True)
        p.grad = torch.randn_like(p)
        opt.step()
        return True

    @attempt("triton")
    def _():
        import triton
        import triton.language as tl

        @triton.jit
        def add(x, y, o, n, B: tl.constexpr):
            i = tl.program_id(0) * B + tl.arange(0, B)
            tl.store(o + i, tl.load(x + i, mask=i < n) + tl.load(y + i, mask=i < n), mask=i < n)
        x = torch.randn(10000, device=dev)
        o = torch.empty_like(x)
        add[(triton.cdiv(10000, 256),)](x, x, o, 10000, B=256)
        return {"version": triton.__version__, "correct": bool(torch.allclose(o, 2 * x))}

    @attempt("project_rasterizer")
    def _():
        from lpp.render import Primitives
        from lpp.render.reference import render_group as ref
        from lpp.render.triton_raster import render_group_triton as tri
        g = torch.Generator(device=dev).manual_seed(0)
        r = lambda *s: torch.rand(*s, device=dev, generator=g)  # noqa: E731
        p = Primitives(xy=r(2, 64, 2) * 256, sigma=2 + r(2, 64, 2) * 30, theta=r(2, 64) * 3.14,
                       beta=torch.ones(2, 64, device=dev), rgb=r(2, 64, 3), alpha=0.3 + 0.6 * r(2, 64))
        (P, T), (Pr, Tr) = tri(p, 64), ref(p, 64)
        return {"max_err": float(max((P - Pr).abs().max(), (T - Tr).abs().max()))}

    if not args.quick:
        @attempt("painter_train_step")
        def _():
            from lpp.codec import Codec
            from lpp.data.sequences import Example, SequenceDataset, collate
            from lpp.model import ModelConfig, Painter
            from lpp.protocol import load_protocol
            proto, codec = load_protocol(), Codec.load()
            g = torch.Generator().manual_seed(0)
            exs = []
            for i in range(32):      # realistic length: ~720 primitives over 6 layers
                lay = torch.sort(torch.randint(0, 6, (720,), generator=g)).values
                tok = torch.stack([torch.randint(v, (720,), generator=g) for v in (256, 256, 4096, 4096, 16)], 1)
                exs.append(Example(tok, lay, torch.zeros(720, dtype=torch.long), i))
            ds = SequenceDataset(exs, proto)
            out = {}
            for name, cfg, B in (("small", ModelConfig.small(), 32), ("base", ModelConfig(), 32)):
                torch.manual_seed(0)
                m = Painter(cfg, proto, codec).to(dev)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
                batch = {k: v.to(dev) for k, v in collate([ds[i] for i in range(B)]).items()}

                def step():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = m(batch)["loss"]
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                t = timeit(step, reps=5)
                out[name] = {"params_M": round(m.num_params() / 1e6, 1), "batch": B,
                             "s_per_step": round(t, 3), "epoch_30k_min": round(t * 30000 / B / 60, 1),
                             "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                del m, opt
                torch.cuda.empty_cache()
            return out

    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
