"""Export a trained Painter to ONNX graphs for a C++ runtime, and check them with onnxruntime.

    python tools/export_onnx.py --ckpt out/small_v2_t0_text_canvas/model_step8000.pt --out out/onnx/small_t0

Writes one .onnx per graph (python/lpp/export.py) plus manifest.json (model config, protocol,
tensor names and dynamic axes) and a parity report: every graph is run with onnxruntime on random
inputs at two sizes and compared with PyTorch (max abs difference).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.export import graphs  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402


def specs(m: Painter, B: int, T: int, P: int, Lt: int) -> dict:
    """Example inputs, input/output names and dynamic axes per graph."""
    c, A = m.cfg, len(m.attr_names)
    hd = c.d_model // c.n_heads
    g = torch.Generator().manual_seed(B * 1000 + T)
    vocab = list(m.proto.attr_vocab_sizes.values())
    attrs = torch.stack([torch.randint(0, n, (B, T), generator=g) for n in vocab], -1)
    out = {
        "embed_record": dict(
            inputs=(torch.randint(1, 4, (B, T), generator=g), torch.randint(0, m.n_events, (B, T), generator=g),
                    torch.randint(-1, m.proto.num_layers, (B, T), generator=g), attrs),
            names=["kind", "event", "layer", "attrs"], outputs=["x"],
            axes={"kind": {0: "B", 1: "T"}, "event": {0: "B", 1: "T"}, "layer": {0: "B", 1: "T"},
                  "attrs": {0: "B", 1: "T"}, "x": {0: "B", 1: "T"}}),
        "embed_text": dict(
            inputs=(torch.randn(B, Lt, c.text_dim, generator=g),), names=["text"], outputs=["x"],
            axes={"text": {0: "B", 1: "Lt"}, "x": {0: "B", 1: "Lt"}}),
        "chunk": dict(
            inputs=(torch.randn(B, T, c.d_model, generator=g), torch.rand(B, T, generator=g) > 0.1,
                    torch.randn(c.n_layers, B, c.n_kv_heads, P, hd, generator=g),
                    torch.randn(c.n_layers, B, c.n_kv_heads, P, hd, generator=g),
                    torch.rand(B, P, generator=g) > 0.3, torch.randint(0, 50, (B,), generator=g),
                    torch.randn(B, (c.canvas_size // 8) ** 2, c.d_model, generator=g),
                    torch.rand(B, T, generator=g) > 0.5),
            names=["x", "valid", "past_k", "past_v", "past_mask", "start_pos", "canvas_tokens", "draw"],
            outputs=["h", "k_new", "v_new"],
            axes={"x": {0: "B", 1: "T"}, "valid": {0: "B", 1: "T"}, "past_k": {1: "B", 3: "P"},
                  "past_v": {1: "B", 3: "P"}, "past_mask": {0: "B", 1: "P"}, "start_pos": {0: "B"},
                  "canvas_tokens": {0: "B"}, "draw": {0: "B", 1: "T"}, "h": {0: "B", 1: "T"},
                  "k_new": {1: "B", 3: "T"}, "v_new": {1: "B", 3: "T"}}),
        "chunk_cached": dict(
            inputs=(torch.randn(B, T, c.d_model, generator=g), torch.rand(B, T, generator=g) > 0.1,
                    torch.randn(c.n_layers, B, c.n_kv_heads, P + T + 2, hd, generator=g),
                    torch.randn(c.n_layers, B, c.n_kv_heads, P + T + 2, hd, generator=g),
                    torch.rand(B, P + T + 2, generator=g) > 0.3, torch.tensor(P, dtype=torch.long),
                    torch.randint(0, 50, (B,), generator=g),
                    torch.randn(B, (c.canvas_size // 8) ** 2, c.d_model, generator=g),
                    torch.rand(B, T, generator=g) > 0.5),
            names=["x", "valid", "cache_k", "cache_v", "cache_mask", "write_pos", "start_pos", "canvas_tokens", "draw"],
            outputs=["h", "cache_k_out", "cache_v_out", "cache_mask_out"],
            axes={"x": {0: "B", 1: "T"}, "valid": {0: "B", 1: "T"}, "cache_k": {1: "B", 3: "Tmax"},
                  "cache_v": {1: "B", 3: "Tmax"}, "cache_mask": {0: "B", 1: "Tmax"}, "start_pos": {0: "B"},
                  "canvas_tokens": {0: "B"}, "draw": {0: "B", 1: "T"}, "h": {0: "B", 1: "T"},
                  "cache_k_out": {1: "B", 3: "Tmax"}, "cache_v_out": {1: "B", 3: "Tmax"},
                  "cache_mask_out": {0: "B", 1: "Tmax"}}),
        "decode_heads": dict(
            inputs=(torch.randn(B, c.d_model, generator=g), torch.randint(0, m.n_events, (B,), generator=g),
                    attrs[:, 0]),
            names=["h", "event", "attrs"], outputs=["event_logits"] + [f"{n}_logits" for n in m.attr_names],
            axes={"h": {0: "B"}, "event": {0: "B"}, "attrs": {0: "B"}, "event_logits": {0: "B"},
                  **{f"{n}_logits": {0: "B"} for n in m.attr_names}}),
        # all attributes of a record in one call (lpp.export.DecodeSample): h has 2B rows (guidance), B samples
        "decode_sample": dict(
            inputs=(torch.randn(2 * B, c.d_model, generator=g),
                    torch.tensor(m.proto.stroke_events)[torch.randint(0, len(m.proto.stroke_events), (B,), generator=g)],
                    torch.rand(B, A, generator=g), torch.tensor([5.0]), torch.tensor([0.6]), torch.tensor([1.0])),
            names=["h", "event", "u", "cfg_scale", "top_p", "temperature"], outputs=["attrs"],
            axes={"h": {0: "R"}, "event": {0: "B"}, "u": {0: "B"}, "attrs": {0: "B"}}),
    }
    if c.canvas:
        out["canvas_enc"] = dict(
            inputs=(torch.rand(B, 3 + m.proto.num_layers, c.canvas_size, c.canvas_size, generator=g),),
            names=["canvas"], outputs=["tokens"], axes={"canvas": {0: "B"}, "tokens": {0: "B"}})
    if A != attrs.shape[-1]:
        raise AssertionError("attribute count mismatch")
    return out


def codec_tables(codec: Codec) -> dict:
    """Everything the C++ runtime needs to turn tokens into pixels, as plain lookup tables.

    Every attribute has a small discrete vocabulary, so decoding = indexing these tables; the
    values are the library's own dequantized float32 numbers (exact in JSON), colours already in
    sRGB, so the runtime reproduces Codec.decode bit for bit without porting Oklab or the bins."""
    from lpp.sequence import KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE, NO_ATTR
    proto = codec.proto
    tables = {}
    with torch.no_grad():
        for name, n in proto.attr_vocab_sizes.items():
            idx = torch.arange(n)
            if name == "color":
                tok = {k: torch.zeros(1, n, dtype=torch.long) for k in proto.attr_names}
                tok["color"] = idx[None]
                tables[name] = codec.decode(tok).rgb[0].tolist()
            else:
                tables[name] = [float(v) for v in proto.attr(name).dequantize(idx).float()]
    return {"attributes": list(proto.attr_names), "tables": tables,
            "canvas_size": proto.canvas_size, "base_color": list(proto.base_color),
            "max_alpha": proto.max_alpha, "cutoff_sigma": proto.cutoff_sigma,
            "sigma_min": proto.sigma_min, "sigma_max": proto.sigma_max,
            "events": list(proto.events), "layers": [{"name": ly.name, "max_primitives": ly.max_primitives}
                                                     for ly in proto.layers],
            "max_text_positions": proto.max_text_positions,
            "kinds": {"BOS": KIND_BOS, "CONTROL": KIND_CONTROL, "PRIMITIVE": KIND_PRIMITIVE}, "no_attr": NO_ATTR}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--step-batches", type=int, nargs="*", default=[],
                    help="also export the static one-step backbone (lpp.export.ChunkStepV2) for these row counts R "
                         "(R = 2 x images with CFG): graphs chunk_step_r<R>, opset 23 + com.microsoft, CUDA Graph capable")
    args = ap.parse_args()
    import onnxruntime as ort
    args.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    proto = _parse(state["protocol"])
    codec = Codec.load(proto=proto, device="cpu")
    m = Painter(ModelConfig(**state["config"]), proto, codec).eval()
    m.load_state_dict(state["model"])
    mods = graphs(m)
    export_specs = specs(m, B=2, T=3, P=5, Lt=4)
    check_specs = [export_specs, specs(m, B=3, T=1, P=17, Lt=9)]
    manifest = {"checkpoint": str(args.ckpt), "step": state.get("step"), "config": state["config"],
                "protocol": state["protocol"], "attr_names": m.attr_names, "opset": args.opset, "graphs": {}}
    report = {}
    for name, mod in mods.items():
        sp = export_specs[name]
        path = args.out / f"{name}.onnx"
        t0 = time.perf_counter()
        torch.onnx.export(mod, sp["inputs"], str(path), input_names=sp["names"], output_names=sp["outputs"],
                          dynamic_axes=sp["axes"], opset_version=args.opset, dynamo=False)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        diffs = []
        for cs in check_specs:
            ins = cs[name]["inputs"]
            with torch.no_grad():
                ref = mod(*ins)
            ref = ref if isinstance(ref, tuple) else (ref,)
            got = sess.run(None, {n: t.numpy() for n, t in zip(sp["names"], ins)})
            diffs.append(max(float(np.abs(g.astype(np.float64) - r.numpy().astype(np.float64)).max())
                             for g, r in zip(got, ref)))
        report[name] = {"max_abs_diff": max(diffs), "bytes": path.stat().st_size,
                        "export_s": round(time.perf_counter() - t0, 2)}
        manifest["graphs"][name] = {"file": path.name, "inputs": sp["names"], "outputs": sp["outputs"],
                                    "dynamic_axes": {k: {str(a): v for a, v in ax.items()} for k, ax in sp["axes"].items()}}
        print(f"{name:13s} max|ort - torch| = {max(diffs):.2e}  ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    if args.step_batches:
        from lpp.export import ChunkCached, ChunkStepV2, exportable
        mx = exportable(m)                       # nn.RMSNorm -> ManualRMSNorm (no aten::rms_norm in ONNX)
        c = m.cfg
        hd, Tmax, NT = c.d_model // c.n_heads, c.max_positions, (c.canvas_size // 8) ** 2
        names = ["x", "valid", "cache_k", "cache_v", "cache_mask", "write_pos", "start_pos", "canvas_tokens", "draw"]
        outs = ["h", "cache_k_out", "cache_v_out", "cache_mask_out"]
        for R in args.step_batches:
            g = torch.Generator().manual_seed(R)

            def example(R, filled):
                cm = torch.zeros(R, Tmax, dtype=torch.bool)
                cm[:, :filled] = torch.rand(R, filled, generator=g) > 0.2     # holes, like padded text prefixes
                return (torch.randn(R, 1, c.d_model, generator=g), torch.rand(R, 1, generator=g) > 0.1,
                        torch.randn(c.n_layers, R, c.n_kv_heads, Tmax, hd, generator=g),
                        torch.randn(c.n_layers, R, c.n_kv_heads, Tmax, hd, generator=g), cm,
                        torch.tensor(filled, dtype=torch.long), torch.randint(0, filled, (R,), generator=g),
                        torch.randn(R, NT, c.d_model, generator=g), torch.rand(R, 1, generator=g) > 0.5)
            name, path = f"chunk_step_r{R}", args.out / f"chunk_step_r{R}.onnx"
            step = ChunkStepV2(mx).eval()
            torch.onnx.export(step, example(R, 40), str(path), input_names=names, output_names=outs,
                              opset_version=23, dynamo=False, custom_opsets={"com.microsoft": 1})
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            diffs = []
            for filled in (7, 300):                    # parity against the dynamic graph's module
                ins = example(R, filled)
                with torch.no_grad():
                    ref = ChunkCached(mx)(*ins)
                got = sess.run(None, {n: t.numpy() for n, t in zip(names, ins)})
                diffs.append(max(float(np.abs(a.astype(np.float64) - b.numpy().astype(np.float64)).max())
                                 for a, b in zip(got, ref)))
            report[name] = {"max_abs_diff_vs_chunk_cached": max(diffs), "bytes": path.stat().st_size, "rows": R}
            manifest["graphs"][name] = {"file": path.name, "inputs": names, "outputs": outs, "static_rows": R,
                                        "static_tmax": Tmax, "opset": 23}
            print(f"{name:13s} max|ort - chunk_cached| = {max(diffs):.2e}  ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
            # in-place variant (runtime v3): reads the cache, returns only the new k / v
            from lpp.export import ChunkStepV3
            name3, path3 = f"chunk_inplace_r{R}", args.out / f"chunk_inplace_r{R}.onnx"
            names3 = ["x", "cache_k", "cache_v", "cache_mask", "start_pos", "canvas_tokens", "draw"]
            pick = lambda t: (t[0], t[2], t[3], t[4], t[6], t[7], t[8])  # noqa: E731
            torch.onnx.export(ChunkStepV3(mx).eval(), pick(example(R, 40)), str(path3), input_names=names3,
                              output_names=["h", "k_new", "v_new"], opset_version=23, dynamo=False,
                              custom_opsets={"com.microsoft": 1})
            sess3 = ort.InferenceSession(str(path3), providers=["CPUExecutionProvider"])
            diffs3 = []
            for filled in (7, 300):
                ins = example(R, filled)
                with torch.no_grad():
                    h_ref, ck_ref, cv_ref, _ = ChunkCached(mx)(*ins)
                h3, kn, vn = sess3.run(None, {n: t.numpy() for n, t in zip(names3, pick(ins))})
                wp = filled
                diffs3.append(max(float(np.abs(h3 - h_ref.numpy()).max()),
                                  float(np.abs(kn - ck_ref[:, :, :, wp:wp + 1].numpy()).max()),
                                  float(np.abs(vn - cv_ref[:, :, :, wp:wp + 1].numpy()).max())))
            report[name3] = {"max_abs_diff_vs_chunk_cached": max(diffs3), "bytes": path3.stat().st_size, "rows": R}
            manifest["graphs"][name3] = {"file": path3.name, "inputs": names3, "outputs": ["h", "k_new", "v_new"],
                                         "static_rows": R, "static_tmax": Tmax, "opset": 23, "cache": "in-place"}
            print(f"{name3:13s} max|ort - chunk_cached| = {max(diffs3):.2e}  ({path3.stat().st_size / 1e6:.1f} MB)", flush=True)
            # static attribute heads (CUDA Graph capture in the runtime's cuda placement)
            name4, path4 = f"decode_heads_r{R}", args.out / f"decode_heads_r{R}.onnx"
            sp4 = specs(mx, B=R, T=1, P=1, Lt=1)["decode_heads"]
            torch.onnx.export(mods["decode_heads"], sp4["inputs"], str(path4), input_names=sp4["names"],
                              output_names=sp4["outputs"], opset_version=args.opset, dynamo=False)
            sess4 = ort.InferenceSession(str(path4), providers=["CPUExecutionProvider"])
            with torch.no_grad():
                ref4 = mods["decode_heads"](*sp4["inputs"])
            got4 = sess4.run(None, {n: t.numpy() for n, t in zip(sp4["names"], sp4["inputs"])})
            d4 = max(float(np.abs(a_ - b_.numpy()).max()) for a_, b_ in zip(got4, ref4))
            report[name4] = {"max_abs_diff": d4, "bytes": path4.stat().st_size, "rows": R}
            manifest["graphs"][name4] = {"file": path4.name, "inputs": sp4["names"], "outputs": sp4["outputs"],
                                         "static_rows": R}
            print(f"{name4:13s} max|ort - torch| = {d4:.2e}  ({path4.stat().st_size / 1e6:.1f} MB)", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "codec.json").write_text(json.dumps(codec_tables(codec), indent=None))
    (args.out / "parity.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
