"""Quantize an exported Painter (tools/export_onnx.py output) for deployment.

    python tools/quantize_onnx.py --src out/onnx/grpo_attr2_s100 --out out/onnx/grpo_attr2_s100_int8 --mode int8
    python tools/quantize_onnx.py --src out/onnx/grpo_attr2_s100 --out out/onnx/grpo_attr2_s100_fp16 --mode fp16

fp16 : every graph's weights and arithmetic in float16, graph inputs/outputs kept float32 (the C++ runtime
       and tools/ort_generate.py load the directory unchanged). For the CUDA execution provider.
int4w / int8w : weight-only, blockwise (--block-size) symmetric int4 / int8 weights in MatMulNBits nodes,
       dequantized on the fly; activations stay float. For memory-bound one-token decode steps.
int8 : dynamic quantization of MatMul/Gemm whose B operand is a weight (per-channel int8 weights, activations
       quantized on the fly per tensor); attention's activation x activation MatMuls stay float32.
       For the CPU execution provider (VNNI / AVX-512 or AVX2 int8 kernels).
Both write the same manifest.json / codec.json, plus quantization.json with sizes per graph.
Quality cannot be judged by token parity (sampling diverges after the first changed draw): use
tools/quant_eval.py for distribution-level comparisons.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["fp16", "int8", "int8w", "int4w"], required=True)
    ap.add_argument("--block-size", type=int, default=32, help="int4w / int8w: weights per quantization block")
    ap.add_argument("--accuracy-level", type=int, default=None,
                    help="int4w / int8w MatMulNBits compute type (None: float; 4: int8 activations on CPU)")
    ap.add_argument("--skip", nargs="*", default=["canvas_enc", "embed_text"],
                    help="graphs left in float32 (small; precision-sensitive inputs)")
    args = ap.parse_args()
    import onnx
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.src / "manifest.json").read_text())
    report = {"mode": args.mode, "source": str(args.src), "graphs": {}}
    for name, g in manifest["graphs"].items():
        src, dst = args.src / g["file"], args.out / g["file"]
        before = src.stat().st_size
        if name in args.skip:
            shutil.copy(src, dst)
        elif args.mode == "fp16":
            from onnxruntime.transformers.float16 import convert_float_to_float16
            model = onnx.load(str(src))
            onnx.save(convert_float_to_float16(model, keep_io_types=True), str(dst))
        elif args.mode in ("int4w", "int8w"):
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            q = MatMulNBitsQuantizer(onnx.load(str(src)), bits=4 if args.mode == "int4w" else 8,
                                     block_size=args.block_size, is_symmetric=True,
                                     accuracy_level=args.accuracy_level, op_types_to_quantize=("MatMul",))
            q.process()
            q.model.save_model_to_file(str(dst), use_external_data_format=False)
        else:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                             op_types_to_quantize=["MatMul", "Gemm"], extra_options={"MatMulConstBOnly": True})
        report["graphs"][name] = {"converted": name not in args.skip, "mb_before": round(before / 2**20, 2),
                                  "mb_after": round(dst.stat().st_size / 2**20, 2)}
        print(name, report["graphs"][name], flush=True)
    for f in ("manifest.json", "codec.json"):
        shutil.copy(args.src / f, args.out / f)
    (args.out / "quantization.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
