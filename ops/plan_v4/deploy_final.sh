#!/bin/bash
# Laptop: plan v4 deployment main line on the final v3 model (base, 126M, protocol v3 with lines), the same toolchain
# as docs/quant_report.md (small, 20M). Answers that report's open question: does weight-only quantization pay off
# once the model is base-sized? Steps (each skipped when its output exists; GPU steps wait on out/gpu.lock):
#   1 export to ONNX + per-graph parity (tools/export_onnx.py)                       CPU
#   2 sampler parity PyTorch vs ORT, same uniforms (tools/ort_generate.py --compare) CPU
#   3 quantized variants fp16 / int8 / int8w / int4w (tools/quantize_onnx.py)        CPU
#   4 benchmark inputs for B = 1 / 8 / 32 (ort_generate --dump)                      CPU
#   5 C++ runtime throughput, median of 3 (bench/runtime_quant_repeat.sh)           GPU + CPU
#   6 quality per variant (tools/quant_eval.py)                                      GPU
# Log: out/deploy_v3/deploy.log
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; C=runs/v3_formal/B_ft/model_step012000.pt; M=out/onnx/v3_final; O=out/deploy_v3
TE=data/models/chinese-clip-vit-base-patch16
mkdir -p $O
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/deploy.log; }
export PYTHONUNBUFFERED=1
if [ ! -f $M/manifest.json ]; then
  "$PY" tools/export_onnx.py --ckpt $C --out $M > $O/export.log 2>&1 && say "1 exported: $(grep -ci 'max' $O/export.log) parity lines (see export.log)" \
    || { say "1 export FAILED (export.log)"; exit 1; }
fi
if [ ! -f $O/parity.done ]; then
  "$PY" tools/ort_generate.py --onnx $M --ckpt $C --text-encoder $TE --n 4 --max-records 400 --cfg-scale 5.0 --compare \
    > $O/parity.log 2>&1 && touch $O/parity.done && say "2 sampler parity: $(tail -1 $O/parity.log)" \
    || say "2 sampler parity FAILED (parity.log)"
fi
for m in fp16 int8 int8w int4w; do
  [ -f ${M}_$m/manifest.json ] || { "$PY" tools/quantize_onnx.py --src $M --out ${M}_$m --mode $m > $O/quant_$m.log 2>&1 \
    && say "3 quantized $m: $(du -sh ${M}_$m | cut -f1)" || say "3 quantize $m FAILED"; }
done
for B in 1 8 32; do
  [ -f $O/in_b$B/uniforms.npy ] || "$PY" tools/ort_generate.py --onnx $M --ckpt $C --text-encoder $TE --n $B \
    --max-records 2 --cfg-scale 5.0 --dump $O/in_b$B > $O/dump_b$B.log 2>&1 || say "4 inputs B=$B FAILED"
done
[ -f $O/summary_median.txt ] && [ $(wc -l < $O/summary_median.txt) -ge 25 ] \
  || { flock out/gpu.lock bash bench/runtime_quant_repeat.sh $M $O > $O/bench.log 2>&1; say "5 C++ throughput done"; }
if [ ! -f $O/quant_eval/summary.json ]; then
  flock out/gpu.lock "$PY" tools/quant_eval.py --text-encoder $TE --captions data/captions/captions_v3.jsonl \
    --split-manifest data/split_v3_faces_anime.jsonl --n 48 --capture 40 --cfg-scale 5.0 --out $O/quant_eval \
    --variant fp32-cpu=$M:CPU --variant int8-cpu=${M}_int8:CPU --variant int4w-cpu=${M}_int4w:CPU \
    --variant fp32-cuda=$M:CUDA --variant fp16-cuda=${M}_fp16:CUDA --variant int8w-cuda=${M}_int8w:CUDA \
    > $O/quant_eval.log 2>&1 && say "6 quality per variant done" || say "6 quant_eval FAILED (quant_eval.log)"
fi
say "deployment run finished"
