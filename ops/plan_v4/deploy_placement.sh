#!/bin/bash
# Laptop, after ops/plan_v4/deploy_final.sh: where the final v3 model's graphs should run, and what the decode step
# costs (docs/quant_report.md did this for the small model; base is 6x larger). Median of 3, 400 records per sample.
#   7 placements: cpu | cuda-all | cuda (backbone + heads on GPU) | cuda-hybrid (backbone only) | cuda-auto
#   8 static one-step graphs + CUDA Graph (export --step-batches 2 16: rows = 2 x images with CFG; 64 rows ran
#     the 14 GB laptop out of memory during export) vs the dynamic chunk_cached path, cuda-auto
#   9 per-graph time at B=1 (LPP_PROFILE=1) for cuda-hybrid and cuda-all
# Summaries: out/deploy_v3/placement.txt, step.txt, profile_*.log
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; C=runs/v3_formal/B_ft/model_step012000.pt; M=out/onnx/v3_final; O=out/deploy_v3
R=$O/rep2; mkdir -p $R
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/deploy.log; }
med3() {    # med3 TAG ONNX PROVIDER B -> median records/s of 3 runs (empty if all failed)
  local vals=() r
  for r in 1 2 3; do
    runtime/run.sh --onnx $2 --provider $3 --text $O/in_b$4/text.npy --text-valid $O/in_b$4/text_valid.npy \
      --uniforms $O/in_b$4/uniforms.npy --max-records 400 --out $R/$1_r$r > $R/$1_r$r.log 2>&1 || continue
    vals+=($(python3 -c "import json; t=json.load(open('$R/$1_r$r/timing.json')); print(round(t['records']/t['seconds']))"))
  done
  printf '%s\n' "${vals[@]}" | sort -n | sed -n 2p
}
exec 9> out/gpu.lock; flock 9                    # the whole script holds the GPU
if [ ! -s $O/placement.txt ]; then
  echo "placement B records_per_s_median" > $O/placement.tmp
  for p in cpu cuda-all cuda cuda-hybrid cuda-auto; do
    for B in 1 8 32; do echo "$p $B $(med3 place_${p}_b$B $M $p $B)" >> $O/placement.tmp; done
  done
  mv $O/placement.tmp $O/placement.txt; say "7 placements done"
fi
if [ ! -s $O/step.txt ]; then
  [ -f ${M}_step/manifest.json ] || "$PY" tools/export_onnx.py --ckpt $C --out ${M}_step --step-batches 2 16 \
    > $O/export_step.log 2>&1 || say "8 step export FAILED (export_step.log)"
  echo "variant B records_per_s_median" > $O/step.tmp
  for B in 1 8; do
    echo "dynamic $B $(med3 dyn_b$B $M cuda-auto $B)" >> $O/step.tmp
    echo "step+graph $B $(med3 step_b$B ${M}_step cuda-auto $B)" >> $O/step.tmp
  done
  mv $O/step.tmp $O/step.txt; say "8 static step graphs done"
fi
for p in cuda-hybrid cuda-all; do
  [ -s $O/profile_$p.log ] || LPP_PROFILE=1 runtime/run.sh --onnx $M --provider $p --text $O/in_b1/text.npy \
    --text-valid $O/in_b1/text_valid.npy --uniforms $O/in_b1/uniforms.npy --max-records 400 --out $R/profile_$p \
    > $O/profile_$p.log 2>&1
done
say "9 profiles done; placement run finished"
