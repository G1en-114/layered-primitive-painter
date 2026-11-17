#!/bin/bash
# Static one-step graphs + CUDA Graph vs the dynamic chunk_cached path (C++ runtime, cuda-auto placement).
# Median of 3 runs; 400 records max per sample. Run with bash (median via sort of space-free lines).
set -u
cd "$(dirname "$0")/.."
O=out/runtime_bench; S=$O/step; mkdir -p $S
for B in 1 8 32; do
  for variant in old:out/onnx/grpo_attr2_s100 step:out/onnx/grpo_attr2_s100_step; do
    name=${variant%%:*}; dir=${variant#*:}; vals=()
    for r in 1 2 3; do
      runtime/run.sh --onnx $dir --provider cuda-auto --text $O/in_b$B/text.npy --text-valid $O/in_b$B/text_valid.npy \
        --uniforms $O/in_b$B/uniforms.npy --max-records 400 --out $S/${name}_b${B}_r$r > $S/${name}_b${B}_r$r.log 2>&1 || continue
      vals+=($(python3 -c "import json; t=json.load(open('$S/${name}_b${B}_r$r/timing.json')); print(round(t['records']/t['seconds']))"))
    done
    med=$(printf '%s\n' "${vals[@]}" | sort -n | sed -n 2p)
    echo "B=$B $name median $med [${vals[*]}]"
  done
done | tee $S/summary.txt
