#!/bin/bash
# C++ runtime throughput for the quantized variants (no Python in the loop).
#   bench/runtime_quant_bench.sh out/onnx/grpo_attr2_s100 out/runtime_bench
# CPU: fp32 vs int8; CUDA: fp32 (TF32 off) vs fp16; batch 1 / 8 / 32; 400 records max per sample.
set -u
M=$1; O=$2; cd "$(dirname "$0")/.."
printf '%-6s %-5s %4s %10s %8s\n' provider prec B records/s seconds | tee $O/summary.txt
for prov in cpu cuda; do
  for prec in fp32 $([ $prov = cpu ] && echo int8 || echo fp16); do
    dir=$M; [ $prec != fp32 ] && dir=${M}_$prec
    for B in 1 8 32; do
      out=$O/${prov}_${prec}_b$B
      runtime/run.sh --onnx $dir --provider $prov --text $O/in_b$B/text.npy --text-valid $O/in_b$B/text_valid.npy \
        --uniforms $O/in_b$B/uniforms.npy --max-records 400 --out $out > $out.log 2>&1 || { echo "$prov $prec B=$B FAILED"; continue; }
      python3 -c "import json; t=json.load(open('$out/timing.json')); print(f\"%-6s %-5s %4d %10.0f %8.1f\" % ('$prov', '$prec', $B, t['records']/t['seconds'], t['seconds']))" | tee -a $O/summary.txt
    done
  done
done
