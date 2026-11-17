#!/bin/bash
# C++ runtime throughput of the weight-only variants (MatMulNBits), appended to out/runtime_bench/summary.txt
set -u
M=$1; O=$2; cd "$(dirname "$0")/.."
for prov in cpu cuda; do for prec in int8w int4w; do for B in 1 8 32; do
  out=$O/${prov}_${prec}_b$B
  runtime/run.sh --onnx ${M}_$prec --provider $prov --text $O/in_b$B/text.npy --text-valid $O/in_b$B/text_valid.npy \
    --uniforms $O/in_b$B/uniforms.npy --max-records 400 --out $out > $out.log 2>&1 || { echo "$prov $prec B=$B FAILED" | tee -a $O/summary.txt; continue; }
  python3 -c "import json; t=json.load(open('$out/timing.json')); print(f\"%-6s %-5s %4d %10.0f %8.1f\" % ('$prov', '$prec', $B, t['records']/t['seconds'], t['seconds']))" | tee -a $O/summary.txt
done; done; done
