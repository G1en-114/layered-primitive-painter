#!/bin/bash
# Median of 3 runs per configuration (C++ runtime, 400 records max per sample), all precisions, B = 1 / 8 / 32.
set -u
M=$1; O=$2; cd "$(dirname "$0")/.."; mkdir -p $O/rep
echo "provider prec B rec_per_s_median runs" > $O/summary_median.txt
for prov in cpu cuda; do
  precs="fp32 int8 int8w int4w"; [ $prov = cuda ] && precs="fp32 fp16 int8w int4w"
  for prec in $precs; do
    dir=$M; [ $prec != fp32 ] && dir=${M}_$prec
    for B in 1 8 32; do
      vals=""
      for r in 1 2 3; do
        out=$O/rep/${prov}_${prec}_b${B}_r$r
        runtime/run.sh --onnx $dir --provider $prov --text $O/in_b$B/text.npy --text-valid $O/in_b$B/text_valid.npy \
          --uniforms $O/in_b$B/uniforms.npy --max-records 400 --out $out > $out.log 2>&1 || continue
        vals="$vals $(python3 -c "import json; t=json.load(open('$out/timing.json')); print(round(t['records']/t['seconds']))")"
      done
      med=$(echo $vals | tr ' ' '\n' | sort -n | sed -n 2p)
      echo "$prov $prec $B $med [$vals ]" | tee -a $O/summary_median.txt
    done
  done
done
