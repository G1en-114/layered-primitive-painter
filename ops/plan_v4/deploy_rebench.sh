#!/bin/bash
# Laptop: rerun every B = 8 / 32 measurement of the final v3 model's deployment run. The first run's "B = 8" and
# "B = 32" inputs had 6 rows (tools/ort_generate.py capped its prompt list at 6; fixed: prompts are cycled to --n), so
# those rows were all B = 6. B = 1 rows were right and are kept. Median of 3, 400 records per sample.
# Writes summary_median_v2.txt, placement_v2.txt, step_v2.txt next to the originals in out/deploy_v3.
set -u
cd "$(dirname "$0")/../.."
M=out/onnx/v3_final; O=out/deploy_v3; R=$O/rep3; mkdir -p $R
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/deploy.log; }
med3() {    # med3 TAG ONNX PROVIDER B
  local vals=() r
  for r in 1 2 3; do
    runtime/run.sh --onnx $2 --provider $3 --text $O/in_b$4/text.npy --text-valid $O/in_b$4/text_valid.npy \
      --uniforms $O/in_b$4/uniforms.npy --max-records 400 --out $R/$1_r$r > $R/$1_r$r.log 2>&1 || continue
    vals+=($(python3 -c "import json; t=json.load(open('$R/$1_r$r/timing.json')); print(round(t['records']/t['seconds']))"))
  done
  printf '%s\n' "${vals[@]}" | sort -n | sed -n 2p
}
exec 9> out/gpu.lock; flock 9
if [ ! -s $O/placement_v2.txt ]; then
  { echo "placement B records_per_s_median"; grep " 1 " $O/placement.txt
    for p in cpu cuda-all cuda cuda-hybrid cuda-auto; do for B in 8 32; do echo "$p $B $(med3 place_${p}_b$B $M $p $B)"; done; done
  } > $O/placement_v2.tmp && mv $O/placement_v2.tmp $O/placement_v2.txt; say "rebench: placements B=8/32 done"
fi
if [ ! -s $O/step_v2.txt ]; then
  { echo "variant B records_per_s_median"; grep " 1 " $O/step.txt
    echo "dynamic 8 $(med3 dyn_b8 $M cuda 8)"; echo "step+graph 8 $(med3 step_b8 ${M}_step cuda 8)"
  } > $O/step_v2.tmp && mv $O/step_v2.tmp $O/step_v2.txt; say "rebench: static step B=8 done"
fi
if [ ! -s $O/summary_median_v2.txt ]; then
  { echo "provider prec B rec_per_s_median"; grep " 1 " $O/summary_median.txt | awk '{print $1, $2, $3, $4}'
    for prov in cuda cpu; do
      precs="fp32 int8 int8w int4w"; [ $prov = cuda ] && precs="fp32 fp16 int8w int4w"
      for prec in $precs; do dir=$M; [ $prec != fp32 ] && dir=${M}_$prec
        for B in 8 32; do echo "$prov $prec $B $(med3 q_${prov}_${prec}_b$B $dir $prov $B)"; done; done
    done
  } > $O/summary_v2.tmp && mv $O/summary_v2.tmp $O/summary_median_v2.txt; say "rebench: precisions B=8/32 done"
fi
say "rebench finished"
