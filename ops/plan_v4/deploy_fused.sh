#!/bin/bash
# Laptop, after deploy_rebench.sh: the fused attribute sampler (decode_sample) on the final v3 model.
#   1 export with decode_sample (out/onnx/v3_final_fused; the other graphs are the same as v3_final)
#   2 parity: the C++ runtime with and without the fused graph (LPP_FUSED=0), same uniforms, B = 8 -> identical records?
#   3 throughput, median of 3, 400 records: cuda and cpu, B = 1 / 8 / 32, fused vs unfused
# Summary: out/deploy_v3/fused.txt
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; C=runs/v3_formal/B_ft/model_step012000.pt; F=out/onnx/v3_final_fused; O=out/deploy_v3
R=$O/rep4; mkdir -p $R
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/deploy.log; }
run() {     # run OUT ONNX PROVIDER B [env...]
  local out=$1 onnx=$2 prov=$3 B=$4; shift 4
  env "$@" runtime/run.sh --onnx $onnx --provider $prov --text $O/in_b$B/text.npy --text-valid $O/in_b$B/text_valid.npy \
    --uniforms $O/in_b$B/uniforms.npy --max-records 400 --cfg-scale 5 --top-p 0.6 --out $out > $out.log 2>&1
}
med3() {    # med3 TAG PROVIDER B [env...]
  local tag=$1 prov=$2 B=$3 vals=() r; shift 3
  for r in 1 2 3; do
    run $R/${tag}_r$r $F $prov $B "$@" || continue
    vals+=($(python3 -c "import json; t=json.load(open('$R/${tag}_r$r/timing.json')); print(round(t['records']/t['seconds']))"))
  done
  printf '%s\n' "${vals[@]}" | sort -n | sed -n 2p
}
exec 9> out/gpu.lock; flock 9
[ -f $F/manifest.json ] || "$PY" tools/export_onnx.py --ckpt $C --out $F > $O/export_fused.log 2>&1 \
  || { say "fused export FAILED (export_fused.log)"; exit 1; }
say "fused: exported, decode_sample $(grep decode_sample $O/export_fused.log | head -1)"
for prov in cpu cuda; do
  run $R/par_${prov}_fused $F $prov 8; run $R/par_${prov}_plain $F $prov 8 LPP_FUSED=0
  say "fused parity $prov B=8: $("$PY" -c "
import numpy as np
a = np.load('$R/par_${prov}_fused/records.npy'); b = np.load('$R/par_${prov}_plain/records.npy')
n = min(a.shape[1], b.shape[1]); same = [bool((a[i, :n] == b[i, :n]).all()) for i in range(a.shape[0])]
first = [int(np.argmax((a[i, :n] != b[i, :n]).any(-1))) if not s else None for i, s in enumerate(same)]
print(f'{sum(same)}/{len(same)} sequences identical; first divergence {first}')")"
done
{ echo "provider B fused unfused"
  for prov in cuda cpu; do for B in 1 8 32; do
    echo "$prov $B $(med3 f_${prov}_b$B $prov $B) $(med3 u_${prov}_b$B $prov $B LPP_FUSED=0)"
  done; done; } > $O/fused.tmp && mv $O/fused.tmp $O/fused.txt
say "fused benchmark finished"
