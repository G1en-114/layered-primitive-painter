#!/bin/bash
# Laptop GPU (RTX 5070, ~3 min per 192-prompt evaluation): the base_v3 sampling sweep that new_box.sh would
# otherwise run on the DCU (same settings, same tags), then prefix quality of base_v3 (layer order).
# Results: out/sweep/sampling_sweep.txt; copy the lines and eval dirs to the box and new_box.sh skips them.
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/sweep; SW=$O/sampling_sweep.txt; mkdir -p $O
J=(--text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48)
FACE=(--data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl)
ANIME=(--data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl)
B=runs/base_v3_snap/model_final.pt
ev() {   # ev NAME face|anime SAMPLING...
  local n=$1 dom=$2; shift 2
  local -a d=("${FACE[@]}"); [ $dom = anime ] && d=("${ANIME[@]}")
  [ -f $O/$n/metrics.json ] || "$PY" -u tools/eval_text.py --ckpt $B --out $O/$n "${J[@]}" "${d[@]}" "$@" > $O/$n.log 2>&1
}
sweep() {   # sweep TAG SAMPLING...
  local t=$1; shift
  grep -q "^$t " $SW 2>/dev/null && return 0
  ev sweep_${t}_face face "$@" && ev sweep_${t}_anime anime "$@" && "$PY" - $t $O >> $SW <<'PY'
import json, sys
t, o = sys.argv[1:]
def part(dom):
    j = json.load(open(f"{o}/sweep_{t}_{dom}/metrics.json"))["judge"]["generated"]
    return f"R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"
print(f"{t:28s} | faces {part('face')} | anime {part('anime')}", flush=True)
PY
}
for cfg in 1.5 2.0 3.0; do for tp in 0.8 0.95; do sweep base_cfg${cfg}_p${tp} --cfg-scale $cfg --top-p $tp; done; done
sweep base_cfg2.0_renyi4 --cfg-scale 2.0 --renyi-alpha 4
echo "sweep done"
[ -f $O/prefix_base_v3/prefix_quality.json ] || "$PY" -u tools/prefix_quality.py --ckpt $B --out $O/prefix_base_v3 "${J[@]}" "${FACE[@]}" \
  > $O/prefix_base_v3.log 2>&1 && echo "prefix quality done" || echo "prefix quality FAILED"
echo "local sweep finished"
