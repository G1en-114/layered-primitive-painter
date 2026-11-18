#!/bin/bash
# Laptop GPU, after local_sweep.sh: can less sampling randomness clean up the paintings? Starting from the best
# setting so far (CFG 3.0, top-p 0.8): lower temperature, tighter top-p, higher CFG, and cooler positions only.
# Shares the GPU with local_grpo_eval.sh through out/gpu.lock. Lines go to out/sweep/sampling_sweep.txt.
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/sweep; SW=$O/sampling_sweep.txt
J=(--text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48)
FACE=(--data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl)
ANIME=(--data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl)
B=runs/base_v3_snap/model_final.pt
ev() {
  local n=$1 dom=$2; shift 2
  local -a d=("${FACE[@]}"); [ $dom = anime ] && d=("${ANIME[@]}")
  [ -f $O/$n/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py --ckpt $B --out $O/$n "${J[@]}" "${d[@]}" "$@" > $O/$n.log 2>&1
}
sweep() {
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
until grep -q "local sweep finished" $O/local_sweep.out 2>/dev/null; do sleep 30; done
sweep base_cfg3.0_p0.8_t0.8 --cfg-scale 3.0 --top-p 0.8 --temperature 0.8
sweep base_cfg3.0_p0.8_t0.6 --cfg-scale 3.0 --top-p 0.8 --temperature 0.6
sweep base_cfg3.0_p0.6 --cfg-scale 3.0 --top-p 0.6
sweep base_cfg4.0_p0.8 --cfg-scale 4.0 --top-p 0.8
sweep base_cfg3.0_p0.8_xy0.6 --cfg-scale 3.0 --top-p 0.8 --attr-temperature x=0.6,y=0.6
echo "local sweep2 finished"
