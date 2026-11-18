#!/bin/bash
# Takes over from new_box.sh's evaluation loop (GRPO v2 keeps training): the laptop GPU evaluates each GRPO
# snapshot (ops/plan_v3/local_grpo_eval.sh, ~3 min per 192-prompt eval vs ~10+ on the DCU, and the DCU is left
# to training) and uploads eval/<name>/metrics.json. Here: wait for each snapshot, give the laptop 45 minutes,
# evaluate on the DCU only if its metrics have not arrived. Then, after GRPO and the draft ablation, the sampling
# sweep for any tag the laptop has not already delivered (eval/sampling_sweep.txt).
set -u
R=/root/private_data; G=$R/runs/grpo_real; LOG=$R/new_box.log
SUM=$R/eval/grpo_real_summary.txt; SW=$R/eval/sampling_sweep.txt
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
say "new_box2 (laptop evaluates, DCU falls back after 45 min) takes over"
J=(--text-encoder $R/models/chinese-clip-vit-base-patch16 --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16
   --captions $R/captions_v3.jsonl --n 192 --batch 48)
FACE=(--data $R/celebamask256_val --split-manifest $R/celebamask256_split_v1.jsonl)
ANIME=(--data $R/anime_val192 --split-manifest $R/anime_val192/split_anime_val192.jsonl)
ev() {   # ev NAME CKPT face|anime [SAMPLING...]
  local n=$1 ck=$2 dom=$3; shift 3
  local -a d=("${FACE[@]}"); [ $dom = anime ] && d=("${ANIME[@]}")
  [ -f $R/eval/$n/metrics.json ] || python -u tools/eval_text.py --ckpt $ck --out $R/eval/$n "${J[@]}" "${d[@]}" "$@" \
    > $R/eval/$n.log 2>&1
}
line() {   # line LABEL FACE_EVAL ANIME_EVAL
  python3 - "$@" <<'PY'
import json, sys
name, face, anime = sys.argv[1:]
def part(n):
    j = json.load(open(f"/root/private_data/eval/{n}/metrics.json"))["judge"]["generated"]
    return f"R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"
print(f"{name:28s} | faces {part(face)} | anime {part(anime)}", flush=True)
PY
}
grpo_alive() { pgrep -f "tools/train_grpo.py" > /dev/null; }

for s in $(seq 25 25 200); do
  n=grpo_real_step$(printf %03d $s)
  grep -q "^$n " $SUM 2>/dev/null && continue
  f=$G/model_step$(printf %03d $s).pt
  while [ ! -f $f ]; do grpo_alive || break 2; sleep 60; done
  for i in $(seq 45); do [ -f $R/eval/$n/metrics.json ] && [ -f $R/eval/anime_$n/metrics.json ] && break; sleep 60; done
  if [ -f $R/eval/$n/metrics.json ] && [ -f $R/eval/anime_$n/metrics.json ]; then src=laptop; else src=DCU; fi
  ev $n $f face && ev anime_$n $f anime && { line $n $n anime_$n >> $SUM; say "evaluated step $s ($src)"; } || say "eval step $s FAILED"
done
while grpo_alive; do sleep 60; done
say "GRPO v2 exited ($(tail -n 1 $G/stdout.log | cut -c1-60))"

until grep -q "draft ablation finished\|FAILED" $R/draft_ablation.log 2>/dev/null && ! pgrep -f draft_ablation.sh > /dev/null; do sleep 120; done
say "draft ablation done; sampling sweep (tags the laptop has not delivered)"
Bf=$R/runs/base_v3/model_final.pt
sweep() {   # sweep TAG SAMPLING...
  local t=$1; shift
  grep -q "^$t " $SW 2>/dev/null && return 0
  ev sweep_${t}_face $Bf face "$@" && ev sweep_${t}_anime $Bf anime "$@" \
    && line $t sweep_${t}_face sweep_${t}_anime >> $SW || say "sweep $t FAILED"
}
for cfg in 1.5 2.0 3.0; do for tp in 0.8 0.95; do sweep base_cfg${cfg}_p${tp} --cfg-scale $cfg --top-p $tp; done; done
sweep base_cfg2.0_renyi4 --cfg-scale 2.0 --renyi-alpha 4
say "new_box finished"
