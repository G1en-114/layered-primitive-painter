#!/bin/bash
# New SCNet box (2026-11-18, new account, Kunshan; ~11 card-hours of balance). The old box's file storage
# (everything but the Qwen captioner) was copied over directly (old -> new rsync, sizes + SHA spot-checked).
#  1. GRPO v2 (grpo_real) resumes from its step-25 checkpoint (the old box stopped at step 49): to 200 steps,
#     faces + anime evaluated every 25 steps -> eval/grpo_real_summary.txt
#  2. in parallel: the draft-order ablation (draft_ablation.sh; it died on FlashAttention before, fixed)
#  3. when both are done: sampling sweep on base_v3 (CFG x top-p, Renyi), faces + anime -> eval/sampling_sweep.txt
# Writes "new_box finished" to $R/new_box.log at the end: the card can be stopped then.
set -u
R=/root/private_data; B=$R/runs/base_v3; G=$R/runs/grpo_real; LOG=$R/new_box.log
SUM=$R/eval/grpo_real_summary.txt; SW=$R/eval/sampling_sweep.txt
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
mkdir -p $R/eval
say "start"

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

setsid bash $R/lpp_dev/ops/plan_v3/draft_ablation.sh > $R/draft_ablation.out 2>&1 < /dev/null &
say "draft ablation started"

if ! grep -q "GRPO v2 exited" $LOG 2>/dev/null; then
  SRC=(); [ -f $R/source_emb_b16.npz.complete ] && SRC=(--source-emb $R/source_emb_b16.npz --w-src 1)
  python -u tools/train_grpo.py --init $B/model_final.pt --out $G --resume \
    --text-encoder $R/models/chinese-clip-vit-base-patch16 --captions $R/captions_v3.jsonl \
    --split-manifest $R/split_v3_faces_anime.jsonl --anime-frac 0.5 "${SRC[@]}" \
    --attr-classifier $R/models/attr_resnet18.pt --attributes $R/celebamask256/attributes.json \
    --adv-mode per-component --w-clip 1 --w-attr 1 --w-orphan 0.5 \
    --steps 200 --prompts 8 --group 8 --micro-batch 16 --lr 1e-5 --beta 0.04 \
    --sample-every 25 --save-every 25 --keep-snapshots >> $G/stdout.log 2>&1 < /dev/null &
  GP=$!
  say "GRPO v2 resumed (pid $GP)"
  for s in $(seq 25 25 200); do
    n=grpo_real_step$(printf %03d $s)
    grep -q "^$n " $SUM 2>/dev/null && continue    # evaluated already (step 25, on the old box)
    f=$G/model_step$(printf %03d $s).pt
    while [ ! -f $f ]; do kill -0 $GP 2>/dev/null || break 2; sleep 60; done
    sleep 30                                     # let the snapshot finish writing
    ev $n $f face && ev anime_$n $f anime && { line $n $n anime_$n >> $SUM; say "evaluated step $s"; } || say "eval step $s FAILED"
  done
  wait $GP; say "GRPO v2 exited ($?)"
fi

until grep -q "draft ablation finished\|FAILED" $R/draft_ablation.log 2>/dev/null && ! pgrep -f draft_ablation.sh > /dev/null; do sleep 120; done
say "draft ablation done; sampling sweep starting"
Bf=$B/model_final.pt
sweep() {   # sweep TAG CKPT SAMPLING...
  local t=$1 ck=$2; shift 2
  grep -q "^$t " $SW 2>/dev/null && return 0
  ev sweep_${t}_face $ck face "$@" && ev sweep_${t}_anime $ck anime "$@" \
    && line $t sweep_${t}_face sweep_${t}_anime >> $SW || say "sweep $t FAILED"
}
for cfg in 1.5 2.0 3.0; do for tp in 0.8 0.95; do sweep base_cfg${cfg}_p${tp} $Bf --cfg-scale $cfg --top-p $tp; done; done
sweep base_cfg2.0_renyi4 $Bf --cfg-scale 2.0 --renyi-alpha 4
say "base sweep done"
say "new_box finished"
