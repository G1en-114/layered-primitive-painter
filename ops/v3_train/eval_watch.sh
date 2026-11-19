#!/bin/bash
# Runs beside ops/v3_train/train.sh: every model snapshot (A_lp final, B_ft every SNAP steps) is evaluated on
# the 192 face and 192 anime validation prompts at the chosen sampling (CFG 5, top-p 0.6), one summary line per
# snapshot in EVAL/summary.txt, compared with base_v3 at the same sampling. A face score clearly below base_v3
# (forgetting) is flagged ALERT; training is not stopped, every snapshot is kept, the best one is picked after.
# Exits once training has ended and every snapshot is evaluated.
set -u
ROOT=${ROOT:-/root/lpp/v3train}; CODE=${CODE:-$ROOT/code}; RUNS=${RUNS:-$ROOT/runs}; EVAL=${EVALDIR:-$ROOT/eval}
PY=${PY:-python}; EVAL_GPU=${EVAL_GPU:-1}; N=${EVAL_N:-192}; EB=${EVAL_BATCH:-48}
D=$ROOT/data; S=$EVAL/summary.txt
mkdir -p $EVAL
J=(--text-encoder $D/models/chinese-clip-vit-base-patch16 --judge-encoder $D/models/chinese-clip-vit-large-patch14-fp16
   --captions $D/captions/captions_v3.jsonl --n $N --cfg-scale 5.0 --top-p 0.6)
FACE=(--data $D/celebamask256_val --split-manifest $D/celebamask256_split_v1.jsonl)
ANIME=(--data $D/anime_val192 --split-manifest $D/anime_val192/split_anime_val192.jsonl)
[ -f $S ] || printf '%-22s | %s\n' "base_v3 (reference)" \
  "faces R@1 0.172 R@5 0.380 FD 0.373 | anime R@1 0.115 R@5 0.375 FD 0.227" > $S
ev() {      # ev NAME CKPT face|anime: retried once at half the batch (the GPU is shared with training)
  local out=$EVAL/$1_$3 b
  local -a d=("${FACE[@]}"); [ $3 = anime ] && d=("${ANIME[@]}")
  for b in $EB $((EB / 2)); do
    [ -f $out/metrics.json ] && return 0
    (cd $CODE && CUDA_VISIBLE_DEVICES=$EVAL_GPU "$PY" -u tools/eval_text.py --ckpt $2 --out $out "${J[@]}" \
      --batch $b "${d[@]}") > $out.log 2>&1
  done
  [ -f $out/metrics.json ]
}
line() {    # line NAME
  "$PY" - "$EVAL" "$1" >> $S <<'PY'
import json, sys
ev, name = sys.argv[1], sys.argv[2]
g = {d: json.load(open(f"{ev}/{name}_{d}/metrics.json"))["judge"]["generated"] for d in ("face", "anime")}
part = lambda j: f"R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"  # noqa: E731
# base_v3 faces: R@5 0.380, FD 0.373; one standard error of R@5 on 192 prompts is ~0.035
f = g["face"]
alert = "  ALERT faces below base_v3" if f["r_precision@5"] < 0.31 or f["clip_fd"] > 0.40 else ""
print(f"{name:22s} | faces {part(f)} | anime {part(g['anime'])}{alert}", flush=True)
PY
}
while true; do
  ended=0; { [ -f $RUNS/train_done ] || [ -f $RUNS/train_failed ]; } && ended=1
  for f in $(ls $RUNS/A_lp/model_step*.pt $RUNS/B_ft/model_step*.pt 2>/dev/null); do
    name=$(basename $(dirname $f))_$(basename $f .pt | sed 's/model_//')
    { grep -q "^$name " $S || [ -f $EVAL/$name.failed ]; } && continue
    if ev $name $f face && ev $name $f anime; then line $name; else touch $EVAL/$name.failed
      echo "$(date -u +%FT%TZ) eval of $name failed (see $EVAL/${name}_*.log)" >> $EVAL/eval_watch.log; fi
  done
  [ $ended = 1 ] && break
  sleep 120
done
echo "$(date -u +%FT%TZ) eval watcher finished" >> $EVAL/eval_watch.log
