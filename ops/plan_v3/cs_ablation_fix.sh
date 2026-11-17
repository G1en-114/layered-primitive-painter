#!/bin/bash
# 2026-11-17: abl12 hit CUDA OOM next to abl4 (9.2 GB) and three extra-face decompositions (3 x 3.55 GB).
# Pause the extra-face loop (SIGSTOP keeps plan11's bookkeeping intact), free its GPU memory, retrain
# abl12, evaluate both runs into /root/plan11_results, then resume the extra-face loop (SIGCONT).
set -u
ulimit -c 0
L=/root/lpp; D=$L/data; SEQ=$D/seq; LOG=/root/plan11.log; RES=/root/plan11_results; A=$D/abl
PY=/root/venv/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
export LPP_RASTER=triton OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $L
# 1. the retry script would evaluate a half-trained abl12: stop it (its abl4 training child keeps running)
for p in $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /cs_ablation_retry.sh$/ {print $1}'); do kill $p; done
# 2. pause the extra-face loop (parent of the decompositions) and free the GPU memory they hold
LOOP=""
for d in $(ps -eo pid,args | awk '/[d]ecompose_dataset.py/ && /celebamask256/ {print $1}'); do
  LOOP=$(ps -o ppid= -p $d | tr -d ' ')
done
[ -n "$LOOP" ] && kill -STOP $LOOP && echo $LOOP > /root/.lpp/extra_loop_pid
for d in $(ps -eo pid,args | awk '/[d]ecompose_dataset.py/ && /celebamask256/ {print $1}'); do kill $d; done
sleep 15
say "ablation fix: extra-face loop $LOOP paused, its decompositions stopped (finished shards are kept)"
train() {   # train N
  local O=/root/runs/abl_small_${1}traj
  rm -rf $O; mkdir -p $O
  $PY -u tools/train.py --data "$SEQ/abl$1/*/decomp_*.npz" --protocol $L/protocol.json --out $O --config small \
    --batch 16 --accum 2 --steps 12000 --lr 3e-4 --min-lr 3e-5 --warmup 500 --workers 3 --seed 0 \
    --captions $A/captions_clean_v1.jsonl --attributes $A/celebamask256/attributes.json \
    --split-manifest $A/celebamask256_split_v1.jsonl --text-encoder $A/chinese-clip-vit-base-patch16 \
    --text-drop 0.1 --cfg-scale 2.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 \
    --eval-every 2000 --save-every 1000 > $O/train.stdout.log 2>&1
}
evaluate() {   # evaluate N
  local O=/root/runs/abl_small_${1}traj
  $PY -c "import torch; s=torch.load('$O/last.pt',map_location='cpu',weights_only=False); assert s['step'] == 12000, s['step']
torch.save({k:s[k] for k in ('model','step','config','protocol')},'$O/model_final.pt')" >> $LOG 2>&1 || { say "abl $1: training did not finish"; return; }
  $PY -u tools/eval_text.py --ckpt $O/model_final.pt --out $O/eval --text-encoder $A/chinese-clip-vit-base-patch16 \
    --judge-encoder $A/chinese-clip-vit-large-patch14-fp16 --attr-classifier $A/models/attr_resnet18.pt \
    --attributes $A/celebamask256/attributes.json --data $D/celebamask256 --captions $A/captions_clean_v1.jsonl \
    --split-manifest $A/celebamask256_split_v1.jsonl --n 192 --batch 48 > $O/eval.log 2>&1 && say "abl $1 evaluated" || say "abl $1 eval FAILED"
  mkdir -p $RES/abl_small_${1}traj
  cp $O/log.jsonl $O/model_final.pt $O/samples_*.png $O/eval/metrics.json $O/eval/*.png $RES/abl_small_${1}traj/ 2>/dev/null
}
# 3. abl12 now (abl4 keeps training); evaluate each as it finishes
train 12 &
T12=$!
say "abl12 restarted (pid $T12)"
A4=$(ps -eo pid,args | awk '/[t]rain.py/ && /abl4/ {print $1}' | head -1)
while [ -n "$A4" ] && kill -0 $A4 2>/dev/null; do sleep 60; done
evaluate 4
wait $T12
evaluate 12
say "ablation done (both runs)"
# 4. resume the extra-face decomposition with the GPU free again
[ -n "$LOOP" ] && kill -CONT $LOOP && say "extra-face loop resumed"
