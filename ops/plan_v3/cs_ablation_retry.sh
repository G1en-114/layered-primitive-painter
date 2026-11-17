#!/bin/bash
# Retry of plan11's ablation (the first run failed: transformers missing in /root/venv).
# Results land in /root/plan11_results, which plan11 pushes to SCNet at its stop time.
set -u
ulimit -c 0
L=/root/lpp; D=$L/data; SEQ=$D/seq; LOG=/root/plan11.log; RES=/root/plan11_results
PY=/root/venv/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
export LPP_RASTER=triton OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $L
mkdir -p $RES
ablation() {
  local A=$D/abl
  local tp=()
  for name in 4 12; do
    O=/root/runs/abl_small_${name}traj; mkdir -p $O
    $PY -u tools/train.py --data "$SEQ/abl$name/*/decomp_*.npz" --protocol $L/protocol.json --out $O --config small \
      --batch 16 --accum 2 --steps 12000 --lr 3e-4 --min-lr 3e-5 --warmup 500 --workers 3 --seed 0 \
      --captions $A/captions_clean_v1.jsonl --attributes $A/celebamask256/attributes.json \
      --split-manifest $A/celebamask256_split_v1.jsonl --text-encoder $A/chinese-clip-vit-base-patch16 \
      --text-drop 0.1 --cfg-scale 2.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 \
      --eval-every 2000 --save-every 1000 > $O/train.stdout.log 2>&1 &
    tp+=($!)
    sleep 5
  done
  say "ablation training started (small, 12000 steps, 4 vs 12 trajectories)"
  for q in "${tp[@]}"; do wait $q; done
  for name in 4 12; do
    O=/root/runs/abl_small_${name}traj
    $PY -c "import torch; s=torch.load('$O/last.pt',map_location='cpu',weights_only=False); print('step', s['step'])
torch.save({k:s[k] for k in ('model','step','config','protocol')},'$O/model_final.pt')" >> $LOG 2>&1 || { say "abl $name: no checkpoint"; continue; }
    $PY -u tools/eval_text.py --ckpt $O/model_final.pt --out $O/eval --text-encoder $A/chinese-clip-vit-base-patch16 \
      --judge-encoder $A/chinese-clip-vit-large-patch14-fp16 --attr-classifier $A/models/attr_resnet18.pt \
      --attributes $A/celebamask256/attributes.json --data $D/celebamask256 --captions $A/captions_clean_v1.jsonl \
      --split-manifest $A/celebamask256_split_v1.jsonl --n 192 --batch 48 > $O/eval.log 2>&1 && say "abl $name evaluated" || say "abl $name eval FAILED"
    mkdir -p $RES/abl_small_${name}traj
    cp $O/log.jsonl $O/model_final.pt $O/samples_*.png $RES/abl_small_${name}traj/ 2>/dev/null
    cp -r $O/eval/metrics.json $O/eval/*.png $RES/abl_small_${name}traj/ 2>/dev/null
  done
  say "ablation done"
}


rm -rf /root/runs/abl_small_4traj /root/runs/abl_small_12traj   # the failed first attempt (no checkpoint)
say "ablation retry started"
ablation
