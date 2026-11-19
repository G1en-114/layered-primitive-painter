#!/bin/bash
# Laptop rehearsal of the formal v3 training: ops/v3_train/train.sh + eval_watch.sh with tiny settings, two DDP
# ranks sharing the one GPU (gloo), one face and one anime shard. Checks the whole chain (v2+v3 data mix,
# LP-FT freeze, DDP, snapshots, both-domain evaluation) before paying for the 5090s. Log: out/v3_dry/dry.log
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/v3_dry
mkdir -p $O
[ -f runs/base_v3_lines_init.pt ] || "$PY" tools/init_v3_from_v2.py --ckpt runs/base_v3_snap/model_final.pt \
  --out runs/base_v3_lines_init.pt
export ROOT=$PWD CODE=$PWD RUNS=$PWD/$O/runs EVALDIR=$PWD/$O/eval PY NGPU=2 LPP_DDP_BACKEND=gloo
export BATCH=2 ACCUM=1 WORKERS=1 STEPS_A=20 STEPS_B=40 SNAP=20 SAVE=20 EVAL=20 EVAL_GPU=0 EVAL_N=16 EVAL_BATCH=8
export FACES="data/seq/v2/decomp_000.npz" ANIME="data/seq/anime_v3_lines_full_t0/decomp_000.npz data/seq/anime_v3_lines_full_t0/decomp_048_rest.npz"
rm -rf $O/runs $O/eval
# stage A on two DDP ranks; stage B on one: two full base models with Adam state (~3.2 GB each) overflow 8 GB
flock out/gpu.lock env STEPS_B=0 bash ops/v3_train/train.sh > $O/dry.log 2>&1
flock out/gpu.lock env NGPU=1 BATCH=2 ACCUM=2 bash ops/v3_train/train.sh >> $O/dry.log 2>&1
flock out/gpu.lock bash ops/v3_train/eval_watch.sh >> $O/dry.log 2>&1
cat $O/eval/summary.txt >> $O/dry.log
