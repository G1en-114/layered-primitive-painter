#!/bin/bash
# Laptop: evaluate every snapshot ops/v3_train/fetch.sh brings back (runs/v3_formal), with the same watcher the box
# would run (eval_watch.sh: 192 face + 192 anime prompts, CFG 5 / top-p 0.6, compared with base_v3). Ends after
# fetch.sh has dropped the train_done / train_failed flag and the last snapshot is evaluated.
# Summary: out/v3_train/eval/summary.txt. Started by launch.sh fetch as the systemd --user unit lpp-v3-eval.
set -u
cd "$(dirname "$0")/../.."
export ROOT=$PWD CODE=$PWD RUNS=$PWD/runs/v3_formal EVALDIR=$PWD/out/v3_train/eval PY="$HOME/AI infra/.venv/bin/python"
export EVAL_GPU=0 EVAL_N=192 EVAL_BATCH=48
exec flock out/gpu.lock bash ops/v3_train/eval_watch.sh
