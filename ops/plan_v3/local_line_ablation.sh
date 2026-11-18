#!/bin/bash
# Laptop GPU: line strokes P1 ablation (docs/line_strokes.md), 16 anime images, 300 steps each.
# Which part helps: contour-first budget (Cole et al.), the skeleton distance term (LIVE's UDF loss), joint
# fitting with the Gaussians (with a same-steps refit of the Gaussians alone as the control); then the piece
# length and the curve budget for the full combination.
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"
run() { local n=$1; shift
  [ -f out/line_ab/$n/line_probe.json ] || flock out/gpu.lock "$PY" -u tools/line_probe.py --seq data/seq/anime_v2_l6_t0/decomp_003.npz \
    --images data/anime_portraits256_v2_l6view --n 16 --steps 300 --out out/line_ab/$n "$@" > out/line_ab/$n.log 2>&1
  echo "$n $(grep '^mean' out/line_ab/$n.log | cut -c6-)"; }
mkdir -p out/line_ab
run v1q
run contour --contour-first
run udf --udf-weight 1e-4
run joint --joint
run all --joint --udf-weight 1e-4 --contour-first
for p in 10 20; do run all_piece$p --joint --udf-weight 1e-4 --contour-first --piece $p; done
for m in 100 300; do run all_max$m --joint --udf-weight 1e-4 --contour-first --max-curves $m; done
echo "line ablation finished"
