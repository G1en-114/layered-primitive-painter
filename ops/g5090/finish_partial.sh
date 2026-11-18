#!/bin/bash
# Laptop, after the 5090 voucher: the last 5090 batch (shards 48-59) took only the first ~365 images of each shard.
# Once the laptop's own conversion (shards 60-106) is done, convert the rest of those shards (--skip N), written as
# decomp_NNN_rest.npz next to the partial outputs (load_shards reads both).
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; D=data/seq/anime_v3_lines_full_t0
until grep -q "decomp_106.npz: written" out/decompose_laptop_t0_rest.log 2>/dev/null; do sleep 300; done
for i in $(seq 48 59); do
  s=$(printf "decomp_%03d" $i); [ -f $D/$s.npz ] || { echo "missing $s.npz (not fetched)"; continue; }
  [ -f $D/${s}_rest.npz ] && continue
  done_n=$("$PY" -c "import numpy as np; print(len(np.load('$D/$s.npz')['ids']))")
  total=$("$PY" -c "import numpy as np; print(len(np.load('data/seq/anime_v2_l6_t0/$s.npz')['ids']))")
  [ "$done_n" -lt "$total" ] || continue
  flock out/gpu.lock "$PY" -u tools/decompose_lines.py --seq data/seq/anime_v2_l6_t0/$s.npz --images data/anime_portraits256_v2_l6view \
    --out $D --steps 200 --skip $done_n >> out/decompose_partial_rest.log 2>&1
  echo "$(date +%T) $s: rest after $done_n of $total done"
done
echo "partial shards finished"
