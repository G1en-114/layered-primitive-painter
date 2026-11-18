#!/bin/bash
# Laptop GPU, early signal for docs/relative_position.md: small model on the 4 local face trajectories,
# absolute vs relative position heads, 3000 steps each, run one after the other (8 GB holds one; same data, seed, hyperparameters).
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"
run() {   # run NAME EXTRA...
  local n=$1; shift
  "$PY" -u tools/train.py --data "data/seq/quick4/*/decomp_*.npz" \
    --out runs/quick_$n --config small --batch 8 --accum 4 --steps 3000 --lr 3e-4 --min-lr 3e-5 --warmup 300 --workers 3 --seed 0 \
    --captions data/captions/captions_clean_v1.jsonl --attributes data/celebamask256/attributes.json \
    --split-manifest data/celebamask256_split_v1.jsonl --text-encoder data/models/chinese-clip-vit-base-patch16 \
    --text-drop 0.1 --cfg-scale 2.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 \
    --eval-every 500 --save-every 1000 "$@" > runs/quick_$n.log 2>&1
}
mkdir -p runs
run rel --rel-pos; run abs
"$PY" - <<'PY'
import json
def vals(p):
    return {r["step"]: r["val"] for r in map(json.loads, open(p)) if "val" in r}
a, b = vals("runs/quick_abs/log.jsonl"), vals("runs/quick_rel/log.jsonl")
for s in sorted(set(a) & set(b)):
    print(f"val step {s:5d}  x {a[s]['x']:.3f} -> {b[s]['x']:.3f}  y {a[s]['y']:.3f} -> {b[s]['y']:.3f}  total {a[s]['loss']:.3f} -> {b[s]['loss']:.3f}")
PY
echo "quick rel finished"
