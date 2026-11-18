#!/bin/bash
# Laptop: protocol v3 at a larger scale (4 anime shards ~1900 images, 6000 steps), after the smoke test and the fixes
# The same ~940 anime images twice, from the same start (a small v2 model converted with init_v3_from_v2.py):
#   lines  : Gaussians + Bezier lines (tools/decompose_lines.py, joint fit)
#   control: Gaussians only, the same joint refit (--max-curves 0), so any difference is the lines
# 2000 fine-tuning steps each, then eval_text on the 192 anime validation prompts (judge R@1/R@5, FD) and samples.
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"
L=data/seq/anime_v3b_lines_t0; C=data/seq/anime_v3b_control_t0; LOG=out/v3_scale.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
SRC="data/seq/anime_v2_l6_t0/decomp_000.npz data/seq/anime_v2_l6_t0/decomp_001.npz data/seq/anime_v2_l6_t0/decomp_002.npz data/seq/anime_v2_l6_t0/decomp_003.npz"
[ -f $L/decomp_003.npz ] || flock out/gpu.lock "$PY" -u tools/decompose_lines.py --seq $SRC --images data/anime_portraits256_v2_l6view --out $L --steps 200 > out/decompose_v3b_lines.log 2>&1
say "data: line shards ready"
[ -f $C/decomp_003.npz ] || flock out/gpu.lock "$PY" -u tools/decompose_lines.py --seq $SRC \
  --images data/anime_portraits256_v2_l6view --out $C --steps 200 --max-curves 0 \
  > out/decompose_v3b_control.log 2>&1
say "control data ready: lines $(grep -o 'mean PSNR [0-9.]* dB' out/decompose_v3b_lines.log | tail -1), control $(grep -o 'mean PSNR [0-9.]* dB' out/decompose_v3b_control.log | tail -1)"
[ -f runs/quick_abs_v3init.pt ] || "$PY" tools/init_v3_from_v2.py --ckpt runs/quick_abs/last.pt --out runs/quick_abs_v3init.pt
train() {   # train NAME DATA
  [ -f runs/v3b_$1/last.pt ] && grep -q '"step": 6000' runs/v3b_$1/log.jsonl 2>/dev/null && return 0
  flock out/gpu.lock "$PY" -u tools/train.py --data "$2/decomp_*.npz" --protocol protocol_v3.json --out runs/v3b_$1 --config small --line-heads \
    --init-weights runs/quick_abs_v3init.pt --batch 8 --accum 4 --steps 6000 --lr 1e-4 --min-lr 1e-5 --warmup 200 --workers 3 --seed 0 \
    --captions data/captions/captions_v3.jsonl --split-manifest data/split_v3_faces_anime.jsonl \
    --text-encoder data/models/chinese-clip-vit-base-patch16 --text-drop 0.1 --cfg-scale 2.0 \
    --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 --eval-every 1000 --save-every 1000 > runs/v3b_$1.log 2>&1 \
    && say "trained $1" || say "training $1 FAILED"
}
train lines $L
train control $C
ev() {      # ev NAME
  "$PY" - runs/v3b_$1/last.pt runs/v3b_$1/model_final.pt <<'PYX'
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
torch.save({k: s[k] for k in ("model", "step", "config", "protocol")}, sys.argv[2])
PYX
  flock out/gpu.lock "$PY" -u tools/eval_text.py --ckpt runs/v3b_$1/model_final.pt --out out/v3b_eval_$1 \
    --text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16 \
    --captions data/captions/captions_v3.jsonl --data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl \
    --n 192 --batch 48 --cfg-scale 5.0 --top-p 0.6 > out/v3b_eval_$1.log 2>&1 && say "evaluated $1" || say "eval $1 FAILED"
}
ev lines
ev control
"$PY" - <<'PYX' | tee -a $LOG
import json
for n in ("lines", "control"):
    j = json.load(open(f"out/v3b_eval_{n}/metrics.json"))["judge"]["generated"]
    print(f"{n:8s} judge R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}")
PYX
say "v3 scale finished"
