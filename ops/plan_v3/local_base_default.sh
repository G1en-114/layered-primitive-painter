#!/bin/bash
# base_v3 at the GRPO evaluation's own sampling (CFG 2.0, top-p 0.9) on the laptop GPU, so GRPO snapshots
# evaluated here have a same-machine reference (the base_v3_final line was measured on the DCU).
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/grpo_eval
J=(--text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48 --ckpt runs/base_v3_snap/model_final.pt)
[ -f $O/base_v3_laptop/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py "${J[@]}" --out $O/base_v3_laptop \
  --data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl > $O/base_v3_laptop.log 2>&1
[ -f $O/anime_base_v3_laptop/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py "${J[@]}" --out $O/anime_base_v3_laptop \
  --data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl > $O/anime_base_v3_laptop.log 2>&1
"$PY" - <<'PY'
import json
def p(n):
    j = json.load(open(f"out/grpo_eval/{n}/metrics.json"))["judge"]["generated"]
    return f"R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"
print(f"base_v3 (laptop)             | faces {p('base_v3_laptop')} | anime {p('anime_base_v3_laptop')}")
PY
