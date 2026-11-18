#!/bin/bash
# Do GRPO and the chosen sampling stack? A GRPO v2 snapshot at CFG 5.0 / top-p 0.6, faces + anime (laptop GPU).
# usage: local_grpo_best.sh STEP
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/sweep; SW=$O/sampling_sweep.txt; s=$(printf %03d $1)
J=(--text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48 --ckpt runs/grpo_real_dl/model_step$s.pt --cfg-scale 5.0 --top-p 0.6)
t=grpo${s}_cfg5.0_p0.6
[ -f $O/sweep_${t}_face/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py "${J[@]}" --out $O/sweep_${t}_face \
  --data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl > $O/sweep_${t}_face.log 2>&1
[ -f $O/sweep_${t}_anime/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py "${J[@]}" --out $O/sweep_${t}_anime \
  --data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl > $O/sweep_${t}_anime.log 2>&1
"$PY" - $t >> $SW <<'PY'
import json, sys
t = sys.argv[1]
def part(dom):
    j = json.load(open(f"out/sweep/sweep_{t}_{dom}/metrics.json"))["judge"]["generated"]
    return f"R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"
print(f"{t:28s} | faces {part('face')} | anime {part('anime')}", flush=True)
PY
