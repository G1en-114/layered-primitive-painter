#!/bin/bash
# Laptop: best-of-4 of the chosen final v3 model (B_ft step 12000) at CFG 5 / top-p 0.6, the same setting as
# base_v3's best result (ops/plan_v3/local_bestof.sh: faces R@5 0.432, anime R@5 0.479). Appends to the summary.
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/v3_train/eval; C=runs/v3_formal/B_ft/model_step012000.pt; t=B_ft_step012000_best4
J=(--text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48 --ckpt $C --cfg-scale 5.0 --top-p 0.6 --best-of 4)
[ -f $O/${t}_face/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py "${J[@]}" --out $O/${t}_face \
  --data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl > $O/${t}_face.log 2>&1
[ -f $O/${t}_anime/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py "${J[@]}" --out $O/${t}_anime \
  --data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl > $O/${t}_anime.log 2>&1
"$PY" - $O $t >> $O/summary.txt <<'PY'
import json, sys
ev, name = sys.argv[1], sys.argv[2]
g = {d: json.load(open(f"{ev}/{name}_{d}/metrics.json"))["judge"]["generated"] for d in ("face", "anime")}
part = lambda j: f"R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"  # noqa: E731
print(f"{name:22s} | faces {part(g['face'])} | anime {part(g['anime'])}   (base_v3 best-of-4: faces R@1 0.193 R@5 0.432 FD 0.375 | anime R@1 0.151 R@5 0.479 FD 0.221)")
PY
tail -1 $O/summary.txt
