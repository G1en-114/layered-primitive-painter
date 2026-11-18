#!/bin/bash
# Laptop GPU, after local_rel_quick.sh: do the finest strokes help or hurt? (tools/stroke_filter_eval.py)
# base_v3 at the chosen sampling (CFG 5.0, top-p 0.6), faces and anime.
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"
until grep -q "quick rel finished" runs/local_rel_quick.out 2>/dev/null; do sleep 30; done
J=(--ckpt runs/base_v3_snap/model_final.pt --text-encoder data/models/chinese-clip-vit-base-patch16
   --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16 --captions data/captions/captions_v3.jsonl
   --cfg-scale 5.0 --top-p 0.6)
flock out/gpu.lock "$PY" -u tools/stroke_filter_eval.py "${J[@]}" --out out/stroke_filter_face \
  --data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl > out/stroke_filter_face.log 2>&1
flock out/gpu.lock "$PY" -u tools/stroke_filter_eval.py "${J[@]}" --out out/stroke_filter_anime \
  --data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl > out/stroke_filter_anime.log 2>&1
echo "stroke filter finished"
