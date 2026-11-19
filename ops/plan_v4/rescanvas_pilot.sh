#!/bin/bash
# Laptop: plan v4 R2, the gate for image -> painting by the model. The final v3 model converted to a residual canvas
# (tools/add_canvas_target.py: identical at init) fine-tuned with its target image in the canvas for half the
# sequences (train.py --canvas-target --p-target 0.5), then evaluated on 192 face + 192 anime validation images:
#   with target  input_psnr (PSNR to the input image) and CLIP image similarity; gate: >= 18 dB (12.7 dB was the
#                best image conditioning before, docs/image_cond_log.md; 18.4 dB is the decomposer-sketch continuation)
#   text only    R@5 / FD as before (must stay near the v3 model: 0.406 / 0.371 faces, 0.417 / 0.206 anime)
# Same evaluation of the unconverted start (target channels at zero) as the reference. Summary: out/plan_v4/rescanvas.txt
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/plan_v4/rescanvas; S=out/plan_v4/rescanvas.txt; R=runs/v4_rescanvas_pilot
STEPS=${STEPS:-3000}; mkdir -p $O
T=data/target_images64_v3
[ -f runs/v3_rescanvas_init.pt ] || "$PY" tools/add_canvas_target.py --ckpt runs/v3_formal/B_ft/model_step012000.pt \
  --out runs/v3_rescanvas_init.pt
DATA=(); for i in $(seq -f %03g 0 9); do DATA+=("data/seq/v2/decomp_$i.npz"); done
for i in $(seq -f %03g 0 19); do DATA+=("data/seq/anime_v3_lines_full_t0/decomp_$i.npz"); done
if ! grep -q "\"step\": $STEPS," $R/log.jsonl 2>/dev/null; then
  RES=(); [ -f $R/last.pt ] && RES=(--resume)
  flock out/gpu.lock "$PY" -u tools/train.py --data "${DATA[@]}" --protocol protocol_v3.json --out $R --config base \
    --line-heads --order draft --init-weights runs/v3_rescanvas_init.pt --canvas --canvas-k 32 --canvas-size 64 \
    --canvas-target --target-images $T --p-target 0.5 --batch 2 --accum 4 --steps $STEPS --lr 1e-4 --min-lr 1e-5 \
    --warmup 100 --workers 3 --seed 0 --captions data/captions/captions_v3.jsonl --attributes data/celebamask256/attributes.json \
    --split-manifest data/split_v3_faces_anime.jsonl --text-encoder data/models/chinese-clip-vit-base-patch16 \
    --text-drop 0.1 --cfg-scale 5.0 --perturb 0.3 --eval-every 1000 --save-every 500 --snapshot-every $STEPS \
    "${RES[@]}" >> $R.out 2>&1 || { echo "training failed (see $R.out)" >> $S; exit 1; }
fi
J=(--text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48 --cfg-scale 5.0 --top-p 0.6)
FACE=(--data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl)
ANIME=(--data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl)
ev() {      # ev NAME CKPT face|anime target|text
  local d=("${FACE[@]}"); [ $3 = anime ] && d=("${ANIME[@]}")
  local t=(); [ $4 = target ] && t=(--target-images $T)
  [ -f $O/$1_$3_$4/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py --ckpt $2 "${J[@]}" "${d[@]}" "${t[@]}" \
    --out $O/$1_$3_$4 > $O/$1_$3_$4.log 2>&1
}
line() {    # line NAME
  "$PY" - $O "$1" >> $S <<'PY'
import json, os, sys
o, n = sys.argv[1], sys.argv[2]
def get(dom, mode):
    p = f"{o}/{n}_{dom}_{mode}/metrics.json"
    return json.load(open(p))["judge"]["generated"] if os.path.exists(p) else None
parts = []
for dom in ("face", "anime"):
    t, x = get(dom, "target"), get(dom, "text")
    parts.append(f"{dom}: input PSNR {t['input_psnr']:.2f} dB, CLIP-to-input {t['clip_image_to_input']:.3f} | "
                 f"text only R@5 {x['r_precision@5']:.3f} FD {x['clip_fd']:.3f}" if t and x else f"{dom}: missing")
print(f"{n:16s} " + " || ".join(parts), flush=True)
PY
}
for d in face anime; do ev start runs/v3_rescanvas_init.pt $d target; done
grep -q "^start " $S 2>/dev/null || {
  for d in face anime; do mkdir -p $O/start_${d}_text
    cp out/v3_train/eval/B_ft_step012000_${d}/metrics.json $O/start_${d}_text/; done; line start; }
F=$R/model_step$(printf %06d $STEPS).pt
for d in face anime; do for m in target text; do ev pilot$STEPS $F $d $m; done; done
line pilot$STEPS
echo "$(date -u +%FT%TZ) residual canvas pilot finished" >> $S
