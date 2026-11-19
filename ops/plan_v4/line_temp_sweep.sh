#!/bin/bash
# Laptop: plan v4 item A1 (stray lines). The final v3 model on the 192 anime prompts (CFG 5 / top-p 0.6) with a lower
# temperature for LINE strokes only (sample.SampleConfig.line_temperature); Gaussians keep temperature 1.
# Reference (line temperature 1): out/v3_train/eval/B_ft_step012000_anime. Then the best setting on faces too.
# Summary: out/plan_v4/line_temp.txt
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; O=out/plan_v4/line_temp; S=out/plan_v4/line_temp.txt; mkdir -p $O
C=runs/v3_formal/B_ft/model_step012000.pt
J=(--ckpt $C --text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48 --cfg-scale 5.0 --top-p 0.6)
ANIME=(--data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl)
FACE=(--data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl)
line() {    # line TAG DIR
  "$PY" - "$1" "$2" >> $S <<'PY'
import json, sys
j = json.load(open(f"{sys.argv[2]}/metrics.json"))["judge"]["generated"]
print(f"{sys.argv[1]:28s} R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}", flush=True)
PY
}
[ -f $S ] || { echo "# anime unless marked; line temperature on the listed attributes, Gaussians at 1" > $S
  line "anime  line_t=1 (reference)" out/v3_train/eval/B_ft_step012000_anime; }
for t in 0.7 0.5 0.3; do
  for set in geom all; do
    a="x=$t,y=$t,theta=$t"; [ $set = all ] && a="x=$t,y=$t,theta=$t,sigma_x=$t,sigma_y=$t,bend=$t,color=$t,alpha=$t"
    n=anime_${set}_$t
    [ -f $O/$n/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py "${J[@]}" "${ANIME[@]}" --line-temperature "$a" \
      --out $O/$n > $O/$n.log 2>&1
    grep -q "^anime  $set t=$t " $S || line "anime  $set t=$t" $O/$n
  done
done
echo "line temperature sweep finished" >> $S
