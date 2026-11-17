#!/bin/bash
# SCNet: per-domain evaluation of base_v3 and every GRPO snapshot (docs: realism gap, 2026-11-17).
# Faces come from grpo_v3.sh's own eval (eval/<name>/metrics.json); this adds anime (192 validation
# images, data/anime_val192) and writes one line per checkpoint with both domains, including the
# CLIP Frechet distance to real images (FD, the realism metric; caption retrieval alone saturates).
set -u
R=/root/private_data; G=$R/runs/grpo_v3; OUT=$R/eval/domain_summary.txt
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
anime() {   # anime NAME CKPT
  [ -f $R/eval/anime_$1/metrics.json ] && return 0
  python -u tools/eval_text.py --ckpt $2 --out $R/eval/anime_$1 --text-encoder $R/models/chinese-clip-vit-base-patch16 \
    --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 --data $R/anime_val192 --captions $R/captions_v3.jsonl \
    --split-manifest $R/anime_val192/split_anime_val192.jsonl --n 192 --batch 48 --sdpa-backend math > $R/eval/anime_$1.log 2>&1
}
line() {    # line NAME FACE_EVAL_DIR
  python3 - "$1" "$2" >> $OUT <<'PY'
import json, sys
name, face = sys.argv[1], sys.argv[2]
def part(p):
    d = json.load(open(p)); j = d["judge"]["generated"]
    return f"judge R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"
print(f"{name:20s} | faces {part(f'/root/private_data/eval/{face}/metrics.json')} | anime {part(f'/root/private_data/eval/anime_{name}/metrics.json')}", flush=True)
PY
}
anime base_v3_final $R/runs/base_v3/model_final.pt && line base_v3_final base_v3_final
for s in $(seq 25 25 200); do
  n=grpo_v3_step$(printf %03d $s)
  while [ ! -f $R/eval/$n/metrics.json ]; do
    pgrep -f grpo_v3.sh > /dev/null || [ -f $R/eval/$n/metrics.json ] || break 2
    sleep 120
  done
  anime $n $G/model_step$(printf %03d $s).pt && line $n $n
done
echo "$(date -u +%FT%TZ) domain eval finished" >> $OUT
