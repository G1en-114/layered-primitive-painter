#!/bin/bash
# SCNet: GRPO v2 ("realism", 2026-11-17) from base_v3, after GRPO v1 (grpo_v3) has its step-100 evaluation.
#  - stops GRPO v1 at step 100 (enough as the control: earlier runs over-optimised after ~100 steps)
#  - prompts: faces and anime, half each (captions_v3, split_v3_faces_anime train)
#  - rewards (per-component group advantages): caption match (B/16), likeness to the prompt's own source
#    image (B/16 image-image cosine, source_emb_b16.npz), attributes (faces only), orphans
#  - every 25 steps: faces (192 validation photos) and anime (192 validation images), independent L/14
#    judge R@1 / R@5 and FD, one line per checkpoint in eval/grpo_real_summary.txt
set -u
R=/root/private_data; B=$R/runs/base_v3; G=$R/runs/grpo_real; LOG=$R/grpo_real.log; SUM=$R/eval/grpo_real_summary.txt
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
until [ -f $R/eval/grpo_v3_step100/metrics.json ] || ! pgrep -f grpo_v3.sh > /dev/null; do sleep 120; done
p=$(pgrep -f "out $R/runs/grpo_v3 --text-encoder")
[ -n "$p" ] && { kill $p; say "GRPO v1 stopped after its step-100 evaluation (pid $p)"; }
[ -e $G ] && { say "$G exists; abort"; exit 1; }
mkdir -p $G
# the source embeddings are computed on the local machine and uploaded; without them v2 still runs
# (mixed domains, anime-safe attributes) and the log says so
SRC=(); [ -f $R/source_emb_b16.npz.complete ] && SRC=(--source-emb $R/source_emb_b16.npz --w-src 1) || say "no source embeddings: running without the source-likeness reward"
python -u tools/train_grpo.py --init $B/model_final.pt --out $G \
  --text-encoder $R/models/chinese-clip-vit-base-patch16 --captions $R/captions_v3.jsonl \
  --split-manifest $R/split_v3_faces_anime.jsonl --anime-frac 0.5 "${SRC[@]}" \
  --attr-classifier $R/models/attr_resnet18.pt --attributes $R/celebamask256/attributes.json \
  --adv-mode per-component --w-clip 1 --w-attr 1 --w-orphan 0.5 \
  --steps 200 --prompts 8 --group 8 --micro-batch 16 --lr 1e-5 --beta 0.04 \
  --sample-every 25 --save-every 25 --keep-snapshots --sdpa-backend math > $G/stdout.log 2>&1 < /dev/null &
GP=$!
say "GRPO v2 started (pid $GP)"
ev() {   # ev NAME CKPT DATA SPLIT
  [ -f $R/eval/$1/metrics.json ] || python -u tools/eval_text.py --ckpt $2 --out $R/eval/$1 \
    --text-encoder $R/models/chinese-clip-vit-base-patch16 --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 \
    --data $3 --captions $R/captions_v3.jsonl --split-manifest $4 --n 192 --batch 48 --sdpa-backend math > $R/eval/$1.log 2>&1
}
line() { python3 - "$@" >> $SUM <<'PY'
import json, sys
name, face, anime = sys.argv[1:]
def part(n):
    d = json.load(open(f"/root/private_data/eval/{n}/metrics.json")); j = d["judge"]["generated"]
    return f"judge R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"
print(f"{name:22s} | faces {part(face)} | anime {part(anime)}", flush=True)
PY
}
[ -f $R/eval/base_v3_final/metrics.json ] && [ -f $R/eval/anime_base_v3_final/metrics.json ] && line base_v3_final base_v3_final anime_base_v3_final
for s in $(seq 25 25 200); do
  f=$G/model_step$(printf %03d $s).pt
  while [ ! -f $f ]; do kill -0 $GP 2>/dev/null || break 2; sleep 60; done
  n=grpo_real_step$(printf %03d $s)
  ev $n $f $R/celebamask256_val $R/celebamask256_split_v1.jsonl && ev anime_$n $f $R/anime_val192 $R/anime_val192/split_anime_val192.jsonl \
    && { line $n $n anime_$n; say "evaluated step $s"; } || say "eval step $s FAILED"
done
wait $GP; say "GRPO v2 exited ($?)"
say "grpo_real finished"
