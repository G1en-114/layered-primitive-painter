#!/bin/bash
# Plan v3 step 2: base (126M) from scratch on faces x12 trajectories + anime (domain word in captions).
# Data: /root/private_data/seq/v3_mix (built by merge_v3.sh). Resume-safe: re-running continues from last.pt
# and re-reads the data glob, so anime t2/t3 can join at a later restart.
set -u
cd /root/private_data/lpp_dev
R=/root/private_data; B=$R/runs/base_v3
mkdir -p $B
RESUME=(); [ -f $B/last.pt ] && RESUME=(--resume)
BA=(tools/train.py --data "$R/seq/v3_mix/*/decomp_*.npz" --protocol $R/lpp_dev/protocol.json
  --out $B --config base --batch 8 --accum 4 --steps 30000 --lr 3e-4 --min-lr 3e-5 --warmup 1000 --workers 6 --seed 0
  --captions $R/captions_v3.jsonl --attributes $R/celebamask256/attributes.json
  --split-manifest $R/split_v3_faces_anime.jsonl --text-encoder $R/models/chinese-clip-vit-base-patch16
  --text-drop 0.1 --cfg-scale 2.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3
  --eval-every 2000 --save-every 500 --sdpa-backend math "${RESUME[@]}")
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nohup python -u "${BA[@]}" >> $B/train.stdout.log 2>&1 < /dev/null &
printf '{"pid": %s, "argv": "%s", "data": "%s", "purpose": "plan v3 base from scratch: faces x12 + anime, domain words"}\n' \
  "$!" "${BA[*]}" "$(find $R/seq/v3_mix -mindepth 1 -maxdepth 1 -type d -printf '%f ' | sort)" >> $B/launch.jsonl
echo "base_v3 launched pid $!"
