#!/bin/bash
# SCNet: GRPO on base_v3, chained after the autopilot's final evaluation.
#  1. wait for "autopilot finished" (base_v3 done, model_final.pt written, final eval run)
#  2. GRPO from base_v3/model_final.pt: face prompts (captions_v3, so the 真人照片 prefix matches training),
#     CLIP + attribute + orphan rewards with per-component advantages, 200 steps, weights-only snapshot every 25
#  3. evaluate every snapshot as it appears (192 face prompts, independent L/14 judge, 4 held-out attributes)
#     and append a comparison row against base_v3_final to eval/grpo_v3_summary.txt
# Earlier runs over-optimised after ~100-175 steps: the snapshot to keep is chosen from the judge, not the reward.
set -u
R=/root/private_data; B=$R/runs/base_v3; G=$R/runs/grpo_v3; LOG=$R/grpo_v3.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
until grep -q "autopilot finished" $R/autopilot_v3.log; do sleep 120; done
[ -f $B/model_final.pt ] || { say "no base_v3 model_final.pt; abort"; exit 1; }
[ -e $G ] && { say "$G exists; abort"; exit 1; }
mkdir -p $G $R/eval
say "base_v3 finished; GRPO starting"
python -u tools/train_grpo.py --init $B/model_final.pt --out $G \
  --text-encoder $R/models/chinese-clip-vit-base-patch16 --captions $R/captions_v3.jsonl \
  --split-manifest $R/celebamask256_split_v1.jsonl \
  --attr-classifier $R/models/attr_resnet18.pt --attributes $R/celebamask256/attributes.json \
  --adv-mode per-component --w-clip 1 --w-attr 1 --w-orphan 0.5 \
  --steps 200 --prompts 8 --group 8 --micro-batch 16 --lr 1e-5 --beta 0.04 \
  --sample-every 25 --save-every 25 --keep-snapshots --sdpa-backend math > $G/stdout.log 2>&1 < /dev/null &
GP=$!
say "GRPO pid $GP"
summary() {   # one comparison row per evaluated checkpoint
  python - "$@" >> $R/eval/grpo_v3_summary.txt <<'PY'
import json, sys
held = ["Wearing_Hat", "Wearing_Earrings", "Bangs", "No_Beard"]
for n in sys.argv[1:]:
    d = json.load(open(f"/root/private_data/eval/{n}/metrics.json")); g = d["generated"]; j = d["judge"]["generated"]
    a = d["attributes"]["per_attribute"]; rew = [k for k in a if k not in held]
    print(n.ljust(24), "B16", g["clip_score"], g["r_precision@5"], "| L14", j["clip_score"], j["r_precision@1"], j["r_precision@5"],
          "| attr reward", round(sum(a[k]["generated"] for k in rew) / len(rew), 4),
          "held-out", round(sum(a[k]["generated"] for k in held) / 4, 4), "| orphan", g["orphan_rate"], flush=True)
PY
}
[ -f $R/eval/base_v3_final/metrics.json ] && summary base_v3_final
for s in $(seq 25 25 200); do
  f=$G/model_step$(printf %03d $s).pt
  while [ ! -f $f ]; do kill -0 $GP 2>/dev/null || break 2; sleep 60; done
  N=grpo_v3_step$(printf %03d $s)
  python -u tools/eval_text.py --ckpt $f --out $R/eval/$N \
    --text-encoder $R/models/chinese-clip-vit-base-patch16 --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 \
    --attr-classifier $R/models/attr_resnet18.pt --attributes $R/celebamask256/attributes.json \
    --data $R/celebamask256_val --captions $R/captions_v3.jsonl --split-manifest $R/celebamask256_split_v1.jsonl \
    --n 192 --batch 48 --sdpa-backend math > $R/eval/$N.log 2>&1 && { summary $N; say "evaluated step $s"; } || say "eval step $s FAILED"
done
wait $GP; say "GRPO exited ($?)"
say "grpo_v3 finished"
