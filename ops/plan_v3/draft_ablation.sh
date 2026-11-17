#!/bin/bash
# SCNet: does the draft painting order help? (docs/draft_order.md)
# Queued after GRPO. Retrains the 12-trajectory ablation model (small, 12000 steps, identical data and
# hyperparameters, see cs_plan11.sh ablation()) with --order draft, then compares it with the layer-order
# run from the 4090 (runs/plan11_4090/plan11_results/abl_small_12traj):
#   - final quality: eval_text.py (192 face prompts, B/16 + independent L/14 judge, attributes)
#   - how early the face appears: prefix_quality.py (CLIP / judge R@5 after 25..400 strokes)
set -u
R=/root/private_data; S=$R/seq/v3_mix; O=$R/runs/abl_small_12traj_draft; L=$R/runs/plan11_4090/plan11_results/abl_small_12traj
LOG=$R/draft_ablation.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
until grep -q "grpo_v3 finished" $R/grpo_v3.log 2>/dev/null; do sleep 300; done
say "GRPO finished; draft ablation starting"
mkdir -p $O $R/eval
COMMON=(--text-encoder $R/models/chinese-clip-vit-base-patch16 --data $R/celebamask256_val
        --captions $R/captions_clean_v1.jsonl --split-manifest $R/celebamask256_split_v1.jsonl)
if [ ! -f $O/model_final.pt ]; then
  python -u tools/train.py --data "$S/face_t*/decomp_*.npz" --protocol $R/lpp_dev/protocol.json --out $O --config small \
    --batch 16 --accum 2 --steps 12000 --lr 3e-4 --min-lr 3e-5 --warmup 500 --workers 3 --seed 0 \
    --captions $R/captions_clean_v1.jsonl --attributes $R/celebamask256/attributes.json \
    --split-manifest $R/celebamask256_split_v1.jsonl --text-encoder $R/models/chinese-clip-vit-base-patch16 \
    --text-drop 0.1 --cfg-scale 2.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 \
    --order draft --draft-rounds 2 --eval-every 2000 --save-every 1000 $( [ -f $O/last.pt ] && echo --resume ) \
    > $O/train.stdout.log 2>&1 < /dev/null || { say "draft training FAILED"; exit 1; }
  python -c "
import torch; s=torch.load('$O/last.pt',map_location='cpu',weights_only=False); assert s['step']==12000, s['step']
torch.save({k:s[k] for k in ('model','step','config','protocol')},'$O/model_final.pt')" || { say "no final checkpoint"; exit 1; }
  say "draft training done"
fi
python -u tools/eval_text.py --ckpt $O/model_final.pt --out $R/eval/abl_draft "${COMMON[@]}" \
  --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 --attr-classifier $R/models/attr_resnet18.pt \
  --attributes $R/celebamask256/attributes.json --n 192 --batch 48 --sdpa-backend math > $R/eval/abl_draft.log 2>&1 \
  && say "draft evaluated" || say "draft eval FAILED"
for run in layer:$L draft:$O; do
  name=${run%%:*}; dir=${run#*:}
  python -u tools/prefix_quality.py --ckpt $dir/model_final.pt --out $R/eval/prefix_abl_$name "${COMMON[@]}" \
    --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 --n 192 --batch 48 \
    > $R/eval/prefix_abl_$name.log 2>&1 && say "prefix quality $name done" || say "prefix quality $name FAILED"
done
python - >> $LOG <<'PY'
import json
R = "/root/private_data/eval"
held = ["Wearing_Hat", "Wearing_Earrings", "Bangs", "No_Beard"]
def final(path):
    d = json.load(open(path)); j = d["judge"]["generated"]; a = d["attributes"]["per_attribute"]
    return f"judge R@1 {j['r_precision@1']} R@5 {j['r_precision@5']} | attributes {round(sum(v['generated'] for v in a.values()) / len(a), 4)}"
print("final  layer:", final("/root/private_data/runs/plan11_4090/plan11_results/abl_small_12traj/metrics.json"))
print("final  draft:", final(f"{R}/abl_draft/metrics.json"))
for name in ("layer", "draft"):
    c = json.load(open(f"{R}/prefix_abl_{name}/prefix_quality.json"))["cuts"]
    print(f"prefix {name}:", "  ".join(f"{k}: judge {v['judge_clip']} R@5 {v['judge_r@5']}" for k, v in c.items()))
PY
say "draft ablation finished"
