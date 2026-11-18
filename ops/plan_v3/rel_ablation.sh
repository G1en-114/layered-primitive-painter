#!/bin/bash
# SCNet: do relative position heads help? (docs/relative_position.md)
# The 12-trajectory small ablation (layer order, 12000 steps; data and hyperparameters as draft_ablation.sh and
# the 4090 baseline runs/plan11_4090/plan11_results/abl_small_12traj), plus --rel-pos. Then the same evaluation
# as the baseline and a validation-loss comparison at matching steps -> rel_ablation.log
set -u
R=/root/private_data; S=$R/seq/v3_mix; O=$R/runs/abl_small_12traj_relpos; L=$R/runs/plan11_4090/plan11_results/abl_small_12traj
LOG=$R/rel_ablation.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
mkdir -p $O $R/eval
say "relpos ablation starting"
if [ ! -f $O/model_final.pt ]; then
  python -u tools/train.py --data "$S/face_t*/decomp_*.npz" --protocol $R/lpp_dev/protocol.json --out $O --config small \
    --batch 16 --accum 2 --steps 12000 --lr 3e-4 --min-lr 3e-5 --warmup 500 --workers 3 --seed 0 \
    --captions $R/captions_clean_v1.jsonl --attributes $R/celebamask256/attributes.json \
    --split-manifest $R/celebamask256_split_v1.jsonl --text-encoder $R/models/chinese-clip-vit-base-patch16 \
    --text-drop 0.1 --cfg-scale 2.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 \
    --rel-pos --eval-every 2000 --save-every 1000 $( [ -f $O/last.pt ] && echo --resume ) \
    > $O/train.stdout.log 2>&1 < /dev/null || { say "relpos training FAILED"; exit 1; }
  python -c "
import torch; s=torch.load('$O/last.pt',map_location='cpu',weights_only=False); assert s['step']==12000, s['step']
torch.save({k:s[k] for k in ('model','step','config','protocol')},'$O/model_final.pt')" || { say "no final checkpoint"; exit 1; }
  say "relpos training done"
fi
python -u tools/eval_text.py --ckpt $O/model_final.pt --out $R/eval/abl_relpos --text-encoder $R/models/chinese-clip-vit-base-patch16 \
  --data $R/celebamask256_val --captions $R/captions_clean_v1.jsonl --split-manifest $R/celebamask256_split_v1.jsonl \
  --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 --attr-classifier $R/models/attr_resnet18.pt \
  --attributes $R/celebamask256/attributes.json --n 192 --batch 48 > $R/eval/abl_relpos.log 2>&1 \
  && say "relpos evaluated" || say "relpos eval FAILED"
python - >> $LOG <<'PY'
import json
def vals(p):
    return {r["step"]: r["val"] for r in map(json.loads, open(p)) if "val" in r}
a = vals("/root/private_data/runs/plan11_4090/plan11_results/abl_small_12traj/log.jsonl")
b = vals("/root/private_data/runs/abl_small_12traj_relpos/log.jsonl")
for s in sorted(set(a) & set(b)):
    print(f"val step {s:5d}  x {a[s]['x']:.3f} -> {b[s]['x']:.3f}  y {a[s]['y']:.3f} -> {b[s]['y']:.3f}  total {a[s]['loss']:.3f} -> {b[s]['loss']:.3f}")
def final(p):
    d = json.load(open(p)); j = d["judge"]["generated"]; at = d["attributes"]["per_attribute"]
    return f"judge R@1 {j['r_precision@1']} R@5 {j['r_precision@5']} FD {j.get('clip_fd')} | attributes {round(sum(v['generated'] for v in at.values()) / len(at), 4)}"
print("final  layer :", final("/root/private_data/runs/plan11_4090/plan11_results/abl_small_12traj/metrics.json"))
print("final  relpos:", final("/root/private_data/eval/abl_relpos/metrics.json"))
PY
say "relpos ablation finished"
