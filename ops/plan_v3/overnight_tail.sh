#!/bin/bash
# SCNet overnight tail, after GRPO v2 (grpo_real) and the draft ablation have both finished:
#  1. sampling sweep on base_v3 (no training): CFG x top-p, plus Renyi adaptive truncation, faces and anime,
#     judge R@1 / R@5 / FD per setting -> eval/sampling_sweep.txt
#  2. the same sweep's best three settings on the last GRPO v2 snapshot
#  3. prefix quality (how early the subject appears) of base_v3 in layer order, as the reference for
#     a later draft-order base model
set -u
R=/root/private_data; LOG=$R/overnight_tail.log; SW=$R/eval/sampling_sweep.txt
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
until grep -q "grpo_real finished" $R/grpo_real.log 2>/dev/null && grep -q "draft ablation finished\|FAILED" $R/draft_ablation.log 2>/dev/null; do sleep 300; done
say "queue empty: sampling sweep starting"
J="--text-encoder $R/models/chinese-clip-vit-base-patch16 --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 --captions $R/captions_v3.jsonl --n 192 --batch 48 --sdpa-backend math"
FACE="--data $R/celebamask256_val --split-manifest $R/celebamask256_split_v1.jsonl"
ANIME="--data $R/anime_val192 --split-manifest $R/anime_val192/split_anime_val192.jsonl"
run() {   # run TAG CKPT SAMPLING...
  local tag=$1 ck=$2; shift 2
  for dom in face anime; do
    local d=FACE; [ $dom = anime ] && d=ANIME
    local o=$R/eval/sweep_${tag}_$dom
    [ -f $o/metrics.json ] || python -u tools/eval_text.py --ckpt $ck --out $o $J ${!d} "$@" > $o.log 2>&1 || { say "sweep $tag $dom FAILED"; return 1; }
  done
  python3 - "$tag" >> $SW <<'PY'
import json, sys
t = sys.argv[1]
def part(dom):
    d = json.load(open(f"/root/private_data/eval/sweep_{t}_{dom}/metrics.json")); j = d["judge"]["generated"]
    return f"R@1 {j['r_precision@1']:.3f} R@5 {j['r_precision@5']:.3f} FD {j['clip_fd']:.3f}"
print(f"{t:28s} | faces {part('face')} | anime {part('anime')}", flush=True)
PY
}
B=$R/runs/base_v3/model_final.pt
for cfg in 1.5 2.0 3.0; do for tp in 0.8 0.95; do run base_cfg${cfg}_p${tp} $B --cfg-scale $cfg --top-p $tp; done; done
for a in 2 4; do run base_cfg2.0_renyi$a $B --cfg-scale 2.0 --renyi-alpha $a; done
say "base sweep done"
last=$(ls $R/runs/grpo_real/model_step*.pt 2>/dev/null | sort | tail -1)
if [ -n "$last" ]; then
  for setting in "--cfg-scale 2.0 --top-p 0.9" "--cfg-scale 3.0 --top-p 0.8" "--cfg-scale 2.0 --renyi-alpha 4"; do
    tag=v2last_$(echo $setting | tr -d ' -' | tr '.' 'p'); run $tag $last $setting
  done
  say "v2 sweep done ($last)"
fi
python -u tools/prefix_quality.py --ckpt $B --out $R/eval/prefix_base_v3 $J $FACE > $R/eval/prefix_base_v3.log 2>&1 \
  && say "prefix quality base_v3 done" || say "prefix quality base_v3 FAILED"
say "overnight tail finished"
