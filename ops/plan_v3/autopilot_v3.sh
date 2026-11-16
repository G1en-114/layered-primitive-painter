#!/bin/bash
# SCNet autopilot for plan v3 (runs with the local machine off). Log: /root/private_data/autopilot_v3.log
#  1. prune the raw anime t0 uploaded from the local machine (once the DCU's trajectories are done)
#  2. wait until the DCU's odd shards t4..t7 are pruned and the even shards t4..t11 are on disk (.complete)
#  3. merge what is available into seq/v3_mix (hard links, protocol SHA + shard counts checked) and launch base_v3
#  4. every 10 min: when new trajectories arrive (odd t8..t11, anime t1..t3 pushed by the 4090), wait for the
#     next checkpoint, stop training, re-merge, resume (same run, re-read data)
#  5. when training reaches its last step: evaluate the final checkpoint (192 face prompts, judge, attributes)
set -u
R=/root/private_data; SEQ=$R/seq; U=$SEQ/upload_v3; M=$SEQ/v3_mix; B=$R/runs/base_v3
LOG=$R/autopilot_v3.log; TL=$R/traj_driver_v3.log
PROTO=bb54b53748557229f1135f26ad5ed9a750ff7410170930c396e47e5787a9d58c
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
cd $R/lpp_dev
export LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2

proto_ok() { [ "$(python -c "import json;print(json.load(open('$1/config.json')).get('protocol_sha256'))")" = "$PROTO" ]; }
nshards() { find "$1" -maxdepth 1 -name 'decomp_*.npz' 2>/dev/null | wc -l; }

# available trajectory sources: prints "name:dir[,dir]" for every complete item
available() {
  for k in 0 1 2 3; do [ "$(nshards $SEQ/v2_clean/t$k)" = 30 ] && echo "face_t$k:$SEQ/v2_clean/t$k"; done
  for k in 4 5 6 7; do
    grep -q "t$k pruned" $TL && [ -f $U/v2_clean_even/t$k/.complete ] && echo "face_t$k:$SEQ/v2_clean/t$k,$U/v2_clean_even/t$k"
  done
  for k in 8 9 10 11; do
    [ -f $U/v2_clean_even/t$k/.complete ] || continue
    if [ -f $U/v2_clean_odd/t$k/.complete ]; then echo "face_t$k:$U/v2_clean_even/t$k,$U/v2_clean_odd/t$k"
    else echo "face_t${k}_even:$U/v2_clean_even/t$k"; fi
  done
  for k in 0 1 2 3; do [ -f $U/anime_v2_l6_clean/t$k/.complete ] && echo "anime_t$k:$U/anime_v2_l6_clean/t$k"; done
}

merge() {   # rebuild v3_mix from the given items (training must be stopped)
  rm -rf $M.new; mkdir -p $M.new
  for it in "$@"; do
    name=${it%%:*}; dirs=${it#*:}; mkdir -p $M.new/$name
    for d in ${dirs//,/ }; do
      proto_ok $d || { say "protocol mismatch in $d: skipped $name"; rm -rf $M.new/$name; continue 2; }
      ln -f $d/decomp_*.npz $M.new/$name/
      [ -f $M.new/$name/config.json ] || cp $d/config.json $M.new/$name/
    done
  done
  rm -rf $M; mv $M.new $M
  say "merged: $(find $M -name 'decomp_*.npz' | wc -l) shards; $(ls $M | tr '\n' ' ')"
}

train_pid() { tail -1 $B/launch.jsonl 2>/dev/null | python -c "import sys,json;print(json.loads(sys.stdin.read())['pid'])" 2>/dev/null; }
running() { p=$(train_pid); [ -n "$p" ] && kill -0 $p 2>/dev/null; }
last_step() { grep -o '"step": [0-9]*' $B/log.jsonl 2>/dev/null | tail -1 | grep -o '[0-9]*'; }

say "autopilot started"
# 1+2: wait for the DCU's own trajectories (and the even halves) before training takes the GPU
until [ "$(grep -c -E 't[4-7] pruned' $TL)" -ge 4 ]; do sleep 120; done
say "DCU odd shards t4-t7 pruned"
RAW=$U/raw/anime_v2_l6_t0
if [ -f $U/anime_v2_l6_clean/t0/.complete ]; then :
elif [ "$(nshards $RAW)" = 107 ]; then
  say "pruning anime t0 on the DCU"
  python -u tools/prune_sequences.py --src $RAW --out $U/anime_v2_l6_clean/t0 --min-visible 2 \
    --protocol $R/lpp_next/protocol.json > $R/prune_anime_t0.log 2>&1 \
    && [ "$(nshards $U/anime_v2_l6_clean/t0)" = 107 ] && touch $U/anime_v2_l6_clean/t0/.complete \
    && say "anime t0 pruned" || say "anime t0 prune FAILED"
else
  say "raw anime t0 not complete ($(nshards $RAW)/107): training starts without it"
fi
for i in $(seq 1 60); do       # the even halves t9..t11 come from the 4090; wait up to 1 h for them
  [ "$(available | grep -c '^face_t\(8\|9\|10\|11\)')" -ge 4 ] && break; sleep 60
done
items=$(available); merge $items; merged="$items"
bash $R/runs/launch_base_v3.sh >> $LOG 2>&1
say "base_v3 launched with: $(echo $merged | tr ' ' '\n' | cut -d: -f1 | tr '\n' ' ')"

# 4: absorb new data at checkpoints; 5: final evaluation
while true; do
  sleep 600
  if ! running; then
    st=$(last_step)
    if [ "${st:-0}" -ge 30000 ]; then say "training finished at step $st"; break; fi
    say "training not running at step ${st:-?}: relaunching (resume)"; bash $R/runs/launch_base_v3.sh >> $LOG 2>&1; continue
  fi
  if [ ! -f $U/anime_v2_l6_clean/t0/.complete ] && [ "$(nshards $RAW)" = 107 ] && [ ! -f $R/.anime_t0_prune_tried ]; then
    touch $R/.anime_t0_prune_tried; say "pruning late anime t0"
    python -u tools/prune_sequences.py --src $RAW --out $U/anime_v2_l6_clean/t0 --min-visible 2       --protocol $R/lpp_next/protocol.json > $R/prune_anime_t0.log 2>&1       && [ "$(nshards $U/anime_v2_l6_clean/t0)" = 107 ] && touch $U/anime_v2_l6_clean/t0/.complete       && say "anime t0 pruned" || say "anime t0 prune FAILED"
  fi
  items=$(available)
  [ "$items" = "$merged" ] && continue
  say "new data: $(comm -13 <(echo "$merged" | tr ' ' '\n' | cut -d: -f1 | sort) <(echo "$items" | tr ' ' '\n' | cut -d: -f1 | sort) | tr '\n' ' ')"
  ref=$(stat -c %Y $B/last.pt 2>/dev/null || echo 0)
  until [ "$(stat -c %Y $B/last.pt 2>/dev/null || echo 0)" -gt "$ref" ]; do sleep 30; running || break; done
  p=$(train_pid); kill $p 2>/dev/null; while kill -0 $p 2>/dev/null; do sleep 5; done
  merge $items; merged="$items"
  bash $R/runs/launch_base_v3.sh >> $LOG 2>&1
  say "resumed at step $(last_step) with new data"
done

python -c "
import torch; s=torch.load('$B/last.pt',map_location='cpu',weights_only=False)
torch.save({k:s[k] for k in ('model','step','config','protocol')},'$B/model_final.pt')"
mkdir -p $R/eval
python -u tools/eval_text.py --ckpt $B/model_final.pt --out $R/eval/base_v3_final \
  --text-encoder $R/models/chinese-clip-vit-base-patch16 --judge-encoder $R/models/chinese-clip-vit-large-patch14-fp16 \
  --attr-classifier $R/models/attr_resnet18.pt --attributes $R/celebamask256/attributes.json \
  --data $R/celebamask256_val --captions $R/captions_v3.jsonl --split-manifest $R/celebamask256_split_v1.jsonl \
  --n 192 --batch 48 --sdpa-backend math > $R/eval/base_v3_final.log 2>&1 && say "final eval done" || say "final eval FAILED"
say "autopilot finished"
