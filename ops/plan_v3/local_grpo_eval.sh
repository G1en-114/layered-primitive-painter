#!/bin/bash
# Laptop side of new_box2.sh: for each GRPO v2 snapshot (step 50..200) on the SCNet box, download it
# (8 parallel 64 MiB ranges, SHA-256 checked), evaluate faces + anime on the local GPU (same settings as the
# box), and upload metrics.json (+ prompts, grid) to eval/<name>/ there, where new_box2.sh picks it up.
# Skips any step the box has already evaluated itself (its 45-minute fallback).
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/AI infra/.venv/bin/python"; R=/root/private_data; D=runs/grpo_real_dl; O=out/grpo_eval
mkdir -p $D $O
X() { timeout ${T:-120} ssh -F /dev/null -S /run/user/1000/scnet.sock -o BatchMode=yes -o LogLevel=ERROR x "$@"; }
say() { echo "$(date -u +%FT%TZ) $*"; }
J=(--text-encoder data/models/chinese-clip-vit-base-patch16 --judge-encoder data/models/chinese-clip-vit-large-patch14-fp16
   --captions data/captions/captions_v3.jsonl --n 192 --batch 48)
FACE=(--data data/celebamask256_val --split-manifest data/celebamask256_split_v1.jsonl)
ANIME=(--data data/anime_val192 --split-manifest data/anime_val192/split_anime_val192.jsonl)

fetch() {   # fetch REMOTE LOCAL: parallel ranged download, verified
  local rf=$1 lf=$2 size chunk=$((64 << 20)) n i
  size=$(X "stat -c %s $rf") || return 1
  n=$(( (size + chunk - 1) / chunk ))
  mkdir -p $lf.parts
  for i in $(seq 0 $((n - 1))); do
    p=$lf.parts/$(printf %04d $i); want=$(( i == n - 1 ? size - i * chunk : chunk ))
    [ -f $p ] && [ $(stat -c %s $p) -eq $want ] && continue
    echo "$i $p $want"
  done | xargs -P 8 -L 1 bash -c 'for try in 1 2 3 4 5 6; do   # the gateway sometimes cuts a stream short: retry
      timeout 600 ssh -F /dev/null -S /run/user/1000/scnet.sock -o BatchMode=yes -o LogLevel=ERROR x \
        "dd if='"$rf"' bs=64M skip=$0 count=1 iflag=fullblock status=none; sleep 3" > $1.tmp
      [ $(stat -c %s $1.tmp) -eq $2 ] && { mv $1.tmp $1; break; }
    done'
  [ $(ls $lf.parts | grep -vc tmp) -eq $n ] || return 1
  cat $lf.parts/* > $lf.tmp
  [ "$(sha256sum < $lf.tmp | cut -c1-64)" = "$(T=300 X "sha256sum < $rf" | cut -c1-64)" ] || { rm -f $lf.tmp; return 1; }
  mv $lf.tmp $lf; rm -rf $lf.parts
}
ev() {   # ev NAME CKPT face|anime
  local -a d=("${FACE[@]}"); [ $3 = anime ] && d=("${ANIME[@]}")
  [ -f $O/$1/metrics.json ] || flock out/gpu.lock "$PY" -u tools/eval_text.py --ckpt $2 --out $O/$1 "${J[@]}" "${d[@]}" > $O/$1.log 2>&1
}
push() {    # push NAME: upload the small files, metrics.json last (the box polls for it)
  X "mkdir -p $R/eval/.up_$1" && tar cf - -C $O/$1 prompts.json grid_generated_over_real.png metrics.json \
    | X "tar xf - -C $R/eval/.up_$1 && rm -rf $R/eval/$1 && mv $R/eval/.up_$1 $R/eval/$1"
}

until grep -q "local sweep finished" out/sweep/local_sweep.out 2>/dev/null; do sleep 60; done
for s in $(seq 50 25 200); do
  n=grpo_real_step$(printf %03d $s); f=$R/runs/grpo_real/model_step$(printf %03d $s).pt; lf=$D/model_step$(printf %03d $s).pt
  say "waiting for $n"
  until X "test -f $f" 2>/dev/null; do
    X "pgrep -f tools/train_grpo.py > /dev/null" 2>/dev/null || { X "test -f $f" || { say "GRPO gone"; exit 0; }; }
    sleep 120
  done
  sleep 60                                                   # let the snapshot finish writing
  X "test -f $R/eval/$n/metrics.json -a -f $R/eval/anime_$n/metrics.json" && { say "$n: box evaluated it"; continue; }
  [ -f $lf ] || fetch $f $lf || fetch $f $lf || { say "$n: download FAILED"; continue; }
  ev $n $lf face && ev anime_$n $lf anime || { say "$n: eval FAILED"; continue; }
  push $n && push anime_$n && say "$n: evaluated and uploaded" || say "$n: upload FAILED"
done
say "local grpo eval finished"
