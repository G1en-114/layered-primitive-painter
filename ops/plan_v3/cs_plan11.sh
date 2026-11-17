#!/bin/bash
# RTX 4090, 2026-11-17: use ~11 h of remaining balance without the local machine.
#   push   : every finished trajectory marked .ready -> SCNet (SHA-checked, .complete there)
#   pull   : face data + CLIP/judge/attribute models from SCNet for the ablation
#   A (GPU): anime t1-t3 (9 jobs, MPS, <= 5 at once)
#   B      : ablation, small model from scratch, 4 vs 12 face trajectories, then eval_text (judge + attributes)
#   C      : extra face trajectories t12+ alongside B (3 processes), new work only while >= 2.2 h remain
#   end    : at the deadline - 30 min stop decomposition, final push, results to SCNet, delete SCNet credentials
set -u
ulimit -c 0
L=/root/lpp; D=$L/data; SEQ=$D/seq; LOG=/root/plan11.log; RES=/root/plan11_results
START=$(date +%s); DEADLINE=$((START + 11 * 3600)); STOP_AT=$((DEADLINE - 1800))
PY=/root/venv/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
left() { echo $(( STOP_AT - $(date +%s) )); }
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY $RES
nvidia-cuda-mps-control -d 2>/dev/null
export LPP_RASTER=triton LPP_DECOMP_COMPILE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
SSHX="ssh -S /root/.lpp/scnet.sock -o LogLevel=ERROR x"
cd $L
say "plan11 start; stop at $(date -u -d @$STOP_AT +%FT%TZ)"

# ---------------------------------------------------------------- push
push_one() {   # push_one KIND T  (KIND = directory under $SEQ and under SCNet seq/upload_v3)
  local kind=$1 t=$2 dst=/root/private_data/seq/upload_v3/$1
  bash /root/.lpp/cs_scnet_master.sh || return 1
  $SSHX "test -f $dst/t$t/.complete" && return 0
  local sums=/root/.lpp/sha_${kind}_t$t
  (cd $SEQ/$kind && find t$t -type f \( -name 'decomp_*.npz' -o -name config.json \) | sort | xargs sha256sum) > $sums
  tar cf - -C $SEQ/$kind t$t | xz -T4 -6 | $SSHX "mkdir -p $dst && cd $dst && xz -d | tar xf -" || return 1
  $SSHX "cd $dst && sha256sum -c --quiet && touch t$t/.complete" < $sums || { say "SHA mismatch $kind/t$t"; return 1; }
  say "pushed $kind/t$t"
}
push_pass() {
  for r in $SEQ/*/t*/.ready; do
    [ -f "$r" ] || continue
    d=$(dirname $r); t=${d##*/t}; kind=$(basename $(dirname $d))
    [ -f $d/.pushed ] && continue
    push_one $kind $t && touch $d/.pushed
  done
}
pusher() { while [ ! -f /root/.lpp/final ]; do push_pass; sleep 120; done; }

# ---------------------------------------------------------------- pull (for the ablation)
pull() {
  bash /root/.lpp/cs_scnet_master.sh || { say "pull: no SCNet connection"; return 1; }
  mkdir -p $SEQ/scnet_v2_clean $D/abl
  for k in 0 1 2 3 4 5 6 7; do
    $SSHX "tar cf - -C /root/private_data/seq/v2_clean t$k" | tar xf - -C $SEQ/scnet_v2_clean || say "pull t$k FAILED"
  done
  for m in chinese-clip-vit-base-patch16 chinese-clip-vit-large-patch14-fp16; do
    $SSHX "tar cf - -C /root/private_data/models $m" | tar xf - -C $D/abl || say "pull $m FAILED"
  done
  $SSHX "tar cf - -C /root/private_data models/attr_resnet18.pt captions_clean_v1.jsonl celebamask256_split_v1.jsonl celebamask256/attributes.json" \
    | tar xf - -C $D/abl || say "pull small files FAILED"
  say "pull done: $(for k in 0 1 2 3 4 5 6 7; do printf 't%s=%s ' $k $(find $SEQ/scnet_v2_clean/t$k -name 'decomp_*.npz' | wc -l); done)"
}

# ---------------------------------------------------------------- A: anime t1-t3
anime() {
  local P=$D/anime_l6_pack SRC=$D/anime_l6_jpegsrc
  until (cd $P && sha256sum -c --quiet /root/anime_pack.sha) > /dev/null 2>&1; do say "anime pack SHA not ok yet"; sleep 120; done
  [ -f $SRC/index.json ] || $PY -u tools/unpack_jpeg.py --pack $P --out $SRC > /root/anime_unpack.log 2>&1 || { say "anime unpack FAILED"; return 1; }
  say "anime unpacked; decomposing t1 t2 t3"
  local pids=()
  for j in 1:1:0/3 1:1:1/3 1:1:2/3 2:0:0/3 2:0:1/3 2:0:2/3 3:1:0/3 3:1:1/3 3:1:2/3; do
    while [ "$(ps -eo args | awk '/[d]ecompose_dataset.py/ && /anime_v2_l6_t/' | wc -l)" -ge 5 ]; do sleep 20; done
    IFS=: read s f p <<< "$j"; fl=""; [ "$f" = "1" ] && fl="--flip"
    $PY -u tools/decompose_dataset.py --data $SRC --out $SEQ/anime_v2_l6_t$s --batch 64 --steps 80 --seed $s $fl --part $p \
      --ids-file $P/decompose_ids_v1.json --protocol $L/protocol.json > /root/plan11_anime_t${s}_${p/\//of}.log 2>&1 &
    pids+=($!)
    sleep 3
  done
  for q in "${pids[@]}"; do wait $q; done      # not a bare wait: the pusher is also a child of this shell
  for s in 1 2 3; do
    n=$(find $SEQ/anime_v2_l6_t$s -name 'decomp_*.npz' | wc -l)
    [ "$n" = 107 ] || { say "anime t$s: only $n/107 shards"; continue; }
    $PY -u tools/prune_sequences.py --src $SEQ/anime_v2_l6_t$s --out $SEQ/anime_v2_l6_clean/t$s --min-visible 2 \
      --protocol $L/protocol.json > /root/plan11_prune_anime_t$s.log 2>&1 \
      && touch $SEQ/anime_v2_l6_clean/t$s/.ready && say "anime t$s done" || say "anime t$s prune FAILED"
  done
}

# ---------------------------------------------------------------- C: extra face trajectories
extra_faces() {
  local s=12
  while [ "$(left)" -gt 7920 ]; do
    for p in 0/2 1/2; do
      while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 30; done
      [ "$(left)" -gt 7920 ] || break 2
      fl=""; [ $((s % 2)) = 1 ] && fl="--flip"
      $PY -u tools/decompose_dataset.py --data $D/celebamask256 --out $SEQ/v2_t$s --batch 64 --steps 80 --seed $s $fl \
        --part $p --protocol $L/protocol.json > /root/plan11_face_t${s}_${p/\//of}.log 2>&1 &
      echo $! >> /root/.lpp/extra_pids
      sleep 3
    done
    say "extra face t$s launched"
    s=$((s + 1))
  done
  wait
}
prune_extra() {
  for d in $SEQ/v2_t1[2-9] $SEQ/v2_t2[0-9]; do
    [ -d $d ] || continue
    s=${d##*/v2_t}; n=$(find $d -name 'decomp_*.npz' | wc -l)
    [ "$n" = 30 ] || { say "extra face t$s incomplete ($n/30 shards): kept raw, not pushed"; continue; }
    [ -f $SEQ/v2_clean_extra/t$s/.ready ] && continue
    $PY -u tools/prune_sequences.py --src $d --out $SEQ/v2_clean_extra/t$s --min-visible 2 --protocol $L/protocol.json \
      > /root/plan11_prune_face_t$s.log 2>&1 && touch $SEQ/v2_clean_extra/t$s/.ready && say "extra face t$s done"
  done
}

# ---------------------------------------------------------------- B: ablation
ablation() {
  local A=$D/abl
  for name in 4 12; do
    rm -rf $SEQ/abl$name; mkdir -p $SEQ/abl$name
    for k in $(seq 0 $((name - 1))); do
      mkdir -p $SEQ/abl$name/t$k
      if [ $k -lt 4 ]; then srcs="$SEQ/scnet_v2_clean/t$k"
      elif [ $k -lt 8 ]; then srcs="$SEQ/scnet_v2_clean/t$k $SEQ/v2_clean_even/t$k"
      else srcs="$SEQ/v2_clean_even/t$k $SEQ/v2_clean_odd/t$k"; fi
      for sd in $srcs; do ln -f $sd/decomp_*.npz $SEQ/abl$name/t$k/; cp -n $sd/config.json $SEQ/abl$name/t$k/; done
    done
    say "ablation data abl$name: $(find $SEQ/abl$name -name 'decomp_*.npz' | wc -l) shards"
  done
  local tp=()
  for name in 4 12; do
    O=/root/runs/abl_small_${name}traj; mkdir -p $O
    $PY -u tools/train.py --data "$SEQ/abl$name/*/decomp_*.npz" --protocol $L/protocol.json --out $O --config small \
      --batch 16 --accum 2 --steps 12000 --lr 3e-4 --min-lr 3e-5 --warmup 500 --workers 3 --seed 0 \
      --captions $A/captions_clean_v1.jsonl --attributes $A/celebamask256/attributes.json \
      --split-manifest $A/celebamask256_split_v1.jsonl --text-encoder $A/chinese-clip-vit-base-patch16 \
      --text-drop 0.1 --cfg-scale 2.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 \
      --eval-every 2000 --save-every 1000 > $O/train.stdout.log 2>&1 &
    tp+=($!)
    sleep 5
  done
  say "ablation training started (small, 12000 steps, 4 vs 12 trajectories)"
  for q in "${tp[@]}"; do wait $q; done
  for name in 4 12; do
    O=/root/runs/abl_small_${name}traj
    $PY -c "import torch; s=torch.load('$O/last.pt',map_location='cpu',weights_only=False); print('step', s['step'])
torch.save({k:s[k] for k in ('model','step','config','protocol')},'$O/model_final.pt')" >> $LOG 2>&1 || { say "abl $name: no checkpoint"; continue; }
    $PY -u tools/eval_text.py --ckpt $O/model_final.pt --out $O/eval --text-encoder $A/chinese-clip-vit-base-patch16 \
      --judge-encoder $A/chinese-clip-vit-large-patch14-fp16 --attr-classifier $A/models/attr_resnet18.pt \
      --attributes $A/celebamask256/attributes.json --data $D/celebamask256 --captions $A/captions_clean_v1.jsonl \
      --split-manifest $A/celebamask256_split_v1.jsonl --n 192 --batch 48 > $O/eval.log 2>&1 && say "abl $name evaluated" || say "abl $name eval FAILED"
    mkdir -p $RES/abl_small_${name}traj
    cp $O/log.jsonl $O/model_final.pt $O/samples_*.png $RES/abl_small_${name}traj/ 2>/dev/null
    cp -r $O/eval/metrics.json $O/eval/*.png $RES/abl_small_${name}traj/ 2>/dev/null
  done
  say "ablation done"
}

# ---------------------------------------------------------------- run
for k in 9 10 11; do [ -d $SEQ/v2_clean_odd/t$k ] && touch $SEQ/v2_clean_odd/t$k/.ready; done
pusher & PUSHER=$!
pull > /root/plan11_pull.out 2>&1 & PULL=$!
anime
say "phase A finished"
extra_faces & EXTRA=$!
wait $PULL
ablation
while kill -0 $EXTRA 2>/dev/null && [ "$(left)" -gt 0 ]; do sleep 60; done
if kill -0 $EXTRA 2>/dev/null; then
  say "stop time: stopping extra decomposition"
  for p in $(cat /root/.lpp/extra_pids 2>/dev/null); do kill $p 2>/dev/null; done
  kill $EXTRA 2>/dev/null; sleep 10
fi
prune_extra
touch /root/.lpp/final; wait $PUSHER 2>/dev/null
push_pass
bash /root/.lpp/cs_scnet_master.sh && tar cf - -C /root plan11_results plan11.log | \
  $SSHX "mkdir -p /root/private_data/runs/plan11_4090 && tar xf - -C /root/private_data/runs/plan11_4090" && say "results pushed to SCNet runs/plan11_4090"
ssh -O exit -S /root/.lpp/scnet.sock x 2>/dev/null
shred -u /root/.lpp/scnet.env /root/.lpp/askpass.sh 2>/dev/null; rm -f /root/.lpp/scnet.env /root/.lpp/askpass.sh
say "SCNet credentials deleted; PLAN11 DONE"
