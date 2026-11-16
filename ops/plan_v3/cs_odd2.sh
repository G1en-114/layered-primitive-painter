#!/bin/bash
# 4090 follow-up v2: after its even shards, the ODD shards of t8..t11 (moved from the DCU).
# Odd shards = part 1/4 + part 3/4; at most MAXP processes (MPS shares the GPU between them).
set -u
MAXP=${1:-5}
L=/root/lpp; LOG=/root/pipeline.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
until grep -q "all even-shard trajectories done" $LOG; do sleep 60; done
cd $L
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LPP_RASTER=triton LPP_DECOMP_COMPILE=1
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
say "odd shards of t8-t11 start (maxp $MAXP)"
for j in 8:0:1/4 8:0:3/4 9:1:1/4 9:1:3/4 10:0:1/4 10:0:3/4 11:1:1/4 11:1:3/4; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 20; done
  IFS=: read s f p <<< "$j"; fl=""; [ "$f" = "1" ] && fl="--flip"
  /root/venv/bin/python -u tools/decompose_dataset.py --data data/celebamask256 --out data/seq/v2_t${s}_odd --batch 64 --steps 80 \
    --seed $s $fl --part $p --protocol $L/protocol.json > /root/decompose_t${s}_odd_${p/\//of}.log 2>&1 &
  sleep 3
done
wait
for s in 8 9 10 11; do
  n=$(find data/seq/v2_t${s}_odd -name 'decomp_*.npz' | wc -l)
  [ "$n" = 15 ] || { say "t$s odd: only $n/15 shards"; continue; }
  /root/venv/bin/python -u tools/prune_sequences.py --src data/seq/v2_t${s}_odd --out data/seq/v2_clean_odd/t$s --min-visible 2 \
    --protocol $L/protocol.json > /root/prune_t${s}_odd.log 2>&1 && say "t$s odd done" || say "t$s odd prune FAILED"
done
say "odd t8-t11 done"
