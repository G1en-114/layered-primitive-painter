#!/bin/bash
# 4090: faces t4..t11 (even shards) with MAXP decomposition processes in parallel (one process per
# trajectory underuses the GPU: CPU-side launch overhead), pruning each trajectory when it finishes.
set -u
MAXP=${1:-4}
L=/root/lpp; LOG=/root/pipeline.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
for p in $(ps -eo pid,args | awk '/[c]s_pipeline.sh/ {print $1}'); do kill $p; done
for p in $(ps -eo pid,args | awk '/[d]ecompose_dataset.py/ {print $1}'); do kill $p; done
sleep 3
cd $L
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=triton
run_one() {
  local s=$1 f=$2 fl=""; [ "$f" = "1" ] && fl="--flip"
  /root/venv/bin/python -u tools/decompose_dataset.py --data data/celebamask256 --out data/seq/v2_t$s --batch 64 --steps 80 \
    --seed $s $fl --part 0/2 --protocol $L/protocol.json >> /root/decompose_t$s.log 2>&1 || { say "t$s decompose FAILED"; return; }
  /root/venv/bin/python -u tools/prune_sequences.py --src data/seq/v2_t$s --out data/seq/v2_clean_even/t$s --min-visible 2 \
    --protocol $L/protocol.json > /root/prune_t$s.log 2>&1 && say "t$s done" || say "t$s prune FAILED"
}
say "parallel driver: maxp=$MAXP"
for j in 4:0 5:1 6:0 7:1 8:0 9:1 10:0 11:1; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 30; done
  run_one ${j%%:*} ${j##*:} &
  sleep 3
done
wait
say "all even-shard trajectories done"
