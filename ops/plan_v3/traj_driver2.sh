#!/bin/bash
# Successor to traj_driver.sh: concurrency limited by counting live decompose processes (so trajectories
# started by the previous driver count too). usage: traj_driver2.sh PART MAXP "7:1 8:0 ..."
set -u
PART=$1; MAXP=$2; JOBS=$3
R=/root/private_data; LOG=$R/traj_driver_v3.log
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
live() { ps -eo args | awk '/[d]ecompose_dataset.py/ && /seq\/v2_t/' | wc -l; }
run_one() {
  local seed=$1 flip=$2 out=$R/seq/v2_t$1
  local fl=""; [ "$flip" = "1" ] && fl="--flip"
  python -u tools/decompose_dataset.py --data $R/celebamask256 --out $out --batch 32 --steps 80 --seed $seed $fl \
    --part $PART --protocol $R/lpp_next/protocol.json > $R/decompose_v2_t${seed}_dcu.log 2>&1 || { say "t$seed decompose FAILED"; return 1; }
  say "t$seed decomposed"
  python -u tools/prune_sequences.py --src $out --out $R/seq/v2_clean/t$seed --data $R/celebamask256 --min-visible 2 \
    --protocol $R/lpp_next/protocol.json > $R/prune_v2_t${seed}.log 2>&1 && say "t$seed pruned" || say "t$seed prune FAILED"
}
say "driver2 part=$PART maxp=$MAXP jobs=[$JOBS]"
for j in $JOBS; do
  while [ "$(live)" -ge "$MAXP" ]; do sleep 30; done
  run_one ${j%%:*} ${j##*:} &
  sleep 20
done
wait
say "driver2 all done"
