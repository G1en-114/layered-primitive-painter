#!/bin/bash
# Finish extra face trajectory t12 after plan11 (5 MPS processes), then prune it. Relay to SCNet is done from the local machine.
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
export LPP_RASTER=triton LPP_DECOMP_COMPILE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=/root/lpp; D=$L/data; SEQ=$D/seq; PY=/root/venv/bin/python; LOG=/root/plan11.log
cd $L
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ulimit -c 0
for p in 0 1 2 3 4; do
  $PY -u tools/decompose_dataset.py --data $D/celebamask256 --out $SEQ/v2_t12 --batch 64 --steps 80 --seed 12 \
    --part $p/5 --protocol $L/protocol.json > /root/t12_part${p}of5.log 2>&1 &
  sleep 3
done
say "extra face t12 resumed by hand (5 processes, 26 shards left)"
wait
n=$(find $SEQ/v2_t12 -name "decomp_*.npz" | wc -l)
[ "$n" = 30 ] || { say "extra face t12 still incomplete ($n/30)"; exit 1; }
$PY -u tools/prune_sequences.py --src $SEQ/v2_t12 --out $SEQ/v2_clean_extra/t12 --min-visible 2 --protocol $L/protocol.json \
  > /root/t12_prune.log 2>&1 && touch $SEQ/v2_clean_extra/t12/.ready && say "extra face t12 done (pruned; relay to SCNet from local)" || say "extra face t12 prune FAILED"
