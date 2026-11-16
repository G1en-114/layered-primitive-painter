#!/bin/bash
# 4090, after the faces: anime trajectory t1 (seed 1, mirrored) after t2/t3 (the local t1 run was cut short) from the uploaded JPEG pack.
set -u
MAXP=${1:-5}
L=/root/lpp; LOG=/root/pipeline.log; P=$L/data/anime_l6_pack; SRC=$L/data/anime_l6_jpegsrc
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
until grep -q "anime t2-t3 done" $LOG; do sleep 60; done
true
say "anime t1 start"
cd $L
true
true
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LPP_RASTER=triton LPP_DECOMP_COMPILE=1
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
for j in 1:1:0/3 1:1:1/3 1:1:2/3; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 20; done
  IFS=: read s f p <<< "$j"; fl=""; [ "$f" = "1" ] && fl="--flip"
  /root/venv/bin/python -u tools/decompose_dataset.py --data $SRC --out data/seq/anime_v2_l6_t$s --batch 64 --steps 80 \
    --seed $s $fl --part $p --ids-file $P/decompose_ids_v1.json --protocol $L/protocol.json \
    > /root/decompose_anime_t${s}_${p/\//of}.log 2>&1 &
  sleep 3
done
wait
for s in 1; do
  n=$(find data/seq/anime_v2_l6_t$s -name 'decomp_*.npz' | wc -l)
  [ "$n" = 107 ] || { say "anime t$s: only $n/107 shards"; continue; }
  /root/venv/bin/python -u tools/prune_sequences.py --src data/seq/anime_v2_l6_t$s --out data/seq/anime_v2_l6_clean/t$s \
    --min-visible 2 --protocol $L/protocol.json > /root/prune_anime_t$s.log 2>&1 && say "anime t$s done" || say "anime t$s prune FAILED"
done
say "anime t1 all done"
