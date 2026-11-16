#!/bin/bash
# 4090, after the faces: anime trajectories t2 (seed 2) and t3 (seed 3, mirrored) from the uploaded JPEG pack.
set -u
MAXP=${1:-5}
L=/root/lpp; LOG=/root/pipeline.log; P=$L/data/anime_l6_pack; SRC=$L/data/anime_l6_jpegsrc
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
until grep -q "odd t8-t11 done" $LOG; do sleep 60; done
until (cd $P && sha256sum -c --quiet /root/anime_pack.sha) > /dev/null 2>&1; do sleep 300; done
say "anime pack complete (216 files, sha ok)"
cd $L
/root/venv/bin/python -u tools/unpack_jpeg.py --pack $P --out $SRC > /root/anime_unpack.log 2>&1 || { say "anime unpack FAILED"; exit 1; }
say "anime unpacked"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LPP_RASTER=triton LPP_DECOMP_COMPILE=1
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
for j in 2:0:0/3 2:0:1/3 2:0:2/3 3:1:0/3 3:1:1/3 3:1:2/3; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 20; done
  IFS=: read s f p <<< "$j"; fl=""; [ "$f" = "1" ] && fl="--flip"
  /root/venv/bin/python -u tools/decompose_dataset.py --data $SRC --out data/seq/anime_v2_l6_t$s --batch 64 --steps 80 \
    --seed $s $fl --part $p --ids-file $P/decompose_ids_v1.json --protocol $L/protocol.json \
    > /root/decompose_anime_t${s}_${p/\//of}.log 2>&1 &
  sleep 3
done
wait
for s in 2 3; do
  n=$(find data/seq/anime_v2_l6_t$s -name 'decomp_*.npz' | wc -l)
  [ "$n" = 107 ] || { say "anime t$s: only $n/107 shards"; continue; }
  /root/venv/bin/python -u tools/prune_sequences.py --src data/seq/anime_v2_l6_t$s --out data/seq/anime_v2_l6_clean/t$s \
    --min-visible 2 --protocol $L/protocol.json > /root/prune_anime_t$s.log 2>&1 && say "anime t$s done" || say "anime t$s prune FAILED"
done
say "anime t2-t3 done"
