#!/bin/bash
# 4090 restart at a shard boundary: MPS (concurrent kernels across processes) + compiled loss + MAXP processes.
# Finished shards are kept (atomic files, skipped on restart); only in-progress shards restart.
MAXP=${1:-5}
LOG=/root/pipeline.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
for p in $(ps -eo pid,args | awk '/[c]s_parallel.sh/ {print $1}'); do kill $p; done
for p in $(ps -eo pid,args | awk '/[d]ecompose_dataset.py/ {print $1}'); do kill $p; done
sleep 5
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
sleep 2
if /root/venv/bin/python -c "import torch; x=torch.ones(1,device='cuda'); print((x+1).item())" > /root/mps_check.log 2>&1; then
  say "MPS on"
else
  echo quit | nvidia-cuda-mps-control; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  say "MPS unavailable ($(tail -1 /root/mps_check.log)); running without it"
fi
sed -i 's/^export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=triton$/export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LPP_RASTER=triton LPP_DECOMP_COMPILE=1/' /root/cs_parallel.sh
sed -i 's/^for p in .*cs_pipeline.sh.*$/true/; s/^for p in .*decompose_dataset.py.*$/true/' /root/cs_parallel.sh
grep -n "^export\|^true" /root/cs_parallel.sh
setsid nohup bash /root/cs_parallel.sh $MAXP > /root/parallel.out 2>&1 < /dev/null &
sleep 3; ps -eo args | grep -c "[d]ecompose_dataset.py"
