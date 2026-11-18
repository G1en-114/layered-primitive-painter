#!/bin/bash
# 5090 box: when the 4th batch (shards 36-47) is written, or at 21:05 UTC at the latest, restart the workers with a
# deadline so the 5th batch writes whatever fits before 21:45 UTC (voucher ends 22:00 UTC = 06:00 Beijing).
cd /root/lpp/code
O=/root/lpp/out/anime_v3_lines_t0
until [ $(ls $O | grep -cE '^decomp_0(3[6-9]|4[0-7])\.npz$') -ge 12 ] || [ $(date -u +%H%M) -ge 2105 ]; do sleep 30; done
echo "$(date -u +%T) batch 4: $(ls $O | grep -cE '^decomp_0(3[6-9]|4[0-7])\.npz$') of 12 written; switching to deadline mode" >> /root/lpp/logs/final_batch.log
for p in $(pgrep -f "python3 ops/g5090/worker.py"); do kill $p; done
sleep 2
for p in $(pgrep -f "tools/decompose_lines.py"); do kill $p; done
sleep 2
for s in $(seq 0 11); do
  setsid -f python3 ops/g5090/worker.py --gpu $((s % 2)) --slot $s --nslots 12 --traj t0 --steps 200 --deadline 21:45 \
    > /root/lpp/logs/worker_launch_$s.out 2>&1 < /dev/null
done
echo "$(date -u +%T) relaunched with deadline 21:45" >> /root/lpp/logs/final_batch.log
