#!/bin/bash
# One command once the box is booted WITH the 2 GPUs (after ops/v3_train/up.sh said ALL READY):
#   bash ops/v3_train/go.sh "ssh root+vm-XXXX@HOST -p PORT"      (omit the line if unchanged)
# Checks GPUs, python and data, starts training + the evaluation watcher on the box (detached: closing the
# laptop does not stop them), starts the laptop's fetch loop (systemd --user unit lpp-v3-fetch), then shows
# live progress (Ctrl-C only stops the display). Later: bash ops/v3_train/launch.sh status
set -u
cd "$(dirname "$0")/../.."
R=/root/lpp/v3train
G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
bash ops/v3_train/connect.sh "$@" || exit 1
n=$($G 'nvidia-smi -L 2>/dev/null | wc -l')
echo "GPUs: $n"; [ "$n" -ge 2 ] || { echo "the box has no GPUs: boot it in GPU mode"; exit 1; }
bash ops/v3_train/launch.sh setup | tail -2
$G "test -f $R/data/models/chinese-clip-vit-large-patch14-fp16/model.safetensors.ok && test -f $R/runs/base_v3_lines_init.pt \
    && ls $R/data/seq/anime_v3_lines_full_t0/decomp_*.npz | wc -l" | grep -qx 129 \
  || { echo "data incomplete: run ops/v3_train/up.sh first"; exit 1; }
bash ops/v3_train/launch.sh start fetch || exit 1
echo "training started; progress every 30 s (Ctrl-C stops only this display)"
while true; do
  $G "tail -1 $R/runs/train.log; f=$R/runs/B_ft/log.jsonl; [ -f \$f ] || f=$R/runs/A_lp/log.jsonl
      [ -f \$f ] && grep records_per_s \$f | tail -1 | cut -c1-110; tail -1 $R/eval/summary.txt 2>/dev/null
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo" 2>/dev/null
  echo "--"
  sleep 30
done
