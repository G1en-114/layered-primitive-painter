#!/bin/bash
# Event stream for Claude's Monitor: new lines of the SCNet autopilot log, base_v3 validation / checkpoint
# failures, the 4090 plan11 log (only with WATCH_4090=1); SSH failures only after 3 in a row. Each event printed once.
SEEN=/tmp/claude-1000/watch_remote.seen; touch $SEEN
emit() { grep -qxF -- "$1" $SEEN || { echo "$1" >> $SEEN; echo "$1"; }; }
fs=0; fc=0
while true; do
  bash /home/martis/picture_train/ops/plan_v3/ensure_masters.sh scnet ${WATCH_4090:+compshare} > /dev/null 2>&1
  out=$(timeout 60 ssh -F /dev/null -S /run/user/1000/scnet.sock -o BatchMode=yes -o LogLevel=ERROR x \
    'df -P /root/private_data | awk "NR==2 && \$5+0 >= 90 {print \"DISK \" \$5 \" used on SCNet private_data\"}"; tail -n 30 /root/private_data/autopilot_v3.log; tail -n 20 /root/private_data/grpo_v3.log 2>/dev/null; tail -n 12 /root/private_data/eval/grpo_v3_summary.txt 2>/dev/null; tail -n 12 /root/private_data/draft_ablation.log 2>/dev/null; tail -n 12 /root/private_data/eval/domain_summary.txt 2>/dev/null; grep -E "\"val\"|checkpoint|step\": [0-9]*000," /root/private_data/runs/base_v3/log.jsonl 2>/dev/null | grep -E "\"val\"|checkpoint" | tail -n 5 | cut -c1-200; true' 2>/dev/null)
  if [ -z "$out" ]; then fs=$((fs+1)); [ $fs = 3 ] && emit "SCNet unreachable x3 $(date -u +%H:%M)"; else fs=0
    while read -r l; do [ -n "$l" ] && emit "scnet: $l"; done <<< "$out"; fi
  [ -n "${WATCH_4090:-}" ] && out=$(timeout 60 ssh -F /dev/null -S /run/user/1000/compshare.sock -o BatchMode=yes -o LogLevel=ERROR c 'df -P / | awk "NR==2 && \$5+0 >= 90 {print \"DISK \" \$5 \" used on the 4090\"}"; tail -n 30 /root/plan11.log; true' 2>/dev/null)
  [ -z "${WATCH_4090:-}" ] && out=skip
  if [ "$out" = skip ]; then :; elif [ -z "$out" ]; then fc=$((fc+1)); [ $fc = 3 ] && emit "4090 unreachable x3 $(date -u +%H:%M)"; else fc=0
    while read -r l; do [ -n "$l" ] && emit "4090: $l"; done <<< "$out"; fi
  sleep 120
done
