#!/bin/bash
# 5090 box -> laptop: every 5 minutes, pull finished v3 line shards (atomic on the box: decompose_lines writes
# a .tmp.npz and renames it) into data/seq/anime_v3_lines_full_<traj>. Run until killed.
set -u
cd "$(dirname "$0")/../.."
G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
while true; do
  for t in t0 t1; do
    dst=data/seq/anime_v3_lines_full_$t; mkdir -p $dst
    for f in $($G "ls /root/lpp/out/anime_v3_lines_$t 2>/dev/null | grep -E '^decomp_[0-9]+\.npz$'"); do
      [ -f $dst/$f ] && continue
      $G "cat /root/lpp/out/anime_v3_lines_$t/$f" > $dst/$f.part && mv $dst/$f.part $dst/$f && echo "$(date +%T) fetched $t/$f"
    done
    [ -f $dst/config.json ] || $G "cat /root/lpp/out/anime_v3_lines_$t/config.json" > $dst/config.json 2>/dev/null
  done
  sleep 300
done
