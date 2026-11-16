#!/bin/bash
# Pull every finished, pruned trajectory from the 4090 to the local machine (download direction only).
# Reads /root/pipeline.log on the 4090; marks each pulled dir with .pulled_ok once its shard count is right.
set -a; . ~/.config/lpp/remotes.env; set +a
L=/home/martis/picture_train/data/seq/from4090
SSH="ssh -F /dev/null -S /run/user/1000/compshare.sock -o BatchMode=yes"   # multiplexed over the master: the gateway caps concurrent TCP connections
R="$COMPSHARE_USER@$COMPSHARE_HOST"
while true; do
  /tmp/claude-1000/-home-martis-picture-train/3ef905ae-8a45-4a59-abc0-5d18646d9715/scratchpad/ensure_masters.sh compshare
  log=$($SSH $R 'cat /root/pipeline.log' 2>/dev/null) || { sleep 120; continue; }
  # "tK done" -> v2_clean_even/tK (15 shards); "tK odd done" -> v2_clean_odd/tK (15); "anime tK done" -> anime (107)
  for item in $(echo "$log" | grep -oE '(anime )?t[0-9]+ (odd )?done' | sed -E 's/^anime t([0-9]+) done/anime_v2_l6_clean:\1:107/; s/^t([0-9]+) odd done/v2_clean_odd:\1:15/; s/^t([0-9]+) done/v2_clean_even:\1:15/'); do
    IFS=: read d t n <<< "$item"
    dst=$L/$d/t$t; [ -f $dst/.pulled_ok ] && continue
    mkdir -p $dst
    rsync -a --partial --timeout=180 -e "$SSH" "$R:/root/lpp/data/seq/$d/t$t/" $dst/ || continue
    got=$(find $dst -name 'decomp_*.npz' | wc -l)
    if [ "$got" = "$n" ]; then touch $dst/.pulled_ok; echo "$(date -u +%FT%TZ) pulled $d/t$t ($got shards)"; fi
  done
  echo "$log" | grep -q "anime t2-t3 done" && [ -f $L/anime_v2_l6_clean/t2/.pulled_ok ] && [ -f $L/anime_v2_l6_clean/t3/.pulled_ok ] && { echo "$(date -u +%FT%TZ) relay complete"; exit 0; }
  sleep 300
done
