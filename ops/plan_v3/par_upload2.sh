#!/bin/bash
# resumable parallel upload of the anime pack to the 4090: 4 streams, each its own TCP connection,
# retried until every stream exits cleanly (the provider's TCP tunnel resets connections now and then)
set -a; . ~/.config/lpp/remotes.env; set +a
S=/tmp/claude-1000/-home-martis-picture-train/3ef905ae-8a45-4a59-abc0-5d18646d9715/scratchpad
cd /home/martis/picture_train/data/anime_l6_pack
SSH="ssh -F /dev/null -o ControlPath=none -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -o ConnectTimeout=30 -i $HOME/.ssh/lpp_compshare -p $COMPSHARE_PORT"
for attempt in $(seq 1 40); do
  pids=()
  for f in $S/anime_q_a?; do
    rsync -a --partial --timeout=180 --files-from=$f -e "$SSH" . "$COMPSHARE_USER@$COMPSHARE_HOST:/root/lpp/data/anime_l6_pack/" &
    pids+=($!)
  done
  ok=1; for p in "${pids[@]}"; do wait $p || ok=0; done
  if [ $ok = 1 ]; then echo "UPLOAD-OK after $attempt attempt(s) $(date)"; exit 0; fi
  echo "attempt $attempt incomplete $(date); retrying in 30 s"; sleep 30
done
echo "UPLOAD-GAVE-UP"
