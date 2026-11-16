#!/bin/bash
# 4090 -> SCNet direct push (the local relay machine is off). For every finished, pruned trajectory:
# tar | xz -> SCNet upload_v3/<kind>/tK, SHA-256 check there, then touch .complete (SCNet's autopilot keys on it).
# When everything is pushed, or after DEADLINE hours, the SCNet credentials are deleted from this machine.
set -u
L=/root/lpp/data/seq; LOG=/root/pipeline.log; PLOG=/root/push.log
DEADLINE=$(( $(date +%s) + 10 * 3600 ))
say() { echo "$(date -u +%FT%TZ) $*" >> $PLOG; echo "$(date -u +%FT%TZ) push: $*" >> $LOG; }
SSH="ssh -S /root/.lpp/scnet.sock -o LogLevel=ERROR x"
wipe() { ssh -O exit -S /root/.lpp/scnet.sock x 2>/dev/null; shred -u /root/.lpp/scnet.env /root/.lpp/askpass.sh 2>/dev/null
         rm -f /root/.lpp/scnet.env /root/.lpp/askpass.sh; say "credentials deleted ($1)"; }
push() {   # push KIND T
  local kind=$1 t=$2 dst=/root/private_data/seq/upload_v3/$1
  bash /root/.lpp/cs_scnet_master.sh || return 1
  $SSH "test -f $dst/t$t/.complete" && return 0
  local sums=/root/.lpp/sha_${kind}_t$t
  (cd $L/$kind && find t$t -type f \( -name 'decomp_*.npz' -o -name config.json \) | sort | xargs sha256sum) > $sums
  local t0=$(date +%s) bytes=$(du -sb $L/$kind/t$t | cut -f1)
  tar cf - -C $L/$kind t$t | xz -T0 -6 | $SSH "mkdir -p $dst && cd $dst && xz -d | tar xf -" || return 1
  $SSH "cd $dst && sha256sum -c --quiet && touch t$t/.complete" < $sums || { say "SHA mismatch $kind/t$t"; return 1; }
  say "pushed $kind/t$t ($((bytes/1048576)) MiB in $(( $(date +%s) - t0 )) s)"
}
# item: KIND:T:LOGPATTERN  (log lines written by cs_parallel / cs_odd2 / cs_anime / cs_anime_t1)
ITEMS=("v2_clean_even:9:Z t9 done$" "v2_clean_even:10:Z t10 done$" "v2_clean_even:11:Z t11 done$"
       "v2_clean_odd:8:Z t8 odd done$" "v2_clean_odd:9:Z t9 odd done$" "v2_clean_odd:10:Z t10 odd done$" "v2_clean_odd:11:Z t11 odd done$"
       "anime_v2_l6_clean:2:Z anime t2 done$" "anime_v2_l6_clean:3:Z anime t3 done$" "anime_v2_l6_clean:1:Z anime t1 done$")
say "push loop started"
while true; do
  left=0
  for item in "${ITEMS[@]}"; do
    kind=${item%%:*}; rest=${item#*:}; t=${rest%%:*}; pat=${rest#*:}
    [ -f /root/.lpp/pushed_${kind}_t$t ] && continue
    left=$((left + 1))
    grep -qE "$pat" $LOG || continue
    push $kind $t && touch /root/.lpp/pushed_${kind}_t$t && left=$((left - 1))
  done
  [ $left = 0 ] && { say "PUSH-ALL-DONE"; wipe "all pushed"; exit 0; }
  [ $(date +%s) -gt $DEADLINE ] && { say "deadline reached with $left items left"; wipe "deadline"; exit 1; }
  sleep 120
done
