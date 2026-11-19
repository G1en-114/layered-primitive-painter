#!/bin/bash
# 5090 box -> laptop, every 5 minutes until training has ended and every snapshot is here: training logs and
# sample sheets (out/v3_train/), every model snapshot (runs/v3_formal/A_lp|B_ft/model_step*.pt, written atomically
# on the box) and the train_done / train_failed flag. The laptop evaluates them (ops/v3_train/local_eval.sh):
# the box's 32 GB are full with training, and this way the box can be shut down as soon as training ends.
set -u
cd "$(dirname "$0")/../.."
G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
R=/root/lpp/v3train; O=out/v3_train; M=runs/v3_formal
mkdir -p $O $M/A_lp $M/B_ft
get() {     # get REMOTE LOCAL: whole file to .part, then rename
  $G "cat $1" > $2.part 2>/dev/null && mv $2.part $2 || { rm -f $2.part; return 1; }
}
while true; do
  bash ops/plan_v3/ensure_masters.sh g5090 > /dev/null 2>&1
  for f in runs/train.log runs/A_lp.out runs/B_ft.out runs/A_lp/log.jsonl runs/B_ft/log.jsonl; do
    get $R/$f $O/$(echo $f | tr / _)
  done
  for f in $($G "cd $R && ls runs/*/samples_*.png 2>/dev/null"); do
    l=$O/$(echo $f | tr / _); [ -f $l ] || get $R/$f $l
  done
  for f in $($G "cd $R/runs && ls */model_step*.pt 2>/dev/null"); do
    [ -f $M/$f ] || { get $R/runs/$f $M/$f && echo "$(date +%T) fetched $f"; }
  done
  ended=$($G "cd $R/runs && ls train_done train_failed 2>/dev/null")
  if [ -n "$ended" ]; then
    # one more pass for the last snapshot, then hand the flag to the local evaluation
    for f in $($G "cd $R/runs && ls */model_step*.pt 2>/dev/null"); do
      [ -f $M/$f ] || { get $R/runs/$f $M/$f && echo "$(date +%T) fetched $f"; }
    done
    touch $M/$ended
    echo "$(date +%T) training ended ($ended); all snapshots fetched: the box can be shut down"
    break
  fi
  sleep 300
done
