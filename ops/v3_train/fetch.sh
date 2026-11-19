#!/bin/bash
# 5090 box -> laptop, every 10 minutes until the box's eval watcher has finished: training logs, the evaluation
# summary, sample sheets (out/v3_train/) and every model snapshot (runs/v3_formal/, written atomically on the box).
set -u
cd "$(dirname "$0")/../.."
G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
R=/root/lpp/v3train; O=out/v3_train; M=runs/v3_formal
mkdir -p $O $M
get() {     # get REMOTE LOCAL: whole file to .part, then rename
  $G "cat $1" > $2.part 2>/dev/null && mv $2.part $2 || rm -f $2.part
}
while true; do
  bash ops/plan_v3/ensure_masters.sh g5090 > /dev/null 2>&1
  for f in runs/train.log runs/A_lp.out runs/B_ft.out runs/A_lp/log.jsonl runs/B_ft/log.jsonl eval/summary.txt eval/eval_watch.log; do
    get $R/$f $O/$(echo $f | tr / _)
  done
  for f in $($G "cd $R && ls runs/*/samples_*.png eval/*/grid*.png 2>/dev/null"); do
    l=$O/$(echo $f | tr / _); [ -f $l ] || get $R/$f $l
  done
  for f in $($G "cd $R/runs && ls */model_step*.pt 2>/dev/null"); do
    l=$M/$(echo $f | tr / _); [ -f $l ] || { get $R/runs/$f $l && echo "$(date +%T) fetched $f"; }
  done
  grep -q "eval watcher finished" $O/eval_eval_watch.log 2>/dev/null && { echo "$(date +%T) all done"; break; }
  sleep 600
done
