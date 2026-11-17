#!/bin/bash
# After plan11's pull: re-fetch any trajectory with the wrong shard count (a master-socket race broke the
# first t0/t1 streams) and any missing model, before the ablation needs them.
SEQ=/root/lpp/data/seq; A=/root/lpp/data/abl; LOG=/root/plan11.log
SSHX="ssh -S /root/.lpp/scnet.sock -o LogLevel=ERROR x"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
until grep -q "pull done" $LOG; do sleep 30; done
for k in 0 1 2 3 4 5 6 7; do
  want=30; [ $k -ge 4 ] && want=15
  for attempt in 1 2 3; do
    n=$(find $SEQ/scnet_v2_clean/t$k -name 'decomp_*.npz' 2>/dev/null | wc -l)
    [ "$n" = $want ] && break
    bash /root/.lpp/cs_scnet_master.sh
    rm -rf $SEQ/scnet_v2_clean/t$k
    $SSHX "tar cf - -C /root/private_data/seq/v2_clean t$k" | tar xf - -C $SEQ/scnet_v2_clean
  done
  say "repair: t$k has $(find $SEQ/scnet_v2_clean/t$k -name 'decomp_*.npz' | wc -l)/$want shards"
done
for m in chinese-clip-vit-base-patch16 chinese-clip-vit-large-patch14-fp16; do
  [ -f $A/$m/config.json ] || { $SSHX "tar cf - -C /root/private_data/models $m" | tar xf - -C $A; say "repair: re-pulled $m"; }
done
for f in models/attr_resnet18.pt captions_clean_v1.jsonl celebamask256_split_v1.jsonl celebamask256/attributes.json; do
  [ -s $A/$f ] || { $SSHX "tar cf - -C /root/private_data $f" | tar xf - -C $A; say "repair: re-pulled $f"; }
done
say "repair done"
