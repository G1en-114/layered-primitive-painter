#!/bin/bash
# Pull the ablation inputs from SCNet with per-file SHA-256 verification (SCNet's ContainerSSH gateway has
# cut tar streams when a push and a pull shared the connection). Retries each item up to 5 times.
SEQ=/root/lpp/data/seq; A=/root/lpp/data/abl; LOG=/root/plan11.log
SSHX="ssh -S /root/.lpp/scnet.sock -o LogLevel=ERROR x"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
fetch() {   # fetch REMOTE_PARENT ITEM LOCAL_PARENT
  local rp=$1 item=$2 lp=$3
  mkdir -p $lp
  $SSHX "cd $rp && find $item -type f ! -name '.*' | sort | xargs sha256sum" > /tmp/want.sha || return 1
  [ -e $lp/$item ] || $SSHX "tar cf - -C $rp $item" | tar xf - -C $lp
  for attempt in 1 2 3 4 5 6 7 8; do
    bad=$(cd $lp && sha256sum -c /tmp/want.sha 2>/dev/null | grep -v ': OK$' | sed 's/: FAILED.*//')
    [ -z "$bad" ] && { say "verify: $item ok ($(wc -l < /tmp/want.sha) files)"; return 0; }
    bash /root/.lpp/cs_scnet_master.sh
    for f in $bad; do              # tar streams get cut near the end: fetch the failing files one by one
      mkdir -p $lp/$(dirname $f); $SSHX "cat $rp/$f" > $lp/$f.part && mv $lp/$f.part $lp/$f
    done
  done
  say "verify: $item FAILED after 5 attempts"; return 1
}
for k in 0 1 2 3 4 5 6 7; do fetch /root/private_data/seq/v2_clean t$k $SEQ/scnet_v2_clean; done
for m in chinese-clip-vit-base-patch16 chinese-clip-vit-large-patch14-fp16; do fetch /root/private_data/models $m $A; done
fetch /root/private_data/models attr_resnet18.pt $A/models
for f in captions_clean_v1.jsonl celebamask256_split_v1.jsonl; do fetch /root/private_data $f $A; done
fetch /root/private_data/celebamask256 attributes.json $A/celebamask256
say "verified pull done"
