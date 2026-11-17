#!/bin/bash
# Relay the hand-finished extra face trajectory t12: 4090 -> local -> SCNet (seq/upload_v3/v2_clean_extra/t12),
# SHA-checked on arrival like the 4090's own pushes, then marked .complete.
set -u
cd /home/martis/picture_train
C=(ssh -F /dev/null -S /run/user/1000/compshare.sock -o BatchMode=yes -o LogLevel=ERROR c)
S=(ssh -F /dev/null -S /run/user/1000/scnet.sock -o BatchMode=yes -o LogLevel=ERROR x)
L=data/seq/from4090/v2_clean_extra/t12; mkdir -p $L
until bash ops/plan_v3/ensure_masters.sh compshare >/dev/null 2>&1 && "${C[@]}" 'test -f /root/lpp/data/seq/v2_clean_extra/t12/.ready'; do sleep 60; done
echo "$(date -u +%T) t12 ready on the 4090"
for i in 1 2 3 4 5; do
  bash ops/plan_v3/ensure_masters.sh compshare >/dev/null 2>&1
  rsync -a --partial --timeout=180 -e "ssh -F /dev/null -S /run/user/1000/compshare.sock -o BatchMode=yes" \
    c:/root/lpp/data/seq/v2_clean_extra/t12/ $L/ && break
done
n=$(find $L -name 'decomp_*.npz' | wc -l); echo "$(date -u +%T) pulled $n shards"; [ "$n" = 30 ] || exit 1
(cd data/seq/from4090/v2_clean_extra && find t12 -type f \( -name 'decomp_*.npz' -o -name config.json \) | sort | xargs sha256sum) > /tmp/claude-1000/t12.sha
for i in 1 2 3 4 5; do
  bash ops/plan_v3/ensure_masters.sh scnet >/dev/null 2>&1
  tar cf - -C data/seq/from4090/v2_clean_extra --exclude=.ready t12 | xz -T4 -6 | \
    "${S[@]}" 'mkdir -p /root/private_data/seq/upload_v3/v2_clean_extra && cd /root/private_data/seq/upload_v3/v2_clean_extra && xz -d | tar xf -' || continue
  "${S[@]}" 'cd /root/private_data/seq/upload_v3/v2_clean_extra && sha256sum -c --quiet && touch t12/.complete' < /tmp/claude-1000/t12.sha && { echo "$(date -u +%T) t12 on SCNet, SHA ok"; exit 0; }
done
echo "$(date -u +%T) t12 upload FAILED"; exit 1
