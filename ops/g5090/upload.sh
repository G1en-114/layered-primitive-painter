#!/bin/bash
# Laptop -> 5090 box: everything the line decomposition needs, compressed, 4 streams, each file atomically
# renamed into place and marked FILE.ok when complete (the workers wait for the markers).
set -u
cd "$(dirname "$0")/../.."
G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
R=/root/lpp/data
put() {   # put LOCAL REMOTE_REL
  local l=$1 r=$R/$2
  $G "test -f $r.ok" 2>/dev/null && return 0
  zstd -q -3 -T2 -c < "$l" | $G "mkdir -p \$(dirname $r) && zstd -q -d -c > $r.tmp && mv $r.tmp $r && touch $r.ok" \
    || { echo "FAILED $l"; return 1; }
}
export -f put; export G R
V=data/anime_portraits256_v2_l6view
put $V/index.json anime_portraits256_v2_l6view/index.json
for t in t0 t1; do
  ls data/seq/anime_v2_l6_$t/ | grep -E "decomp_[0-9]+.npz|config.json" | sed "s|^|data/seq/anime_v2_l6_$t/ |" \
    | xargs -P 4 -L 1 bash -c 'put "$0$1" "seq/anime_v2_l6_'$t'/$1"'
  $G "touch $R/seq/upload_done_$t"
  echo "$(date +%T) seq $t uploaded"
  [ $t = t0 ] || continue
  for k in $(seq -f %03g 0 107); do echo "$V/ images_$k.npy"; echo "$V/ labels_$k.npy"; done \
    | xargs -P 4 -L 1 bash -c 'put "$0$1" "anime_portraits256_v2_l6view/$1"'
  echo "$(date +%T) images uploaded"
done
echo "upload finished"
