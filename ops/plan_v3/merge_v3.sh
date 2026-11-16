#!/bin/bash
# Build the base-v3 training set on SCNet from hard links (no extra space):
#   faces tK (K=0..11): odd shards (DCU v2_clean/tK, or 4090 upload_v3/v2_clean_odd/tK for K>=8) + even shards
#                       (v2_clean/tK for K<4, upload_v3/v2_clean_even/tK for K>=4) -> v3_mix/face_tK (30 shards)
#   anime tK:           anime t0/t1 from the local machine, t2/t3 from the 4090 -> v3_mix/anime_tK (107 shards)
# Every directory must carry the same protocol SHA; prints a summary; exits non-zero if anything is missing.
# usage: merge_v3.sh [ANIME_TRAJ...]   (default: 0 1)
set -u
R=/root/private_data/seq; M=$R/v3_mix; U=$R/upload_v3
PROTO=bb54b53748557229f1135f26ad5ed9a750ff7410170930c396e47e5787a9d58c
ANIME=${*:-0 1}
mkdir -p $M
fail=0
link_from() {   # link_from SRC DST: hard-link decomp_*.npz (and config.json if absent)
  local src=$1 dst=$2
  [ -d "$src" ] || { echo "missing $src"; fail=1; return; }
  local sha; sha=$(python -c "import json;print(json.load(open('$src/config.json')).get('protocol_sha256'))")
  [ "$sha" = "$PROTO" ] || { echo "protocol mismatch in $src: $sha"; fail=1; return; }
  mkdir -p "$dst"
  for f in "$src"/decomp_*.npz; do ln -f "$f" "$dst/"; done
  [ -f "$dst/config.json" ] || cp "$src/config.json" "$dst/config.json"
}
for k in $(seq 0 11); do
  d=$M/face_t$k
  if [ $k -lt 4 ]; then link_from $R/v2_clean/t$k $d
  else
    link_from $U/v2_clean_even/t$k $d
    if [ $k -lt 8 ]; then link_from $R/v2_clean/t$k $d; else link_from $U/v2_clean_odd/t$k $d; fi
  fi
  n=$(find $d -name 'decomp_*.npz' 2>/dev/null | wc -l)
  [ "$n" = 30 ] || { echo "face_t$k: $n/30 shards"; fail=1; }
done
for k in $ANIME; do
  d=$M/anime_t$k
  link_from $U/anime_v2_l6_clean/t$k $d
  n=$(find $d -name 'decomp_*.npz' 2>/dev/null | wc -l)
  [ "$n" = 107 ] || { echo "anime_t$k: $n/107 shards"; fail=1; }
done
echo "v3_mix: $(find $M -name 'decomp_*.npz' | wc -l) shards in $(find $M -mindepth 1 -maxdepth 1 -type d | wc -l) trajectories; fail=$fail"
exit $fail
