#!/bin/bash
# up_scnet.sh LOCAL_PARENT REMOTE_PARENT NAME...: stream NAME dirs (tar | xz) to SCNet over the master
# socket, then verify every file's SHA-256 remotely. Prints throughput.
set -u
LP=$1; RP=$2; shift 2
S=/tmp/claude-1000/-home-martis-picture-train/3ef905ae-8a45-4a59-abc0-5d18646d9715/scratchpad
SSH="ssh -F /dev/null -S /run/user/1000/scnet.sock -o BatchMode=yes -o LogLevel=ERROR x"
/tmp/claude-1000/-home-martis-picture-train/3ef905ae-8a45-4a59-abc0-5d18646d9715/scratchpad/ensure_masters.sh scnet >&2
cd "$LP"
sums=$S/up_$(echo "$*" | tr ' /' '__').sha
find "$@" -type f ! -name '.pulled_ok' -print0 | sort -z | xargs -0 sha256sum > "$sums"
bytes=$(find "$@" -type f ! -name '.pulled_ok' -printf '%s\n' | awk '{s+=$1} END {print s}')
t0=$(date +%s)
tar cf - --exclude=.pulled_ok "$@" | xz -T0 -6 | $SSH "mkdir -p $RP && cd $RP && xz -d | tar xf -" || { echo "UPLOAD FAILED: $*"; exit 1; }
t1=$(date +%s)
$SSH "cd $RP && sha256sum -c --quiet" < "$sums" && echo "OK $* -> $RP: $((bytes/1048576)) MiB raw in $((t1-t0)) s ($((bytes/1024/(t1-t0+1))) KiB/s raw)" || echo "SHA MISMATCH: $*"
