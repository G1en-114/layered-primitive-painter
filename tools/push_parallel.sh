#!/usr/bin/env bash
# Upload files to the DCU box over an existing SSH ControlMaster, one rsync stream per
# file (the link is throttled per connection), resuming partial files.
#   tools/push_parallel.sh <local_dir> <remote_dir> [max_parallel]
set -euo pipefail
src=$1; dst=$2; par=${3:-6}
cp="$XDG_RUNTIME_DIR/scnet.sock"
ssh -o ControlPath="$cp" -o LogLevel=ERROR x "mkdir -p '$dst'"
cd "$src"
find -L . -maxdepth 1 -type f ! -name 'aria2.*' -printf '%s %P\n' | sort -rn | cut -d' ' -f2- |
  xargs -P "$par" -I{} rsync -L --partial --append-verify -e "ssh -o ControlPath=$cp -o LogLevel=ERROR" \
    "{}" "x:$dst/{}"
echo "upload done: $src -> $dst"
