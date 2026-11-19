#!/bin/bash
# Connect to the 5090 box: connect.sh ["ssh root+vm-XXXX@HOST -p PORT"]. Updates G5090_USER/HOST/PORT in
# ~/.config/lpp/remotes.env when the line differs (the password stays), forgets the old host key (the console
# may reissue it after a mode switch) and opens the SSH master at /run/user/1000/g5090.sock.
set -u
cd "$(dirname "$0")/../.."
ENV=~/.config/lpp/remotes.env; G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
if [ $# -ge 1 ]; then
  [[ $1 =~ ([^[:space:]@]+)@([^[:space:]]+)[[:space:]]+-p[[:space:]]*([0-9]+) ]] || { echo "cannot parse: $1"; exit 1; }
  u=${BASH_REMATCH[1]} h=${BASH_REMATCH[2]} p=${BASH_REMATCH[3]}
  . $ENV
  if [ "$u" != "$G5090_USER" ] || [ "$h" != "$G5090_HOST" ] || [ "$p" != "$G5090_PORT" ]; then
    sed -i "s|^G5090_USER=.*|G5090_USER=$u|; s|^G5090_HOST=.*|G5090_HOST=$h|; s|^G5090_PORT=.*|G5090_PORT=$p|" $ENV
    echo "box address updated in remotes.env"
  fi
fi
ssh -O exit -S /run/user/1000/g5090.sock x 2>/dev/null       # a reboot leaves a dead master behind
rm -f ~/.config/lpp/known_hosts_g5090                         # the console may hand out a new host key
bash ops/plan_v3/ensure_masters.sh g5090
$G true || { echo "cannot connect (box still booting? retry in a minute)"; exit 1; }
