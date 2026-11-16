#!/bin/bash
# open (or reuse) the 4090 -> SCNet master connection; password via askpass from /root/.lpp/scnet.env (600)
. /root/.lpp/scnet.env
SOCK=/root/.lpp/scnet.sock
ssh -O check -S $SOCK x 2>/dev/null && exit 0
rm -f $SOCK
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  SSH_ASKPASS=/root/.lpp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid \
  ssh -F /dev/null -fNM -S $SOCK -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
  -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o LogLevel=ERROR \
  -p "$SCNET_PORT" "$SCNET_USER@$SCNET_HOST" < /dev/null
