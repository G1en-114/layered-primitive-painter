#!/bin/bash
# (Re)open the SSH master connections if they are gone. Credentials come from ~/.config/lpp/remotes.env;
# the SCNet password is fed by the askpass helper and never appears on a command line.
set -a; . ~/.config/lpp/remotes.env; set +a
for which in "$@"; do
  case $which in
    scnet)
      ssh -O check -S /run/user/1000/scnet.sock x 2>/dev/null && continue
      rm -f /run/user/1000/scnet.sock
      LPP_PW_VAR=SCNET_PW SSH_ASKPASS=$HOME/.config/lpp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid \
        ssh -F /dev/null -fNM -S /run/user/1000/scnet.sock -o ControlPersist=yes -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=10 -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no \
        -o LogLevel=ERROR -p "$SCNET_PORT" "$SCNET_USER@$SCNET_HOST" < /dev/null && echo "$(date -u +%FT%TZ) reopened scnet master" ;;
    compshare)
      ssh -O check -S /run/user/1000/compshare.sock c 2>/dev/null && continue
      rm -f /run/user/1000/compshare.sock
      ssh -F /dev/null -fNM -S /run/user/1000/compshare.sock -o ControlPersist=yes -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=10 -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR -i $HOME/.ssh/lpp_compshare \
        -p "$COMPSHARE_PORT" "$COMPSHARE_USER@$COMPSHARE_HOST" && echo "$(date -u +%FT%TZ) reopened compshare master" ;;
  esac
done
