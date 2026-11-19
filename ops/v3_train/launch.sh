#!/bin/bash
# Laptop -> rented 2x5090 box: the formal v3 training (ops/v3_train/train.sh) and its evaluation watcher.
#   ops/v3_train/launch.sh all         connect, set up python, upload (resumable), start training + watcher, fetch
#   ops/v3_train/launch.sh STEP...     any of: connect setup upload models start fetch status
# Box credentials: G5090_HOST/PORT/USER/PW in ~/.config/lpp/remotes.env (a NEW box: update them and pass
# NEW_BOX=1 to forget the old host key). Code goes up as `git archive HEAD` (commit first).
# On the box: /root/lpp/v3train/{code,data,runs,eval}; logs runs/train.log, runs/*.out, eval/summary.txt.
# Fetched to the laptop (every 10 min, systemd --user unit lpp-v3-fetch): out/v3_train/ (logs, summary,
# sample sheets) and runs/v3_formal/ (every model snapshot).
set -u
cd "$(dirname "$0")/../.."
G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
R=/root/lpp/v3train
connect() {
  [ "${NEW_BOX:-0}" = 1 ] && rm -f ~/.config/lpp/known_hosts_g5090 && ssh -O exit -S /run/user/1000/g5090.sock x 2>/dev/null
  bash ops/plan_v3/ensure_masters.sh g5090
  $G "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; nproc; free -g | sed -n 2p"
}
setup() {   # torch comes with the image; add what is missing without letting pip move numpy (numpy 2 broke torch)
  $G 'python - <<EOF
import importlib
miss = [m for m in ("numpy", "PIL", "transformers", "triton", "safetensors") if importlib.util.find_spec(m) is None]
import torch, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs; numpy", numpy.__version__)
print("missing:", " ".join(miss) or "-")
EOF'
  local miss; miss=$($G "python -c 'import importlib.util as u; print(\" \".join(m for m in (\"PIL\",\"transformers\",\"safetensors\") if u.find_spec(m) is None))'")
  [ -z "$miss" ] && return 0
  miss=${miss/PIL/pillow}
  # the box's torch is NVIDIA's 2.6.0a0 build: transformers 5.x rejects it (needs >= 2.6) and 4.50+ refuse to
  # torch.load the text encoder's pytorch_model.bin below 2.6, so 4.48.3; system python (PEP 668) -> --break-system-packages
  miss=${miss/transformers/transformers==4.48.3}
  $G "pip install -q --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple $miss \"numpy==\$(python -c 'import numpy; print(numpy.__version__)')\""
}
put() {     # put LOCAL REMOTE_REL: atomic, skipped when already there (FILE.ok)
  local l=$1 r=$R/$2
  $G "test -f $r.ok" 2>/dev/null && return 0
  # zstd: the token shards halve; the laptop's uplink (~0.3 MB/s through the proxy) is the bottleneck
  zstd -q -9 -T4 -c < "$l" | $G "mkdir -p \$(dirname $r) && zstd -q -d -c > $r.tmp && mv $r.tmp $r && touch $r.ok" \
    || { echo "FAILED $l"; return 1; }
}
upload() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || echo "WARNING: uncommitted changes are NOT uploaded"
  git archive HEAD | $G "rm -rf $R/code.new && mkdir -p $R/code.new && tar -x -C $R/code.new && rm -rf $R/code && mv $R/code.new $R/code"
  echo "$(date +%T) code $(git rev-parse --short HEAD) uploaded"
  [ -f runs/base_v3_lines_init.pt ] || "$HOME/AI infra/.venv/bin/python" tools/init_v3_from_v2.py \
    --ckpt runs/base_v3_snap/model_final.pt --out runs/base_v3_lines_init.pt
  local f
  {
    echo runs/base_v3_lines_init.pt
    for d in v2 v2_t1 v2_t2 v2_t3 anime_v3_lines_full_t0; do ls data/seq/$d/decomp_*.npz data/seq/$d/config.json; done
    ls data/captions/captions_v3.jsonl data/celebamask256/attributes.json data/split_v3_faces_anime.jsonl \
       data/celebamask256_split_v1.jsonl data/celebamask256_val/* data/anime_val192/*
    ls data/models/chinese-clip-vit-base-patch16/SHA256SUMS data/models/chinese-clip-vit-large-patch14/SHA256SUMS
  } > out/v3_upload_list.txt
  export -f put; export G R
  sed 's|.*|& &|' out/v3_upload_list.txt | xargs -P 4 -L 1 bash -c 'put "$0" "$1"'
  local n; n=$($G "find $R -name '*.ok' | wc -l")
  echo "$(date +%T) data uploaded: $n of $(wc -l < out/v3_upload_list.txt) files on the box"
  [ "$n" -ge "$(wc -l < out/v3_upload_list.txt)" ]
}
models() {  # the public text encoders: the box downloads them itself (hf-mirror ~8 MB/s), checked against SHA256SUMS;
            # the judge is the large model cast to fp16 (as data/models/chinese-clip-vit-large-patch14-fp16)
  $G "test -f $R/data/models/chinese-clip-vit-large-patch14-fp16/model.safetensors.ok" && return 0
  $G "test -f $R/data/models/chinese-clip-vit-large-patch14/SHA256SUMS.ok" || { echo "upload first"; return 1; }
  $G 'set -e; M='$R'/data/models
    for m in chinese-clip-vit-base-patch16 chinese-clip-vit-large-patch14; do
      cd $M/$m
      for f in $(awk "{print \$2}" SHA256SUMS); do
        sha256sum -c --status <(grep " $f\$" SHA256SUMS) 2>/dev/null && continue
        { curl -sfL --retry 3 -o $f.part https://hf-mirror.com/OFA-Sys/$m/resolve/main/$f \
          || curl -sfL --retry 3 -o $f.part https://www.modelscope.cn/models/AI-ModelScope/$m/resolve/master/$f; } \
          && mv $f.part $f
      done
      sha256sum -c SHA256SUMS
    done
    cd $M && python3 -c "
import torch; from transformers import ChineseCLIPModel
ChineseCLIPModel.from_pretrained(\"chinese-clip-vit-large-patch14\").half().save_pretrained(\"chinese-clip-vit-large-patch14-fp16\")"
    cp chinese-clip-vit-large-patch14/{configuration.json,preprocessor_config.json,vocab.txt} chinese-clip-vit-large-patch14-fp16/
    touch chinese-clip-vit-large-patch14-fp16/model.safetensors.ok; ls -la chinese-clip-vit-large-patch14-fp16'
}
start() {   # training only: the box's 32 GB per GPU are full with it, the laptop evaluates (local_eval.sh)
  $G "pgrep -f '[v]3_train/train.sh' > /dev/null" && { echo "training already running"; return 0; }
  $G "cd $R || exit 1; setsid nohup bash code/ops/v3_train/train.sh > runs_train.nohup 2>&1 < /dev/null &
      sleep 3; pgrep -af '[v]3_train/train.sh'"
}
fetch() {   # detached loops on the laptop (survive closing Claude and the terminal): pull results, evaluate them
  systemctl --user is-active -q lpp-v3-fetch || systemd-run --user --unit lpp-v3-fetch --same-dir bash ops/v3_train/fetch.sh
  systemctl --user is-active -q lpp-v3-eval || systemd-run --user --unit lpp-v3-eval --same-dir bash ops/v3_train/local_eval.sh
}
status() {
  cat out/v3_train/eval/summary.txt 2>/dev/null
  $G "tail -3 $R/runs/train.log; tail -2 $R/runs/B_ft/log.jsonl 2>/dev/null || tail -2 $R/runs/A_lp/log.jsonl; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
}
steps=("$@"); [ "${1:-all}" = all ] && steps=(connect setup upload models start fetch)
for s in "${steps[@]}"; do $s || { echo "step $s failed"; exit 1; }; done
