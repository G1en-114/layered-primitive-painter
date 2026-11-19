#!/bin/bash
# One command, laptop -> 5090 box (boot it in no-GPU mode first, the upload is bound by the laptop's uplink):
#   bash ops/v3_train/up.sh "ssh root+vm-XXXX@HOST -p PORT"      (the SSH line from the console; omit it if unchanged)
# 1. connects (updates G5090_HOST/PORT/USER in ~/.config/lpp/remotes.env when the line differs; password unchanged)
# 2. checks the box's python (installs transformers 4.48.3 if missing)
# 3. the box downloads the public text encoders itself (hf-mirror, SHA256-checked), in the background
# 4. code (git HEAD) + data as ONE zstd tar stream with a progress bar (pv); files already on the box at full
#    size are skipped, so re-running after a cut resumes
# 5. waits for the models, verifies every file's size, then says it is safe to switch to the GPUs
set -u
cd "$(dirname "$0")/../.."
R=/root/lpp/v3train
G="ssh -F /dev/null -S /run/user/1000/g5090.sock -o BatchMode=yes -o LogLevel=ERROR x"
bash ops/v3_train/connect.sh "$@" || exit 1
echo "== 1/4 box: $($G 'nvidia-smi -L 2>/dev/null | wc -l') GPUs, $($G "df -h / | tail -1 | awk '{print \$4}'") free"

echo "== 2/4 python"
bash ops/v3_train/launch.sh setup | tail -2

echo "== 3/4 text encoders: the box downloads them (background)"
for m in chinese-clip-vit-base-patch16 chinese-clip-vit-large-patch14; do
  $G "mkdir -p $R/data/models/$m && cat > $R/data/models/$m/SHA256SUMS && touch $R/data/models/$m/SHA256SUMS.ok" \
    < data/models/$m/SHA256SUMS
done
setsid bash ops/v3_train/launch.sh models > out/v3_models.log 2>&1 < /dev/null &   # own session: Ctrl-C here does not kill it
MODELS=$!

echo "== 4/4 code + data"
echo "sending code ($(( $(git archive HEAD | wc -c) / 1024 )) KB)..."
git archive HEAD | pv -N code -pterab | $G "rm -rf $R/code.new && mkdir -p $R/code.new && tar -x -C $R/code.new && rm -rf $R/code && mv $R/code.new $R/code"
echo "code $(git rev-parse --short HEAD) uploaded"
[ -f runs/base_v3_lines_init.pt ] || "$HOME/AI infra/.venv/bin/python" tools/init_v3_from_v2.py \
  --ckpt runs/base_v3_snap/model_final.pt --out runs/base_v3_lines_init.pt
{
  echo runs/base_v3_lines_init.pt
  for d in v2 v2_t1 v2_t2 v2_t3 anime_v3_lines_full_t0; do ls data/seq/$d/decomp_*.npz data/seq/$d/config.json; done
  ls data/captions/captions_v3.jsonl data/celebamask256/attributes.json data/split_v3_faces_anime.jsonl \
     data/celebamask256_split_v1.jsonl data/celebamask256_val/* data/anime_val192/*
} > out/v3_upload_list.txt
sizes() { $G "cd $R && while read f; do [ -f \$f ] && echo \"\$f \$(stat -c %s \$f)\"; done" < out/v3_upload_list.txt; }
echo "checking which files the box already has..."
sizes > out/v3_remote_sizes.txt
todo=out/v3_upload_todo.txt; : > $todo; bytes=0
while read -r f; do
  s=$(stat -Lc %s "$f")
  grep -qxF "$f $s" out/v3_remote_sizes.txt || { echo "$f" >> $todo; bytes=$((bytes + s)); }
done < out/v3_upload_list.txt
echo "$(wc -l < $todo) of $(wc -l < out/v3_upload_list.txt) files to send, $((bytes / 1048576)) MB before compression"
if [ -s $todo ]; then
  tar -chf - -T $todo | pv -s $bytes -N upload -pterab | zstd -q -6 -T4 \
    | $G "mkdir -p $R && zstd -q -d | tar -x -C $R" || { echo "upload cut: run the same command again to resume"; exit 1; }
fi
sizes > out/v3_remote_sizes.txt
bad=0
while read -r f; do grep -qxF "$f $(stat -Lc %s "$f")" out/v3_remote_sizes.txt || { echo "missing/short: $f"; bad=1; }; done < out/v3_upload_list.txt
[ $bad = 0 ] || { echo "some files are incomplete: run the same command again"; exit 1; }
echo "data complete on the box"

echo "waiting for the text encoders..."
wait $MODELS; tail -3 out/v3_models.log
$G "test -f $R/data/models/chinese-clip-vit-large-patch14-fp16/model.safetensors.ok" \
  || { echo "text encoder download failed (out/v3_models.log): run the same command again"; exit 1; }
echo
echo "ALL READY. Shut the box down, boot it with the 2 GPUs, then: bash ops/v3_train/go.sh \"ssh ...new line...\""
