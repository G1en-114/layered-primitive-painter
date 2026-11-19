#!/bin/bash
# Formal protocol-v3 training (docs/line_strokes.md, "formal training"): base_v3 converted to v3, trained on
# faces (4 Gaussian-only v2 trajectories, widened on load: replay against forgetting) + anime with Bezier lines
# (trajectory t0, 50k images), draft paint order. LP-FT in two stages, DDP over NGPU GPUs:
#   A  only the new parts train (LINE heads, bend, event tables), trunk frozen          STEPS_A steps
#   B  everything, low learning rate, from A's weights; model snapshots every SNAP steps   STEPS_B steps
# Resume-safe: re-running skips a finished stage and resumes an interrupted one from its last.pt.
# Paths and sizes come from the environment (defaults: the rented 2x5090 box); ops/v3_train/dry_run.sh runs
# the same script on the laptop with tiny settings. Ends by writing RUNS/train_done (or RUNS/train_failed).
set -u
ROOT=${ROOT:-/root/lpp/v3train}; CODE=${CODE:-$ROOT/code}; RUNS=${RUNS:-$ROOT/runs}
PY=${PY:-python}; NGPU=${NGPU:-2}; BATCH=${BATCH:-16}; ACCUM=${ACCUM:-1}; WORKERS=${WORKERS:-6}
STEPS_A=${STEPS_A:-1000}; STEPS_B=${STEPS_B:-12000}; SNAP=${SNAP:-2000}; SAVE=${SAVE:-500}; EVAL=${EVAL:-1000}
INIT=${INIT:-$ROOT/runs/base_v3_lines_init.pt}
D=$ROOT/data
read -ra FACES <<< "${FACES:-$D/seq/v2/decomp_*.npz $D/seq/v2_t1/decomp_*.npz $D/seq/v2_t2/decomp_*.npz $D/seq/v2_t3/decomp_*.npz}"
read -ra ANIME <<< "${ANIME:-$D/seq/anime_v3_lines_full_t0/decomp_*.npz}"
# the parts protocol v3 adds (tools/init_v3_from_v2.py): LP-FT stage A trains only these
NEW=line_heads.,heads.bend.,heads.event.,event_emb.,dec_event_emb.,attr_emb.mods.bend.,dec_attr_emb.mods.bend.,dec_step
mkdir -p $RUNS
rm -f $RUNS/train_done $RUNS/train_failed
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $RUNS/train.log; }
# NGC 25.01's NCCL 2.25.1 needs more shared memory than an RTX 5090 (sm_120) has ("ncclMaxSharedMem 82240 exceeds
# 79856" -> "Cuda failure 1 'invalid argument'"); a newer NCCL beside it (pip nvidia-nccl-cu12 --target) is preloaded
NCCL_LIB=${NCCL_LIB:-/root/nccl/nvidia/nccl/lib/libnccl.so.2}
[ -f "$NCCL_LIB" ] && export LD_PRELOAD=$NCCL_LIB
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONUNBUFFERED=1
LAUNCH=("$PY" -u); [ "$NGPU" -gt 1 ] && LAUNCH=("$PY" -u -m torch.distributed.run --standalone --nproc_per_node=$NGPU)
COMMON=(--data "${FACES[@]}" "${ANIME[@]}" --protocol $CODE/protocol_v3.json --config base --line-heads --order draft
  --batch $BATCH --accum $ACCUM --workers $WORKERS --seed 0
  --captions $D/captions/captions_v3.jsonl --attributes $D/celebamask256/attributes.json
  --split-manifest $D/split_v3_faces_anime.jsonl --text-encoder $D/models/chinese-clip-vit-base-patch16
  --text-drop 0.1 --cfg-scale 5.0 --canvas --canvas-k 32 --canvas-size 64 --perturb 0.3 --save-every $SAVE)
stage() {   # stage NAME STEPS ARGS...: skip when its final snapshot exists, resume when interrupted
  local name=$1 steps=$2; shift 2
  local out=$RUNS/$name
  local snap=$out/model_step$(printf %06d $steps).pt
  [ -f $snap ] && { say "$name already done"; return 0; }
  local -a resume=(); [ -f $out/last.pt ] && resume=(--resume)
  say "$name: start (${resume[*]:-fresh}), $steps steps, $NGPU GPU x batch $BATCH x accum $ACCUM"
  (cd $CODE && "${LAUNCH[@]}" tools/train.py "${COMMON[@]}" --out $out --steps $steps "$@" "${resume[@]}") \
    >> $out.out 2>&1
  [ -f $snap ] && { say "$name: done"; return 0; }
  say "$name: FAILED (see $out.out)"; return 1
}
mkdir -p $RUNS/A_lp $RUNS/B_ft
stage A_lp $STEPS_A --init-weights $INIT --train-only $NEW --lr 3e-4 --min-lr 1e-4 --warmup 100 \
    --eval-every $EVAL --snapshot-every $STEPS_A \
  && { [ "$STEPS_B" = 0 ] || stage B_ft $STEPS_B --init-weights $RUNS/A_lp/model_step$(printf %06d $STEPS_A).pt \
    --lr 1e-4 --min-lr 1e-5 --warmup 300 --eval-every $EVAL --snapshot-every $SNAP; } \
  && { say "training finished"; touch $RUNS/train_done; } || { say "training stopped"; touch $RUNS/train_failed; }
