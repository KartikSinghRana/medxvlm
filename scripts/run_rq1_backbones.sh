#!/usr/bin/env bash
# RQ1 encoder comparison (docs/v2_rebuild.md §3.8): the v2 Stage 2 recipe
# (configs/classifier_v2.yaml: image-only, per-label attention head, ASL, split LR,
# early stopping) on three more ViT-B towers, straight from public weights.
# Phase A: one seed per tower. Phase B: a second seed for the BiomedCLIP-vs-CLIP
# pair (the literal RQ1 question), only if Phase A took <= 75 min (keeps the
# whole chain ~2 h, the window the owner approved on 2026-09-26).
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src
unset PYTORCH_CUDA_ALLOC_CONF
log() { echo "[$(date '+%H:%M:%S')] $*"; }
run() {  # backbone seed
  log "train $1 seed $2"
  python scripts/train_classifier.py --config configs/classifier_v2.yaml --vision-backbone "$1" --seed "$2" --run-name "rq1_$1_s$2"
  log "train $1 seed $2 exit=$?"
}
t0=$(date +%s)
for b in biomedclip clip_openai imagenet_vit; do run "$b" 0; done
phase_a_min=$(( ($(date +%s) - t0) / 60 ))
log "phase A took ${phase_a_min} min"
if [ "$phase_a_min" -le 75 ]; then
  for b in biomedclip clip_openai; do run "$b" 1; done
else
  log "skipping phase B (over budget)"
fi
log "CHAIN DONE"
