#!/usr/bin/env bash
# Post-Phase-5 rebuild: serialized so the 16GB GPU / 15GiB WSL RAM is never shared.
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== 1/4 Stage 1 v2 (XrayDINOv2, frozen, frontal, 15 ep) ==="
python3 scripts/train_align.py --config configs/align_v2.yaml 2>&1 | grep -v "Loading weights"
log "=== 2/4 Stage 2 v2 (image-only, ASL, frontal) ==="
python3 scripts/train_classifier.py --config configs/classifier_v2.yaml 2>&1 | grep -v "Loading weights"
log "=== 3/4 baseline v2: ImageNet ViT-B/16 + linear ==="
python3 scripts/train_baseline.py --config configs/baseline_v2.yaml --backbone vit_base_patch16_224 2>&1 | grep -v "Loading weights"
log "=== 4/4 baseline v2: XrayDINOv2 + linear ==="
python3 scripts/train_baseline.py --config configs/baseline_v2.yaml --backbone xraydinov2 2>&1 | grep -v "Loading weights"
log "=== REBUILD CHAIN DONE ==="
