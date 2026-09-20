#!/usr/bin/env bash
# Wait for the NIH ChestX-ray14 download+extract, then run ONLY the external
# eval (~15 min GPU). Deliberately does NOT start any training -- Kartik wants
# to be asked before multi-hour GPU jobs (memory: feedback-ask-before-gpu-training).
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "waiting for NIH download (marker DOWNLOADED_TO in data/raw_nih/download3.log)"
until grep -q "DOWNLOADED_TO" data/raw_nih/download3.log 2>/dev/null; do sleep 20; done
log "download done: $(grep DOWNLOADED_TO data/raw_nih/download3.log)"
sleep 5
log "=== NIH ChestX-ray14 external eval ==="
python3 scripts/eval_external_chestxray14.py
log "NIH eval exit code $?"
log "EVAL CHAIN DONE (no training launched)"
