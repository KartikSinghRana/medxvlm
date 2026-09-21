#!/usr/bin/env bash
# Serialized chain (one heavy job at a time -- see memory note 2026-09-19):
#   1. wait for the kagglehub NIH ChestX-ray14 download+extract to finish
#   2. run the external ChestX-ray14 eval (GPU, ~15 min)
#   3. relaunch Stage 3 v2b training alone (GPU, ~4.5 h)
echo "HISTORICAL (2026-09-20): this chain auto-launched a 4.5 h training run after an eval step."
echo "That is no longer allowed here -- training needs an explicit go-ahead (docs/project_status_report.md §3)."
echo "Kept for the record only. Use scripts/run_nih_eval_only.sh for the eval step."
exit 1

set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for NIH download to finish (marker DOWNLOADED_TO in data/raw_nih/download3.log)"
until grep -q "DOWNLOADED_TO" data/raw_nih/download3.log 2>/dev/null; do sleep 30; done
log "download done: $(grep DOWNLOADED_TO data/raw_nih/download3.log)"
sleep 10

log "=== NIH ChestX-ray14 external eval ==="
python3 scripts/eval_external_chestxray14.py
log "NIH eval exit code $?"
sleep 10

log "=== Stage 3 v2b training (abnormal_fraction 0.3) ==="
python3 scripts/train_report.py --config configs/report_gen_v2b.yaml
log "training exit code $?"
log "CHAIN DONE"
