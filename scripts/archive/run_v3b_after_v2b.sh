#!/usr/bin/env bash
# After the v2b chain: Stage 1 v3b (finding sentences only, ~25 min) + the
# zero-shot table. No PYTORCH_CUDA_ALLOC_CONF (see run_v2b_then_score.sh).
set -u
cd "$(dirname "$0")/../.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH=src
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() { log "=== START $1 ==="; shift; "$@"; local rc=$?; log "=== END rc=$rc ==="; sleep 15; return $rc; }
log "waiting for V2B CHAIN DONE"
until grep -q "V2B CHAIN DONE" runs/v2b_chain.log 2>/dev/null; do sleep 60; done
sleep 30
run "Stage 1 v3b (findings-only sentences)" python3 scripts/train_align.py --config configs/align_v3b_findings.yaml --run-name findings
run "classification table incl. v3b zero-shot" python3 scripts/eval_v2_classification.py
log "V3B CHAIN DONE"
