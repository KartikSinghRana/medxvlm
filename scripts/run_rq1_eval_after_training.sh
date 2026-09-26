#!/usr/bin/env bash
# Inference only: waits for run_rq1_backbones.sh to finish, then evaluates every
# encoder checkpoint (test / NIH / pointing game) and recomputes the paired bootstrap.
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "waiting for CHAIN DONE in runs/rq1_backbones.log"
until grep -q "CHAIN DONE" runs/rq1_backbones.log; do sleep 30; done
log "evaluating encoders"
python scripts/eval_rq1_backbones.py; log "eval exit=$?"
log "bootstrap"
python scripts/eval_bootstrap_classification.py; log "bootstrap exit=$?"
log "EVAL CHAIN DONE"
