#!/usr/bin/env bash
# Resume the v2c chain after the 2026-09-21 WSL restart killed it at epoch 1:
# continue training from checkpoints/report_gen_v2c/last.pt (epochs 2..5), then
# run the inference + scoring steps. Launch only with the owner's go-ahead.
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src
unset PYTORCH_CUDA_ALLOC_CONF
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "resuming v2c training from last.pt"
python scripts/train_report.py --config configs/report_gen_v2c.yaml --resume checkpoints/report_gen_v2c/last.pt; rc=$?
log "training exit=$rc"
[ $rc -ne 0 ] && { log "CHAIN DONE (training failed)"; exit $rc; }
bash scripts/run_v2c_eval_only.sh checkpoints/report_gen_v2c/best.pt
log "CHAIN DONE"
