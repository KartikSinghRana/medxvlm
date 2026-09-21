#!/usr/bin/env bash
# Stage 3 v2c (findings-conditioned) -- one serialized chain, launched only after
# the owner's explicit go-ahead (2026-09-21):
#   1. train  (configs/report_gen_v2c.yaml, ~4-5 h, selection by generation CE F1)
#   2. generate on the 150-image Phase 5 subsample: beam-3 (project default) + greedy
#      + a no-findings ablation (same checkpoint, plain prompt)
#   3. score everything side by side -> docs/phase5_artifacts/v2/report_generation_scores_v2c.json
# Markers: "Done in" (training), "CHAIN DONE" (all).
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src
unset PYTORCH_CUDA_ALLOC_CONF   # expandable_segments breaks near-cap runs on WSL2 (docs/v2_rebuild.md §3.6.4)
log() { echo "[$(date '+%H:%M:%S')] $*"; }
V2=docs/phase5_artifacts/v2

log "1/3 training v2c"
python scripts/train_report.py --config configs/report_gen_v2c.yaml; rc=$?
log "training exit=$rc"
[ $rc -ne 0 ] && { log "CHAIN DONE (training failed)"; exit $rc; }

log "2/3 generating (beam-3, greedy, no-findings ablation)"
python scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2c/best.pt --num-beams 3 --repetition-penalty 1.2 --out $V2/medxvlm_stage3_v2c_beam3_generations.csv; log "beam exit=$?"
python scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2c/best.pt --out $V2/medxvlm_stage3_v2c_generations.csv; log "greedy exit=$?"
python scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2c/best.pt --num-beams 3 --repetition-penalty 1.2 --no-findings --out $V2/medxvlm_stage3_v2c_nofindings_beam3_generations.csv; log "ablation exit=$?"

log "3/3 scoring"
python scripts/eval_v2_score_reports.py \
  --extra v2_stage3_beam3=$V2/medxvlm_stage3_v2_beam3_generations.csv \
  --extra v2b_stage3=$V2/medxvlm_stage3_v2b_generations.csv \
  --extra v2c_stage3=$V2/medxvlm_stage3_v2c_generations.csv \
  --extra v2c_stage3_beam3=$V2/medxvlm_stage3_v2c_beam3_generations.csv \
  --extra v2c_stage3_nofindings_beam3=$V2/medxvlm_stage3_v2c_nofindings_beam3_generations.csv \
  --extra classifier_verbalised=$V2/classifier_verbalised_generations.csv \
  --out $V2/report_generation_scores_v2c.json; log "score exit=$?"
log "CHAIN DONE"
