#!/usr/bin/env bash
# Steps 2-3 of run_v2c_findings_conditioned.sh only (inference + scoring, no training).
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src
unset PYTORCH_CUDA_ALLOC_CONF
log() { echo "[$(date '+%H:%M:%S')] $*"; }
V2=docs/phase5_artifacts/v2
CK=${1:-checkpoints/report_gen_v2c/best.pt}
log "generating with $CK (beam-3, greedy, no-findings ablation)"
python scripts/eval_v2_report_generation.py --checkpoint $CK --num-beams 3 --repetition-penalty 1.2 --out $V2/medxvlm_stage3_v2c_beam3_generations.csv; log "beam exit=$?"
python scripts/eval_v2_report_generation.py --checkpoint $CK --out $V2/medxvlm_stage3_v2c_generations.csv; log "greedy exit=$?"
python scripts/eval_v2_report_generation.py --checkpoint $CK --num-beams 3 --repetition-penalty 1.2 --no-findings --out $V2/medxvlm_stage3_v2c_nofindings_beam3_generations.csv; log "ablation exit=$?"
log "scoring"
python scripts/eval_v2_score_reports.py \
  --extra v2_stage3_beam3=$V2/medxvlm_stage3_v2_beam3_generations.csv \
  --extra v2b_stage3=$V2/medxvlm_stage3_v2b_generations.csv \
  --extra v2c_stage3=$V2/medxvlm_stage3_v2c_generations.csv \
  --extra v2c_stage3_beam3=$V2/medxvlm_stage3_v2c_beam3_generations.csv \
  --extra v2c_stage3_nofindings_beam3=$V2/medxvlm_stage3_v2c_nofindings_beam3_generations.csv \
  --extra classifier_verbalised=$V2/classifier_verbalised_generations.csv \
  --out $V2/report_generation_scores_v2c.json; log "score exit=$?"
log "EVAL DONE"
