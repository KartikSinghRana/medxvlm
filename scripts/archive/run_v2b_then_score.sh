#!/usr/bin/env bash
# Stage 3 v2b, third attempt: WITHOUT PYTORCH_CUDA_ALLOC_CONF=expandable_segments
# (the two "CUDA driver error: device not ready" failures on 2026-09-20 both had
# it set; the 30-batch smoke without it passed). Then its generation + scoring.
set -u
cd "$(dirname "$0")/../.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH=src
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() { log "=== START $1 ==="; shift; "$@"; local rc=$?; log "=== END rc=$rc ==="; sleep 15; return $rc; }
G=docs/phase5_artifacts/v2
run "Stage 3 v2b (abnormal_fraction 0.3), no expandable_segments" python3 scripts/train_report.py --config configs/report_gen_v2b.yaml
run "v2b generation, greedy" python3 scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt --out $G/medxvlm_stage3_v2b_generations.csv
run "v2b generation, beam3+rep1.2" python3 scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt --num-beams 3 --repetition-penalty 1.2 --out $G/medxvlm_stage3_v2b_beam3_generations.csv
run "report scoring (all systems)" env CUDA_VISIBLE_DEVICES="" python3 scripts/eval_v2_score_reports.py \
  --extra v2b_stage3=$G/medxvlm_stage3_v2b_generations.csv --extra v2_stage3_beam3=$G/medxvlm_stage3_v2_beam3_generations.csv \
  --extra v2b_stage3_beam3=$G/medxvlm_stage3_v2b_beam3_generations.csv --out $G/report_generation_scores_v2_overnight.json
log "V2B CHAIN DONE"
