#!/usr/bin/env bash
# Follow-up to run_overnight_2026-09-20.sh: Stage 3 v2b hit a transient
# "CUDA driver error: device not ready" 32 s after launch (GPU had just been
# released by smoke tests). Re-run it after the main chain finishes, then
# its dependent evals. Still one heavy job at a time.
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() { log "=== START $1 ==="; shift; "$@"; local rc=$?; log "=== END rc=$rc ==="; sleep 15; return $rc; }

log "waiting for main chain (marker OVERNIGHT CHAIN DONE)"
until grep -q "OVERNIGHT CHAIN DONE" runs/overnight_2026-09-20.log 2>/dev/null; do sleep 60; done
sleep 30  # let the GPU settle after the last eval released it

rm -rf checkpoints/report_gen_v2b runs/report_gen_v2b
run "Stage 3 v2b (abnormal_fraction 0.3) -- retry" \
  python3 scripts/train_report.py --config configs/report_gen_v2b.yaml
G=docs/phase5_artifacts/v2
run "v2b generation, greedy" \
  python3 scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt --out $G/medxvlm_stage3_v2b_generations.csv
run "v2b generation, beam3+rep1.2" \
  python3 scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt --num-beams 3 --repetition-penalty 1.2 --out $G/medxvlm_stage3_v2b_beam3_generations.csv
run "report scoring (all systems)" \
  env CUDA_VISIBLE_DEVICES="" python3 scripts/eval_v2_score_reports.py \
    --extra v2b_stage3=$G/medxvlm_stage3_v2b_generations.csv \
    --extra v2_stage3_beam3=$G/medxvlm_stage3_v2_beam3_generations.csv \
    --extra v2b_stage3_beam3=$G/medxvlm_stage3_v2b_beam3_generations.csv \
    --out $G/report_generation_scores_v2_overnight.json
log "PART2 DONE"
