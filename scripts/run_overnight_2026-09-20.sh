#!/usr/bin/env bash
# Overnight chain, launched 2026-09-20 with Kartik's go-ahead. Strictly one heavy
# job at a time (memory: feedback-background-training-survival). Longest job
# first so the most important artefact exists earliest; ends with evals only.
#   1. Stage 3 v2b  -- abnormal_fraction 0.3 (hallucination vs collapse)   ~4.5 h
#   2. Stage 2 v2 seeds 1,2,3  -- for a seed ensemble + error bars          ~1.7 h
#   3. Stage 1 v3   -- sentence-level multi-positive alignment (zero-shot)  ~0.5 h
#   4. Evals: classifier ensemble (home + NIH), zero-shot variants, v2b
#      generation + decoding sweep, report scoring                          ~0.6 h
set -u
cd "$(dirname "$0")/.."
source /home/kartik/miniconda3/etc/profile.d/conda.sh && conda activate ml
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() { log "=== START $1 ==="; shift; "$@"; local rc=$?; log "=== END rc=$rc ==="; sleep 15; return $rc; }

log "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) | F: $(df -h /mnt/f | awk 'NR==2{print $4}') free"

run "1/4 Stage 3 v2b (abnormal_fraction 0.3)" \
  python3 scripts/train_report.py --config configs/report_gen_v2b.yaml

for s in 1 2 3; do
  run "2/4 Stage 2 v2 seed $s" \
    python3 scripts/train_classifier.py --config configs/classifier_v2.yaml --seed "$s" --run-name "seed$s"
done

run "3/4 Stage 1 v3 sentence-level" \
  python3 scripts/train_align.py --config configs/align_v3_sentence.yaml --run-name sentence

log "=== 4/4 EVALS ==="
run "ensemble (home + NIH)" \
  python3 scripts/eval_classifier_ensemble.py \
    checkpoints/classifier_v2/local_on/best.pt checkpoints/classifier_v2/seed1/best.pt \
    checkpoints/classifier_v2/seed2/best.pt checkpoints/classifier_v2/seed3/best.pt --nih
run "classification table incl. zero-shot variants" \
  python3 scripts/eval_v2_classification.py
G=docs/phase5_artifacts/v2
run "v2b generation, greedy" \
  python3 scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt --out $G/medxvlm_stage3_v2b_generations.csv
run "v2 generation, beam3+rep1.2" \
  python3 scripts/eval_v2_report_generation.py --num-beams 3 --repetition-penalty 1.2 --out $G/medxvlm_stage3_v2_beam3_generations.csv
run "v2b generation, beam3+rep1.2" \
  python3 scripts/eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt --num-beams 3 --repetition-penalty 1.2 --out $G/medxvlm_stage3_v2b_beam3_generations.csv
run "report scoring (all systems)" \
  env CUDA_VISIBLE_DEVICES="" python3 scripts/eval_v2_score_reports.py \
    --extra v2b_stage3=$G/medxvlm_stage3_v2b_generations.csv \
    --extra v2_stage3_beam3=$G/medxvlm_stage3_v2_beam3_generations.csv \
    --extra v2b_stage3_beam3=$G/medxvlm_stage3_v2b_beam3_generations.csv \
    --out $G/report_generation_scores_v2_overnight.json
log "OVERNIGHT CHAIN DONE"
