#!/usr/bin/env bash
# Orchestrates the Phase 3 real training runs sequentially (single GPU).
# Assumes train_baseline.py is already running separately; waits for it,
# then runs: align(on) -> align(off) -> classifier(on) -> classifier(off) -> report(on).
set -uo pipefail
cd "$(dirname "$0")/../.."

BASELINE_PID="${1:?usage: run_phase3_training.sh <baseline_pid>}"

echo "ORCH: waiting for baseline (pid $BASELINE_PID) to finish..."
while kill -0 "$BASELINE_PID" 2>/dev/null; do sleep 10; done
echo "STAGE_DONE:baseline"

run_stage() {
  local name="$1"; shift
  echo "ORCH: starting $name"
  if python "$@"; then
    echo "STAGE_DONE:$name"
  else
    echo "STAGE_FAILED:$name"
  fi
}

run_stage "align_local_on" scripts/train_align.py --use-local-loss
run_stage "align_local_off" scripts/train_align.py --no-use-local-loss
run_stage "classifier_local_on" scripts/train_classifier.py --stage1-checkpoint checkpoints/align/local_on/best.pt
run_stage "classifier_local_off" scripts/train_classifier.py --stage1-checkpoint checkpoints/align/local_off/best.pt
run_stage "report_gen" scripts/train_report.py --stage1-checkpoint checkpoints/align/local_on/best.pt

echo "ORCH: all stages attempted"
