# Archived run scripts

One-off orchestration chains used to produce specific results (overnight runs, resumed jobs, eval-after-training
waits). They are kept for provenance: the docs cite them as the exact commands behind reported numbers. Each still
runs from here (paths point two levels up to the repo root), but none is needed to reproduce the project:

| Script | What it ran | Cited in |
|---|---|---|
| `run_phase3_training.sh` | Phase 3 training chain (baseline, Stage 1 ×2, Stage 2 ×2, Stage 3) | `status_report.md` |
| `run_nih_eval_only.sh` | NIH ChestX-ray14 external evaluation | `project_status_report.md` |
| `run_nih_then_v2b.sh` | superseded chain (disabled; exits immediately) | `project_status_report.md` |
| `run_overnight_2026-09-20.sh`, `_part2.sh` | seed ensemble, decoding sweep, zero-shot v3 | `v2_rebuild.md` §3.6 |
| `run_v2b_then_score.sh`, `run_v3b_after_v2b.sh` | Stage 3 v2b and zero-shot v3b | `v2_rebuild.md` §3.6 |
| `run_v2c_resume.sh`, `run_v2c_eval_only.sh` | resumed v2c training and its evaluation | `v2_rebuild.md` §3.7 |
| `run_rq1_eval_after_training.sh` | RQ1 evaluation + bootstrap after training | `v2_rebuild.md` §3.9 |

The canonical recipes stay in `scripts/`: `run_v2_rebuild.sh` (v2 Stage 1/2/baselines), `run_rq1_backbones.sh`
(encoder comparison), `run_v2c_findings_conditioned.sh` (v2c), plus `hosting.sh` for the demo.
