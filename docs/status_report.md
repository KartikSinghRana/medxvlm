# MedXVLM — Repository Status Report

Generated: 2026-09-18; §6/§11/§12 updated 2026-09-19 after the Stage 3 v2 retrain. By direct inspection of the working tree at `/home/kartik/DNN Project 2/medxvlm`. Every number below cites the file or command it came from. Nothing in this document was estimated.

## Plain-language summary

MedXVLM is a three-stage chest X-ray vision-language pipeline (contrastive alignment → 14-class classifier → LoRA report generator) trained on the public Indiana University / Open-i dataset. **All three stages are implemented and have real, logged training runs; the repository is far past "skeleton" stage.** The project has been through five documented phases plus a post-Phase-5 rebuild ("v2"). The headline honest result is a **blind, image-only classifier scoring 0.855 macro AUROC on the held-out frontal test split (377 images) across the 8 classes shared with RadZero, and 0.829 across all 13 measurable CheXpert classes** — measured against corrected labels, with no report text given to the model. On identical images and labels, the published RadZero model scores 0.868. An earlier headline figure (0.865, Phase 3) has been formally superseded because it was measured with the ground-truth report fed into the classifier; tested image-only, that same checkpoint scores 0.507. Report generation (Stage 3) was retrained on 2026-09-19: the Phase 3 model's **mode collapse** (106 of 150 test generations identical; 5 distinct outputs) is broken (61 distinct), and clinical-efficacy macro F1 on the 70 frontal test images is now **0.152, level with CheXOne's 0.149** on identical images and corrected labels — but the new model **hallucinates catheters/lines/surgical changes in 10% of frontal outputs** (CheXOne: 0%), and both systems remain far from usable (macro F1 ≈ 0.15). BLEU-4 and ROUGE-L were computed for the first time and slightly favour the collapsed Phase 3 model, an artefact of n-gram metrics on templated reports. Two external published models (RadZero, CheXOne) were actually run on this project's data with real numbers; the reverse direction (this model on their datasets) was not run because those datasets require credentialed access. **Notable housekeeping gap: the repo is `git init`-ed but has zero commits** — nothing is version-controlled yet. A plain-language PDF summary and five phase documents exist under `docs/`.

---

## 1. Repo identity and environment

### 1.1 Git

- `git rev-parse --is-inside-work-tree` → `true`; branch `master`.
- `git log --oneline -10` → **`fatal: your current branch 'master' does not have any commits yet`**.
- `git rev-list --all --count` → `0`. `git ls-files | wc -l` → `19` (staged, never committed). `git status --short | wc -l` → `63` entries.
- `.gitignore` excludes `data/raw/`, `data/processed/`, `checkpoints/`, `runs/`, `*.log`, `*.egg-info/` — i.e. all datasets, weights, and logs are intentionally unversioned.

**Consequence for a report writer:** there is no commit history to cite. Dates below come from file mtimes and the `Date:` lines inside the docs.

### 1.2 Environment files

- `requirements.txt` (pinned; also carries an sm_120/RTX 5080 wheel-index note): `torch==2.11.0`, `torchvision==0.26.0`, `timm==1.0.28`, `transformers==5.8.0`, `peft==0.20.0`, `open_clip_torch==3.3.0`, `opencv-python-headless==5.0.0.93`, `scikit-learn==1.8.0`, `sacrebleu==2.6.0`, `rouge-score==0.1.2`, `nltk==3.10.2`, `pandas==2.3.3`, `numpy==2.4.3`, `pillow==12.1.1`, `pyyaml==6.0.3`, `kagglehub==1.0.2`, `tensorboard==2.21.0`, `streamlit==1.61.1`.
- `pyproject.toml`: package `medxvlm` v0.1.0, `requires-python = ">=3.10"`, dependencies dynamic from `requirements.txt`, packages under `src/`.
- No `environment.yml` (`ls environment.yml` → not found).

### 1.3 Runtime (commands run in conda env `ml`)

- `python --version` → `Python 3.11.15`
- `pip list` (installed, matches pins): torch 2.11.0, torchvision 0.26.0, transformers 5.8.0, timm 1.0.28, peft 0.20.0, open_clip_torch 3.3.0, scikit-learn 1.8.0, pandas 2.3.3, numpy 2.4.3, accelerate 1.13.0, huggingface_hub 1.14.0, tensorboard 2.21.0, weasyprint 69.0, pypdf 6.16.2.
- `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv` → `NVIDIA GeForce RTX 5080, 16303 MiB, 3132 MiB, 610.47`
- `uname -srm` → `Linux 6.18.33.2-microsoft-standard-WSL2 x86_64`; `/etc/os-release` → `Ubuntu 26.04 LTS`.
- `docs/environment_check.md` (dated 2026-08-30) records the same GPU and `[PASS] torch -> 2.11.0+cu130`, `[PASS] torchvision -> 0.26.0+cu130`, `[PASS] open_clip_torch -> 3.3.0` from `scripts/verify_env.py`.

### 1.4 Folder tree (`find . -maxdepth 2 -type d`, excluding .git/__pycache__/venv)

```
./app                      (contains only .gitkeep — no demo UI exists)
./checkpoints/{align, align_v2, baseline, baseline_v2, classifier, classifier_v2, report_gen}
./configs
./data/{processed, raw}
./docs/phase5_artifacts
./runs/{align, align_v2, baseline, baseline_v2, classifier, classifier_v2, report_gen}
./scripts
./src/medxvlm, ./src/medxvlm.egg-info
```

`src/medxvlm/` modules: `data/{dataset,labeler,toy_dataset}.py`, `engine/{loop_utils,sanity_trainer}.py`, `eval/metrics.py`, `explain/attention_maps.py`, `models/{alignment,baseline_classifier,classifier_head,labels,report_generator,text_encoder,vision_encoder}.py`.
`scripts/`: `download_dataset.py, prepare_data.py, validate_labeler.py, sanity_check_data.py, sanity_check_models.py, verify_env.py, train_baseline.py, train_align.py, train_classifier.py, train_report.py, run_phase3_training.sh, run_v2_rebuild.sh, eval_phase5_classification.py, eval_phase5_report_generation.py, eval_phase5_score_reports.py, eval_v2_classification.py`.

Outside the repo root, `/home/kartik/DNN Project 2/external_repro/` holds the RadZero and CheXOne clones, two isolated venvs, and their run scripts/outputs (referenced in §9).

---

## 2. Phase / exit-checklist status

The repo does not contain the original phase prompts; checklist items below are reconstructed from what each phase's own deliverable document says it set out to do. `README.md` §"Project status" still says **"Phase 3"** and is stale relative to the docs (it predates Phases 4–5 and the v2 rebuild).

### Phase 1 — repo skeleton, environment, literature review, proposal
| Item | Status | Evidence |
|---|---|---|
| Repo skeleton (`src/`, `scripts/`, `configs/`, `docs/`, `app/`) | Done | folder tree §1.4 |
| Environment validated | Done | `docs/environment_check.md` (2026-08-30), `scripts/verify_env.py` |
| Literature review | Done | `docs/literature_review.md` (13,516 B), `docs/literature_review_student.pdf` |
| Project proposal | Done | `docs/project_proposal.md` (12,431 B, mtime 2026-09-05) |
| Dataset download + smoke test | Done | `scripts/download_dataset.py`, `scripts/sanity_check_data.py`; 7,470 PNGs on disk |
| Version control | **Not done** | 0 commits (§1.1) |

### Phase 2 — baseline + model skeletons, wiring check, code review
| Item | Status | Evidence |
|---|---|---|
| Baseline classifier, Stage 1, Stage 2 modules | Done | `src/medxvlm/models/*.py` |
| Overfit-one-batch wiring check | Done | `scripts/sanity_check_models.py` docstring: "forward + loss + backward + optimizer step over ~50 steps on one repeated batch … PASS/FAIL line per component" |
| Code review notes | Done | `docs/code_review_notes.md` (2026-08-30), headings: per-module rationale, "What the sanity checks verify — and what they don't", "Immediate next milestone" |
| Stage 3 module | Done (built in Phase 3) | `src/medxvlm/models/report_generator.py` |
| Demo app | **Not started** | `app/` contains only `.gitkeep` |

### Phase 3 — systematic training (deliverable: `docs/training_summary.md`, `docs/ablation_study.md`)
| Item | Status | Evidence |
|---|---|---|
| Real data pipeline + rule-based labeler + uid-level split | Done | `scripts/prepare_data.py`, `data/processed/manifest.csv` |
| Baseline trained | Done | `checkpoints/baseline/best.pt`; log "Done in … Best val macro AUROC: 0.6939" (`runs/phase3_orchestrator.log` / `runs/train_baseline.log`) |
| Stage 1 trained (×2, local loss on/off) | Done | `checkpoints/align/{local_on,local_off}/best.pt`, 19.9 / 19.1 min |
| Stage 2 trained (×2) | Done | `checkpoints/classifier/{local_on,local_off}/best.pt`, 35.6 / 36.2 min, val macro AUROC 0.8645 / 0.8648 |
| Stage 3 trained | Done | `checkpoints/report_gen/best.pt` (epoch 2), 210.7 min |
| Ablation study written | Done (one of two ablations run) | `docs/ablation_study.md`; §"Second ablation: Asymmetric Loss vs. plain BCE" says **"Not run in this pass"** |
| Training summary written | Done, later superseded-banner added | `docs/training_summary.md` |

**Critical caveat surfaced later (Phase 5):** the Phase 3 Stage 2 numbers were produced with the paired ground-truth report text fused into the classifier at train *and* validation time (`scripts/train_classifier.py` `Stage2Model.forward` → `ClassifierHead(text_embeds=…)`). `docs/training_summary.md` now opens with a banner: *"Superseded for classification (2026-09-17)… tested image-only, the same checkpoint scores 0.51 (chance)."*

### Phase 4 — external benchmark reproduction (deliverables: `docs/external_repro_feasibility.md`, `docs/external_baseline_reproduction.md`)
| Item | Status | Evidence |
|---|---|---|
| Feasibility check written before setup | Done | `docs/external_repro_feasibility.md` (mtime 2026-09-06 05:44, precedes reproduction doc at 06:39) |
| Isolated venv per repo | Done | `external_repro/{radzero_venv,chexone_venv}` (listed in reproduction doc §2) |
| RadZero run on native benchmark | Done | `external_repro/radzero_openi_eval_n1000_seed0_summary.json` |
| CheXOne quantitative benchmark | **Not done — blocked by data access** | reproduction doc §4; qualitative Quick Start run only |
| Comparison table with reproduced-vs-reported labels | Done | reproduction doc §3–§4 |

### Phase 5 — cross-dataset evaluation (deliverable: `docs/cross_dataset_evaluation.md`)
| Item | Status | Evidence |
|---|---|---|
| Direction explicit (flipped, why) | Done | doc §1 |
| Every cell labelled apples-to-apples / not comparable | Done | column headers of doc §4/§5 tables carry the tags |
| MedXVLM on their datasets | **Not run** (credentialed data) | doc §1, §7 |
| Their models on MedXVLM's test split | Done | `external_repro/radzero_on_medxvlm_testsplit_summary.json`, `external_repro/chexone_testsplit_generations.csv` |
| Week-7 drop-in table | Done | doc §8 |

### Post-Phase-5 "v2 rebuild" (deliverable: `docs/v2_rebuild.md`, 2026-09-17)
Not one of the five planned phases; triggered by Phase 5's findings. Fully documented; results in §5 below.

---

## 3. Dataset statistics (computed from files in the repo)

Source: `data/raw/indiana_reports.csv`, `data/raw/indiana_projections.csv`, `data/processed/manifest.csv`, via a pandas script run for this report.

| Quantity | Value | Source |
|---|---:|---|
| Raw report rows / unique studies (uid) | 3,851 / 3,851 | `indiana_reports.csv` (3,852 lines incl. header) |
| Raw projection rows | 7,466 | `indiana_projections.csv` |
| PNG files on disk | 7,470 | `ls data/raw/images/images_normalized \| wc -l` |
| Raw data size | 14 GB | `du -sh data/raw` |
| Studies in manifest | **3,826** | `manifest.csv`, `uid.nunique()` |
| Image rows in manifest | **7,426** | `manifest.csv` |
| Frontal / Lateral image rows | 3,794 / 3,632 | `manifest.projection.value_counts()` |
| Split — studies (train/val/test) | **3,060 / 382 / 384** | uid-level, seed 0 (`scripts/prepare_data.py` `--train-frac 0.8 --val-frac 0.1`) |
| Split — image rows | 5,929 / 749 / 748 | same |
| Split — frontal rows only (what v2 trains/evals on) | 3,030 / 387 / **377** | `manifest[projection=='Frontal'].split.value_counts()` |

The dataset is a single-image-per-study-view collection with no patient ID beyond `uid`; "patient count" is not derivable from the files (uid = study).

**Data-quality accounting (computed):**
- 25 reports have both `findings` and `impression` empty → dropped by `prepare_data.py` (3,851 → 3,826 studies).
- 40 projection rows belong to those 25 studies → dropped (7,466 → 7,426 image rows).
- 0 projection rows reference a missing image file.
- 489 reports have `findings` missing but `impression` present; 6 the reverse (both kept; text is the join of whatever exists).
- De-identification token "XXXX": present in 3,425 of 7,426 rows in the Phase 3 manifest (`manifest_v1_phase3_labels.csv`); **0 rows** in the current `manifest.csv` after `_clean_report_text()` was added to `prepare_data.py` on 2026-09-17.

**Label distribution — study-level, current manifest (`manifest.csv`), raw labeler values and the U-Ones training rate:**

| Class | pos (1) | neg (0) | unc (−1) | NaN | U-Ones pos % |
|---|---:|---:|---:|---:|---:|
| No Finding | 2642 | 1184 | 0 | 0 | 69.1 |
| Enlarged Cardiomediastinum | 8 | 35 | 0 | 3783 | 0.2 |
| Cardiomegaly | 297 | 5 | 1 | 3523 | 7.8 |
| Lung Opacity | 471 | 571 | 16 | 2768 | 12.7 |
| Lung Lesion | 136 | 175 | 13 | 3502 | 3.9 |
| Edema | 83 | 264 | 16 | 3463 | 2.6 |
| Consolidation | 37 | 1113 | 3 | 2673 | 1.0 |
| Pneumonia | 60 | 171 | 24 | 3571 | 2.2 |
| Atelectasis | 312 | 7 | 56 | 3451 | 9.6 |
| Pneumothorax | 27 | 2451 | 2 | 1346 | 0.8 |
| Pleural Effusion | 161 | 2663 | 27 | 975 | 4.9 |
| Pleural Other | 34 | 0 | 7 | 3785 | 1.1 |
| Fracture | 103 | 88 | 11 | 3624 | 3.0 |
| Support Devices | 98 | 0 | 2 | 3726 | 2.6 |

Produced by `src/medxvlm/data/labeler.py::aggregate_report` via `scripts/prepare_data.py`; validated by `scripts/validate_labeler.py` (24 cases, 42 assertions, all pass — run output quoted in `docs/v2_rebuild.md` §1).

**Label-fix delta (Phase 3 manifest → current), study-level U-Ones %:** Pneumothorax 10.2 → 0.8; Pleural Effusion 19.6 → 4.9; Consolidation 2.8 → 1.0; Pneumonia 3.8 → 2.2; No Finding 51.0 → 69.1. Cause: a negation-scope bug (`docs/v2_rebuild.md` §1 row 2). The old manifest is preserved as `data/processed/manifest_v1_phase3_labels.csv`. Split membership is identical between the two manifests (verified in that doc).

---

## 4. Stage 1 — Contrastive pretraining

### Checkpoints (`find checkpoints -name "*.pt" -printf …`)
| Path | Size | Modified | Recorded in ckpt |
|---|---:|---|---|
| `checkpoints/align/local_on/best.pt` | 784,407,363 B | 2026-09-05 14:35 | epoch 5, val_recall@1_mean 0.0127 |
| `checkpoints/align/local_off/best.pt` | 784,407,363 B | 2026-09-05 14:54 | epoch 5, val_recall@1_mean 0.0194 |
| `checkpoints/align_v2/local_on/best.pt` | 784,138,134 B | 2026-09-17 16:52 | epoch 15, val_recall@1_mean 0.0517; cfg: xraydinov2, frozen vision, Frontal |
| `checkpoints/align_v2/local_on_valloss/best.pt` | 784,138,262 B | 2026-09-17 18:33 | epoch 5, val_loss 4.678, select_by val_loss |
| `checkpoints/align_v2/local_on_frozen_text/best.pt` | 784,138,262 B | 2026-09-17 19:34 | epoch 9, val_loss 5.622 |
(each has a matching `last.pt`)

### Training logs, duration, final loss
- Phase 3 `local_on`: `runs/align/local_on/metrics.csv`, 5 epochs; final row `epoch 5: train/loss 3.7344, val/loss 4.6833`. Log: `Done in 19.9 min. Best val recall@1 (mean): 0.0127.` (`runs/phase3_orchestrator.log`)
- Phase 3 `local_off`: 5 epochs; final `train/loss 1.4670, val/loss 2.1621`. `Done in 19.1 min. Best val recall@1 (mean): 0.0194.`
- v2 `local_on`: `runs/align_v2/local_on/metrics.csv`, 15 epochs; epoch 7 val loss minimum 4.6745; final `epoch 15: train/loss 1.8087, val/loss 5.9729`. `Done in 36.9 min. Best val recall@1 (mean): 0.0517.` (`runs/v2_rebuild.log`)
- v2 `local_on_valloss`: 10 epochs; `Done in 20.7 min. select_by=val_loss | best val recall@1 (mean): 0.0362 | best val loss: 4.6783.` (`runs/align_v2_valloss.log`)
- v2 `local_on_frozen_text`: 10 epochs; `Done in 21.3 min. … best val loss: 5.6218.` (`runs/align_v2_frozen_text.log`)

### Retrieval evaluation (computed every epoch on the val split by `train_align.py`; quoted from the metrics CSVs)
| Run | epoch | R@1 i2t | R@5 i2t | R@10 i2t | R@1 t2i | R@5 t2i | R@10 t2i |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 3 local_on | 5 | 0.0134 | 0.0601 | 0.0921 | 0.0120 | 0.0668 | 0.1041 |
| Phase 3 local_off | 5 | 0.0174 | 0.0841 | 0.1322 | 0.0214 | 0.0668 | 0.1215 |
| v2 local_on | 15 | 0.0439 | 0.1318 | 0.2171 | 0.0594 | 0.1705 | 0.2403 |
| v2 local_on (val-loss best, ep 7) | 7 | 0.0362 | 0.1292 | 0.2506 | 0.0439 | 0.1576 | 0.2248 |
| v2 local_on_valloss | 5 | 0.0284 | 0.1421 | 0.2429 | 0.0439 | 0.1421 | 0.2171 |
| v2 local_on_frozen_text | 9 | 0.0155 | 0.0594 | 0.1008 | 0.0052 | 0.0413 | 0.1034 |

Val set for these = 749 (Phase 3, all views) or 387 (v2, frontal) image-report pairs. No retrieval evaluation has been run on the **test** split.

---

## 5. Stage 2 — Multi-label classification

### Checkpoints
| Path | Size | Modified | Recorded best |
|---|---:|---|---|
| `checkpoints/classifier/local_on/best.pt` | 797,845,787 B | 2026-09-05 15:30 | epoch 8, val macro AUROC 0.8645 (text-fusion protocol) |
| `checkpoints/classifier/local_off/best.pt` | 797,845,787 B | 2026-09-05 16:07 | epoch 8, 0.8648 (text-fusion) |
| `checkpoints/classifier_v2/local_on/best.pt` | 352,488,997 B | 2026-09-17 17:11 | epoch 8, val macro AUROC **0.8332**; cfg `use_text_fusion: False, loss: asl, projections: [Frontal]` |
| `checkpoints/baseline/best.pt` | 343,290,805 B | 2026-09-05 14:14 | epoch 5, 0.6939 |
| `checkpoints/baseline_v2/vit_base_patch16_224/best.pt` | 343,290,933 B | 2026-09-17 17:42 | epoch 7, 0.7812 |
| `checkpoints/baseline_v2/xraydinov2/best.pt` | 343,025,288 B | 2026-09-17 18:08 | epoch 7, 0.7761 |

### Training logs
- Phase 3 Stage 2 `local_on`: 8 epochs, `Done in 35.6 min. Best val macro AUROC: 0.8645.`; `local_off`: `Done in 36.2 min … 0.8648.` (`runs/phase3_orchestrator.log`)
- v2 Stage 2: 14 epochs run of 25 (early-stopped: log line `Early stopping: no val macro AUROC improvement for 6 epochs.`), `Done in 33.8 min. Best val macro AUROC: 0.8332.`; `runs/classifier_v2/local_on/metrics.csv` epoch 8 row: `val/macro_auroc 0.833229, val/macro_auprc 0.351212, val/macro_f1 0.290631`. Train macro AUROC reached 0.9828 by epoch 14 while val fell to 0.7931 (overfitting, documented).
- Baseline v2 ImageNet ViT: 12 epochs, `Done in 25.9 min. … 0.7812.`; XrayDINOv2+linear: `Done in 26.1 min. … 0.7761.`

### Validation-split per-class metrics at best epoch (from `runs/*/metrics.csv`)
Format AUROC / AUPRC / F1@0.5. "v2 S2" = `classifier_v2/local_on` (image-only). "P3 S2" = `classifier/local_on` (text-fusion — **not** an image-only number).

| Class | v2 S2 (img-only) | P3 S2 (text-fusion) | Baseline P3 | Baseline v2 ImageNet | Baseline v2 XrayDINOv2 |
|---|---|---|---|---|---|
| No Finding | 0.848 / 0.910 / 0.849 | 0.919 / 0.922 / 0.829 | 0.656 / 0.635 / 0.692 | 0.788 / 0.861 / 0.840 | 0.762 / 0.843 / 0.817 |
| Enlarged Cardiomediastinum | 0.982 / 0.125 / 0.000 | 0.912 / 0.022 / 0.000 | 0.892 / 0.018 / 0.000 | 0.902 / 0.026 / 0.000 | 0.977 / 0.100 / 0.000 |
| Cardiomegaly | 0.926 / 0.722 / 0.658 | 0.929 / 0.711 / 0.074 | 0.848 / 0.422 / 0.000 | 0.922 / 0.496 / 0.314 | 0.886 / 0.454 / 0.367 |
| Lung Opacity | 0.838 / 0.456 / 0.450 | 0.937 / 0.675 / 0.676 | 0.685 / 0.236 / 0.000 | 0.790 / 0.372 / 0.424 | 0.741 / 0.268 / 0.000 |
| Lung Lesion | 0.805 / 0.234 / 0.182 | 0.828 / 0.085 / 0.000 | 0.644 / 0.040 / 0.000 | 0.550 / 0.053 / 0.000 | 0.738 / 0.107 / 0.000 |
| Edema | 0.902 / 0.668 / 0.650 | 0.833 / 0.373 / 0.000 | 0.807 / 0.301 / 0.000 | 0.851 / 0.466 / 0.000 | 0.847 / 0.439 / 0.000 |
| Consolidation | 0.830 / 0.120 / 0.000 | 0.829 / 0.098 / 0.000 | 0.723 / 0.079 / 0.000 | 0.809 / 0.113 / 0.000 | 0.696 / 0.056 / 0.000 |
| Pneumonia | 0.837 / 0.174 / 0.133 | 0.758 / 0.104 / 0.000 | 0.614 / 0.056 / 0.000 | 0.741 / 0.049 / 0.000 | 0.569 / 0.029 / 0.000 |
| Atelectasis | 0.872 / 0.433 / 0.450 | 0.949 / 0.706 / 0.627 | 0.724 / 0.207 / 0.000 | 0.821 / 0.373 / 0.327 | 0.755 / 0.262 / 0.000 |
| Pneumothorax | 0.641 / 0.016 / 0.000 | 0.840 / 0.486 / 0.086 | 0.558 / 0.116 / 0.000 | 0.674 / 0.040 / 0.000 | 0.821 / 0.051 / 0.000 |
| Pleural Effusion | 0.916 / 0.561 / 0.514 | 0.837 / 0.602 / 0.455 | 0.541 / 0.260 / 0.000 | 0.904 / 0.442 / 0.414 | 0.828 / 0.340 / 0.211 |
| Pleural Other | 0.752 / 0.295 / 0.000 | 0.827 / 0.029 / 0.000 | 0.644 / 0.015 / 0.000 | 0.812 / 0.045 / 0.000 | 0.815 / 0.061 / 0.000 |
| Fracture | 0.654 / 0.047 / 0.000 | 0.866 / 0.132 / 0.000 | 0.693 / 0.039 / 0.000 | 0.624 / 0.039 / 0.000 | 0.690 / 0.072 / 0.000 |
| Support Devices | 0.860 / 0.156 / 0.182 | 0.839 / 0.122 / 0.000 | 0.686 / 0.102 / 0.000 | 0.748 / 0.149 / 0.000 | 0.743 / 0.222 / 0.000 |
| **Macro** | **0.833 / 0.351 / 0.291** | 0.864 / 0.362 / 0.196 | 0.694 / 0.180 / 0.049 | 0.781 / 0.252 / 0.166 | 0.776 / 0.236 / 0.100 |

Note: Phase 3 and v2 rows use different labels (old vs corrected manifest) and different view sets (all vs frontal) — they are not directly comparable to each other; the held-out test table below is the like-for-like comparison.

### Held-out TEST-split evaluation (`scripts/eval_v2_classification.py` → `docs/phase5_artifacts/v2/v2_classification_summary.json`, mtime 2026-09-17 19:39)
Frontal test rows, n = **377**, corrected labels, every system image-only. Per-class positive support: No Finding 268, Cardiomegaly 36, Lung Opacity 42, Atelectasis 36, Pleural Effusion 15, Edema 12, Lung Lesion 9, Pneumonia 8, Fracture 8, Support Devices 8, Consolidation 5, Pleural Other 3, Pneumothorax 1, Enlarged Cardiomediastinum 0 (undefined AUROC).

| System | Macro AUROC, 8 RadZero-shared classes | …of those with ≥10 positives (5 classes) | Macro AUROC, all 13 valid classes |
|---|---:|---:|---:|
| **v2 Stage 2, image-only** | **0.8548** | **0.8621** | **0.8287** |
| RadZero zero-shot (re-scored from Phase 5 saved probabilities) | 0.8684 | 0.8750 | n/a (8 classes only) |
| Baseline v2: ImageNet ViT-B/16 + linear | 0.8043 | 0.8172 | 0.7383 |
| Baseline v2: XrayDINOv2 + linear | 0.7767 | 0.7948 | 0.7341 |
| Phase 3 Stage 2 run blind | 0.5718 | 0.5255 | 0.5469 |
| Phase 3 Stage 1 zero-shot | 0.6693 | 0.7085 | 0.5623 |
| v2 Stage 1 zero-shot (ep-15 ckpt) | 0.5650 | 0.5474 | 0.5265 |
| v2 Stage 1 zero-shot (val-loss ckpt) | 0.5725 | 0.5192 | 0.5039 |
| v2 Stage 1 zero-shot (frozen text) | 0.4838 | 0.5200 | 0.4682 |

v2 Stage 2 per-class test AUROC (from the JSON): No Finding 0.820, Cardiomegaly 0.939, Lung Opacity 0.854, Lung Lesion 0.653, Edema 0.879, Consolidation 0.832, Pneumonia 0.771, Atelectasis 0.851, Pneumothorax 0.949 (n=1), Pleural Effusion 0.787, Pleural Other 0.664, Fracture 0.808, Support Devices 0.966.

Earlier Phase 5 test evaluation on the **old** labels, all 748 test rows (`docs/phase5_artifacts/medxvlm_testsplit_classification_summary.json`, 2026-09-06): Phase 3 Stage 2 blind 0.5074; with ground-truth text fusion 0.8259; Stage 1 zero-shot 0.5358 (13 valid classes). No test-split AUPRC/F1 table has been produced; test evaluation reports AUROC only.

---

## 6. Stage 3 — Report generation

Two trained models exist. The **v2** model (2026-09-19) is current; the Phase 3 model is kept as the "before".

### 6.1 Checkpoints and training logs
| | Phase 3 | v2 |
|---|---|---|
| Checkpoint | `checkpoints/report_gen/best.pt`, 88,148,017 B, 2026-09-05 21:57, epoch 2, val_loss 0.9987 | `checkpoints/report_gen_v2/best.pt`, 88,148,145 B, 2026-09-19 14:15, **epoch 3, val_loss 1.1321** |
| Contents | LoRA adapters + projector only (`trainable_state_dict`); base `Qwen/Qwen2.5-1.5B-Instruct` not stored | same |
| Vision tower (frozen) | Phase 3 Stage 1 BiomedCLIP ViT-B/16 (`checkpoints/align/local_on/best.pt`) | **v2 Stage 2 fine-tuned XrayDINOv2** (`checkpoints/classifier_v2/local_on/best.pt`) |
| Training data | all 5,929 train rows (both views), old manifest ("XXXX" present) | 3,030 frontal rows, cleaned text, **abnormal studies oversampled to 50% of sampling mass** (`WeightedRandomSampler`) |
| Visual tokens / batch | 7×7 = 49 / batch 2 × accum 8 | 8×8 = 64 / batch 4 × accum 4 |
| Trainable / frozen params | 22,006,784 / 1,629,098,240 | 22,006,784 / 1,629,025,280 |
| Log | `runs/train_report_gen.log`, `runs/report_gen/metrics.csv` | `runs/train_report_gen_v2.log`, `runs/report_gen_v2/metrics.csv` |
| Val loss by epoch | 1.022, **0.999**, 1.106 | 1.157, 1.156, **1.132**, 1.200, 1.339 (`Early stopping: no val loss improvement for 2 epochs.`) |
| Wall time | `Done in 210.7 min.` | `Done in 278.6 min.` |
| Config | `configs/report_gen.yaml` | `configs/report_gen_v2.yaml` |

(Phase 3 and v2 val losses are not comparable: different data, tokens, and sampling.)

### 6.2 Saved samples
- Phase 3: `runs/report_gen/samples/epoch_00{1,2,3}.txt`; v2: `runs/report_gen_v2/samples/epoch_00{1..5}.txt` (3 val samples each). The 3 fixed val samples are all normal studies, so they do not show abnormal behaviour; see the test-set generations below.
- Test-subsample generations, identical 150 images (`docs/phase5_artifacts/report_gen_subsample.csv`, 80 lateral / 70 frontal): Phase 3 `docs/phase5_artifacts/medxvlm_stage3_generations.csv`; v2 `docs/phase5_artifacts/v2/medxvlm_stage3_v2_generations.csv`; CheXOne `/home/kartik/DNN Project 2/external_repro/chexone_testsplit_generations.csv`.
- Verbatim example, abnormal frontal case `2919_IM-1321-3001` (labels Cardiomegaly, Edema):
  > REFERENCE: The heart is significantly enlarged. Prominent pulmonary vascularity. No focal airspace consolidation, suspicious pulmonary opacity, or definite pleural effusion. No pneumothorax. …
  > PHASE 3: The heart size and pulmonary vascularity appear within normal limits. The lungs are free of focal airspace disease. No pleural effusion or pneumothorax is seen. 1. No evidence of active disease.
  > V2: Cardiomegaly. No infiltrates or effusions. Right catheter tip upper SVC. The long right IJ catheter tip is at the lower SVC.
  (v2 gets the cardiomegaly the Phase 3 model missed, and hallucinates two catheters the reference does not mention.)

### 6.3 Scores — all three systems, same 150 images, corrected labels
Source: `scripts/eval_v2_score_reports.py` → `docs/phase5_artifacts/v2/report_generation_scores_v2.json` (run 2026-09-19). CE = `medxvlm.data.labeler` on generated text vs. manifest label vector, U-Ones. BLEU-4 = sacrebleu corpus BLEU (0–100); ROUGE-L = rouge_score F, mean over reports.

| System | Subset | CE macro F1 | CE micro F1 | BLEU-4 | ROUGE-L | Distinct / n | Modal-output share | Mean words |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 3 Stage 3 | frontal 70 | 0.078 | 0.603 | 12.08 | 0.304 | 5 / 70 | 74% | 29.6 |
| CheXOne (instruct) | frontal 70 | 0.149 | 0.599 | 10.24 | 0.286 | 36 / 70 | 26% | 28.8 |
| **Stage 3 v2** | frontal 70 | **0.152** | 0.575 | 9.76 | 0.280 | 26 / 70 | 54% | 35.4 |
| Phase 3 Stage 3 | all 150 | 0.066 | 0.614 | 12.15 | 0.308 | 5 / 150 | 71% | 29.7 |
| CheXOne (instruct) | all 150 | 0.107 | 0.479 | 9.61 | 0.278 | 87 / 150 | 13% | 29.0 |
| **Stage 3 v2** | all 150 | **0.119** | 0.424 | 7.09 | 0.244 | 61 / 150 | 39% | 45.6 |

Per-class CE F1, frontal 70 (support; Phase 3 / CheXOne / v2): No Finding (46) 0.800 / 0.828 / 0.849; Cardiomegaly (6) 0.286 / 0.000 / 0.333; Lung Opacity (9) 0.000 / 0.364 / 0.364; Lung Lesion (4) 0 / 0 / 0.333; Pleural Effusion (4) 0 / 0.400 / 0.250; Support Devices (1) 0 / 0.500 / 0; Edema (3), Pneumonia (2), Atelectasis (7), Fracture (3), Pleural Other (1): 0 for all.

**Hallucination rate** (computed for this report: generation mentions catheter/PICC/tube/line/clips/wires/pacemaker/lobectomy/sternotomy while the reference does not): Phase 3 1/70 frontal; CheXOne 0/70; **v2 7/70 (10%)**, and 40/150 (27%) including laterals. Training-set explanation: device/surgery words occur in 15% of abnormal frontal reports vs 4% of normal ones, so oversampling abnormal studies taught the decoder a device-mention shortcut.

The **superseded** Phase 5 scores on the old labels (`docs/phase5_artifacts/report_generation_clinical_efficacy.json`, 2026-09-06): Phase 3 macro F1 0.075 / CheXOne 0.105. METEOR: not computed (no implementation in repo).

## 7. Explainability outputs

- **Mechanism implemented:** per-label attention weights from `ClassifierHead` (`ClassifierOutput.attention_weights`, `[B, num_labels, num_patches]`) overlaid as JET heatmaps — `src/medxvlm/explain/attention_maps.py`. Its docstring states this is *"this project's Grad-CAM/'segmentation map' equivalent, not a separate gradient-based CAM."*
- **Not implemented:** gradient-based Grad-CAM, attention rollout, text-grounded heatmaps, sentence attribution — `grep -rn -i "gradcam\|rollout\|text_grounded" src scripts` returns only comments/docstrings, no code. These remain proposal items (`docs/project_proposal.md` lines 45, 100–101).
- **Files on disk:** 120 PNGs total (`find runs -path "*attention_maps*" -name "*.png" | wc -l`), 12 per snapshot directory (4 samples × top-3 labels):
  - `runs/classifier/attention_maps/local_on/epoch_{002,004,006,008}/` and `…/local_off/epoch_{002,004,006,008}/` — 96 PNGs (Phase 3)
  - `runs/classifier_v2/attention_maps/local_on/epoch_{005,010}/` — 24 PNGs (v2). Example: `runs/classifier_v2/attention_maps/local_on/epoch_010/sample0_Fracture.png`.
- **Sanity check logged:** yes, qualitative only — `docs/ablation_study.md` §"Qualitative: grounding-map sharpness" spot-compares two samples at epoch 8: *"Pleural Effusion, sample 0: both conditions concentrate attention on the lower/lateral lung fields near the costophrenic angles — the clinically correct region"*; *"Pneumothorax, sample 1 … local-off looks sharper on this sample — the opposite of the hypothesis in docs/project_proposal.md RQ2."* No quantitative grounding metric (pointing game, IoU) has been computed; the v2 snapshots have not been inspected in any doc.

---

## 8. Ablation study

`docs/ablation_study.md` (2026-09-05). Config compared: Stage 1 local patch-word loss ON vs OFF, everything else fixed; each Stage 1 checkpoint then fed to an otherwise-identical Stage 2 run. Source runs: `runs/align/{local_on,local_off}/metrics.csv`, `runs/classifier/{local_on,local_off}/metrics.csv`. Quoted tables:

**Stage 1 retrieval, final epoch 5/5**
| Metric | Local loss ON | Local loss OFF |
|---|---:|---:|
| Train loss (final) | 3.7344 (global 1.8735 + local 1.8609) | 1.4670 |
| Val loss | 4.6833 | 2.1621 |
| Recall@1 (i2t) | 0.0134 | 0.0174 |
| Recall@5 (i2t) | 0.0601 | 0.0841 |
| Recall@10 (i2t) | 0.0921 | 0.1322 |
| Recall@1 (t2i) | 0.0120 | 0.0214 |
| Recall@5 (t2i) | 0.0668 | 0.0668 |
| Recall@10 (t2i) | 0.1041 | 0.1215 |

**Stage 2 classification, best epoch 8/8** (text-fusion protocol, old labels — see Phase 3 caveat)
| Metric | ON | OFF | Δ (on − off) |
|---|---:|---:|---:|
| Macro AUROC | 0.8645 | 0.8648 | −0.0003 |
| Macro AUPRC | 0.3619 | 0.3549 | +0.0070 |
| Macro F1 | 0.1966 | 0.2008 | −0.0042 |

Per-class AUROC deltas (on − off) range from −0.0657 (Support Devices) to +0.0570 (Consolidation); full 14-row table at doc lines 63–78.

**Second ablation (Asymmetric Loss vs BCE on the baseline):** doc §"Second ablation" says **"Not run in this pass."** No `--loss asl` baseline checkpoint exists (`checkpoints/baseline/` has one run, cfg `loss: bce`). Note that the v2 Stage 2 run *does* use ASL (`configs/classifier_v2.yaml: loss: asl`), but no controlled ASL-vs-BCE comparison was performed.

The v2 rebuild's baseline-vs-pipeline rows (§5 test table: XrayDINOv2+linear 0.777 vs v2 Stage 2 0.855 on the same encoder/data) function as an encoder-controlled ablation of the Stage 2 head, but are documented in `docs/v2_rebuild.md`, not in the ablation doc.

---

## 9. External benchmark reproduction (RadZero / CheXOne)

### Feasibility check — `docs/external_repro_feasibility.md` (2026-09-06)
Conclusion (verbatim from its summary table and text): both repos public with public weights; RadZero ~0.2B params fp32 "fits in 16GB trivially"; CheXOne ~3B Qwen2.5-VL "bf16 weights ≈ 6GB, comfortably inside the 16GB"; RadZero native quantitative eval **"Yes"** (OpenI is on disk); CheXOne native quantitative eval **"No, not the paper's benchmark numbers"** because its benchmarks need PhysioNet-credentialed MIMIC-CXR plus gated annotation files. *"Neither repo is blocked outright … That is the actual constraint, not GPU memory."*

### Actually reproduced in this project (all under `/home/kartik/DNN Project 2/external_repro/`)
| What | Command / script | Result | Source file |
|---|---|---|---|
| RadZero zero-shot classification, RadZero's own OpenI split, 1,000-image random subsample (seed 0), 18 classes, keyword-derived labels from RadZero's `custom.csv` | `python3 run_radzero_openi_eval.py --n 1000 --seed 0` | **macro AUROC 0.8524** over 17 valid classes (Abnormal Lesion had 0 positives); 2,647 s | `radzero_openi_eval_n1000_seed0_summary.json` |
| RadZero on MedXVLM's 748-image test split, 8 shared classes, MedXVLM's **old** labels | `python3 run_radzero_on_medxvlm_testsplit.py` | macro AUROC **0.7367**; 1,949 s. Per class: Atelectasis 0.889, Cardiomegaly 0.941, Pleural Effusion 0.554, Pneumonia 0.750, Pneumothorax 0.512, Edema 0.775, Fracture 0.621, Lung Opacity 0.851 | `radzero_on_medxvlm_testsplit_summary.json` |
| RadZero, same saved probabilities re-scored on the 377 frontal test rows with **corrected** labels | `scripts/eval_v2_classification.py` | macro AUROC **0.8684** | `docs/phase5_artifacts/v2/v2_classification_summary.json` |
| CheXOne Quick Start (README `transformers` path), 2 images | `python3 run_chexone_demo.py <img>` | ran; 7.51 GB VRAM; 20.9 s / 257 tokens reasoning mode; qualitative only | `docs/external_baseline_reproduction.md` §4 |
| CheXOne findings generation, MedXVLM 150-image test subsample, instruct mode | `python3 run_chexone_on_medxvlm_testsplit.py` | 150 generations; CE macro F1 **0.105** (see §6) | `chexone_testsplit_generations.csv`, `report_generation_clinical_efficacy.json` |

Weights used: `Deepnoid/RadZero` rev `1e6e7d6295a13aa0a4a7e1c1e2bb36b9ef4e02e8`; `StanfordAIMI/CheXOne` rev `0c350e6852ea08f9d9baf3b7595c1a10d4849927` (recorded in the reproduction doc §3–§4). Environment deviation recorded there: RadZero's pinned `torch==2.2.1` cannot run on the sm_120 GPU, so `torch 2.14.0+cu130` and `transformers==4.46.3` were used instead.

### Paper-reported only — NOT reproduced here (labelled as such in the docs)
- RadZero: OpenI mean AUC **0.847** (arXiv:2504.07416, Table 5, full 7,470-image set, authors' labels).
- CheXOne (arXiv:2604.00493): ReXVQA presence 0.947, anatomical localization 0.931, negation 0.988, differential 0.951, geometric 0.883; view classification 0.998; findings generation on ReXGradient 1/RadCliQ 1.116 / BertScore 0.483 / BLEU 0.229 / RadGraph 0.21; reader study "comparable to or better than resident-written reports in 55% of cases". Every one of these is cited in `docs/external_baseline_reproduction.md` §4 with the label "reported by the authors, not reproduced here". No IU-Xray number for CheXOne is available from the source consulted.

---

## 10. Cross-dataset evaluation

`docs/cross_dataset_evaluation.md` (2026-09-06, addendum §9 on 2026-09-17).

- **Accessibility check:** §1–§2 conclude the only dataset both external models evaluate on that is reachable here is OpenI/IU X-ray — MedXVLM's own home dataset. MIMIC-CXR, CheXpert, PadChest, ReXGradient, ReXVQA, MS-CXR all require credentialing or license forms and were not obtained. **Direction run: flipped (their models on MedXVLM's held-out test split) for both models; MedXVLM was not run on any external dataset.**
- **Adapter:** label-taxonomy mapping CheXpert-14 ↔ RadZero-18, 8 shared concepts (Atelectasis, Cardiomegaly, Pleural Effusion, Pneumonia, Pneumothorax, Lung Opacity, Fracture/Bone Fracture, Edema/Pulmonary Edema), each row tagged (doc §3).
- **Results produced** (doc §4, old labels, 748 images; column headers carry the tags `[apples-to-apples]` / `[NOT COMPARABLE — see §3]`):

| System | Macro AUROC, 8 shared | Tag |
|---|---:|---|
| RadZero zero-shot | 0.737 | apples-to-apples |
| MedXVLM Stage 1 zero-shot | 0.603 | apples-to-apples |
| MedXVLM Stage 2 blind | 0.512 | apples-to-apples |
| MedXVLM Stage 2 text-fusion | 0.851 | **not comparable** (sees ground-truth text) |
| RadZero own reported, full OpenI | 0.847 | reported by authors, not reproduced |

  Report generation (doc §5, 150 identical images, tagged apples-to-apples): CheXOne 0.105 vs MedXVLM 0.075 macro F1 (§6 above).
- **Addendum §9** records the label fix and the post-rebuild replacement row: RadZero 0.868 vs MedXVLM v2 image-only 0.855 (frontal test, corrected labels), and states that §4's Pneumothorax row "should be disregarded (it was almost entirely false positives)".
- **Skip list** (doc §7): direction flipped; RadZero's other datasets not run; only 8/18 and 8/14 classes comparable; CheXOne ran in instruct not reasoning mode; no BLEU/RadGraph/BERTScore computed.
- **Update 2026-09-20 — the un-run direction was run.** v2 Stage 2 evaluated as-is on the **official NIH ChestX-ray14 test split** (25,596 images / 2,797 patients, Kaggle mirror `nih-chest-xrays/data`, never seen in training): **macro AUROC 0.714** over 8 concept-identical classes (per class 0.634 Pneumothorax … 0.785 Effusion); vs 0.855 at home. Source: `scripts/eval_external_chestxray14.py` → `docs/phase5_artifacts/v2/chestxray14_external_eval_summary.json` (`elapsed_sec` 132). Labelled *approximate* (different hospital/labeler/class subset). RadZero's paper-reported 0.804 on this dataset is cited only.

---

## 11. Known issues / incomplete items

**Code markers** (`grep -rn -E "TODO|FIXME|HACK|\bXXX\b" --include="*.py" src scripts app`):
- `src/medxvlm/models/classifier_head.py:5` — docstring says *"No text fusion yet … Gated text fusion … is the next milestone; see the # TODO below"* — **stale**: text fusion was implemented (`text_proj`/`gate` at lines 97–113 of that file) and later deliberately disabled for v2 via `use_text_fusion: false`.
- `src/medxvlm/models/text_encoder.py:68` — `# TODO: swap in PubMedBERT weights if the checkpoint isn't reachable in this environment.` (fallback path to `bert-base-uncased`; the PubMedBERT/BiomedBERT weights do load in this environment per training logs).
- No `raise NotImplementedError`, no `pass  # TODO` stubs found (`grep` returned nothing).

**Documented limitations / skipped work** (from the docs' own "skip list" / "didn't finish" / "negative result" sections):
1. **Stage 3 v2 hallucinates support devices / surgical history** in 10% of frontal generations (27% incl. laterals), a failure the Phase 3 model and CheXOne do not have; the modal normal report still covers 54% of frontal outputs; recall on atelectasis/edema/fracture/pneumonia remains 0 — `docs/v2_rebuild.md` §3.4, §6 above. (The Phase 3 model's mode collapse — 5 distinct outputs in 150 — is resolved: 61 distinct.)
2. **Phase 3 Stage 2 headline (0.865) superseded** — measured with ground-truth report as input — banner in `docs/training_summary.md`.
3. **Zero-shot head is a documented negative result** (0.48–0.57 across three Stage 1 variants) — `docs/v2_rebuild.md` §3.3.
4. **Test-set support is thin after the label fix**: Pneumothorax n=1, Pleural Other n=3, Consolidation n=5, Enlarged Cardiomediastinum n=0 → per-class AUROCs for these are not meaningful — `docs/v2_rebuild.md` §3.2.
5. **Stage 1 checkpoint selection by recall@1 is noisy** (picked the most-overfit epoch 15) — `docs/v2_rebuild.md` §2; a `--select-by val_loss` option was added.
6. **ASL-vs-BCE ablation never run** — `docs/ablation_study.md` §"Second ablation".
7. **No METEOR / RadGraph for report generation.** BLEU-4 and ROUGE-L were added on 2026-09-19 (`scripts/eval_v2_score_reports.py`); they slightly favour the collapsed Phase 3 model and are reported but not used to rank systems (`docs/v2_rebuild.md` §3.4 item 4).
8. **No retrieval or classification metrics on the test split for Phase 3-era Stage 1** beyond the zero-shot rows; retrieval R@k exists on val only.
9. **Explainability limited to attention-weight overlays**; no Grad-CAM / rollout / text-grounded heatmaps / sentence attribution; no quantitative grounding metric.
10. **`app/` demo UI not started** (only `.gitkeep`; `streamlit` pinned but unused).
11. **MedXVLM never evaluated on an external dataset** — `docs/cross_dataset_evaluation.md` §1/§7.
12. **Phase 3 infrastructure incident**: first Stage 3 attempt killed at model load by session teardown; restarted with `setsid` — `docs/training_summary.md` §"What didn't finish".
13. **`README.md` "Project status" is stale** (says Phase 3; no mention of Phases 4–5, the v2 rebuild, or the superseded number).
14. **Zero git commits** — no history, no reproducible snapshot of which code produced which checkpoint beyond file mtimes and the `config` dicts saved inside each `.pt`.
15. **Storage incident (2026-09-19/20).** `df` inside WSL reports the virtual disk's 1 TB nominal size, not the host drive; the 45 GB ChestX-ray14 download grew `F:\WSL\Ubuntu\ext4.vhdx` to 193 GB and filled the 466 GB host F: drive to 542 MB free, causing a reboot that killed a detached download and the queued Stage 3 v2b retrain. Freed ~100 GB inside WSL afterwards (86,524 non-test NIH images, pip/conda/CheXOne-HF caches, both `external_repro` venvs, all `last.pt` checkpoints — `best.pt` kept everywhere). The `.vhdx` does not shrink on its own; host-side compaction is required to return the space to F:.
16. **A residual labeler false positive** was observed during Stage 3 inspection: `1367_IM-0237-1001` is labelled Pleural Effusion = 1 from *"…no pleural line to suggest pneumothorax or costophrenic blunting to suggest large pleural effusion"* — negation does not reach across "…to suggest". The 2026-09-17 fix covered the dominant coordinated-list pattern, not every scope pattern. Not fixed.

---

## 12. Summary table

| Component | Status | Key number available |
|---|---|---|
| Stage 1 — contrastive alignment | Done (Phase 3 ×2, v2 ×3 runs) | yes — v2 val R@1 i2t/t2i 0.0439/0.0594 (ep 15), val loss min 4.6745 (ep 7); Phase 3 local_off R@1 0.0174/0.0214 (`runs/align*/…/metrics.csv`) |
| Stage 2 — classification | Done (v2 image-only is the current model) | yes — **test macro AUROC 0.855 (8 shared) / 0.829 (13 classes)**, val 0.833 (`docs/phase5_artifacts/v2/v2_classification_summary.json`, `runs/classifier_v2/local_on/metrics.csv`) |
| Stage 3 — report generation | Done — **retrained 2026-09-19 (v2)**; collapse resolved, hallucination issue open; a v2b run (30% abnormal sampling) is prepared (`configs/report_gen_v2b.yaml`) but **not run** — first attempt crashed (cuBLAS, disk-contention), awaiting a GPU window | yes — v2 CE macro F1 **0.152** (70 frontal) / 0.119 (150) vs CheXOne 0.149 / 0.107; BLEU-4 9.76, ROUGE-L 0.280 (frontal); 61 distinct / 150; val loss 1.132 (ep 3) (`docs/phase5_artifacts/v2/report_generation_scores_v2.json`) |
| Explainability | In progress — attention-weight heatmaps only | no quantitative number; 120 PNGs on disk; qualitative spot-check in `docs/ablation_study.md` |
| Ablation | Done (local-loss on/off); ASL-vs-BCE not run | yes — Stage 2 macro AUROC 0.8645 vs 0.8648 (text-fusion protocol); Stage 1 R@1 i2t 0.0134 vs 0.0174 (`docs/ablation_study.md`) |
| External benchmark (RadZero / CheXOne) | Done for RadZero; CheXOne qualitative + one generation run only | yes — RadZero 0.852 (own OpenI subsample), 0.868 (MedXVLM frontal test, corrected labels); CheXOne CE F1 0.105 on 150 MedXVLM images; CheXOne benchmarks: paper-reported only |
| Cross-dataset | Done in both directions (flipped 2026-09-06; MedXVLM→NIH 2026-09-20) | yes — RadZero 0.868 vs MedXVLM 0.855 on identical images; **MedXVLM on NIH ChestX-ray14 test split: 0.714** macro AUROC, 8 classes (`chestxray14_external_eval_summary.json`) |
| Demo app | Not started | no |
| Version control | Not started (0 commits) | — |
