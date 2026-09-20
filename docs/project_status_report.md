# MedXVLM — Comprehensive Project Status Report

Generated 2026-09-20 from direct inspection of the working tree at `medxvlm/` (git `412c3b3`). Companion documents: [`status_report.md`](status_report.md) is the line-by-line sourced inventory of every number and file; [`v2_rebuild.md`](v2_rebuild.md) is the narrative of what changed after Phase 5 and why. Where this report states a number, its source is one of those two documents or a file named inline.

---

## 1. Project overview

- **Name:** MedXVLM
- **Purpose:** A three-stage vision-language system for chest X-rays — align images with radiology report text (Stage 1), classify 14 CheXpert-style findings from the image alone (Stage 2), and generate a free-text findings paragraph (Stage 3) — built and evaluated as a B.Sc. major project, with the explicit goal of honest measurement against two published external models (RadZero, CheXOne).
- **Stack:** Python 3.11.15, PyTorch 2.11.0 (cu130, RTX 5080 / sm_120), HuggingFace `transformers` 5.8.0, `timm` 1.0.28, `open_clip_torch` 3.3.0, `peft` 0.20.0 (LoRA), `scikit-learn` 1.8.0, `pandas`, `sacrebleu` 2.6.0, `rouge-score` 0.1.2, `tensorboard`, `weasyprint` (PDF), `kagglehub` (data). Ubuntu 26.04 under WSL2 on Windows; conda env `ml`.
- **Dataset:** Indiana University / Open-i chest X-ray collection (public, 3,826 usable studies / 7,426 images after cleaning), plus the NIH ChestX-ray14 official test split (25,596 images) used once, for an external-generalisation test.

---

## 2. Current state — working vs. incomplete

**Working right now (trained, evaluated, checkpoint on disk):**

| Component | Checkpoint | Honest headline number |
|---|---|---|
| Stage 2 image-only classifier (v2) | `checkpoints/classifier_v2/local_on/best.pt` | **0.855 macro AUROC** on the held-out frontal test split (377 images, 8 classes shared with RadZero); 0.829 over all 13 measurable classes. RadZero on identical images: 0.868. |
| Same classifier on an unseen hospital (NIH ChestX-ray14 test split) | same | **0.714 macro AUROC**, 8 mapped classes — the generalisation number. |
| Stage 3 report generator (v2) | `checkpoints/report_gen_v2/best.pt` (LoRA + projector, 88 MB) | Clinical-efficacy macro F1 **0.152** on 70 frontal test images (CheXOne: 0.149); 61 distinct outputs / 150 (Phase 3 model: 5). |
| Stage 1 contrastive alignment (v2) | `checkpoints/align_v2/local_on/best.pt` | Val loss 4.67 at epoch 7; retrieval R@1 ≈ 0.04–0.06 — functional as an encoder initialiser, **not** as a zero-shot classifier (see §6). |
| Rule-based CheXpert labeler | `src/medxvlm/data/labeler.py` | 24/24 validation cases pass (`scripts/validate_labeler.py`); negation-scope bug fixed 2026-09-17. |
| Two baselines (v2) | `checkpoints/baseline_v2/{vit_base_patch16_224,xraydinov2}/best.pt` | 0.804 / 0.777 macro AUROC on the same test split. |
| Attention-map explainability | `src/medxvlm/explain/attention_maps.py` | 120 heatmap PNGs under `runs/*/attention_maps/`; qualitative spot-check only. |

**Incomplete / not started / superseded:**

- **Stage 3 hallucination problem is open.** The v2 model invents catheters/lines/surgical history in 10 % of frontal generations (27 % including lateral views); the modal "normal" report still accounts for 54 % of frontal outputs; recall on atelectasis/edema/fracture/pneumonia is 0 on the test subsample. A mitigation run (`configs/report_gen_v2b.yaml`, 30 % abnormal sampling instead of 50 %) is prepared but **not run** — the first attempt crashed with `CUBLAS_STATUS_INTERNAL_ERROR` during a concurrent 45 GB download; it needs a ~4.5 h GPU window and the owner's go-ahead.
- **Zero-shot classification head does not work** at this data scale (0.48–0.57 AUROC across three Stage 1 variants) — documented as a negative result, not a bug.
- **Phase 3 numbers are superseded.** The 0.865 Stage 2 AUROC in `training_summary.md` was measured with the ground-truth report fused into the classifier; tested image-only, that checkpoint scores 0.507. The file carries a banner saying so.
- **`app/` demo UI:** built 2026-09-20 (`app/app.py` over `src/medxvlm/inference.py`); passes the 4-path headless check in `scripts/sanity_check_app.py` and serves via `streamlit run`.
- **Grad-CAM / attention rollout / text-grounded heatmaps / sentence attribution:** listed in the proposal, never implemented; only Stage 2's own per-label attention weights are visualised.
- **ASL-vs-BCE ablation on the baseline:** never run (`docs/ablation_study.md` §"Second ablation" says "Not run in this pass").
- **METEOR / RadGraph** for report generation: not computed. BLEU-4 and ROUGE-L were added on 2026-09-19.

**Broken / environmental:**

- The host **F: drive is at 542 MB free** (466 GB total). The WSL virtual disk grew to 193 GB during the NIH download; ~100 GB was freed *inside* WSL on 2026-09-20 but the `.vhdx` will not return that space to Windows until compacted from the host side. Until then, any Windows-side write is at risk. (See §6.)

---

## 3. Architecture summary

Data flow: `indiana_reports.csv` + `indiana_projections.csv` → `scripts/prepare_data.py` (clean text, run labeler, uid-level 80/10/10 split) → `data/processed/manifest.csv` → one `Dataset` class per stage → `scripts/train_*.py` → `checkpoints/*/best.pt` + `runs/*/metrics.csv` → `scripts/eval_*.py` → `docs/phase5_artifacts/**`.

### `src/medxvlm/models/`
- **`labels.py`** — `CHEXPERT_LABELS` (14-tuple) and `NUM_CHEXPERT_LABELS`; the single source of truth for label order, imported by every head.
- **`vision_encoder.py`** (152 lines) — `VisionEncoder(pretrained, backbone)` at L82. Two backbones: `"biomedclip"` (timm ViT-B/16 trunk from BiomedCLIP, `_load_biomedclip_trunk` L52, ImageNet fallback L62) and `"xraydinov2"` (`StanfordAIMI/dinov2-base-xray-224` via HF `Dinov2Model`, `_load_xraydinov2` L68). `forward` (L121) returns `VisionEncoderOutput(patch_tokens, pooled)`. `infer_backbone_from_state_dict` (L140) sniffs checkpoint key prefixes so pre-v2 checkpoints load without a stored backbone name.
- **`text_encoder.py`** (97 lines) — `TextEncoder` at L40 wraps `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` with `bert-base-uncased` fallback; `tokenize` L81, `forward` L90 returns per-token embeddings + pooler output.
- **`alignment.py`** (166 lines) — `ContrastiveAlignment` L41: projects both towers to a 512-d space (`encode_image` L70, `encode_text` L74), symmetric InfoNCE in `forward` L78; optional FILIP-style `local_patch_word_loss` L122 (the Phase 3 ablation switch).
- **`classifier_head.py`** (116 lines) — `PerLabelAttentionPooling` L36 (one learned query per label over patch tokens) → `ClassifierHead` L68 with a shared `Linear(D,1)`. Optional gated text fusion (`text_proj`/`gate`, L97–L101) is applied only when `forward(text_embeds=…)` receives text (L103–L107). **v2 never passes text**; Phase 3 passed the ground-truth report (the leak).
- **`baseline_classifier.py`** (99 lines) — `BaselineClassifier` L28 (backbone + linear head; supports `"xraydinov2"` via `VisionEncoder`), `multilabel_bce_loss` L68, `asymmetric_loss` L73 (ASL, used by v2 Stage 2).
- **`report_generator.py`** (217 lines) — `ReportGenerator` L40: frozen `VisionEncoder` → adaptive-avg-pool of the patch grid to `pooled_grid_size²` tokens → 2-layer MLP projector → spliced at a literal `<image>` token into `Qwen/Qwen2.5-1.5B-Instruct` (L22) with LoRA on q/k/v/o/gate/up/down (L25). `forward` L182 (teacher-forced, loss masked to answer tokens), `generate` L189 (greedy).

### `src/medxvlm/data/`
- **`labeler.py`** (300 lines) — rule-based CheXpert-style labeler. Cue tables at L38–L84 (`NEGATION_PRE_WINDOW = 12`, `SCOPE_RESET_TOKENS` L48, `PRE_NEGATION_CUES` L55, `LABEL_PHRASES` L84). `_pre_cue_in_scope` L210 is the 2026-09-17 fix that lets a leading "no" govern a whole coordinated list until a scope-reset token. `find_mentions` L266, `aggregate_report` L286 (positive-wins aggregation; derives "No Finding").
- **`dataset.py`** (216 lines) — `CLAHETransform` L34, `build_transforms` L57 (train: CLAHE+rotation+jitter+erasing; no horizontal flip — laterality), `_load_split` L82 with a `projections` filter (v2 trains frontal-only), `labels_to_training_target` L121 (U-Ones policy: uncertain → 1, NaN → 0), and `AlignmentDataset` L137 / `ClassificationDataset` L161 / `ReportGenDataset` L193.
- **`toy_dataset.py`** — Phase 1 smoke-test loader; still used by `scripts/sanity_check_models.py` only.

### `src/medxvlm/engine/`
- **`loop_utils.py`** (128 lines) — `load_config` L22 (YAML + CLI overrides), `save_checkpoint` L44 / `load_checkpoint` L52, `load_stage1_encoders` L56 (rebuilds both towers from a Stage 1 ckpt), `load_vision_encoder` L76 (vision tower from *any* ckpt — needed because v2 Stage 2 checkpoints have no text encoder), `EpochLogger` L92 (CSV + TensorBoard).
- **`sanity_trainer.py`** — `run_sanity_check` L39, the Phase 2 overfit-one-batch wiring test.

### `src/medxvlm/eval/metrics.py` (75 lines)
- `classification_metrics` L17 (per-class + macro AUROC/AUPRC/F1, skipping single-class columns), `retrieval_recall_at_k` L56.

### `src/medxvlm/explain/attention_maps.py` (75 lines)
- `save_grounding_snapshots` L43 — overlays `ClassifierOutput.attention_weights` as JET heatmaps; grid side derived from token count (`_overlay_heatmap` L31), so it works for both 14×14 and 16×16 towers.

### `scripts/` (2,818 lines total)
- Data: `download_dataset.py`, `prepare_data.py` (202; `_clean_report_text` strips "XXXX"), `validate_labeler.py` (187; 24 cases), `sanity_check_data.py`, `verify_env.py`.
- Training: `train_baseline.py` (123), `train_align.py` (194; `--select-by {recall1,val_loss}`, `--freeze-text-encoder`), `train_classifier.py` (227; `Stage2Model` L61 with `use_text_fusion` flag, ASL, split LR, cosine, early stopping), `train_report.py` (200; `make_balanced_sampler` L56 with `abnormal_fraction`, early stopping on val loss, `--limit-train-batches`).
- Evaluation: `eval_v2_classification.py` (221; all classifiers on the frontal test split + RadZero re-scoring), `eval_v2_report_generation.py` (101), `eval_v2_score_reports.py` (137; CE F1 + BLEU-4 + ROUGE-L + collapse diagnostics), `eval_external_chestxray14.py` (164; `LABEL_MAP` L60), and the Phase 5 originals `eval_phase5_*.py`.
- Orchestration: `run_phase3_training.sh`, `run_v2_rebuild.sh`, `run_nih_then_v2b.sh` (**do not use as-is** — it auto-launches training; superseded by the owner's request to be asked first), `run_nih_eval_only.sh`.

### `configs/`
Nine YAMLs: `{baseline,align,classifier,report_gen}.yaml` (Phase 3 recipes) and `*_v2.yaml` (current recipes), plus `report_gen_v2b.yaml` (prepared, unrun).

---

## 4. Completed features

1. **End-to-end data pipeline** — download, text cleaning, rule-based labelling with a regression suite, uid-level split, single manifest. Split verified byte-identical before and after the label fix.
2. **Stage 1 contrastive pretraining** with global InfoNCE + optional local patch-word loss; trained five times (Phase 3 ×2, v2 ×3) with per-epoch retrieval metrics.
3. **Stage 2 image-only multi-label classifier** with per-label attention pooling, ASL loss, early stopping — the project's strongest result (0.855 test AUROC, 1.3 points behind RadZero on identical images and labels).
4. **Stage 3 LoRA report generator** (LLaVA-style) trained twice; v2 breaks the mode collapse and matches CheXOne-instruct on clinical efficacy.
5. **Two supervised baselines** on the same data/encoder, giving an encoder-controlled attribution of what the pipeline adds (+5–8 points over a linear head).
6. **Local-loss ablation** (`docs/ablation_study.md`) — a near-tie at the macro level with a mixed per-class picture.
7. **External-model reproduction** — RadZero run from public weights on its own OpenI benchmark (0.852 on a 1,000-image sample) and on MedXVLM's test split (0.868, corrected labels); CheXOne run for generation on 150 images. Every paper-reported number is labelled as such.
8. **Two-direction cross-dataset evaluation** — their models on our data (Phase 5) and our model on NIH ChestX-ray14 (0.714, 2026-09-20).
9. **Explainability snapshots** from the classifier's own attention weights, saved every N epochs.
10. **Documentation set** — nine markdown docs, a sourced status inventory, and a plain-language PDF (HTML source kept in `docs/`).
11. **Version control** — initialised 2026-09-19; two commits; data/checkpoints/logs excluded by `.gitignore`.

---

## 5. In-progress / incomplete work

| Item | State | What's missing |
|---|---|---|
| Stage 3 v2b (30 % abnormal sampling) | Config and code ready (`configs/report_gen_v2b.yaml`, `abnormal_fraction` in `train_report.py` L56) | A ~4.5 h GPU run, then `eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt` and re-scoring. First attempt crashed under disk contention; owner has asked to schedule GPU use around gaming. |
| Host disk reclamation | ~100 GB freed inside WSL | `wsl --shutdown` + `diskpart … compact vdisk` on `F:\WSL\Ubuntu\ext4.vhdx` from Windows, by the owner. |
| Demo app (`app/`) | Done (2026-09-20) | Browser walkthrough by a human not yet done — verified headlessly (AppTest) and via HTTP health only. |
| Grad-CAM / rollout / text-grounded heatmaps | Not started | Design and implementation; `explain/attention_maps.py` is the only explainability code. |
| Report-generation metrics | CE F1, BLEU-4, ROUGE-L done | METEOR, RadGraph/RadCliQ (would need external model downloads). |
| Labeler | Dominant bug fixed | A residual false-positive pattern ("…no … to suggest large pleural effusion") is documented in `v2_rebuild.md` §3.4 item 6, not fixed. |
| Explainability validation | 120 PNGs on disk, 2 spot-checked qualitatively | Any quantitative grounding metric; v2 snapshots (`runs/classifier_v2/attention_maps/`) have never been inspected in a doc. |

---

## 6. Known issues / bugs

**In code (grep `TODO|FIXME|HACK|XXX` over `src/`, `scripts/`, `configs/` — two hits, no stubs):**
- `src/medxvlm/models/classifier_head.py:5` — docstring says *"No text fusion yet … is the next milestone"*. **Stale**: fusion exists at L97–L113 and was the source of the Phase 3 leak; v2 disables it via `use_text_fusion: false`.
- `src/medxvlm/models/text_encoder.py:68` — `# TODO: swap in PubMedBERT weights if the checkpoint isn't reachable`. The weights do load in this environment; the TODO guards a fallback path that has not been exercised in a real run.

**Model / data:**
1. **Stage 3 hallucination** — 7/70 frontal generations (10 %) and 40/150 overall mention devices or surgery absent from the reference; cause traced to abnormal-report oversampling (device words in 15 % of abnormal vs 4 % of normal training reports). CheXOne: 0 %.
2. **Stage 3 residual collapse** — modal normal report = 54 % of frontal outputs.
3. **Thin test support after the label fix** — on the 377-image frontal test split: Pneumothorax n=1, Pleural Other n=3, Consolidation n=5, Enlarged Cardiomediastinum n=0. Per-class AUROCs for these are not meaningful; the ≥10-positive macro (0.862) is the reliable summary.
4. **Stage 1 checkpoint selection by recall@1 is noisy** (picked the most-overfit epoch); `--select-by val_loss` was added but the shipped v2 Stage 2 was built on a frozen tower, so it is unaffected.
5. **Overfitting sets in early in every stage** (Stage 2 v2 by epoch 9, Stage 3 v2 by epoch 4) — the ~3k-image regime; early stopping handles it, more epochs will not.
6. **n-gram metrics mislead here** — BLEU-4/ROUGE-L favour the collapsed Phase 3 model (12.1/0.30 vs 9.8/0.28); reported, not used to rank.
7. **`df` inside WSL is not the real free space** (see §2 "Broken"); the NIH download filled the host drive and caused a reboot on 2026-09-20 that killed detached jobs.
8. **Stale docstring in `README.md`'s dataset section?** — not checked line-by-line; the status section was rewritten on 2026-09-19 and is current.

---

## 7. Recent activity (git log — the repository has 2 commits)

The repo was `git init`-ed on 2026-08-30 but nothing was committed until 2026-09-19, so the log does not cover the project's history; the docs' `Date:` lines do.

| Commit | Date | Summary |
|---|---|---|
| `412c3b3` | 2026-09-20 | Stage 3 v2 retrain (collapse fixed, CE F1 0.152), BLEU/ROUGE scorer, NIH ChestX-ray14 external eval (0.714), `report_gen_v2b.yaml` prepared, docs/PDF updated. |
| `25de569` | 2026-09-19 | Initial commit: everything through Phase 5 and the v2 rebuild (77 files). |

Chronology of work by document date: Phase 1–2 (2026-08-30) → Phase 3 training (09-05) → Phase 4 external reproduction and Phase 5 cross-dataset (09-06) → v2 rebuild of the labeler, Stage 1 and Stage 2 (09-17) → status inventory (09-18) → Stage 3 v2 retrain (09-19) → NIH external eval, disk incident and cleanup (09-20).

---

## 8. File / folder structure

```
medxvlm/
├── README.md                      status + doc index (current as of 2026-09-19)
├── requirements.txt / pyproject.toml
├── .gitignore                     excludes data/, checkpoints/, runs/, *.log
├── app/                           EMPTY (.gitkeep) — demo UI never started
├── configs/                       9 YAML training recipes (Phase 3 + *_v2 + report_gen_v2b)
├── data/
│   ├── raw/                       Indiana CSVs + 7,470 PNGs (14 GB, unversioned)
│   ├── processed/manifest.csv     current corrected labels; manifest_v1_phase3_labels.csv = old
│   └── raw_nih/                   download logs only (NIH images live in ~/.cache/kagglehub, test split only kept)
├── checkpoints/                   13 × best.pt, 6.6 GB (last.pt files deleted 2026-09-20)
│   ├── {align,classifier,baseline,report_gen}/        Phase 3
│   └── {align,classifier,baseline,report_gen}_v2/     current
├── runs/                          metrics.csv + TensorBoard per run; attention_maps/ PNGs; *.log
├── docs/
│   ├── status_report.md           sourced inventory (every number → file)
│   ├── v2_rebuild.md              what changed post-Phase 5 and why; §3.4 Stage 3, §3.5 NIH
│   ├── cross_dataset_evaluation.md, external_baseline_reproduction.md, external_repro_feasibility.md
│   ├── training_summary.md (superseded banner), ablation_study.md, code_review_notes.md
│   ├── literature_review.md, project_proposal.md, environment_check.md
│   ├── project_summary_plain_language.{html,pdf}
│   └── phase5_artifacts/          raw eval outputs (CSV/JSON); v2/ holds the current ones
├── scripts/                       19 scripts (see §3)
└── src/medxvlm/{data,engine,eval,explain,models}/   the package (see §3)
../external_repro/                 RadZero + CheXOne clones and their eval outputs (venvs deleted 2026-09-20)
```

---

## 9. Dependencies (pinned in `requirements.txt`; installed versions match)

| Package | Version | Used for |
|---|---|---|
| torch / torchvision | 2.11.0 / 0.26.0 (cu130) | everything; sm_120 wheels required (note in `requirements.txt` header) |
| transformers | 5.8.0 | BiomedBERT, DINOv2, Qwen2.5 decoder |
| timm | 1.0.28 | ViT-B/16 trunk (BiomedCLIP / ImageNet) |
| open_clip_torch | 3.3.0 | pulling BiomedCLIP's image tower |
| peft | 0.20.0 | LoRA for Stage 3 |
| scikit-learn | 1.8.0 | AUROC / AUPRC / F1 |
| sacrebleu / rouge-score / nltk | 2.6.0 / 0.1.2 / 3.10.2 | BLEU-4, ROUGE-L (nltk unused) |
| pandas / numpy / pillow / opencv-python-headless | 2.3.3 / 2.4.3 / 12.1.1 / 5.0.0.93 | data, CLAHE, heatmaps |
| kagglehub | 1.0.2 | dataset download |
| tensorboard | 2.21.0 | run logging |
| streamlit | 1.61.1 | pinned, **unused** (no app) |
| weasyprint / pypdf | 69.0 / 6.16.2 | installed in env, not pinned — PDF generation/verification |

External model weights (HF Hub): `StanfordAIMI/dinov2-base-xray-224` (Apache-2.0), `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, `Qwen/Qwen2.5-1.5B-Instruct`, `microsoft/BiomedCLIP-…` (Phase 3 only), `Deepnoid/RadZero` (CC BY-NC 4.0), `StanfordAIMI/CheXOne` (cache deleted 2026-09-20; re-downloadable).

---

## 10. Next steps

Ordered by how directly they follow from the code and open issues:

1. **Reclaim host disk** (owner, Windows side): `wsl --shutdown`; `diskpart` → `select vdisk file="F:\WSL\Ubuntu\ext4.vhdx"` → `attach vdisk readonly` → `compact vdisk` → `detach vdisk`. Expected to return ~100 GB to F:. Nothing GPU-related should start before this.
2. **Run Stage 3 v2b** when a ~4.5 h GPU window is confirmed: `PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/train_report.py --config configs/report_gen_v2b.yaml`, then `eval_v2_report_generation.py --checkpoint checkpoints/report_gen_v2b/best.pt --out docs/phase5_artifacts/v2/medxvlm_stage3_v2b_generations.csv` and add it to `SYSTEMS` in `eval_v2_score_reports.py`. The question it answers: does 30 % sampling cut the 10 % hallucination rate without restoring the collapse?
3. **Fix the two stale/unhelpful TODOs**: rewrite the `classifier_head.py` module docstring (L1–L7) to describe the fusion path and why v2 disables it; either exercise or remove the `text_encoder.py:68` fallback TODO.
4. ~~Decide the fate of `app/`~~ — built 2026-09-20; remaining: a human browser pass.
5. **Labeler residual pattern** — add a regression case for *"no … to suggest <finding>"* to `validate_labeler.py` and extend `SCOPE_RESET_TOKENS`/cue handling; re-run `prepare_data.py` (split is seed-stable, so it will not move).
6. **Quantitative grounding check** — the 24 v2 attention-map PNGs have never been looked at; at minimum a documented spot-check, ideally a pointing-game score against a few hand-drawn boxes.
7. **RadGraph / RadCliQ** for report generation, if the ~1 GB RadGraph model download is acceptable given disk constraints.
8. **Optional**: re-run RadZero on the NIH test split with its own prompts to turn the 0.714-vs-0.804 context into a like-for-like comparison (needs the deleted `radzero_venv` recreated per `external_baseline_reproduction.md` §2; ~1 h GPU).
