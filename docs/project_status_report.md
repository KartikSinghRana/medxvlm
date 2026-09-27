# MedXVLM — Comprehensive Project Status Report

Generated 2026-09-21 from direct inspection of the working tree at `medxvlm/`; updated the same day after the explainability evaluation. Supersedes the 2026-09-20 version of this file (git `412c3b3`). Remote: `github.com/KartikSinghRana/medxvlm` (private). Companion documents: [`status_report.md`](status_report.md) is the line-by-line sourced inventory of every number and file; [`v2_rebuild.md`](v2_rebuild.md) is the narrative of what changed after Phase 5 and why — its §3.6 covers the 2026-09-20 overnight run that changed the headline numbers below. Where this report states a number, its source is one of those two documents or a file named inline.

---

## 1. Project overview

- **Name:** MedXVLM
- **Purpose:** A three-stage vision-language system for chest X-rays — align images with radiology report text (Stage 1), classify 14 CheXpert-style findings from the image alone (Stage 2), and generate a free-text findings paragraph (Stage 3) — built and evaluated as a B.Sc. major project, with the explicit goal of honest measurement against two published external models (RadZero, CheXOne). A Streamlit demo (`app/`) serves the trained models.
- **Stack:** Python 3.11.15, PyTorch 2.11.0 (cu130, RTX 5080 / sm_120), HuggingFace `transformers` 5.8.0, `timm` 1.0.28, `open_clip_torch` 3.3.0, `peft` 0.20.0 (LoRA), `scikit-learn` 1.8.0, `pandas`, `sacrebleu` 2.6.0, `rouge-score` 0.1.2, `tensorboard`, `streamlit` 1.61.1 + `altair` 6.2.2 (demo), `weasyprint` (PDF), `kagglehub` (data). Ubuntu 26.04 under WSL2 on Windows; conda env `ml`.
- **Dataset:** Indiana University / Open-i chest X-ray collection (public, 3,826 usable studies / 7,426 images after cleaning), plus the NIH ChestX-ray14 official test split (25,596 images) used only for an external-generalisation test.

---

## 2. Current state — working vs. incomplete

**Working right now (trained, evaluated, checkpoint on disk):**

| Component | Checkpoint | Honest headline number |
|---|---|---|
| Stage 2 image-only classifier (v2), single model | `checkpoints/classifier_v2/{local_on,seed1,seed2,seed3}/best.pt` | **0.837 ± 0.015 macro AUROC** (mean ± std over 4 seeds: 0.855 / 0.833 / 0.839 / 0.819) on the held-out frontal test split (377 images, 8 classes shared with RadZero); 0.804 ± 0.019 over all 13 measurable classes. Source: `docs/phase5_artifacts/v2/classifier_seed_ensemble.json`. The 0.855 quoted in earlier documents is seed 0, the best of the four. |
| Stage 2 encoder comparison (RQ1, 2026-09-26) | `checkpoints/classifier_v2/rq1_{biomedclip,clip_openai,imagenet_vit}_s0/best.pt` | Same recipe on four encoders: X-ray DINOv2 0.837 ± 0.015 (4 seeds) > BiomedCLIP 0.802 > OpenAI CLIP 0.792 > ImageNet ViT 0.784 (1 seed each); same order on NIH (0.704 → 0.633) and on the pointing game (32% → 13%). BiomedCLIP vs CLIP not significant. `v2_rebuild.md` §3.9. |
| Stage 2, 4-seed probability ensemble | the four checkpoints above, averaged (`src/medxvlm/inference.py` `DEFAULT_CLASSIFIER_ENSEMBLE`) | **0.868 macro AUROC** (8 shared) / 0.867 (5 shared classes with ≥10 positives) / 0.835 (all 13). **RadZero on identical images and labels: 0.868** (0.8684 vs 0.8682). This is what the demo app serves. |
| Same classifier on an unseen hospital (NIH ChestX-ray14 test split, 8 mapped classes) | same | single seed **0.704 ± 0.009** (seed 0: 0.714); **ensemble 0.719** — the generalisation number. |
| Stage 3 report generator (v2) with beam-3 + repetition-penalty 1.2 decoding — **app default** | `checkpoints/report_gen_v2/best.pt` (LoRA + projector, 88 MB); decoding defaults in `inference.py` `DEFAULT_DECODING` | Clinical-efficacy macro F1 **0.176** on 70 frontal test images (CheXOne-instruct: 0.149; same checkpoint with greedy decoding: 0.152); device/surgery hallucination **3 %** of frontal cases (greedy: 10 %); 14 distinct outputs / 70 frontal, modal "normal" report 61 %. Source: `report_generation_scores_v2_overnight.json`, `v2_rebuild.md` §3.6.2. |
| Stage 3 v2c, findings-conditioned (2026-09-21) — selectable in the app | `checkpoints/report_gen_v2c/best.pt` (epoch 1, val-selected by generation CE F1) | CE macro F1 **0.165** greedy (beam 0.135), micro 0.610, **0 % hallucination**; the classifier's flagged findings verbalised as a bare list score 0.234 on the same 70 images (the ceiling; only 18/70 flagged); remove the findings from the prompt and it collapses to 1 output. Bootstrap CIs ±0.10 → not distinguishable from v2. `v2_rebuild.md` §3.7. |
| Stage 1 zero-shot prompt classifier (v3b, finding-sentence alignment) | `checkpoints/align_v3/findings/best.pt` | **0.734 macro AUROC** (8 shared, positive-prompt scoring); 0.693 on the ≥10-positive classes; 0.652 all 13. Up from 0.565 (v2 report-level) and 0.568 (v3 all-sentence). RadZero zero-shot: 0.868. Source: `v2_classification_summary.json` key `v3b_stage1_findings_zero_shot`. |
| Stage 1 contrastive alignment (v2) | `checkpoints/align_v2/local_on/best.pt` | Encoder initialiser for the v2 classifier; val loss 4.67 at epoch 7; retrieval R@1 ≈ 0.04–0.06; 0.565 zero-shot (not usable as a classifier). |
| Rule-based CheXpert labeler | `src/medxvlm/data/labeler.py` | 24/24 validation cases pass (`scripts/validate_labeler.py`); negation-scope bug fixed 2026-09-17. |
| Two baselines (v2) | `checkpoints/baseline_v2/{vit_base_patch16_224,xraydinov2}/best.pt` | 0.804 / 0.777 macro AUROC on the same test split. |
| Demo app | `app/app.py` over `src/medxvlm/inference.py` | Upload or pick a test image → ensemble probabilities (thin-support classes greyed), heatmaps in three methods each labelled with its measured accuracy, optional report generation (lazy-loads ~7 GB; beam decoding default, device-mention warning flag) with per-sentence attribution, model card. `scripts/sanity_check_app.py` (headless `streamlit.testing`) passes 5/5. Run: `conda activate ml && streamlit run app/app.py`. |
| Explainability — heatmaps, **validated** | `src/medxvlm/explain/saliency.py`; results `docs/phase5_artifacts/v2/grounding_chestxray14_summary.json` | Pointing game on **984 NIH radiologist boxes**: Stage 2 attention pooling (ensemble) **41.2 % [38.1, 44.3]** vs random point 7.5 % / image centre 21.5 %; Grad-CAM 19.0 %; class-agnostic rollout 6.4 %; text-grounded v3b 11.3 %; Phase 3 local-loss on/off 5.3 / 3.8 % (RQ2: no gain). Write-up `docs/explainability_evaluation.md`. |
| Explainability — sentence attribution | `src/medxvlm/explain/sentence_attribution.py`; `sentence_attribution_summary.json` | Per-sentence visual attention of the Stage 3 generator; see `docs/explainability_evaluation.md` §3 for the measured shares. |

**Negative results, closed (documented, not to be re-run as-is):**

- **Stage 3 v2b (30 % abnormal sampling instead of 50 %)** — trained 2026-09-20 (131 min, best epoch 2, val loss 1.108 < v2's 1.132) and **collapsed completely**: one identical sentence for all 150 test images, CE macro F1 0.057. Lessons recorded in `v2_rebuild.md` §3.6.4: the 50 % oversampling is load-bearing, and teacher-forced val NLL selects the collapsed model, so it is the wrong checkpoint-selection signal for Stage 3. Checkpoint kept at `checkpoints/report_gen_v2b/best.pt` for reference only.
- **Stage 1 v3 (all-sentence multi-positive InfoNCE)** — 0.568 zero-shot; cause measured: only 1,683 of 16,375 training sentences (10.3 %) contain a finding. Fixed by v3b (finding sentences only, above).
- **Positive-minus-negative prompt scoring** — helps v2 (0.565 → 0.604) but hurts v3b (0.734 → 0.605) and Phase 3 (0.669 → 0.642); not a reliable gain, so pos-only is the reported mode.

**Incomplete / not started / superseded:**

- **Stage 3 is now bounded by the classifier, not the decoder:** the findings-conditioned v2c removes hallucination (0 %) but cannot exceed what the classifier's own findings score (0.234 on the 70-image subsample, where only 18 images get any flag). Further gains need classifier recall on rare findings (data), which this dataset cannot supply. The v2 default still has 61 % modal share and 3 % hallucination.
- **Zero-shot gap to RadZero (0.734 vs 0.868)** is assessed as a data-scale limit (900 finding-bearing training images vs RadZero's ~100×), not something further recipe changes will close; the project's classification claim is the supervised head/ensemble.
- **Phase 3 numbers are superseded.** The 0.865 Stage 2 AUROC in `training_summary.md` was measured with the ground-truth report fused into the classifier; image-only it scores 0.507. The file carries a banner saying so.
- **ASL-vs-BCE ablation on the baseline:** never run (`docs/ablation_study.md` §"Second ablation").
- **METEOR / RadGraph** for report generation: not computed. BLEU-4 and ROUGE-L are computed but documented as misleading here (they favour collapsed outputs).
- **Human browser pass of the demo app:** not yet done; verified only headlessly (5 AppTest checks).
- **In-distribution grounding:** Indiana has no box annotations, so all localisation numbers are out-of-distribution (NIH); no reader study.

**Environmental:**

- **Host disk is healthy again.** The F: drive (466 GB) hit 542 MB free on 2026-09-20 and forced a reboot; ~100 GB was freed inside WSL and the owner compacted the `.vhdx` (193 → 88 GB). `df /mnt/f` now reports 100 GB free; the WSL ext4 image uses 90 GB. `df /` inside WSL still reports a nominal 1 TB and is not the real limit.
- All `last.pt` files were deleted (2026-09-21); `checkpoints/` is 9.2 GB of `best.pt` only.
- **2026-09-27 local cleanup:** deleted the negative-result checkpoints `align_v2/local_on_valloss`, `align_v2/local_on_frozen_text`, `align_v3/sentence`, `report_gen_v2b/` and `report_gen_v2c/epoch3.pt`, plus unused model caches (Qwen2.5-0.5B, rad-dino, CXR-BERT, RRG_scorers) and the RadZero weights. Their results remain in `docs/phase5_artifacts/v2/`; re-running those specific evaluations (or RadZero) would need retraining or re-downloading. Everything the demo app uses is kept.
- No GPU job is running; any further training requires the owner's go-ahead with a duration (standing constraint since 2026-09-20).

---

## 3. Architecture summary

Data flow: `indiana_reports.csv` + `indiana_projections.csv` → `scripts/prepare_data.py` (clean text, run labeler, uid-level 80/10/10 split) → `data/processed/manifest.csv` → one `Dataset` class per stage → `scripts/train_*.py` → `checkpoints/*/best.pt` + `runs/*/metrics.csv` → `scripts/eval_*.py` → `docs/phase5_artifacts/**`. Serving: `src/medxvlm/inference.py` → `app/app.py`.

### `src/medxvlm/models/`
- **`labels.py`** — `CHEXPERT_LABELS` (14-tuple) and `NUM_CHEXPERT_LABELS`; the single source of truth for label order, imported by every head.
- **`vision_encoder.py`** (152 lines) — `VisionEncoder(pretrained, backbone)` at L82. Two backbones: `"biomedclip"` (timm ViT-B/16 trunk from BiomedCLIP, `_load_biomedclip_trunk` L52, ImageNet fallback L62) and `"xraydinov2"` (`StanfordAIMI/dinov2-base-xray-224` via HF `Dinov2Model`, `_load_xraydinov2` L68). `forward` (L121) returns `VisionEncoderOutput(patch_tokens, pooled)`. `infer_backbone_from_state_dict` (L140) sniffs checkpoint key prefixes so pre-v2 checkpoints load without a stored backbone name.
- **`text_encoder.py`** (97 lines) — `TextEncoder` at L40 wraps `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` with `bert-base-uncased` fallback; `tokenize` L81, `forward` L90 returns per-token embeddings + pooler output.
- **`alignment.py`** (193 lines) — `ContrastiveAlignment` L41: projects both towers to a 512-d space (`encode_image` L70, `encode_text` L74); `forward` L78 takes an optional `positive_mask` — `None` gives the original symmetric InfoNCE, a `[B,B]` bool mask routes to `_multi_positive_infonce_loss` L123 (sum over all positives in the softmax numerator, diagonal forced positive; used by the v3/v3b sentence-level recipes). Optional FILIP-style `local_patch_word_loss` L149 (the Phase 3 ablation switch).
- **`classifier_head.py`** (116 lines) — `PerLabelAttentionPooling` L36 (one learned query per label over patch tokens) → `ClassifierHead` L68 with a shared `Linear(D,1)`. Optional gated text fusion (`text_proj`/`gate`, L97–L101) is applied only when `forward(text_embeds=…)` L103 receives text. **v2 never passes text**; Phase 3 passed the ground-truth report (the leak).
- **`baseline_classifier.py`** (99 lines) — `BaselineClassifier` L28 (backbone + linear head; supports `"xraydinov2"` via `VisionEncoder`), `multilabel_bce_loss` L68, `asymmetric_loss` L73 (ASL, used by v2 Stage 2).
- **`report_generator.py`** (222 lines) — `ReportGenerator` L40: frozen `VisionEncoder` → adaptive-avg-pool of the patch grid to `pooled_grid_size²` tokens → 2-layer MLP projector → spliced at a literal `<image>` token into `Qwen/Qwen2.5-1.5B-Instruct` (L22) with LoRA on q/k/v/o/gate/up/down (L25). `forward` L182 (teacher-forced, loss masked to answer tokens), `generate` L189 now accepts `**generate_kwargs` (beam search, repetition penalty) — greedy is no longer the only path.

### `src/medxvlm/data/`
- **`labeler.py`** (326 lines) — rule-based CheXpert-style labeler. Cue tables at L38–L84 (`NEGATION_PRE_WINDOW = 12`, `SCOPE_RESET_TOKENS` L48, `PRE_NEGATION_CUES` L55, `LABEL_PHRASES` L84). `split_report_sentences` L171 (strips enumeration prefixes, dedups, `min_words=3`) feeds the sentence-level Stage 1 datasets. `_pre_cue_in_scope` L236 is the 2026-09-17 fix that lets a leading "no" govern a whole coordinated list until a scope-reset token. `find_mentions` L292, `aggregate_report` L312 (positive-wins aggregation; derives "No Finding").
- **`dataset.py`** (270 lines) — `CLAHETransform` L35, `build_transforms` L58 (train: CLAHE+rotation+jitter+erasing; no horizontal flip — laterality), `_load_split` L83 with a `projections` filter (v2 trains frontal-only), `labels_to_training_target` L122 (U-Ones policy: uncertain → 1, NaN → 0), `AlignmentDataset` L138 (one report per image), `SentenceAlignmentDataset` L162 (one random sentence per image per epoch, worker-safe via `torch.randint`; `findings_only=True` keeps only sentences the labeler marks positive/uncertain and substitutes a canonical normal sentence otherwise — the v3b recipe), `ClassificationDataset` L215, `ReportGenDataset` L247.
- **`toy_dataset.py`** — Phase 1 smoke-test loader; still used by `scripts/sanity_check_models.py` only.

### `src/medxvlm/engine/`
- **`loop_utils.py`** (128 lines) — `load_config` L22 (YAML + CLI overrides), `set_seed` L37, `save_checkpoint` L44 / `load_checkpoint` L52, `load_stage1_encoders` L56 (rebuilds both towers from a Stage 1 ckpt), `load_vision_encoder` L76 (vision tower from *any* ckpt — needed because v2 Stage 2 checkpoints have no text encoder), `EpochLogger` L92 (CSV + TensorBoard).
- **`sanity_trainer.py`** — `run_sanity_check` L39, the Phase 2 overfit-one-batch wiring test.

### `src/medxvlm/eval/metrics.py` (75 lines)
- `classification_metrics` L17 (per-class + macro AUROC/AUPRC/F1, skipping single-class columns), `retrieval_recall_at_k` L56.

### `src/medxvlm/explain/`
- **`attention_maps.py`** (85 lines) — `overlay_attention` L43 (single-image overlay used everywhere), `save_grounding_snapshots` L53 (training-time snapshots); grid side derived from token count, so both 14×14 and 16×16 towers work.
- **`saliency.py`** (added 2026-09-21) — `attention_pooling_map`, `gradcam_map` (Grad-CAM on the final patch tokens via `torch.autograd.grad`), `attention_rollout` (DINOv2 only; switches the backbone to eager attention), `TextGrounder` / `text_grounded_map` (patch–prompt cosine in the Stage 1 space), `upsample_map`, `minmax`.
- **`sentence_attribution.py`** (added 2026-09-21) — `attribute_sentences(model, pixel_values, instruction, report)`: teacher-forced decoder pass with `output_attentions=True`, per-sentence mean attention onto the 64 visual-token positions, raw and sink-corrected visual shares.

### `src/medxvlm/inference.py` (added 2026-09-20, extended 2026-09-21)
- `MedXVLMPipeline`: loads every present member of `DEFAULT_CLASSIFIER_ENSEMBLE` (missing members are skipped and listed in `classifier_meta["skipped"]`); `classify` averages sigmoid probabilities and attention maps across members (`ClassificationResult.heatmap(label)`); `heatmaps(image, labels, method)` for `HEATMAP_METHODS = ("attention", "gradcam", "text")` — the text method lazily loads `DEFAULT_TEXT_GROUNDING_CHECKPOINT` (`align_v3/findings`); `load_report_generator` lazily builds Stage 3; `generate_report` applies `DEFAULT_DECODING = {"num_beams": 3, "repetition_penalty": 1.2}` and attaches honesty flags; `attribute_report(image, text)` returns per-sentence `SentenceExplanation(sentence, visual_share, overlay)`. The decoder is switched between sdpa (decoding) and eager (attribution) attention as needed.

### `app/app.py` (added 2026-09-20, extended 2026-09-21)
- Thin Streamlit view over the pipeline: radio-button input source (upload vs held-out test image with its reference report and true labels), Altair probability bar chart with thin-support classes greyed (Pneumothorax n=1, Pleural Other n=3, Consolidation n=5, Enlarged Cardiomediastinum n=0), heatmap-method radio whose option labels carry the measured pointing-game accuracy read from `grounding_chestxray14_summary.json`, optional report generation with a decoding radio (beam default), the hallucination warning and a sentence-attribution expander, model card with the numbers in §2 (now including grounding and attribution).

### `scripts/` (3,277 lines total)
- Data: `download_dataset.py`, `prepare_data.py` (202; `_clean_report_text` strips "XXXX"), `validate_labeler.py` (187; 24 cases), `sanity_check_data.py`, `verify_env.py`.
- Training: `train_baseline.py` (123), `train_align.py` (225; `text_unit: report|sentence` L142, `findings_only` L148, `make_collate(..., multi_positive)` L54 builds the positive mask from normalised sentence text, `--select-by {recall1,val_loss}` L43, `--freeze-text-encoder` L46, `--run-name`, early stopping), `train_classifier.py` (228; `Stage2Model` L52 with `use_text_fusion` flag, ASL, split LR, cosine, early stopping, `--seed`, `--run-name`), `train_report.py` (200; `make_balanced_sampler` L56 with `abnormal_fraction`, early stopping on val loss, `--limit-train-batches`).
- Evaluation: `eval_grounding_chestxray14.py` (pointing game / mass ratio / area-matched IoU of every saliency method on the 984 NIH boxes, with random and centre baselines), `eval_sentence_attribution.py` (visual-attention share of generated sentences on the 70 frontal test images; box-matching test on a stratified NIH sample), `eval_v2_classification.py` (273; all classifiers on the frontal test split + RadZero re-scoring; `load_zero_shot(path, device, mode)` L105 with `POS_TEMPLATES`/`NEG_TEMPLATES` L72–L73; systems include v3/v3b), `eval_classifier_ensemble.py` (138; per-seed spread + probability ensemble, `--nih` for the external split), `eval_v2_report_generation.py` (109; `--num-beams --repetition-penalty --no-repeat-ngram-size --limit --out`), `eval_v2_score_reports.py` (146; CE F1 + BLEU-4 + ROUGE-L + collapse diagnostics; `--extra NAME=PATH`), `eval_external_chestxray14.py` (164; `LABEL_MAP` L60), and the Phase 5 originals `eval_phase5_*.py`.
- Demo check: `sanity_check_app.py` (five `AppTest` paths — cold start, upload, test-image, heatmap methods, report generation + sentence attribution).
- Orchestration (bash, all serialised one-GPU-job-at-a-time): `scripts/archive/run_phase3_training.sh`, `run_v2_rebuild.sh`, `scripts/archive/run_nih_eval_only.sh`, `scripts/archive/run_overnight_2026-09-20.sh` + `_part2.sh` (seeds, ensemble eval, decoding sweep, v3), `scripts/archive/run_v2b_then_score.sh` (unsets `PYTORCH_CUDA_ALLOC_CONF`), `scripts/archive/run_v3b_after_v2b.sh`. `scripts/archive/run_nih_then_v2b.sh` is historical and now exits immediately with an explanation (it used to auto-launch training).

### `configs/`
Eleven YAMLs: `{baseline,align,classifier,report_gen}.yaml` (Phase 3 recipes), `*_v2.yaml` (current supervised recipes), `report_gen_v2b.yaml` (30 % sampling — collapsed, keep for the record), `align_v3_sentence.yaml` (all-sentence — negative), `align_v3b_findings.yaml` (finding-sentence — the 0.734 zero-shot head).

---

## 4. Completed features

1. **End-to-end data pipeline** — download, text cleaning, rule-based labelling with a regression suite, uid-level split, single manifest. Split verified byte-identical before and after the label fix.
2. **Stage 1 contrastive pretraining** — report-level InfoNCE with optional local patch-word loss (Phase 3 ×2, v2 ×3) and sentence-level multi-positive InfoNCE (v3, v3b); per-epoch retrieval metrics.
3. **Stage 2 image-only multi-label classifier** with per-label attention pooling, ASL loss, early stopping — trained with four seeds; **0.837 ± 0.015 single model, 0.868 as a 4-seed ensemble, equal to RadZero on identical images and labels**.
4. **Stage 3 LoRA report generator** (LLaVA-style) trained three times; v2 breaks the Phase 3 mode collapse, and with beam-3 decoding scores CE macro F1 0.176 vs CheXOne-instruct 0.149 on the same images at 3 % device hallucination.
5. **Decoding sweep** — established on the same checkpoint that beam-3 + repetition penalty 1.2 cuts hallucination 10 % → 3 % and lifts CE F1 0.152 → 0.176; now the default everywhere.
6. **Zero-shot prompt classifier** — from 0.565 (report-level) to 0.734 by aligning only finding-bearing sentences; the failure of the intermediate all-sentence recipe was diagnosed quantitatively (10.3 % of sentences carry a finding).
7. **Two supervised baselines** on the same data/encoder, giving an encoder-controlled attribution of what the pipeline adds (+5–8 points over a linear head).
8. **Local-loss ablation** (`docs/ablation_study.md`) — a near-tie at the macro level with a mixed per-class picture.
9. **External-model reproduction** — RadZero run from public weights on its own OpenI benchmark (0.852 on a 1,000-image sample) and on MedXVLM's test split (0.868, corrected labels); CheXOne run for generation on 150 images. Every paper-reported number is labelled as such.
10. **Two-direction cross-dataset evaluation** — their models on our data (Phase 5) and our model on NIH ChestX-ray14 (0.704 ± 0.009 single, 0.719 ensemble).
11. **Seed-variance reporting and confidence intervals** — every headline classification number carries a mean ± std over four seeds, and a paired bootstrap gives 95% CIs and paired tests (`v2_rebuild.md` §3.8): ensemble vs RadZero Δ 0.000 [−0.033, +0.033].
11b. **RQ1 answered** — the classifier retrained on BiomedCLIP, OpenAI CLIP and ImageNet ViT encoders (`v2_rebuild.md` §3.9); all four research questions now have measured answers.
12. **Explainability, implemented and validated** — attention pooling, Grad-CAM, attention rollout, text-grounded heatmaps and Stage 3 sentence attribution; scored against 984 NIH radiologist boxes (attention ensemble 41 % pointing game vs 7.5 % random) — the proposal's Week-10/13 items and the first measured answers to RQ2 and RQ4.
13. **Streamlit demo app** with a reusable inference module and a headless test harness (proposal Week 14 deliverable).
14. **Documentation set** — thirteen markdown docs (incl. `explainability_evaluation.md`), a sourced status inventory, and a plain-language PDF regenerated 2026-09-20 from `docs/project_summary_plain_language.html`.
15. **Version control** — pushed to GitHub (`KartikSinghRana/medxvlm`, private, 2026-09-21); data/checkpoints/logs excluded by `.gitignore`.

---

## 5. In-progress / incomplete work

| Item | State | What's missing |
|---|---|---|
| Stage 3 quality beyond 0.176 | Closed: v2c (findings-conditioned, generation-based selection) run 2026-09-21 — 0 % hallucination, CE F1 0.165 vs 0.176, ceiling 0.234 | Nothing decoder-side left that this data can resolve; the metric is capped by classifier recall on rare findings. |
| Demo app (`app/`) | Done and committed (`39c71e9`, `1b1e168`) | A human browser walkthrough — verified only via `AppTest` and HTTP health. |
| Report-generation metrics | CE F1, BLEU-4, ROUGE-L done | METEOR, RadGraph/RadCliQ (external model downloads; disk is no longer the blocker). |
| Labeler | Dominant bug fixed | A residual false-positive pattern ("…no … to suggest large pleural effusion") is documented in `v2_rebuild.md` §3.4 item 6, not fixed. |
| Explainability validation | Done out-of-distribution (984 NIH boxes) | In-distribution boxes do not exist for Indiana; a reader study would be the next level of evidence. |
| Disk hygiene | F: at 100 GB free after compaction; stray `last.pt` files deleted | — |

---

## 6. Known issues / bugs

**In code (grep `TODO|FIXME|HACK` over `src/`, `scripts/`, `app/`, `configs/` — no hits as of 2026-09-21; the two stale docstrings were rewritten).**

**Model / data:**
1. **Stage 3 residual collapse** — modal normal report = 61 % of frontal outputs with the default beam decoding (54 % greedy); 14 distinct outputs / 70 frontal. Recall 0 on Lung Lesion, Pneumonia, Fracture, Pleural Other in the 150-image test subsample.
2. **Stage 3 hallucination** — 3 % of frontal generations mention devices/surgery absent from the reference with beam decoding (10 % greedy; CheXOne 0 %). Cause: abnormal-report oversampling (device words in 15 % of abnormal vs 4 % of normal training reports) — but removing the oversampling (v2b) collapses the model entirely, so the trade-off is real.
3. **Validation NLL is the wrong selection signal for Stage 3** — the fully collapsed v2b had *lower* val loss (1.108) than v2 (1.132). `train_report.py` still early-stops and selects on val loss; any future Stage 3 run should select on a generation-based criterion.
4. **Seed variance is ±1.5 AUROC points** for the v2 classifier recipe; single-seed comparisons at the 1–2 point level are not meaningful. Always quote mean ± std and the ensemble.
5. **Thin test support after the label fix** — on the 377-image frontal test split: Pneumothorax n=1, Pleural Other n=3, Consolidation n=5, Enlarged Cardiomediastinum n=0. Per-class AUROCs for these are not meaningful; the ≥10-positive macro (0.855 ± 0.005 single / 0.867 ensemble) is the reliable summary. The app greys these classes out.
6. **Zero-shot scoring must match the training text distribution** — pos-minus-neg prompts help the report-level v2 head but hurt v3b (0.734 → 0.605), which never saw negated sentences. Reported mode is pos-only.
7. **Stage 1 checkpoint selection by recall@1 is noisy** (picked the most-overfit epoch); `--select-by val_loss` exists; the shipped v2 Stage 2 was built on a frozen tower and is unaffected.
8. **Overfitting sets in early in every stage** (Stage 2 v2 by epoch 9, Stage 3 v2 by epoch 4, v3b by epoch 5) — the ~3k-image regime; early stopping handles it, more epochs will not.
9. **Heatmaps are only as good as the classifier**: pointing-game accuracy is at chance for Pneumothorax (4 %) and near it for Nodule (8 %), matching those classes' weak AUROCs; Grad-CAM is markedly worse than the attention-pooling weights on this architecture (19 % vs 41 %); the Stage 1 patch–prompt maps (incl. the local-loss ablation pair) are near or below a random point.
10. **The report writer barely looks at the image** — see `docs/explainability_evaluation.md` §3 (sentence attribution).
11. **n-gram metrics mislead here** — BLEU-4/ROUGE-L favour the collapsed Phase 3 model (12.1/0.30 vs 8.7/0.26 for v2 beam); reported, not used to rank.

**Environment / ops (all documented in `v2_rebuild.md` §3.6.4 and the status inventory):**
12. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks near-VRAM-cap runs under WSL2** (`CUDA driver error: device not ready` at the first backward). Stage 3 must run without it; `scripts/archive/run_v2b_then_score.sh` unsets it.
13. **`df` inside WSL is not the real free space** — the 2026-09-20 NIH download filled the 466 GB host drive and forced a reboot that killed detached jobs. Check `df /mnt/f` before large downloads.
14. **Stage 3 epoch time is ~33–56 min** at batch 4 fp32 depending on the sampler; short smoke tests under-estimate it by ~4×.

---

## 7. Recent activity (git log — six commits)

The repo was `git init`-ed on 2026-08-30 but nothing was committed until 2026-09-19, so the log does not cover the project's early history; the docs' `Date:` lines do.

| Commit | Date | Summary |
|---|---|---|
| `1b1e168` | 2026-09-20 | Overnight results: v3b zero-shot 0.734, v2b collapse documented, app defaults switched to the 4-seed ensemble + beam decoding, README/status/PDF updated. |
| `abbceab` | 2026-09-20 | Overnight results part 1: seed ensemble (0.837 ± 0.015 / 0.868), decoding sweep (0.176, 3 %), v3 zero-shot negative result. |
| `4602f98` | 2026-09-20 | Sentence-level Stage 1 (multi-positive InfoNCE, `SentenceAlignmentDataset`), seed/ensemble tooling, decoding options in the generator and eval scripts. |
| `39c71e9` | 2026-09-20 | Streamlit demo app + reusable `inference.py` + `sanity_check_app.py`. |
| `412c3b3` | 2026-09-20 | Stage 3 v2 retrain (collapse fixed, CE F1 0.152), BLEU/ROUGE scorer, NIH ChestX-ray14 external eval (0.714), `report_gen_v2b.yaml` prepared. |
| `25de569` | 2026-09-19 | Initial commit: everything through Phase 5 and the v2 rebuild (77 files). |

Chronology of work by document date: Phase 1–2 (2026-08-30) → Phase 3 training (09-05) → Phase 4 external reproduction and Phase 5 cross-dataset (09-06) → v2 rebuild of the labeler, Stage 1 and Stage 2 (09-17) → status inventory (09-18) → Stage 3 v2 retrain (09-19) → NIH external eval, disk incident and cleanup, demo app, overnight seed/decoding/zero-shot/v2b runs (09-20).

---

## 8. File / folder structure

```
medxvlm/
├── README.md                      status + doc index (headline numbers current as of 2026-09-20)
├── requirements.txt / pyproject.toml
├── .gitignore                     excludes data/, checkpoints/, runs/, *.log
├── app/app.py                     Streamlit demo (2026-09-20)
├── configs/                       11 YAML recipes (Phase 3, *_v2, report_gen_v2b, align_v3_sentence, align_v3b_findings)
├── data/
│   ├── raw/                       Indiana CSVs + 7,470 PNGs (14 GB, unversioned)
│   ├── processed/manifest.csv     current corrected labels; manifest_v1_phase3_labels.csv = old
│   └── raw_nih/                   download logs only (NIH test images live in ~/.cache/kagglehub, ~10 GB)
├── checkpoints/                   19 × best.pt, 9.2 GB
│   ├── {align,classifier,baseline,report_gen}/        Phase 3
│   ├── {align,classifier,baseline,report_gen}_v2/     current; classifier_v2/{local_on,seed1,seed2,seed3} = the ensemble
│   ├── align_v3/{sentence,findings}/                  zero-shot attempts; findings = 0.734
│   └── report_gen_v2b/                                collapsed (reference only)
├── runs/                          metrics.csv + TensorBoard per run; attention_maps/ PNGs (216); *.log incl. overnight chain logs
├── docs/
│   ├── project_status_report.md   this file
│   ├── explainability_evaluation.md   heatmaps vs 984 NIH boxes; sentence attribution (2026-09-21)
│   ├── status_report.md           sourced inventory (every number → file)
│   ├── v2_rebuild.md              what changed post-Phase 5 and why; §3.4 Stage 3, §3.5 NIH, §3.6 overnight
│   ├── cross_dataset_evaluation.md, external_baseline_reproduction.md, external_repro_feasibility.md
│   ├── training_summary.md (superseded banner), ablation_study.md, code_review_notes.md
│   ├── literature_review.md, project_proposal.md, environment_check.md
│   ├── project_summary_plain_language.{html,pdf}   9-page plain-language summary (PDF regenerated 2026-09-20)
│   └── phase5_artifacts/          raw eval outputs; v2/ holds the current ones (classification summary + raw,
│                                  classifier_seed_ensemble.json, report scores, 4 generation CSVs, NIH summary + raw,
│                                  grounding_chestxray14_{summary,raw} + grounding_examples/, sentence_attribution_* + examples/)
├── scripts/                       28 scripts (see §3)
└── src/medxvlm/{data,engine,eval,explain,models}/ + inference.py   the package (see §3; explain/ now has saliency + sentence_attribution)
../external_repro/                 RadZero + CheXOne clones and their eval outputs (venvs deleted 2026-09-20)
```

---

## 9. Dependencies (pinned in `requirements.txt` unless noted; installed versions match)

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
| streamlit | 1.61.1 | demo app + `streamlit.testing` harness |
| altair | 6.2.2 | app probability chart (pinned 2026-09-21) |
| weasyprint / pypdf | 69.0 / 6.16.2 | installed in env, not pinned — PDF generation/verification |

External model weights (HF Hub): `StanfordAIMI/dinov2-base-xray-224` (Apache-2.0), `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, `Qwen/Qwen2.5-1.5B-Instruct`, `microsoft/BiomedCLIP-…` (Phase 3 only), `Deepnoid/RadZero` (CC BY-NC 4.0), `StanfordAIMI/CheXOne` (cache deleted 2026-09-20; re-downloadable).

---

## 10. Next steps

What remains, in order. Nothing here needs GPU training.

1. **Final report and presentation** (Week 16 of the proposal) — the write-up now has every number it needs: all four research questions answered (RQ1 §3.9, RQ2/RQ4 `explainability_evaluation.md`, RQ3 §3.7), confidence intervals (§3.8), the explainability chapter and the v2c result. No GPU work remains on the critical path.
1b. **Optional**: second seeds for the BiomedCLIP and OpenAI CLIP encoders (~1.7 h GPU) would tighten the one unresolved RQ1 contrast (BiomedCLIP − CLIP +0.010, CI −0.025 to +0.086).
2. **Human browser pass of the app** — all six paths (incl. the v2c report model) are verified headlessly only.
3. **Labeler residual pattern** — add a regression case for *"no … to suggest <finding>"* to `validate_labeler.py` and extend `SCOPE_RESET_TOKENS`/cue handling; re-run `prepare_data.py` (split is seed-stable, so it will not move).
4. **Optional**: re-run RadZero on the NIH test split with its own prompts to turn the 0.719-vs-0.804 context into a like-for-like comparison (needs the deleted `radzero_venv` recreated per `external_baseline_reproduction.md` §2; ~1 h GPU inference). RadZero's own similarity maps could also be scored on the same 984 boxes with `eval_grounding_chestxray14.py`'s metrics for a like-for-like grounding comparison.
5. **RadGraph / RadCliQ** for report generation (~1 GB model download).
6. **Ops**: Stage 3 training on this machine should always use batch 2 × grad-accum 8 — the near-cap batch-4 runs were being paged by the WSL2 GPU layer (epoch 70 min vs 10.5 min, `v2_rebuild.md` §3.7).
