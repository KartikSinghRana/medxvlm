# MedXVLM: complete briefing for writing the final project report

*Prepared 2026-09-26 from the project repository (`github.com/KartikSinghRana/medxvlm`, private, commit `ea45135` and later). Every number below is copied from the project's own result files or documentation, with its source named. This document is the only input you need.*

---

## 0. Read this first: instructions for the report writer

**Who this is for.** You are an AI assistant (Claude) helping **Kartik Singh Rana**, a B.Sc. student in AI & Data Science, write the final report and a summary for his major project, **MedXVLM**. You do not have access to the code or the result files, so this document gives you everything, including every comparison, every number, and the caveats attached to each one.

**What to produce (unless Kartik asks otherwise):**
1. A **full project report** in the usual B.Sc. major-project structure. A suggested outline, mapped section by section to this document, is in §17.
2. A **1–2 page executive summary** for a non-specialist reader.
3. Optionally, a **presentation outline** of 12–15 slides (see §17.3).

**Ground rules. These matter more than style.**
1. **Never invent a number.** Use only the figures in this document. If a number the report seems to need is not here, say it was not measured. Do not estimate it.
2. **Keep the comparability labels.** Every comparison below is marked as one of:
   - **apples-to-apples**: same images, same labels, same metric;
   - **approximate**: same task and metric, but different data or protocol;
   - **reported by the authors, not reproduced here**: a number from a published paper.
   Never put a paper-reported number in the same column as a measured one without saying so.
3. **Quote the honest headline numbers** (§2). The project went through several corrections, and some older numbers look better but are wrong or superseded. §16 lists the ones **not to use**.
4. **Negative results are part of the contribution, not something to hide.** The project's strongest quality is that it caught its own errors: a data leak, a labeling bug, output collapse, and a failed zero-shot attempt. It diagnosed each one, measured the cause and fixed it. The report should tell that story plainly (§6).
5. **No clinical claims.** This is a research prototype trained on about 3,000 X-rays from one hospital. It is not a medical device and must not be described as ready for clinical use.
6. **Use statistical language correctly.** Where a confidence interval (CI) includes zero, say "not statistically significant" or "indistinguishable". Do not say "better". Say "within noise" for differences smaller than the stated CI width.
7. Leave `[University]`, `[Supervisor]`, `[Department]` and similar as placeholders for Kartik to fill in.

**Conventions in this document.**
- **AUROC**: area under the ROC curve (0.5 = chance, 1.0 = perfect).
- **Macro AUROC**: the average of per-class AUROCs.
- "**8 shared classes**": the 8 findings that both MedXVLM and RadZero predict: Atelectasis, Cardiomegaly, Pleural Effusion, Pneumonia, Pneumothorax, Lung Opacity, Fracture, Edema. This is the headline classification metric.
- "**5 stable classes**": the 5 of those 8 with at least 10 positive test images: Cardiomegaly, Lung Opacity, Atelectasis, Pleural Effusion, Edema. This macro is more stable because it doesn't depend on classes with 1–8 positives.
- "**Test split**": unless stated otherwise, the **377 held-out frontal test images** (never used in training or tuning).
- **CE F1** = clinical-efficacy F1. The rule-based labeler is run on a *generated* report, and the findings it extracts are scored against the true labels. This is the main report-quality metric.

---

## 1. Project identity

| Item | Value |
|---|---|
| Title | **MedXVLM: A Multimodal Vision-Language System for Chest X-Ray Understanding** |
| Author | Kartik Singh Rana |
| Programme | B.Sc. Artificial Intelligence & Data Science: major project |
| Institution / supervisor | `[University]`, `[Supervisor]` (fill in) |
| Duration | 2026-08-30 (Phase 1) → 2026-09-26 (final experiments); proposal planned 16 weeks |
| Hardware | One NVIDIA RTX 5080 (16 GB VRAM, Blackwell `sm_120`), Windows + WSL2 (Ubuntu), conda env |
| Software | Python 3.11.15, PyTorch 2.11.0 (CUDA 13.0), Hugging Face `transformers` 5.8.0, `timm` 1.0.28, `peft` 0.20.0 (LoRA), `open_clip_torch` 3.3.0, scikit-learn 1.8.0, `sacrebleu` 2.6.0, `rouge-score` 0.1.2, Streamlit 1.61.1 (demo) |
| Code | Modular, typed PyTorch package `src/medxvlm/` (data, models, engine, eval, explain, inference) + `scripts/` + `configs/` (YAML) + `app/` (Streamlit) + `docs/` |
| Data | Indiana University / Open-i chest X-rays (public). NIH ChestX-ray14 test split used only for external testing |
| External systems compared | **RadZero** (Deepnoid, arXiv:2504.07416, NeurIPS 2025 poster): zero-shot classifier, reproduced here from public weights. **CheXOne** (Stanford AIMI, arXiv:2604.00493): 3B report generator, run here from public weights |

---

## 2. The headline results (the numbers to quote)

| What | Result | Comparison | Label |
|---|---|---|---|
| **Disease classification, 4-model ensemble** | **Macro AUROC 0.868** (8 shared classes), 95% CI 0.827–0.902 | RadZero on identical images: **0.868** (CI 0.823–0.903). Paired difference **0.000, 95% CI −0.033 to +0.033**: statistically indistinguishable | apples-to-apples |
| Disease classification, single model | **0.837 ± 0.015** (mean ± sd over 4 random seeds; seeds: 0.855, 0.833, 0.839, 0.819) | — | apples-to-apples |
| All 13 measurable classes | Ensemble 0.835; single 0.804 ± 0.019 | RadZero n/a (no prompts for the other 5 classes) | — |
| **External hospital** (NIH ChestX-ray14, 25,596 unseen images, 8 mapped classes) | Ensemble **0.719**; single **0.704 ± 0.009** | vs 0.868 at home: a domain-shift drop of about 0.15. RadZero's paper reports 0.804 on this dataset with its own protocol | approximate / RadZero figure reported by the authors, not reproduced here |
| Contribution of the project's classifier head | Attention head beats a fine-tuned "baseline head" on the same encoder by **+0.078** (CI +0.028 to +0.119) | Baselines: ImageNet ViT 0.804, X-ray DINOv2 0.777 | apples-to-apples |
| **Report generation** (best config: v2 + beam search) | **CE macro F1 0.176** on 70 frontal test images; **3%** of reports invent a device/surgery | CheXOne (3B) on same images: **0.149**, 0% invented devices. Difference within the noise band (report-level CIs are about ±0.10) | apples-to-apples |
| Report generation, findings-conditioned (v2c) | CE F1 **0.165**, **0% invented devices** | Ceiling for any report built on the classifier's findings: **0.234** | apples-to-apples |
| **Zero-shot (prompt) classification** | Best variant **0.734** (8 shared) | RadZero **0.868**; difference +0.135 (CI +0.091 to +0.208) in RadZero's favour | apples-to-apples |
| **Heatmap localisation** (pointing game, 984 NIH radiologist boxes, unseen hospital) | Classifier attention maps (ensemble) hit the box **41.2%** (CI 38.1–44.3) | Random point **7.5%**; always-centre **21.5%**; Grad-CAM **19.0%** | approximate (out-of-distribution by design) |
| Report writer's use of the image | Generated sentences put **1.3%** of their attention on the image tokens | — | measured |
| **RQ1: does encoder pretraining matter?** | X-ray DINOv2 0.837 ± 0.015 > BiomedCLIP 0.802 > OpenAI CLIP 0.792 > ImageNet 0.784. Pointing game 32% / 27% / 24% / 13% | BiomedCLIP vs CLIP not significant (one seed each) | apples-to-apples |

**One-sentence summary of the project:** a three-stage chest X-ray vision-language system trained on about 3,000 X-rays. Its 4-model image-only classifier is statistically indistinguishable from a published system trained on about 100× more data (RadZero, 0.868 vs 0.868 macro AUROC on identical images). Its heatmaps are validated against radiologist boxes (41% vs 7.5% chance). Its report writer matches a 3B-parameter published model on clinical-efficacy F1 while remaining far from clinically usable. All of it was evaluated with an unusually strict honesty protocol that uncovered and fixed a data leak, a labeling bug and output collapse along the way.

---

## 3. Timeline (for the report's project-management section)

| Date | Phase | What happened |
|---|---|---|
| 2026-08-30 | Phase 1–2 | Repository skeleton, environment validation (RTX 5080 / `sm_120` PyTorch), toy data loader, literature review, proposal; model wiring and sanity checks (overfit one batch per stage), code review |
| 2026-09-05 | Phase 3 | Real data pipeline and rule-based labeler; trained baseline, Stage 1 (×2 for the local-loss ablation), Stage 2 (×2), Stage 3. Headline then: 0.865 macro AUROC (later found to be leaky) |
| 2026-09-06 | Phase 4 | Reproduced RadZero from public weights (0.852 on its own OpenI benchmark vs 0.847 paper-reported); ran CheXOne from public weights |
| 2026-09-06 | Phase 5 | Head-to-head on MedXVLM's held-out split. **Found the text-fusion data leak** (0.865 → 0.51 image-only), **found the report-generator collapse** (5 distinct outputs for 150 images) |
| 2026-09-17 | v2 rebuild | Found and fixed a **labeler negation bug**; switched to X-ray DINOv2 encoder, frontal-only, image-only classifier with asymmetric loss → 0.855 (seed 0) vs RadZero 0.868 |
| 2026-09-18 | — | Sourced status inventory of every number and file |
| 2026-09-19 | Stage 3 v2 | Report generator retrained with abnormal-case oversampling → collapse broken, CE F1 0.152 ≈ CheXOne 0.149, but 10% invented devices |
| 2026-09-20 | External + overnight runs | NIH ChestX-ray14 external test (0.714); 4-seed ensemble (**0.868 = RadZero**; single 0.837 ± 0.015); beam-search decoding (CE 0.176, invented devices 10% → 3%); v2b ablation (collapse, showing oversampling was essential); zero-shot v3 (failed) and v3b (**0.734**); Streamlit demo app |
| 2026-09-21 | Explainability + v2c | Grad-CAM, attention rollout, text-grounded maps, sentence attribution, all scored on 984 NIH radiologist boxes; findings-conditioned report writer v2c (0% invented devices, ceiling 0.234) |
| 2026-09-24 | Demo app | Redesigned into a two-page app (Analyze / Model card), light minimal theme, "focus" heatmaps |
| 2026-09-26 | Statistics + RQ1 | Paired-bootstrap confidence intervals for all classification claims; **RQ1 answered** by retraining the classifier on three more encoders (BiomedCLIP, OpenAI CLIP, ImageNet ViT) |

---

## 4. Problem, motivation, objectives, research questions

### 4.1 Problem statement (from the proposal, condensed)
Chest radiography is the most frequently performed diagnostic imaging study worldwide, and the radiologist's narrative report is what drives clinical decisions. Most deep-learning work treats the image and the report as separate problems. Classifiers predict a fixed label set from pixels (ChestX-ray14, CheXNet, CheXpert). Report generators are trained end-to-end on image→text pairs, without a reusable, jointly aligned representation. That throws away supervision that every hospital archive already has for free: the report itself.

MedXVLM's premise: one well-aligned image–text representation, learned with contrastive pretraining, can serve as the shared backbone for **both** multi-label classification **and** free-text report generation. An explainability layer is added so that both outputs can be audited against image regions and report phrases, because a model meant to support a radiologist (not replace one) must be inspectable.

**Stated contribution:** not a new method, but **integration at small scale**. Alignment, classification and generation are built around one shared vision backbone, on hardware and a timeline a single student can manage, with explainability instrumented across the stages. The main closest works are GLoRIA (loss design), CheXagent (overall scope, at ~100× the scale), and RadZero (zero-shot alignment with built-in explainability).

**Contribution as it turned out (worth stating in the conclusion):** besides the integrated system, the project contributes a **rigorous, honest evaluation**. It compares head-to-head with two published systems on identical images, reports confidence intervals and seed variance, tests on an external hospital, validates heatmaps against radiologist boxes, and documents every failure with its measured cause.

### 4.2 Objectives and whether each was met

| # | Objective (proposal §2) | Outcome |
|---|---|---|
| 1 | Stage 1 contrastive alignment usable for retrieval and as a visual backbone | **Met, with a qualification.** Several Stage 1 variants were trained. Retrieval works but is weak at this data scale (val recall@1 ≈ 0.04–0.06, §7.1). In the final system the Stage 1 alignment powers the zero-shot head (0.734) and the text-grounded heatmaps. The classifier's image encoder is the pretrained X-ray DINOv2 tower, which Stage 1 v2 kept frozen, so Stage 1 passes it through unchanged |
| 2 | 14-class CheXpert classifier; measure gain over a vision-only baseline | **Met.** 0.837 ± 0.015 single / 0.868 ensemble; +0.05 to +0.08 over fine-tuned baselines, statistically significant |
| 3 | LoRA-tuned report generator sharing the visual backbone (LLaVA-style) | **Met.** Qwen2.5-1.5B + LoRA on the classifier's vision tower; CE F1 0.176 (≈ CheXOne 0.149); v2c variant with 0% invented devices |
| 4 | Explainability at every stage: Grad-CAM, attention rollout, text-grounded heatmaps, sentence attribution | **Met, and quantified.** All four were implemented and scored on 984 radiologist boxes (§11), which the proposal only planned qualitatively |
| 5 | Clean, modular, typed PyTorch code with a reproducible pipeline | **Met.** Package + configs + scripts; seeded splits; every result file in the repository; headless app tests; git history |
| (Week 14) | Streamlit demo | **Met.** Two-page app (§14) |

### 4.3 Research questions, hypotheses, and final answers

| RQ | Question and hypothesis (proposal §5) | Answer | Where |
|---|---|---|---|
| **RQ1** | *Does domain-specific contrastive initialisation matter at this scale?* Hypothesis: BiomedCLIP init beats general-domain CLIP for downstream classification (and retrieval). | **Yes, the pretraining domain matters, but the specific BiomedCLIP-over-CLIP contrast is not resolved.** The same classifier on four encoders: X-ray DINOv2 (chest-X-ray self-supervised) 0.837 ± 0.015 > BiomedCLIP 0.802 > OpenAI CLIP 0.792 > ImageNet ViT 0.784. X-ray DINOv2's lead over both general-domain encoders is significant. BiomedCLIP vs CLIP is +0.010 (CI −0.025 to +0.086): in the predicted direction, not significant with one run each. The effect is largest for localisation: pointing game 32% vs 13% for ImageNet. Retrieval was not re-tested. | §9 |
| **RQ2** | *Does a local (patch–word) contrastive term improve grounding, not just accuracy?* Hypothesis: yes, as in GLoRIA. | **No measurable effect at this scale.** Text-grounded pointing game with vs without the local loss: 5.3% vs 3.8%. Both are *below* a random point (7.5%). Classification with vs without: 0.8645 vs 0.8648 (Phase 3, tie). The direction matches the hypothesis; the magnitude is unusable. | §11.1, §12 |
| **RQ3** | *Can a frozen vision tower + LoRA-tuned 1.5B LLM produce coherent findings on a narrow report style?* | **Coherent: yes. Visually grounded: mostly no.** The reports are fluent and correctly formatted, and match CheXOne-3B on CE F1 (0.176 vs 0.149). But sentence attribution shows the decoder puts only 1.3% of its attention on the image and reuses one "glance" for all sentences. Its output is mostly language prior. Conditioning on the classifier's findings (v2c) removes invented findings but is capped by the classifier (ceiling 0.234). | §10, §11.2 |
| **RQ4** | *Does explainability quality track the Stage 1 design / encoder choice, not just task accuracy?* | **Yes, and more strongly than accuracy does.** (a) Across encoders, heatmap localisation ranges 13%–32% while AUROC ranges only 0.78–0.84. (b) Text-grounded localisation improves across successive alignment versions (Phase 3 5.3% → v2 8.7% → v3b 11.3%). This is not monotonic in zero-shot AUROC (Phase 3 0.669 > v2 0.565), and Phase 3 used a different encoder. It stays far below the supervised attention maps (41%). (c) The report writer's visual grounding is weak regardless. | §9, §11 |

---

## 5. Data

### 5.1 Indiana University / Open-i chest X-ray collection (training and main evaluation)
Public; accessed via the Kaggle mirror `raddar/chest-xrays-indiana-university`. Paired frontal/lateral radiographs with de-identified reports (indication, comparison, findings, impression), from the Indiana Network for Patient Care. Chosen over **MIMIC-CXR**, which needs PhysioNet credentialing and is about 377k images, and over **ChestX-ray14**, which has no report text, so it cannot train the alignment or report stages. (Source: `docs/project_proposal.md` §4, `docs/status_report.md` §3.)

| Quantity | Value |
|---|---:|
| Raw reports / projection rows / PNG files | 3,851 / 7,466 / 7,470 (14 GB) |
| Studies after cleaning (25 with empty findings and impression dropped) | **3,826** |
| Image rows after cleaning | **7,426** (3,794 frontal / 3,632 lateral) |
| Split (study-level, seed 0, 80/10/10) | 3,060 / 382 / 384 studies; 5,929 / 749 / 748 images |
| **Frontal images per split (what the final models use)** | **3,030 train / 387 val / 377 test** |
| Report-generation evaluation subsample (fixed, seed 0) | 150 test images (70 frontal, 80 lateral) |

The split is at study level (`uid`), so the two views of a study never cross splits. The dataset has no patient ID beyond the study ID. The split was verified byte-identical before and after the label fix, so no leakage was introduced by relabeling.

**Cleaning:** 3,425 of 7,426 report rows contained the de-identification token "XXXX", which the Phase 3 report generator learned to reproduce. It was stripped in the v2 pipeline (0 remaining).

**Labels:** produced by the project's own rule-based CheXpert-style labeler (`src/medxvlm/data/labeler.py`) for the 14 CheXpert classes. It uses positive / negative / uncertain states, and training uses the **U-Ones** policy (uncertain → positive, not mentioned → negative). The labeler is validated by a 24-case / 42-assertion regression suite. Study-level label distribution (current labels):

| Class | Positive | Negative | Uncertain | Not mentioned | Training positive rate (U-Ones) |
|---|---:|---:|---:|---:|---:|
| No Finding | 2,642 | 1,184 | 0 | 0 | 69.1% |
| Enlarged Cardiomediastinum | 8 | 35 | 0 | 3,783 | 0.2% |
| Cardiomegaly | 297 | 5 | 1 | 3,523 | 7.8% |
| Lung Opacity | 471 | 571 | 16 | 2,768 | 12.7% |
| Lung Lesion | 136 | 175 | 13 | 3,502 | 3.9% |
| Edema | 83 | 264 | 16 | 3,463 | 2.6% |
| Consolidation | 37 | 1,113 | 3 | 2,673 | 1.0% |
| Pneumonia | 60 | 171 | 24 | 3,571 | 2.2% |
| Atelectasis | 312 | 7 | 56 | 3,451 | 9.6% |
| Pneumothorax | 27 | 2,451 | 2 | 1,346 | 0.8% |
| Pleural Effusion | 161 | 2,663 | 27 | 975 | 4.9% |
| Pleural Other | 34 | 0 | 7 | 3,785 | 1.1% |
| Fracture | 103 | 88 | 11 | 3,624 | 3.0% |
| Support Devices | 98 | 0 | 2 | 3,726 | 2.6% |

**Severe class imbalance:** 10 of 14 classes are under 5% training prevalence with the corrected labels. **Test-set positives among the 377 frontal test images** (important caveat for every per-class number): No Finding 268, Lung Opacity 42, Cardiomegaly 36, Atelectasis 36, Pleural Effusion 15, Edema 12, Lung Lesion 9, Pneumonia 8, Fracture 8, Support Devices 8, Consolidation 5, Pleural Other 3, **Pneumothorax 1**, Enlarged Cardiomediastinum 0. Per-class AUROCs on single-digit counts are noise.

**The labeling bug and its fix (a key episode, §6).** The original labeler's negation scope was 5 tokens, and it matched cue substrings (so "no" matched inside "nodular"). In *"No focal consolidation, pleural effusion, or pneumothorax"*, the later items escaped the negation and were labeled **positive**. After the fix (a leading negation now governs the whole coordinated list until a scope-reset word, with word-boundary cue matching), study-level positive rates changed:

| Class | Before fix | After fix |
|---|---:|---:|
| Pneumothorax | 10.2% | **0.8%** |
| Pleural Effusion | 19.6% | **4.9%** |
| Consolidation | 2.8% | 1.0% |
| Pneumonia | 3.8% | 2.2% |
| No Finding | 51.0% | **69.1%** |

Other classes moved by less than 1 point. **Independent confirmation:** RadZero's saved predictions, unchanged, agree far better with the corrected labels. At a fixed image set the label fix alone adds about +0.09 macro AUROC (all 748 test images 0.737 → 0.824; 377 frontal 0.780 → 0.868). The frequently quoted 0.737 → 0.868 also includes the frontal-only restriction (+0.04), so do not attribute all of it to the fix. (Do not cite a "published normal rate": the dataset paper reports 38% normal reports under a broader coder definition, which is not comparable to CheXpert "No Finding"; see §22.) One residual labeler pattern is known and was deliberately left unfixed, because relabeling this late would shift every result: *"…costophrenic blunting to suggest large pleural effusion"* (negation does not reach across "to suggest").

### 5.2 NIH ChestX-ray14 (external test only; never trained on)
Official patient-disjoint test split: **25,596 images, 2,797 patients**, different hospital and scanners, NLP-mined labels from NIH's own labeler. Used for (a) external classification AUROC and (b) the heatmap validation, because it ships **984 radiologist-drawn bounding boxes on 880 test images** for 8 findings: Atelectasis 180, Cardiomegaly 146, Effusion 153, Infiltrate 123, Mass 85, Nodule 79, Pneumonia 120, Pneumothorax 98. Indiana has no box annotations, so all localisation numbers are out-of-distribution by design.

Class mapping for external AUROC, concept-identical classes only: Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion→Effusion, No Finding. Mapping for the bounding boxes: Effusion→Pleural Effusion; Mass and Nodule→Lung Lesion; Infiltrate→Lung Opacity (**approximate**).

---

## 6. The project's story: what went wrong and how it was fixed

This is the narrative spine of the report. Present it as the scientific process working, not as a list of failures.

1. **Phase 3 looked like a success.** The classifier reported **0.865 macro AUROC** (val) vs a 0.694 baseline.
2. **External benchmarking exposed a data leak (Phase 5).** To compare fairly with RadZero, which sees only the image, the classifier was re-tested image-only and scored **0.51, which is chance**. Cause: its "gated text fusion" was fed the **paired ground-truth report** at training and validation time, so it had learned to read the answer from the text. Fix: an image-only classifier (`use_text_fusion: false`); the text encoder is never built.
3. **The labels were wrong (v2 rebuild).** The negation bug above mislabeled thousands of normal studies. Fixed and regression-tested; RadZero independently confirmed the fix.
4. **Other v2 fixes:**
   - Frontal-only training: half of the images were lateral views, which inherit the frontal image's report but carry little signal.
   - X-ray-pretrained encoder (X-ray DINOv2, the same tower RadZero uses) instead of BiomedCLIP.
   - Asymmetric loss (ASL) for the imbalance; split learning rates (encoder 1e-5, head 1e-3); cosine schedule; early stopping.
   - "XXXX" tokens stripped from the reports.
   - Result, on the same 377 frontal test images: **0.572 → 0.855 (seed 0)**. The 0.51 figure above belongs to a different set (748 all-view images, old labels): the leak itself is shown there as 0.851 with the report text vs 0.512 without (C16).
5. **A single seed overstated it.** Four seeds gave 0.837 ± 0.015; seed 0 was the luckiest. Averaging the four models: **0.868 = RadZero**. Every headline since quotes mean ± sd or the ensemble.
6. **The report generator had collapsed.** Phase 3 produced **5 distinct reports for 150 images**, mostly one normal boilerplate. Fix: oversample abnormal studies to 50% of training batches → 61 distinct outputs, CE F1 doubled (0.078 → 0.152). **But** it now invented support devices in 10% of frontal reports, a shortcut learned because device words are 4× more common in abnormal reports.
7. **Decoding fixed most of the hallucination for free.** Beam search (3 beams, repetition penalty 1.2) on the same checkpoint: invented devices 10% → 3%, CE F1 0.152 → 0.176.
8. **Removing the oversampling (v2b) collapsed the model completely:** one identical sentence for all 150 images. Yet its validation loss was *better* (1.108 vs 1.132). Lesson: **validation loss is the wrong model-selection signal for report generation**, and the oversampling was essential.
9. **Zero-shot classification failed, then was fixed as far as the data allows.** Report-level alignment gave 0.565. Sentence-level alignment (v3) gave 0.568, and the cause was measured: only **1,683 of 16,375 training sentences (10.3%)** mention any finding. Training on finding sentences only (v3b) gave **0.734**.
10. **Explainability was measured, not assumed.** The classifier's heatmaps hit radiologist boxes 41% of the time (random 7.5%). Grad-CAM, the textbook method, was worse (19%). Sentence attribution showed the report writer barely looks at the image (1.3% attention).
11. **Findings-conditioning (v2c)** fed the classifier's findings to the report writer as text: 0% invented devices, but capped at a 0.234 ceiling by the classifier's recall.
12. **Statistics and RQ1** (final step): confidence intervals turned "equal to RadZero" into "indistinguishable (Δ 0.000 ± 0.033)", and the encoder comparison answered RQ1.

---

## 7. Methods: system architecture (final version) and every variant

```
                 ┌──────────────── Stage 1: contrastive alignment ────────────────┐
 chest X-ray ──► │ ViT-B vision tower ──► 512-d projection ┐                       │
                 │                                         ├─ InfoNCE (+ local     │──► zero-shot head, text-grounded heatmaps
 report text ──► │ BiomedBERT text tower ─► 512-d proj.  ──┘   patch-word loss)    │
                 └────────────────────────────────────────────────────────────────┘
 chest X-ray ──► ViT-B tower (X-ray DINOv2) ──► 256 patch tokens ──► Stage 2: per-label attention pooling ──► 14 probabilities (+14 attention maps)
                                     │                                  × 4 seeds, probability-averaged = ensemble
                                     └──► 8×8 avg-pool = 64 tokens ──► MLP projector ──► Stage 3: Qwen2.5-1.5B-Instruct + LoRA ──► findings paragraph
                                                                        (v2c: prompt also lists the classifier's flagged findings)
```

### 7.1 Stage 1: contrastive image–report alignment
- **Model:** `ContrastiveAlignment` = vision tower + text tower, each followed by a linear projection to a **512-d** shared space. Learnable temperature (CLIP init, 1/0.07). Symmetric **InfoNCE** loss. Optional **local patch–word loss** in the FILIP/GLoRIA style: each word is matched to its most similar patch, the matches are averaged over words, and InfoNCE is applied to the resulting scores.
- **Text tower:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT).
- **Variants trained:**

| Variant | Vision tower | Text unit | Notes | Val result |
|---|---|---|---|---|
| Phase 3 local-on / local-off | BiomedCLIP ViT-B/16, fine-tuned | full report | 5 epochs, batch 16, LR 2e-5, all views; the RQ2 ablation pair | val R@1 (image→text) 0.0134 / 0.0174; val loss flat (did not converge) |
| **v2** | **X-ray DINOv2 (frozen)** | full report | 15 epochs, batch 32, text LR 2e-5, projection LR 1e-3, frontal, local loss on | val loss 5.96 → 4.67 (epoch 7); val R@1 i→t 0.044, t→i 0.059 (epoch 15) |
| v2, val-loss-selected / frozen-text | as v2 | full report | tested whether the weak zero-shot came from checkpoint choice or text drift | neither helped (§8.4) |
| v3 | X-ray DINOv2 (fine-tuned, LR 1e-5; projections 1e-3; batch 32; local loss off) | **one sentence per image per epoch**; **multi-positive InfoNCE** (identical sentences count as mutual positives) | 12 epochs | val loss 3.19 → 2.94 (barely moved) |
| **v3b** | X-ray DINOv2 (fine-tuned, LR 1e-5; projections 1e-3; batch 32; local loss off) | **finding sentences only**: 900 images have one; the other 2,130 normal images all get the canonical sentence "No acute cardiopulmonary abnormality." | 8 epochs (early-stopped, best 5), 17.1 min | **0.734 on the 8 shared classes (best there and on all 13); Phase 3 is better on the 5 stable classes (0.708 vs 0.693)** |

- **Retrieval (val split; no test-split retrieval evaluation was run).** R@1 / R@5 / R@10, image→text:
  - Phase 3 local-on: 0.013 / 0.060 / 0.092
  - Phase 3 local-off: 0.017 / 0.084 / 0.132
  - v2: 0.044 / 0.132 / 0.217

  Weak in absolute terms, as expected with about 3,000 pairs and near-duplicate normal reports.
- **Zero-shot head:** cosine similarity between the image embedding and the prompt "There is {finding}." For No Finding the prompt is "There is no acute cardiopulmonary abnormality." A variant subtracting negative prompts ("There is no {finding}.") was also tested (§8.4).

### 7.2 Stage 2: image-only multi-label classifier (the project's strongest component)
- **Encoder:** `StanfordAIMI/dinov2-base-xray-224` ("X-ray DINOv2", Apache-2.0). A DINOv2 ViT-B/14 self-supervised on chest X-rays, and the same tower RadZero uses. 85.7M parameters; 224 px input gives a 16×16 grid of 256 patch tokens plus a class token.
- **Head (`ClassifierHead`, 2.38M parameters):**
  - **Per-label attention pooling:** 14 learned query vectors, one per finding, attend over the patch tokens (8-head multi-head attention + LayerNorm). This gives one pooled vector per finding, scored by a shared linear layer.
  - Each finding therefore has its own spatial attention map, which doubles as the heatmap.
  - Optional gated text fusion exists in the code but is **disabled** (it caused the Phase 3 leak).
- **Training (`configs/classifier_v2.yaml`):**
  - Frontal images only (3,030), U-Ones labels.
  - **Asymmetric Loss** (Ben-Baruch/Ridnik et al., ICCV 2021; γ+ = 1, γ− = 4, clip 0.05).
  - AdamW with **split learning rates**: encoder 1e-5, head 1e-3; weight decay 0.05; cosine schedule.
  - Batch 32, up to 25 epochs, **early stopping on validation macro AUROC** (patience 6).
  - Preprocessing: CLAHE contrast enhancement → resize to 224 → ImageNet normalisation. Train-time augmentation: ±7° rotation, brightness jitter, random erasing. **No horizontal flip**, because anatomy is not left–right symmetric.
- **Seeds / ensemble:** the recipe was trained with seeds 0–3 (about 30–39 min each). The **ensemble** averages the four models' sigmoid probabilities.
- **Operating thresholds** (for the app and v2c only; AUROC needs none): per class, the F1-optimal point on the frontal validation split for the 8 classes with ≥10 val positives (val macro F1 0.566 there). The other 6 classes use the median threshold 0.538.
- **Baselines ("baseline head"; `configs/baseline_v2.yaml`):** the whole encoder fine-tuned with a single linear layer on the pooled image embedding, BCE loss, LR 5e-5, 12 epochs. **These are fully fine-tuned baselines, not frozen linear probes.** Two were trained: ImageNet ViT-B/16 and X-ray DINOv2. They isolate what the attention head plus the ASL recipe add.

### 7.3 Stage 3: report generation (LLaVA-style)
- **Architecture (`ReportGenerator`):**
  - Frozen vision tower: the **seed-0 Stage 2 classifier's fine-tuned X-ray DINOv2**, which has demonstrably learned this dataset's findings.
  - Its 16×16 patch grid is average-pooled to **8×8 = 64 visual tokens**.
  - A 2-layer MLP projector (768 → 1536 → 1536, GELU) maps them into the language model.
  - The tokens are spliced at an `<image>` placeholder in the chat-template prompt: *"Generate a radiology findings report for this chest X-ray."*
  - Decoder: **Qwen2.5-1.5B-Instruct** with **LoRA** (rank 16, α 32, dropout 0.05) on q/k/v/o/gate/up/down projections.
  - Trainable 22.0M (LoRA + projector) vs 1.63B frozen. fp32; about 7 GB at inference.
- **Training:** loss only on answer tokens; LR 2e-4; effective batch 16; frontal only; cleaned text; **abnormal-study oversampling to 50%** of sampled batches (v2); early stopping.
- **Variants:**

| Variant | Change | Outcome |
|---|---|---|
| Phase 3 | BiomedCLIP Stage 1 tower, 7×7 = 49 tokens, all views, no oversampling, text contained "XXXX" | collapsed: 5 distinct outputs / 150 |
| **v2** | tower, token count, cleaning and 50% oversampling as above; selected by val loss (best epoch 3 of 5; 278.6 min) | collapse broken; CE 0.152 (greedy) / **0.176 (beam)** |
| v2b | as v2 but 30% abnormal (≈ natural rate) | total collapse (1 output for all images) despite lower val loss |
| **v2c** | prompt adds *"Classifier findings: X, Y."* from the ensemble + thresholds (each finding dropped with p = 0.1 during training); **checkpoint chosen by clinical-efficacy F1 of generated text** on a 48-image val subset, not by loss | 0% invented devices; CE 0.165 (greedy); verbaliser of the classifier |

- **Decoding:** greedy vs **beam search (3 beams, repetition penalty 1.2)**. Beam is best for v2 and greedy is best for v2c (§10.2).

### 7.4 Explainability methods (all in `src/medxvlm/explain/`)

| Method | What it shows | Class-specific? |
|---|---|---|
| **Attention pooling** | Stage 2's own per-finding attention weights over patches (the decision mechanism itself) | yes |
| **Grad-CAM** | gradient of the finding's logit with respect to the final patch tokens (channel-weighted, ReLU) | yes |
| **Attention rollout** (Abnar & Zuidema 2020) | CLS-to-patch information flow through all 12 DINOv2 layers | no (one map per image) |
| **Text-grounded maps** | cosine between each projected patch and the prompt "There is {finding}." in the Stage 1 space | yes |
| **Sentence attribution** | re-run the decoder over its own report with attention weights exposed; per sentence, the attention mass on the 64 visual tokens (share, sink-corrected) and its 8×8 map | per sentence |

In the app, heatmaps use a **"focus" overlay**: only high-attention regions are tinted (inferno colormap), so the anatomy stays readable. Docs and training snapshots use the classic JET overlay.

### 7.5 Evaluation protocol and metrics
- **Classification:**
  - Per-class AUROC and macro averages over the 8 shared / 5 stable / all 13 measurable classes. Enlarged Cardiomediastinum has 0 test positives, so it is excluded.
  - **Paired bootstrap** CIs: 2,000 resamples of the 377 test images, each system scored on the same resample. In the 8-class macro, a class with no positives in a resample is skipped for that resample.
  - **Seed variance:** 4 seeds.
- **External:** macro AUROC over 8 mapped NIH classes, with no fine-tuning or recalibration.
- **Report generation**, on a fixed 150-image test subsample; the headline is on its **70 frontal** images because every final model is frontal-only:
  - **CE macro F1:** the project's labeler run on the generated text vs the image's true 14-label vector (U-Ones). **Macro over all 14 classes**, so classes with no support count as 0.
  - CE micro F1; BLEU-4 (sacrebleu); ROUGE-L.
  - **Distinct outputs** and **modal-output share**, which measure collapse.
  - **Device/surgery hallucination rate:** regex (catheter, PICC, tube, line, clip, wire, pacemaker, lobectomy, sternotomy, port) present in the generation but absent from the reference.
  - Bootstrap CIs on CE F1.
- **Heatmap localisation**, on the 984 NIH boxes:
  - **Pointing game:** is the heatmap's hottest pixel inside the box? (Zhang et al.) 95% Wilson CIs.
  - **Mass ratio:** share of heatmap mass inside the box ÷ box area share; 1 = uniform.
  - **Area-matched IoU.**
  - Baselines: a uniformly random point (expected hit = box area fraction) and always-the-image-centre.

### 7.6 Compute (wall-clock, single RTX 5080)

| Run | Time |
|---|---:|
| Phase 3: baseline / Stage 1 ×2 / Stage 2 ×2 / Stage 3 | 20.4 min / ~20 min each / ~36 min each / 210.7 min |
| v2 Stage 1 | 36.9 min |
| v2 Stage 2 (per seed) | 30–39 min |
| v2 baselines | ~26 min each |
| v2 Stage 3 / v2b / v2c | 278.6 / 131 / 91.7 min (v2c: 70.4 + 21.3; see §15 on GPU memory paging) |
| Stage 1 v3 / v3b | 26.1 / 17.1 min |
| RQ1 encoders: BiomedCLIP / OpenAI CLIP / ImageNet | 45.0 / 50.8 / 27.5 min |
| RadZero reproduction (1,000 images × 18 classes) / on the 748-image test split | 44 / 32.5 min |
| CheXOne, 150 reports | 11.9 min |
| NIH external eval (25,596 images) | 132 s |
| Pointing game, all methods | 185 s |
| Sentence attribution (377 + 310 images) | 18 min |

---

## 8. Results: disease classification (377 held-out frontal test images, corrected labels, image only)

Sources: `docs/v2_rebuild.md` §3.1–3.3, §3.6, §3.8; result files `v2_classification_summary.json`, `classifier_seed_ensemble.json`, `classification_bootstrap.json`.

### 8.1 Main comparison table (all rows apples-to-apples: same images, same labels, no text input)

| System | 8 shared classes [95% CI] | 5 stable classes [95% CI] | All 13 classes |
|---|---:|---:|---:|
| **MedXVLM, 4-model ensemble** | **0.868** [0.827, 0.902] | 0.867 [0.818, 0.910] | **0.835** |
| RadZero (published zero-shot model, public weights, re-scored on these images) | **0.868** [0.823, 0.903] | 0.875 [0.827, 0.916] | n/a |
| MedXVLM, single model, mean ± sd of 4 seeds | 0.837 ± 0.015 | 0.855 ± 0.005 | 0.804 ± 0.019 |
| MedXVLM, single model, seed 0 | 0.855 [0.805, 0.888] | 0.862 [0.820, 0.901] | 0.829 |
| Baseline head, ImageNet ViT (fine-tuned) | 0.804 [0.741, 0.843] | 0.817 [0.766, 0.863] | 0.738 |
| Baseline head, X-ray DINOv2 (fine-tuned) | 0.777 [0.731, 0.816] | 0.795 [0.744, 0.838] | 0.734 |
| MedXVLM zero-shot prompts (v3b) | 0.734 [0.656, 0.771] | 0.693 [0.629, 0.746] | 0.652 |
| Phase 3 classifier tested image-only (the "before") | 0.572 | 0.525 | 0.547 |

Per-seed values (8 shared / 5 stable / all 13 / NIH): seed 0 0.855 / 0.862 / 0.829 / 0.714; seed 1 0.833 / 0.850 / 0.792 / 0.701; seed 2 0.839 / 0.856 / 0.808 / 0.706; seed 3 0.819 / 0.853 / 0.786 / 0.693. Validation macro AUROC (13 classes) of the four seeds: 0.833 / 0.828 / 0.850 / 0.810.

### 8.2 Paired differences (bootstrap, 2,000 resamples)

| Comparison (a − b) | Δ, 8 shared [95% CI] | Significant? | Δ, 5 stable [95% CI] |
|---|---:|:---:|---:|
| Ensemble − RadZero | **0.000** [−0.033, +0.033] | no (indistinguishable) | −0.008 [−0.031, +0.016] |
| Single model (seed 0) − RadZero | −0.014 [−0.058, +0.024] | no | −0.013 [−0.039, +0.012] |
| Ensemble − single model (seed 0) | +0.013 [−0.004, +0.038] | no (P(a>b) = 0.94) | +0.005 [−0.015, +0.021] |
| Attention head − baseline head, same encoder (X-ray DINOv2) | **+0.078** [+0.028, +0.119] | **yes** | +0.067 [+0.032, +0.105] |
| Attention head (X-ray DINOv2) − baseline head (ImageNet ViT) | +0.051 [+0.011, +0.100] | **yes** | +0.045 [+0.014, +0.081] |
| Baseline head: ImageNet − X-ray DINOv2 | +0.028 [−0.022, +0.059] | no | +0.022 [−0.015, +0.059] |
| RadZero − MedXVLM zero-shot (v3b) | +0.135 [+0.091, +0.208] | **yes** | +0.182 [+0.132, +0.237] |

**How to phrase it:**
- "On identical images and labels, the ensemble and RadZero are statistically indistinguishable (difference 0.000, 95% CI −0.033 to +0.033). The CI rules out a true gap larger than about 3 AUROC points in either direction."
- **Do not claim MedXVLM beats RadZero.** On the stable 5-class macro, RadZero is nominally ahead (not significant).
- The per-system AUROC intervals are ±0.04–0.05 wide. This test set cannot separate good systems that differ by 1–2 points, which is why the seed spread (±0.015) matters.

### 8.3 Per-class AUROC (test; n+ = positives among 377)

| Class | n+ | Ensemble | Single (seed 0) | Baseline ImageNet | Baseline X-ray DINOv2 | Phase 3 blind | RadZero | Zero-shot v3b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 268 | 0.826 | 0.820 | 0.811 | 0.773 | 0.603 | — | 0.23 |
| Cardiomegaly | 36 | **0.948** | 0.939 | 0.928 | 0.871 | 0.627 | **0.964** | 0.74 |
| Lung Opacity | 42 | **0.878** | 0.854 | 0.801 | 0.776 | 0.358 | 0.859 | 0.58 |
| Atelectasis | 36 | 0.853 | 0.851 | 0.771 | 0.787 | 0.580 | **0.883** | 0.66 |
| Pleural Effusion | 15 | 0.792 | 0.787 | 0.743 | 0.678 | 0.640 | **0.833** | 0.75 |
| Edema | 12 | **0.865** | 0.879 | 0.843 | 0.861 | 0.422 | 0.835 | 0.73 |
| Lung Lesion | 9 | 0.657 | 0.653 | 0.442 | 0.517 | 0.476 | — | 0.58 |
| Pneumonia | 8 | 0.873 | 0.771 | 0.748 | 0.797 | 0.466 | **0.911** | 0.72 |
| Fracture | 8 | **0.835** | 0.808 | 0.641 | 0.637 | 0.577 | 0.752 | 0.69 |
| Support Devices | 8 | 0.981 | 0.966 | 0.900 | 0.851 | 0.411 | — | 0.61 |
| Consolidation | 5 | 0.882 | 0.832 | 0.622 | 0.797 | 0.422 | — | 0.73 |
| Pleural Other | 3 | 0.570 | 0.664 | 0.389 | 0.392 | 0.622 | — | 0.46 |
| Pneumothorax | 1 | 0.902 | 0.949 | 0.960 | 0.806 | 0.904 | 0.910 | 0.99 |
| Enlarged Cardiomediastinum | 0 | — | — | — | — | — | — | — |

Rows with n+ < 10 are indicative only. The Pneumothorax row rests on **one** positive image and should not be interpreted.

### 8.4 Zero-shot classification (prompt-based, from the Stage 1 alignment)

| Stage 1 model | Prompt "There is {finding}." (8 shared) | Positive − negative prompts (3 templates) | 5 stable |
|---|---:|---:|---:|
| Phase 3 (BiomedCLIP, report-level, all views) | 0.669 | 0.642 | 0.708 |
| v2 (X-ray DINOv2 frozen, report-level) | 0.565 | 0.604 | 0.547 |
| v2, val-loss-selected checkpoint | 0.572 | — | — |
| v2, both towers frozen (projections only) | 0.484 | — | — |
| v3 (sentence-level, multi-positive) | 0.568 | 0.558 | — |
| **v3b (finding sentences only)** | **0.734** | 0.605 | 0.693 |
| RadZero (~353k training pairs) | 0.868 | — | 0.875 |

**Findings:**
1. Two hypotheses for the failure of the report-level head were tested and refuted: bad checkpoint selection, and drift of the text tower away from short prompts.
2. The measured cause of the v3 failure is **signal density**. Only 10.3% of training sentences carry a finding (20% are negations, 70% boilerplate), so about 3 sentences in a batch of 32 carry class signal.
3. Restricting training to finding sentences (v3b) fixed it: 0.565 → 0.734. v3b beats v2 on every finding (e.g. Pneumonia 0.38 → 0.72, Atelectasis 0.41 → 0.66). The one regression is "No Finding" (0.34 → 0.23), a negative concept that single-prompt cosine similarity handles badly.
4. **Negative prompts only help a model trained on negated text:** they help v2 (full reports) but hurt v3b (0.734 → 0.605).
5. The remaining 13-point gap to RadZero is attributed to data scale: about 1,700 finding sentences from 900 images vs RadZero's roughly 100× more data. This is an explanation, not a measured quantity.

### 8.5 External hospital: NIH ChestX-ray14 (25,596 unseen images; approximate comparison)

| NIH class | n+ | Single model (seed 0) | 4-model ensemble |
|---|---:|---:|---:|
| Effusion | 4,658 | 0.785 | 0.786 |
| Edema | 925 | 0.781 | 0.793 |
| Cardiomegaly | 1,069 | 0.765 | 0.788 |
| Atelectasis | 3,279 | 0.708 | 0.719 |
| No Finding | 9,861 | 0.698 | 0.709 |
| Consolidation | 1,815 | 0.690 | 0.696 |
| Pneumonia | 555 | 0.650 | 0.651 |
| Pneumothorax | 2,665 | 0.634 | 0.611 |
| **Macro, 8 mapped** | | **0.714** (seed mean 0.704 ± 0.009) | **0.719** |

Context: a drop of about 0.15 from home (0.868 → 0.719) is the domain-shift cost of one outpatient site and 3,030 training images. **Reported by the authors, not reproduced here:** RadZero's paper gives 0.804 on ChestX-ray14 with its own protocol and label set. Supervised models trained *on* ChestX-ray14 typically report about 0.80–0.84 (general literature context, not measured here). Pneumothorax is weakest, consistent with only 23 positive training studies (27 in the whole corpus).

---

## 9. Results: RQ1, encoder comparison

Source: `docs/v2_rebuild.md` §3.9, `rq1_backbones_summary.json`, `classification_bootstrap.json`. **Design:** the identical Stage 2 recipe (`classifier_v2.yaml`; seed 0) was trained on four ViT-B encoders that differ only in their pretraining. All use the same ImageNet-normalised preprocessing, which slightly handicaps the two CLIP towers since their native normalisation differs; fine-tuning largely absorbs this. One run per new encoder (budget). X-ray DINOv2 has four seeds.

| Encoder | Pretraining | Runs | Test AUROC, 8 shared | Seed-0 [95% CI] | 5 stable | NIH AUROC | Pointing game (984 boxes) | Best epoch | Train time |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **X-ray DINOv2** (used by MedXVLM) | chest X-rays, self-supervised | 4 | **0.837 ± 0.015** | 0.855 [0.805, 0.888] | 0.855 ± 0.005 | **0.704 ± 0.009** | **31.5% ± 6.1** (seeds 22.6–36.5%) | 8, 8, 10, 12 | 30–39 min |
| BiomedCLIP | PubMed figure–caption pairs, contrastive | 1 | 0.802 | 0.802 [0.764, 0.854] | 0.844 | 0.687 | 26.7% | 15 | 45.0 min |
| OpenAI CLIP | web image–text, contrastive | 1 | 0.792 | 0.792 [0.726, 0.834] | 0.824 | 0.667 | 24.3% | 18 | 50.8 min |
| ImageNet ViT | ImageNet-21k→1k, supervised | 1 | 0.784 | 0.784 [0.719, 0.821] | 0.798 | 0.633 | 13.3% | 7 | 27.5 min |

Paired differences (seed 0 vs seed 0):

| Comparison | Δ 8 shared [95% CI] | Sig.? | Δ 5 stable [95% CI] | Sig.? |
|---|---:|:---:|---:|:---:|
| BiomedCLIP − OpenAI CLIP (**the literal RQ1 contrast**) | +0.010 [−0.025, +0.086] | no | +0.020 [−0.009, +0.049] | no |
| X-ray DINOv2 − BiomedCLIP | +0.053 [−0.006, +0.085] | no (P = 0.95) | +0.018 [−0.007, +0.048] | no |
| X-ray DINOv2 − OpenAI CLIP | +0.063 [+0.029, +0.105] | **yes** | +0.038 [+0.011, +0.067] | **yes** |
| X-ray DINOv2 − ImageNet ViT | +0.071 [+0.031, +0.119] | **yes** | +0.064 [+0.028, +0.108] | **yes** |
| BiomedCLIP − ImageNet ViT | +0.018 [−0.009, +0.089] | no | +0.046 [+0.014, +0.084] | **yes** |
| ImageNet ViT − OpenAI CLIP | −0.008 [−0.056, +0.039] | no | −0.026 [−0.062, +0.008] | no |

Per class (seed 0; test positives in brackets):

| Finding | X-ray DINOv2 | BiomedCLIP | OpenAI CLIP | ImageNet ViT |
|---|---:|---:|---:|---:|
| Cardiomegaly (36) | 0.94 | 0.94 | 0.89 | 0.82 |
| Lung Opacity (42) | 0.85 | 0.85 | 0.82 | 0.80 |
| Edema (12) | 0.88 | 0.87 | 0.88 | 0.82 |
| Atelectasis (36) | 0.85 | 0.83 | 0.79 | 0.79 |
| Pleural Effusion (15) | 0.79 | 0.74 | 0.74 | 0.76 |
| Pneumonia (8) | 0.77 | 0.87 | 0.70 | 0.78 |
| Fracture (8) | 0.81 | 0.65 | 0.57 | 0.56 |

**Interpretation:**
1. Pretraining domain matters: modestly for in-distribution accuracy, clearly for the external hospital (same ranking, wider spread), and most of all for localisation (32% vs 13%).
2. The literal RQ1 contrast (BiomedCLIP > CLIP) is in the predicted direction on every measure but not significant. RQ1's "retrieval" part was not re-tested.
3. X-ray DINOv2's *worst* seed (0.819) beats every other encoder's run. Caveat: the paired tests use its seed 0, which is its best seed. Against its 4-seed mean (0.837), the leads are +0.035 over BiomedCLIP, +0.045 over CLIP and +0.053 over ImageNet.
4. **The attention head needs a good encoder to help.** With X-ray DINOv2 it beats the baseline head by 0.06–0.08. With ImageNet it doesn't (0.784 vs 0.804, within noise; the two recipes also differ in loss and learning rate). The head pools *patch* tokens, so it helps only when the patches already encode chest anatomy.
5. General-domain encoders do worst on subtle findings (Fracture 0.56–0.65 vs 0.81; only 8 positives, so indicative).

**Optional follow-up (not done):** a second run each for BiomedCLIP and CLIP (about 1.7 h of GPU) would tighten the unresolved contrast.

---

## 10. Results: report generation

Sources: `docs/v2_rebuild.md` §3.4, §3.6.2, §3.6.4, §3.7; `report_generation_scores_v2c.json`, `report_ce_f1_bootstrap.json`. All systems generated reports for the **same fixed 150 test images** and were scored with the same scorer against corrected labels. The headline is the **70 frontal** images.

### 10.1 All systems (frontal 70 unless stated)

| System | CE macro F1 | CE micro F1 | CE macro F1, all 150 | Distinct outputs / 70 | Modal-output share | Invented devices / surgery | BLEU-4 | ROUGE-L | Mean words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Ceiling: the classifier's flagged findings as a bare list* | *0.234* | *0.656* | *0.224* | *13* | *74%* | *0%* | *0.0* | *0.10* | — |
| **v2 + beam search (best of the configurations compared on test, post hoc; app default)** | **0.176** | **0.650** | 0.148 | 14 | 61% | **3%** | 8.7 | 0.26 | 33.1 |
| v2c, findings-conditioned, greedy (official checkpoint) | 0.165 | 0.610 | **0.182** | 13 | 74% | **0%** | 9.2 | 0.29 | — |
| v2, greedy | 0.152 | 0.575 | 0.119 | 26 | 54% | 10% | 9.8 | 0.28 | 35.4 |
| **CheXOne, 3B (instruct mode, public weights)** | 0.149 | 0.599 | 0.107 | 36 | 26% | 0% | 10.2 | 0.29 | 28.8 |
| v2c, beam | 0.135 | 0.642 | 0.142 | 12 | 74% | 0% | 8.1 | 0.29 | — |
| Phase 3 (before fixes) | 0.078 | 0.603 | 0.066 | 5 | 74% | 1% | 12.1 | 0.30 | 29.6 |
| v2b (30% oversampling) | 0.057 | 0.590 | 0.057 | 1 | 100% | 0% | 6.7 | 0.29 | 17.0 |
| v2c with findings removed from the prompt (ablation) | 0.057 | 0.590 | 0.057 | 1 | 100% | 0% | 6.7 | 0.29 | — |
| *v2c epoch 3, greedy (post hoc, not selected)* | *0.231* | *0.651* | *0.191* | *13* | *74%* | *0%* | *8.7* | *0.29* | — |

**Distinct outputs over all 150 images:** Phase 3: 5; v2 greedy: 61; CheXOne: 87; v2 beam: 25; v2b: 1.

**Per-class CE F1** (frontal 70; support in brackets; v2 greedy / CheXOne / Phase 3):
- No Finding (46): 0.849 / 0.828 / 0.800
- Cardiomegaly (6): 0.333 / 0.000 / 0.286
- Lung Opacity (9): 0.364 / 0.364 / 0.000
- Lung Lesion (4): 0.333 / 0 / 0
- Pleural Effusion (4): 0.250 / 0.400 / 0
- Support Devices (1): 0 / 0.500 / 0
- Edema, Pneumonia, Atelectasis, Fracture, Pleural Other: 0 for all three.

A second per-class comparison (v2c epoch 3 greedy / v2c epoch 1 greedy / v2 beam / classifier ceiling):
- Cardiomegaly 0.67 / 0.55 / 0.73 / 0.71
- Lung Opacity 0.50 / 0.35 / 0.18 / 0.60
- Atelectasis 0.33 / 0.14 / 0.22 / 0.43
- Pleural Effusion 0.40 / 0.44 / 0.00 / 0.40
- Edema 0 / 0 / 0.50 / 0.29
- Pneumonia 0.50 / 0 / 0 / 0

**Bootstrap CIs on CE F1** (70 images; here macro is over *supported* classes only, so the values are higher than the 14-class macro in the table above):
- v2 beam 0.223 [0.127, 0.342]; v2c epoch 1 0.210 [0.142, 0.322]; v2c epoch 3 0.293 [0.185, 0.442]; classifier ceiling 0.297 [0.219, 0.431].
- Differences: v2c epoch 1 − v2 beam −0.007 [−0.127, +0.127]; v2c epoch 3 − v2 beam +0.070 [−0.047, +0.196].
- **Only "ceiling − v2 beam" (+0.087, CI excluding 0) is significant.**
- Report-level differences smaller than about ±0.10 are noise.

### 10.2 Findings, in order of importance
1. **Collapse was the first problem, and oversampling abnormal studies fixed it.** Phase 3 had 5 distinct outputs; v2 had 61. The v2b ablation proves the oversampling was essential: at the natural 30% rate the model emits one sentence for every image (*"The heart is normal in size. The mediastinum is unremarkable. The lungs are clear. No acute disease."*).
2. **Validation loss is the wrong selection criterion.** The fully collapsed v2b had *lower* validation loss (1.108) than v2 (1.132), because it fits the 70%-normal majority better. v2c therefore selected checkpoints by clinical-efficacy F1 of generated text.
3. **Oversampling introduced a hallucination shortcut.** v2 greedy mentioned a device or surgery absent from the reference in 10% of frontal reports (27% of all 150; 41% on the lateral views it never trained on). Device words appear in 15% of abnormal vs 4% of normal training reports, so "abnormal ⇒ mention a line" became a shortcut.
4. **Beam search halves the problem at zero cost:** invented devices 10% → 3%, CE F1 0.152 → 0.176. It also concentrates the output (26 → 14 distinct).
5. **CheXOne comparison:** MedXVLM v2-beam 0.176 vs CheXOne 0.149 on identical images. This is within the noise band, so the fair phrasing is "comparable". CheXOne varies its output more (36 distinct, 26% modal) and never invents devices. Caveat: CheXOne ran in instruct mode; its paper evaluates mainly in reasoning mode, which it reports as more accurate, so this likely understates it.
6. **Findings-conditioned generation (v2c):**
   - It is a verbaliser of the classifier. With the findings removed from the prompt it collapses to one output, so the image reaches the report only through the classifier.
   - It invents **0% devices** in every variant, the one unambiguous gain.
   - It is **capped at 0.234**: the ensemble flags a finding on only 18 of the 70 frontal images, per-class support is 1–9, and the scorer divides by 14 classes.
   - **Greedy beats beam for v2c** (0.165 vs 0.135), the opposite of v2.
   - Validation and test disagree about epoch 1 vs epoch 3 (validation 0.270 vs 0.229; test 0.165 vs 0.231). Epoch 1 is official because it won on the pre-registered criterion and on the full validation split. The epoch-3 test figure is labeled post hoc.
7. **BLEU-4 and ROUGE-L mislead here.** They favour the collapsed Phase 3 model (12.1 BLEU), because its boilerplate matches the modal reference phrasing. They are reported, not used for ranking.
8. **Absolute quality is low for every system**, including CheXOne. Most specific abnormal findings are missed; CE F1 around 0.15–0.18 is far from clinical use.
9. **Out of distribution** (NIH images), the generator is more collapsed: 90 distinct reports for 310 images, including one "opacities in the left lower lobe…" template for 67 of them.

### 10.3 The larger-decoder question (considered and declined)
The student asked whether a bigger language model (e.g. 7B) would write better or longer reports. After analysis it was declined, for four reasons:
- **CheXOne, a 3B model trained on far more data, scored lower** (0.149) than the 1.5B model here.
- **The decoder puts about 1% of its attention on the image**, so a bigger language prior would make reports more fluent, not more correct.
- **v2c's ceiling is set by the classifier.**
- **Report length follows the training references** (37 words mean, 33 median), not model size.

The report may mention this as a reasoned design decision.

---

## 11. Results: explainability (the proposal's Week 10/13 items, quantified)

Source: `docs/explainability_evaluation.md`; `grounding_chestxray14_summary.json`, `sentence_attribution_summary.json`.

### 11.1 Heatmap localisation: pointing game on 984 NIH radiologist boxes (880 images, unseen hospital)

| Method | Hit rate [95% CI] | Macro over 8 findings | Macro excluding Cardiomegaly | Mass ratio | Area-matched IoU |
|---|---:|---:|---:|---:|---:|
| **Classifier attention pooling, 4-model ensemble** | **41.2%** [38.1, 44.3] | 36.8% | 28.7% | 2.68 | 0.183 |
| Classifier attention pooling, single model (seed 0) | 33.7% [30.9, 36.8] | 28.2% | 20.2% | 2.33 | 0.127 |
| *Baseline: always the image centre* | *21.5%* [19.1, 24.2] | *19.3%* | *8.2%* | — | — |
| Grad-CAM, ensemble | 19.0% [16.7, 21.6] | 16.9% | 11.2% | 1.96 | 0.081 |
| Text-grounded, v3b alignment | 11.3% [9.5, 13.4] | 10.9% | 9.2% | 1.25 | 0.088 |
| Grad-CAM, single model | 9.9% [8.1, 11.9] | 8.2% | 6.7% | 1.91 | 0.051 |
| Text-grounded, v2 alignment | 8.7% [7.1, 10.7] | 7.7% | 3.9% | 1.05 | 0.060 |
| *Baseline: random point (expected)* | *7.5%* | *7.1%* | *5.6%* | *1.00* | — |
| Attention rollout (class-agnostic) | 6.4% [5.0, 8.1] | 5.6% | 2.2% | 1.19 | 0.066 |
| Text-grounded, Phase 3 **with** local loss (RQ2) | 5.3% [4.1, 6.9] | 5.6% | 5.3% | 0.95 | 0.034 |
| Text-grounded, Phase 3 **without** local loss (RQ2) | 3.8% [2.7, 5.1] | 3.9% | 3.4% | 0.96 | 0.027 |

Per finding (hit rate %): attention ensemble / Grad-CAM ensemble / text v3b / random / centre, with box counts:

| Finding (boxes) | Attention ens. | Grad-CAM ens. | Text v3b | Random | Centre |
|---|---:|---:|---:|---:|---:|
| Atelectasis (180) | **33.3** | 12.8 | 1.7 | 3.2 | 6.1 |
| Cardiomegaly (146) | 93.8 | 56.8 | 23.3 | 17.6 | **97.3** |
| Effusion (153) | **41.8** | 13.1 | 14.4 | 6.9 | 2.6 |
| Infiltrate (123) (≈ Lung Opacity) | **61.8** | 32.5 | 13.0 | 9.7 | 22.8 |
| Mass (85) (→ Lung Lesion) | 12.9 | 8.2 | 10.6 | 4.2 | 10.6 |
| Nodule (79) (→ Lung Lesion) | 7.6 | 0.0 | 1.3 | 0.5 | 0.0 |
| Pneumonia (120) | **39.2** | 10.8 | 15.0 | 9.4 | 14.2 |
| Pneumothorax (98) | 4.1 | 1.0 | 8.2 | 5.2 | 1.0 |

**Findings:**
1. **The classifier's own attention maps localise real findings.** They hit 41% vs 7.5% random. Excluding Cardiomegaly, whose boxes cover the image centre and inflate the centre baseline to 97%, they hit 28.7% vs 8.2% centre and 5.6% random. For Atelectasis, Effusion, Infiltrate and Pneumonia they are about 4–10× better than chance. This is the first validated explainability claim in the project; before this, the heatmaps had only been eyeballed.
2. **Ensembling improves localisation** (33.7% → 41.2%), as it improves AUROC.
3. **Where the classifier is weak, the maps are at chance.** Pneumothorax is 4%, matching its weakest external AUROC and its 23 training positives. A heatmap cannot explain a decision the model doesn't make.
4. **Grad-CAM, the method the proposal specified, is worse** than the attention weights (19% vs 41%), and noisier across seeds. The attention weights *are* the decision mechanism; Grad-CAM approximates it with gradients.
5. **Class-agnostic attention rollout is at chance** (6.4%). Generic "salient region" maps don't localise a named finding.
6. **RQ2:** the local patch–word loss gives no measurable grounding benefit at this scale. With it 5.3%, without it 3.8%, both below a random point.
7. **RQ4:** text-grounded localisation improves as the alignment improves (Phase 3 5.3% → v2 8.7% → v3b 11.3%; only v3b beats chance), but stays far below the supervised maps. Across *encoders* (§9), localisation ranges from 13% to 32%, a much sharper spread than AUROC.

### 11.2 Sentence attribution: how much does the report writer look at the image?
Each generated report is re-read by the decoder with attention weights exposed. For each sentence, the attention mass on the 64 visual tokens is averaged over all 28 layers × 12 heads. The first position (the "attention sink" every decoder-only model has) is removed. **Scale:** the visual tokens make up about half of the attendable positions, so equal use of image and text would give about 0.5. The prompt's roughly 30 text tokens receive about 0.10.

| Sentence type (377 frontal test images, 2,032 sentences) | n | Visual attention share | Attention on prompt text | Map entropy (1 = uniform) |
|---|---:|---:|---:|---:|
| Names a finding | 100 | **2.7%** | 10.0% | 0.88 |
| Negation only | 459 | 1.3% | 9.9% | 0.93 |
| Other / boilerplate | 1,473 | 1.2% | 11.2% | 0.92 |
| **All** | 2,032 | **1.3%** (90th percentile 2.0%) | | |

- **One glance per image.** On 60 reports, maps of different sentences *within* a report correlate at **0.91** (median 0.93), versus 0.60 between different images. The decoder does not look at different regions for different sentences.
- **When a sentence names a finding that a radiologist boxed** (NIH, 310 images, 320 boxes, only 97 such sentences), its visual map hits the box **54.6%** [45, 64] of the time, versus 32.0% for the centre baseline. Excluding Cardiomegaly: 40.6% vs 4.3%.
- So the visual information is present, because the frozen tower is the classifier's, but the decoder uses it faintly and uniformly. This is the mechanism behind the modal-report share and the v2b collapse, and it motivated v2c.

---

## 12. Ablation: local patch–word loss (Phase 3; RQ2)
Source: `docs/ablation_study.md`. **Caveat:** these were Phase 3 runs, before the fixes. They used old labels, all views, a BiomedCLIP encoder, one seed, validation split, and the **leaky text-fusion classifier**. Treat them as historical.

| Metric | Local loss ON | Local loss OFF |
|---|---:|---:|
| Stage 1 val recall@1 image→text / text→image | 0.0134 / 0.0120 | 0.0174 / 0.0214 |
| Stage 1 val recall@10 image→text | 0.0921 | 0.1322 |
| Stage 2 val macro AUROC (text-fusion protocol) | 0.8645 | 0.8648 |
| Stage 2 val macro AUPRC / macro F1 | 0.3619 / 0.1966 | 0.3549 / 0.2008 |

The per-class split was mixed: the local loss helped focal findings (Consolidation +0.057, Pleural Other +0.036, Fracture +0.032, Lung Opacity +0.030) and hurt rare or diffuse ones (Support Devices −0.066, Enlarged Cardiomediastinum −0.052). The later, quantitative grounding test (§11.1: 5.3% vs 3.8%, both below chance) supersedes the original two-image visual check. **Conclusion for RQ2:** no measurable benefit at this scale; the direction is consistent with the hypothesis.

A second planned ablation (asymmetric loss vs BCE on the baseline) **was not run**. The RQ1 encoder comparison is the project's other controlled ablation.

---

## 13. External systems: what was reproduced vs cited

### 13.1 RadZero (Deepnoid; arXiv:2504.07416; NeurIPS 2025 poster), zero-shot classification
- About 0.2B parameters (DINOv2 vision + mpnet text), CC BY-NC 4.0. Run from public weights in an isolated environment. Pinned to `transformers` 4.46.3; torch had to be newer than the repo's pin for RTX 5080 support.
- **Reproduced on its own OpenI benchmark:** macro AUROC **0.852** over 17 classes, on a disclosed random 1,000-image subsample (seed 0; the full set would take 4+ hours), with ground truth from the project's own keyword mapping of MeSH terms. The paper reports **0.847** on the full set with its own mapping (**reported by the authors, not reproduced here**). The two are independent measurements that agree within 0.5 points. Per-class: 0.616 (Lung Nodule) to 0.995 (Hernia).
- **On MedXVLM's test split:** 0.737 against the old labels (Phase 5), **0.868 against the corrected labels** (same saved predictions, re-scored). This is the RadZero number used throughout.
- **Paper-reported on ChestX-ray14:** 0.804 (not reproduced).

### 13.2 CheXOne (Stanford AIMI; arXiv:2604.00493), report generation
- Qwen2.5-VL-3B-Instruct backbone post-trained with SFT + GRPO; public weights; ran in 7.51 GB of VRAM. Its quantitative benchmarks require PhysioNet-credentialed MIMIC-CXR and gated annotations, so they **were not reproduced**.
- **Run here:** instruct-mode reports for the same 150 test images, scored with MedXVLM's scorer. CE macro F1 0.149 (frontal 70) and 0.107 (all 150); 0% invented devices; 87 distinct outputs.
- **Paper-reported numbers (not reproduced here; different datasets and metrics; context only):**
  - ReXGradient findings generation: 1/RadCliQ 1.116, BERTScore 0.483, BLEU 0.229, RadGraph 0.21.
  - Reader study: reports rated comparable to or better than residents' in 55% of cases; resident reporting time 64.9 ± 34.2 s with CheXOne vs 178.6 ± 116.6 s without.
  - ReXVQA accuracies 0.88–0.99.

### 13.3 Other systems in the literature (context only; no numbers claimed for them)
CheXzero (zero-shot, MIMIC-CXR), GLoRIA (local + global alignment), BioViL (CXR-BERT), CheXagent (multi-task foundation model), LLaVA / LLaVA-Med (visual instruction tuning), R2Gen (report generation on IU X-ray). See §18 for citations.

---

## 14. Master comparison matrix (every comparison in one place)

| # | Comparison | Data | Result | Label |
|---|---|---|---|---|
| C1 | MedXVLM ensemble vs RadZero, classification | 377 Indiana frontal test | 0.868 vs 0.868; Δ 0.000 [−0.033, +0.033] | apples-to-apples |
| C2 | MedXVLM single model vs RadZero | same | 0.837 ± 0.015 (seed 0: 0.855) vs 0.868; Δ(seed 0) −0.014 [−0.058, +0.024] | apples-to-apples |
| C3 | MedXVLM attention head vs fine-tuned baseline heads | same | 0.855 vs 0.777 (same encoder) / 0.804 (ImageNet); both significant | apples-to-apples |
| C4 | Four encoders, same recipe (RQ1) | same + NIH + boxes | 0.837 / 0.802 / 0.792 / 0.784; pointing 32 / 27 / 24 / 13% | apples-to-apples |
| C5 | MedXVLM zero-shot vs RadZero zero-shot | 377 test | 0.734 vs 0.868; Δ +0.135 [+0.091, +0.208] for RadZero | apples-to-apples |
| C6 | MedXVLM home vs external hospital | Indiana vs NIH (25,596) | 0.868 → 0.719 (ensemble) | approximate |
| C7 | MedXVLM vs RadZero on NIH | NIH | 0.719 (measured) vs 0.804 (paper) | RadZero reported by the authors, not reproduced here: **not like-for-like** |
| C8 | RadZero reproduced vs paper, own benchmark | OpenI | 0.852 (1,000-image subsample, own labels) vs 0.847 (full, paper labels) | approximate |
| C9 | MedXVLM report writer vs CheXOne | 150 test images (70 frontal) | CE F1 0.176 (v2 beam) vs 0.149; within noise; devices 3% vs 0% | apples-to-apples |
| C10 | MedXVLM v2c vs v2 vs classifier ceiling | same | 0.165 vs 0.176 vs 0.234; devices 0% vs 3% | apples-to-apples |
| C11 | Before vs after the rebuild, classification | 377 test | 0.572 (Phase 3 image-only) → 0.855 / 0.837 ± 0.015 / 0.868 | apples-to-apples |
| C12 | Before vs after, report generation | 150 test | 5 → 61 distinct; CE 0.078 → 0.152 (0.176 beam) | apples-to-apples |
| C13 | Heatmap methods vs baselines | 984 NIH boxes | attention 41% > centre 21.5% > Grad-CAM 19% > text 11% > random 7.5% > rollout 6.4% | approximate (out-of-distribution) |
| C14 | Local loss on vs off (RQ2) | boxes; Phase 3 val | 5.3% vs 3.8% pointing; 0.8645 vs 0.8648 AUROC | apples-to-apples (historical protocol for AUROC) |
| C15 | Greedy vs beam decoding | 150 test | v2: 0.152 → 0.176, devices 10% → 3%; v2c: beam worse (0.135 vs 0.165) | apples-to-apples |
| C16 | Phase 3 with the leak vs image-only | Phase 5 test (748, old labels) | 0.851 with ground-truth text vs 0.512 image-only | demonstrates the leak |
| C17 | Labels before vs after fix, as judged by RadZero | 748 all-view / 377 frontal | label fix alone: 0.737 → 0.824 (748) and 0.780 → 0.868 (377) | independent validation |

---

## 15. Limitations, threats to validity, and engineering challenges

### 15.1 Scientific limitations (state these in the report)
1. **Small, single-site data.** 3,030 frontal training images from one outpatient hospital. The external drop (0.868 → 0.719) shows the limits of generalisation.
2. **Thin test support for rare findings.** Pneumothorax has 1 test positive, Pleural Other 3, Consolidation 5, and Pneumonia, Fracture and Support Devices 8 each. Per-class numbers for these are unreliable, which is why the 5-stable-class macro is also reported.
3. **Labels come from a rule-based labeler, not radiologists.** The labeler is validated against 24 hand-written cases, and one known residual pattern is left unfixed. All metrics are relative to these automatic labels.
4. **Wide confidence intervals.** Classification CIs are about ±0.04–0.05 per system; report-level CE F1 CIs are about ±0.10 on 70 images. Many differences are within noise.
5. **RQ1 uses one run per new encoder.** The BiomedCLIP-vs-CLIP contrast is unresolved. The paired tests use X-ray DINOv2's best seed.
6. **Localisation is validated only out of distribution** (NIH boxes). Indiana has no box annotations, and no reader study was done.
7. **Report generation is weak in absolute terms** for every system tested. The report writer mostly paraphrases and barely uses the image. v2c removes invented devices but inherits the classifier's misses.
8. **Zero-shot is data-limited** (0.734 vs 0.868). Retrieval was evaluated only on the validation split.
9. **Not all planned metrics were computed.** RadGraph F1, RadCliQ and METEOR are missing, and the ASL-vs-BCE ablation was not run.
10. **External-model comparisons are partial.** CheXOne ran in instruct mode, not its stronger reasoning mode, and its own benchmarks weren't reproduced. The RadZero reproduction used a 1,000-image subsample and a custom label mapping.
11. **Model selection.** Checkpoints were chosen on the validation split, and the v2c epoch-3 test number is labeled post hoc. **But the decoding strategy was not:** beam vs greedy was compared on the 70-image test subsample itself (§10). The pre-specified protocol was greedy decoding (v2: 0.152). Describe 0.176 as the best of the configurations compared on the test subsample (a post hoc choice).

### 15.2 Engineering challenges and how they were solved (good material for a "Challenges" section)
- **New GPU architecture.** The RTX 5080 (Blackwell, `sm_120`) needed PyTorch builds newer than the external repositories pinned, so versions were adjusted and documented in isolated virtual environments.
- **Disk exhaustion.** The 45 GB NIH download silently filled the Windows drive hosting WSL2, because `df` inside WSL reports the virtual disk's nominal 1 TB size. The machine rebooted and killed jobs. Fixed by pruning about 100 GB (only the 25,596 NIH test images were kept) and compacting the virtual disk.
- **GPU memory under WSL2:**
  - The allocator setting `expandable_segments` crashed near-capacity runs ("device not ready").
  - Windows applications holding VRAM caused host-side out-of-memory (`dxgkio_make_resident -12`).
  - Running at 15.9 of 16.3 GB made WSL page GPU memory silently: 70 min per epoch at batch 4 vs **10.5 min at batch 2 × gradient accumulation 8**, with the same effective batch.
- **Long jobs surviving interruptions.** Detached processes (`setsid`), serialized chains with one GPU job at a time, log markers, and a `--resume` option. A WSL restart killed the v2c run after epoch 1, and it was resumed from its checkpoint.
- **Streamlit and caching.** Caching a result object broke after a code hot-reload (class identity changed during pickling). Fixed by caching plain values.

### 15.3 Lessons learned (useful for the conclusion)
1. Always evaluate with exactly the inputs available at deployment (image only). Fusing the answer text hid a chance-level model behind 0.865.
2. Check the labels before the model: a 5-token negation bug created thousands of false positives.
3. Report seed variance. A single lucky seed overstated the model by about 0.02.
4. Validation loss can select a collapsed generator. Use generation-based selection.
5. Decoding choices can matter as much as training (beam search removed two-thirds of the hallucinations).
6. Validate explanations quantitatively. The textbook method (Grad-CAM) lost to the model's own attention.
7. Measure *why* something fails (sentence signal density; decoder attention share), not just that it fails.

---

## 16. Numbers NOT to use, and inconsistencies in older documents

| Do not quote | Why | Use instead |
|---|---|---|
| **0.865** macro AUROC (Phase 3 headline, `training_summary.md`) | Leaky: the classifier saw the ground-truth report | 0.837 ± 0.015 single / 0.868 ensemble |
| 0.851 (Phase 5 "text-fusion") | Same leak | — (cite only as evidence of the leak) |
| **0.855** alone as "the" classifier accuracy | Seed 0, the best of four | 0.837 ± 0.015, or 0.868 ensemble; 0.855 only as "seed 0" |
| Phase 5 tables (RadZero 0.737, MedXVLM blind 0.512 on 748 images, CE F1 0.075 vs 0.105) | Old, buggy labels; all views | The corrected v2 tables (§8, §10) |
| 0.694 baseline (Phase 3) | Old labels, 5 epochs, all views | 0.804 / 0.777 v2 baselines |
| "Linear probe" for the baselines | Misnomer: the encoders were fully fine-tuned | "Baseline head (fine-tuned encoder + linear layer)" |
| "MedXVLM beats CheXOne" | 0.176 vs 0.149 is within the ±0.10 noise band | "comparable to CheXOne on CE F1" |
| "MedXVLM beats RadZero" | Δ 0.000, CI ±0.033 | "statistically indistinguishable" |
| Qwen2.5-**0.5B** (mentioned in the literature review's LLaVA-Med and LoRA paragraphs) | The proposal upgraded to **1.5B** during Phase 2 | Qwen2.5-1.5B-Instruct |
| "Stage 1 initialised from BiomedCLIP" as the final design | True for Phase 3 only; v2 uses X-ray DINOv2 | Describe both; final = X-ray DINOv2 |
| Report CE F1 bootstrap values (0.223 etc.) next to the table values (0.176 etc.) | Different macro definitions (supported classes vs all 14) | Use one definition per table and say which |
| Pneumothorax per-class AUROC as a finding | One positive test image | Mention only as a caveat |

---

## 17. Suggested structure for the final report (with sources in this document)

### 17.1 Report outline
1. **Title page, abstract.** Abstract of about 250 words from §2 and the one-sentence summary.
2. **Introduction.** Problem and motivation (§4.1); objectives (§4.2); research questions (§4.3); contributions, stated honestly (§4.1 last paragraph); report organisation.
3. **Literature review.** Themes from §18: supervised CXR classification; contrastive vision-language pretraining; zero-shot; report generation; explainability; parameter-efficient fine-tuning. Close with the gap: integration at small scale.
4. **Dataset.** §5 (statistics, split, labeler, label distribution, the negation fix, NIH external set and boxes).
5. **Methodology.** §7: architecture diagram, Stage 1 / 2 / 3, explainability methods, training details, evaluation metrics and protocol, statistical testing.
6. **Implementation.** Code structure (§19.2), environment and hardware (§1), demo app (§19.1), engineering challenges (§15.2).
7. **Experiments and results.**
   - 7.1 Development history and the corrections (§6, compressed).
   - 7.2 Classification (§8.1–8.3, with CIs).
   - 7.3 External validation (§8.5).
   - 7.4 Encoder comparison, RQ1 (§9).
   - 7.5 Zero-shot (§8.4).
   - 7.6 Report generation (§10).
   - 7.7 Explainability (§11).
   - 7.8 Ablations (§12, §10's v2b / v2c ablations).
   - 7.9 Comparison with published systems (§13, §14).
8. **Discussion.** Answers to RQ1–RQ4 (§4.3); what worked and why; limitations (§15.1); lessons (§15.3).
9. **Conclusion and future work** (§20).
10. **References** (§18). **Appendices:** hyperparameter tables (§7), per-class tables, compute table (§7.6), reproduction commands (§19.3).

### 17.2 Figures and tables to include
Kartik can supply these image files from the repository:
- **Demo screenshots:** `docs/assets/app_analyze.jpg`, `docs/assets/app_model_card.jpg`.
- **Heatmap vs radiologist-box examples** (4 methods side by side): `docs/phase5_artifacts/v2/grounding_examples/*.jpg`.
- **Sentence-attribution examples:** `docs/phase5_artifacts/v2/sentence_attribution_examples/*.jpg`.
- **Plain-language summary PDF (14 pages):** `docs/project_summary_plain_language.pdf`, which has bar-chart figures.

Suggested figures to draw:
1. The three-stage architecture diagram (§7).
2. Classification bar chart with CI error bars (§8.1).
3. Encoder comparison: AUROC and pointing game side by side (§9).
4. Pointing-game bar chart with random and centre baselines (§11.1).
5. Report-generation comparison, CE F1 with the ceiling line (§10.1).
6. The before-and-after "correction timeline" (§6).
7. Label prevalence before and after the fix (§5.1).
8. The external domain-shift per-class chart (§8.5).

Key tables: §8.1, §8.2, §9, §10.1, §11.1, §14.

### 17.3 Slide outline (12–15 slides)
1. Title.
2. Problem and motivation.
3. The three-stage system.
4. Data and the labeling fix.
5. The honesty story (leak → rebuild).
6. Classification = RadZero (with CI).
7. External hospital.
8. RQ1: encoders.
9. Report generation vs CheXOne, plus the hallucination fix.
10. Findings-conditioning and its ceiling.
11. Explainability validated on radiologist boxes.
12. Report writer barely looks at the image.
13. Demo.
14. Limitations.
15. Conclusions and future work.

---

## 18. References (verify formatting before submission)
From the project's literature review, plus the methods actually used. Kartik should double-check venues and years in the final bibliography; entries marked † were added for methods used after the literature review was written.

- Wang, Peng, Lu, Lu, Bagheri, Summers (2017). *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks…* CVPR. (ChestX-ray14 and its bounding boxes.)
- Rajpurkar, Irvin, Zhu et al. (2017). *CheXNet.* arXiv:1711.05225.
- Irvin, Rajpurkar, Ko et al. (2019). *CheXpert.* AAAI.
- Smit, Jain, Rajpurkar et al. (2020). *CheXbert.* EMNLP.
- Radford, Kim, Hallacy et al. (2021). *CLIP: Learning transferable visual models from natural language supervision.* ICML.
- Zhang, Jiang, Miura, Manning, Langlotz (2020/2022). *ConVIRT.* arXiv:2010.00747; MLHC 2022.
- Huang, Shen, Lungren, Yeung (2021). *GLoRIA.* ICCV.
- Boecking, Usuyama, Bannur et al. (2022). *BioViL.* ECCV.
- Zhang, Xu, Usuyama et al. (2023). *BiomedCLIP.* arXiv:2303.00915 (NEJM AI 2025).
- Tiu, Talius, Patel, Langlotz, Ng, Rajpurkar (2022). *CheXzero.* Nature Biomedical Engineering.
- Park, Yoon, Kim, Choi (2025). *RadZero.* arXiv:2504.07416 (NeurIPS 2025 poster).
- CheXOne (2026). arXiv:2604.00493. Stanford AIMI; model `StanfordAIMI/CheXOne`. (Verify the full title and author list from arXiv.)
- Chen, Song, Chang, Wan (2020). *R2Gen.* EMNLP.
- Liu, Li, Wu, Lee (2023). *LLaVA: Visual instruction tuning.* NeurIPS.
- Li, Wong, Zhang et al. (2023). *LLaVA-Med.* NeurIPS Datasets & Benchmarks.
- Chen, Varma, Xu et al. (2024). *CheXagent.* arXiv:2401.12208.
- Selvaraju et al. (2017). *Grad-CAM.* ICCV.
- Abnar, Zuidema (2020). *Quantifying attention flow in transformers* (attention rollout). ACL.
- Hu, Shen, Wallis et al. (2021/2022). *LoRA.* ICLR 2022.
- † Oquab et al. (2023). *DINOv2: Learning robust visual features without supervision.* arXiv:2304.07193. (The X-ray DINOv2 encoder is `StanfordAIMI/dinov2-base-xray-224` on Hugging Face.)
- † Ben-Baruch, Ridnik, Zamir et al. (2021). *Asymmetric loss for multi-label classification.* ICCV.
- † Gu et al. (2021). *Domain-specific language model pretraining for biomedical NLP* (PubMedBERT / BiomedBERT). ACM Transactions on Computing for Healthcare.
- † Qwen Team (2024). *Qwen2.5 technical report.* arXiv:2412.15115.
- † Yao et al. (2021). *FILIP: Fine-grained interactive language-image pre-training.* (Local-loss formulation.)
- † Zhang, Bargal, Lin, Brandt, Shen, Sclaroff (2016/2018). *Top-down neural attention by excitation backprop* (the pointing game). ECCV 2016 / IJCV 2018.
- † Xiao et al. (2023). *Efficient streaming language models with attention sinks.* (Why the first position is excluded in sentence attribution.)
- † Efron (1979), bootstrap; Wilson (1927), score interval for proportions.
- Datasets: Demner-Fushman et al. (2016), *Preparing a collection of radiology examinations for distribution and retrieval*, JAMIA (Indiana / Open-i). Kaggle mirrors `raddar/chest-xrays-indiana-university` and `nih-chest-xrays/data`.

---

## 19. Demo application, code and reproducibility

### 19.1 Streamlit demo app (proposal Week 14)
- **Two pages.** "Analyze an X-ray" and "Model card", with a light, minimal theme.
- **Analyze page:**
  - Pick a held-out test image (optionally only abnormal studies, or at random) or upload a frontal X-ray.
  - Shows the 4-model ensemble's 14 probabilities, each class's validation-chosen threshold, the model's call vs the radiologist's label, and a summary line ("Model flags" vs "Radiologist").
  - Heatmaps in three methods (attention, Grad-CAM, text-grounded), with their validated accuracies stated. The heatmaps use a "focus" overlay: only high-attention regions are coloured.
  - Report generation with a choice of model (v2 or v2c) and decoding (beam or greedy; each defaults to its measured best). The model's report appears beside the radiologist's; v2c also shows the findings it was given.
  - Optional per-sentence attention maps.
  - The report model loads on first use (about 7 GB of VRAM); the classifier ensemble needs about 1.4 GB.
- **Model card page:** every headline number, read live from the evaluation files. Includes comparison charts against RadZero, CheXOne and the baselines, the encoder comparison, pointing-game results, and a limitations tab.
- **Testing:** an automated headless test (Streamlit AppTest) covers 7 user paths. All pass.
- **Disclaimer:** shown throughout: "research prototype, not a medical device".

### 19.2 Code structure
```
medxvlm/
  src/medxvlm/
    data/        dataset classes, rule-based labeler, findings-prompt builder
    models/      vision encoder (4 backbones), text encoder, contrastive alignment, classifier head, report generator, baselines
    engine/      config loading, checkpointing, logging
    eval/        metrics
    explain/     attention overlays, Grad-CAM / rollout / text grounding, sentence attribution
    inference.py MedXVLMPipeline (ensemble classify, heatmaps, generate_report, attribute_report)
  scripts/       prepare_data, validate_labeler, train_{align,classifier,baseline,report}, eval_* (classification, bootstrap, RQ1,
                 external NIH, grounding, sentence attribution, report generation + scoring), run_* chains, sanity checks
  configs/       YAML recipes (Phase 3, v2, v2b, v2c, v3, v3b)
  app/           Streamlit demo (app.py, app_pages/, common.py, theme)
  docs/          proposal, literature review, phase write-ups, v2_rebuild (main results), explainability evaluation,
                 status reports, plain-language PDF, all result files under phase5_artifacts/v2/
```
Checkpoints (about 11 GB) and data are not in the repository.

### 19.3 Reproducibility
- Seeded, study-level split; the label fix was verified to leave the split unchanged.
- Every metric is computed by a script whose output file is committed.
- Paired bootstrap with a fixed seed.
- External-model environments are documented with exact commit hashes and versions.
- Training commands: `scripts/run_v2_rebuild.sh` (v2), `scripts/run_rq1_backbones.sh` (RQ1), `scripts/run_v2c_findings_conditioned.sh` (v2c).

---

## 20. Future work (ranked by expected value)
1. **More data.** CheXpert Plus (about 223k image–report pairs; free after registration) is the only lever expected to substantially improve every stage: rare findings, zero-shot and report grounding. Cost: days of work and full retraining.
2. **Report-grounding objectives.** Make the decoder use the image (more visual tokens, an unfrozen projector with a grounding auxiliary loss), since it currently puts about 1% of its attention there.
3. **Better report metrics.** RadGraph F1 / RadCliQ, and a small radiologist reader study.
4. **Tighten RQ1.** Second seeds for BiomedCLIP and CLIP (about 1.7 h of GPU).
5. **In-distribution localisation labels, and calibration.** Probability calibration and threshold validation for clinical-style operating points.
6. **Finish the planned comparisons.** The ASL-vs-BCE ablation; RadZero on the NIH boxes and classes under a like-for-like protocol.

---

## 21. Glossary (for the non-specialist summary)
- **AUROC:** probability that the model ranks a random positive case above a random negative one; 0.5 = coin flip, 1 = perfect.
- **Macro average:** mean over classes, so each finding counts equally.
- **Confidence interval (95%):** the range the true value plausibly lies in, given this test set.
- **Paired bootstrap:** re-drawing the same test images thousands of times and scoring both systems on each draw, to test whether their difference is real.
- **Seed / ensemble:** a model trained with different random starting points / the average of several such models.
- **Zero-shot:** classifying by comparing the image with a text description, with no labeled training for that task.
- **Contrastive alignment (InfoNCE):** training image and text encoders so that matching pairs end up close together in a shared space.
- **LoRA:** training small low-rank adapter matrices instead of the whole language model.
- **Clinical-efficacy F1:** findings extracted from a generated report and compared with the true findings.
- **Hallucination (here):** the report mentions a device or surgery that the radiologist did not report.
- **Pointing game:** does a heatmap's hottest point fall inside the radiologist's box?
- **Attention sink:** the first token, which decoder language models attend to by default; it is excluded when measuring attention to the image.
- **Mode collapse (report generation):** producing the same report regardless of the image.
- **Domain shift:** performance drop when moving to data from a different hospital or scanner.

---

---

## 22. Corrections and additions (2026-09-27)

These answer an audit of the briefing; the fixes are also applied in the sections above.

**Decisions:**
- **Decoding was chosen on test data.** Beam vs greedy was compared on the 70-image test subsample (the pre-specified protocol was greedy). Report v2 greedy **0.152** as the pre-specified result and v2 + beam **0.176** as "the best of four configurations compared on the test subsample (post hoc)". Also post hoc: "greedy is best for v2c". The v2c *checkpoint* (epoch 1) was selected on validation, with greedy decoding, so v2c greedy **0.165** is its pre-specified number.
- **RadZero label check** (same saved predictions, recomputed 2026-09-27):

  | | All 748 test images | 377 frontal |
  |---|---:|---:|
  | Old labels | 0.737 | 0.780 |
  | Corrected labels | 0.824 | 0.868 |

  The label fix alone is worth about +0.09; the frontal restriction about +0.04.
- **Before/after:** use **0.572 → 0.855** (same 377 images; C11) for the rebuild. Show the leak as **0.851 vs 0.512** on the Phase 5 set (748 all-view images, old labels; C16).
- **RQ4(b):** "improves across successive alignment versions", not "with alignment quality".
- **Zero-shot "best":** v3b is best on the 8 shared (0.734) and all 13 (0.652, vs Phase 3 0.562); Phase 3 is best on the 5 stable classes (0.708 vs 0.693). Report both.

**Fixes:**
- **Head contribution:** +0.078 uses seed 0, the best of 4. The seed-mean version is 0.837 − 0.777 = +0.060, so quote "+0.06 to +0.08". The baselines are single runs.
- **Imbalance:** 10 of 14 classes are under 5% training prevalence.
- **Pneumothorax:** 23 positive studies in the training split (25 counting uncertain); 27 in the whole corpus.
- **RadZero classes:** 18 prompts/classes in its vocabulary. The OpenI macro is over **17** because "Abnormal Lesion" had 0 positives in the 1,000-image subsample. Both numbers are right in context.
- **Pointing-game multiplier:** "about 4–10× better than chance".
- **Duplicate rows (v2b vs v2c without findings): not an error.** Both models output the identical sentence ("The heart is normal in size. The mediastinum is unremarkable. The lungs are clear. No acute disease.") for all 150 images (verified per image), so every metric matches.
- **Normal rate:** drop the "~2/3 published" claim; it was wrong. Demner-Fushman et al. (2016) report 1,526 of 3,996 reports (38%) as normal "as judged by 2 coders", under a broader definition that counts non-CheXpert findings. Not comparable to 69.1% CheXpert "No Finding".
- **"Shared backbone":** agreed. The final system shares one vision tower between Stage 2 and Stage 3 (Stage 3 uses the seed-0 classifier's fine-tuned X-ray DINOv2). Stage 1 is a parallel branch: v2 froze that tower, so Stage 1 passes it through unchanged, and the alignment powers only zero-shot and text-grounded maps. The proposal's premise (one contrastive alignment as the backbone for both tasks) was only partly realised.
- **Rounding:** paired Δs are computed from unrounded AUROCs and may differ by ±0.001 from subtracting the rounded values.

**Missing information:**
- **Planned 16-week schedule** (proposal §6):

  | Week | Plan |
  |---|---|
  | 1–2 | Skeleton, environment, toy loader, literature review, proposal |
  | 3–4 | Data pipeline, study-level split, rule-based labeler |
  | 5–6 | Stage 1 (BiomedCLIP init, global + local loss), retrieval/zero-shot checks |
  | 7 | Stage 1 evaluation (RQ1/RQ2 groundwork) |
  | 8–9 | Stage 2 (attention pooling, gated fusion) vs vision-only baseline |
  | 10 | Grad-CAM / rollout on Stage 2; localisation checks (RQ2) |
  | 11–12 | Stage 3 (LLaVA-style projection, Qwen2.5-1.5B + LoRA) |
  | 13 | Stage 3 evaluation (BLEU/ROUGE + clinical checks), sentence attribution (RQ3/RQ4) |
  | 14 | Streamlit demo |
  | 15 | Ablations for RQ1–RQ4, error analysis |
  | 16 | Buffer, final report, presentation |

  Actual: see §3. Phases 1–3 finished by 2026-09-05. The evaluation phases (4–5) then triggered the v2 rebuild. Grad-CAM/rollout (planned week 10) and the RQ ablations (week 15) happened last (2026-09-21 and 09-26).
- **v3 / v3b hyperparameters:** both use batch 32, encoder LR 1e-5 (vision and text both fine-tuned), projection/temperature LR 1e-3, weight decay 0.01, max text length 64, frontal images only, **local loss off**, selection by validation loss, early-stopping patience 3, up to 12 epochs.
  - v3: all sentences, multi-positive InfoNCE; best epoch 10.
  - v3b: finding sentences + canonical normal sentence; best epoch 5 of 8.
- **v2c:** up to 5 epochs, early-stopping patience 2 on the generation criterion. It ran 3 epochs (epoch 1 at batch 4 × accumulation 4; epochs 2–3 at batch 2 × 8; effective batch 16 throughout); best epoch 1. It **kept 50% abnormal oversampling**. Findings dropout 0.1; LR 2e-4. Generation-based selection used 48 validation images with max 96 new tokens.
- **The 310 NIH images for sentence attribution:** stratified random sample of **40 boxes per finding** × 8 findings = 320 boxes (pandas `sample`, `random_state=0`). Some images carry several boxes, which leaves 310 unique images.
- **Classes with F1-optimal thresholds** (≥10 frontal validation positives, with count): No Finding (258), Lung Opacity (51), Cardiomegaly (39), Atelectasis (37), Edema (21), Pleural Effusion (15), Lung Lesion (14), Support Devices (11). The other six (Enlarged Cardiomediastinum 1, Pneumothorax 3, Pleural Other 4, Consolidation 8, Pneumonia 8, Fracture 9) use the median threshold 0.538.
- **Licences:**

  | Resource | Licence / terms | Source |
  |---|---|---|
  | Indiana University / Open-i collection (Kaggle mirror `raddar/chest-xrays-indiana-university`) | **CC BY-NC-ND 4.0**; cite the original Open-i source | Hugging Face mirror card [verify on the Kaggle page] |
  | NIH ChestX-ray14 | "The usage of the data set is unrestricted. But you should provide the link to our original download site, acknowledge the NIH Clinical Center and provide a citation to our CVPR 2017 paper." | NIH FAQ file shipped with the dataset |
  | X-ray DINOv2 (`StanfordAIMI/dinov2-base-xray-224`) | Apache-2.0 | project docs |
  | RadZero weights | CC BY-NC 4.0 | project docs |
  | CheXOne | Apache 2.0 framework; model/data follow original sources | project docs |
  | Qwen2.5-1.5B-Instruct, BiomedCLIP, BiomedBERT, OpenAI CLIP, timm ViT | [verify on each model card] | — |

**Reference entries** (checked 2026-09-27):
- **CheXOne:** Y. Zhang, C. Wang, Y. Gao, J. Liu, M. Varma, J. Xu, S. Ostmeier, J. Long, S. Gatidis, S. Dehkharghani, A. Michalson, E. K. Hong, C. Bluethgen, H. H. Guo, A. V. Ortiz, S. Altmayer, S. Bodapati, J. D. Janizek, K. Chang, J.-B. Delbrouck, A. S. Chaudhari, C. P. Langlotz, "A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation," arXiv:2604.00493, Apr. 2026.
- **CheXpert Plus:** P. Chambon, J.-B. Delbrouck, T. Sounack, S.-C. Huang, Z. Chen, M. Varma, S. Q. H. Truong, C. T. Chuong, C. P. Langlotz, "CheXpert Plus: Augmenting a Large Chest X-ray Dataset with Text Radiology Reports, Patient Demographics and Additional Image Formats," arXiv:2405.19538, 2024.
- **MIMIC-CXR:** A. E. W. Johnson, T. J. Pollard, S. J. Berkowitz, N. R. Greenbaum, M. P. Lungren, C.-y. Deng, R. G. Mark, S. Horng, "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports," *Scientific Data*, vol. 6, 317, 2019. [verify]
- **Indiana dataset paper:** D. Demner-Fushman, M. D. Kohli, M. B. Rosenman, S. E. Shooshan, L. Rodriguez, S. Antani, et al., "Preparing a collection of radiology examinations for distribution and retrieval," *JAMIA*, vol. 23, no. 2, pp. 304–310, 2016.

*End of briefing. If anything the report needs is missing here, ask Kartik rather than filling the gap: the underlying files are listed in each section's "Source" line.*
