# v2 Rebuild — Fixing the Image-Only Classifier (post-Phase 5)

Date: 2026-09-17
Trigger: Phase 5 ([`cross_dataset_evaluation.md`](cross_dataset_evaluation.md)) showed MedXVLM's Stage 2 classifier scored **0.51 macro AUROC** when tested honestly (image only) — chance level — and that the 0.865 headline in [`training_summary.md`](training_summary.md) depended on feeding the classifier the ground-truth report. The goal of this rebuild was to get a *real*, blind, image-based number as high as legitimately possible, and re-compare against RadZero on identical images.

Every number here is on data the models never trained on, with no report text given to any classifier. The Phase 3 checkpoints are untouched (`checkpoints/{align,classifier,baseline}/`); everything new lives under `*_v2/`.

## 1. Root causes found, and what was changed

| # | Cause | Evidence | Fix |
|---|---|---|---|
| 1 | **Stage 2 was trained with the answer key.** `train_classifier.py` fused the *paired ground-truth report's* text embedding into the classifier at train and val time. The head learned to read labels out of the text, not the image. | Blind re-test: 0.51. Same checkpoint with text: 0.83. | New `use_text_fusion: false` mode — the text encoder is never built; the head sees only patch tokens (`configs/classifier_v2.yaml`). |
| 2 | **Labeler negation-scope bug → massive label noise.** The rule-based labeler's negation window was 5 tokens, so in *"No focal consolidation, pleural effusion, or pneumothorax"* the last item(s) escaped negation and were labelled **positive**. Cue matching was also substring-based ("no" matched inside "nodular"/"normally"). | Study-level positive rate, before → after fix: Pneumothorax **10.2% → 0.8%**, Pleural Effusion **19.6% → 4.9%**, Consolidation 2.8% → 1.0%, Pneumonia 3.8% → 2.2%, No Finding 51% → 69%. (The dataset paper reports 38% of reports as normal, but under a broader coder definition that counts non-CheXpert findings such as granulomas, so it is not comparable; an earlier version of this doc wrongly cited a "published ~2/3 normal rate".) Other classes moved <1 point — the fix is targeted. Independent check: RadZero's fixed, saved predictions agree far better with the corrected labels: at a fixed image set the label fix alone raises its macro AUROC by about +0.09 (all 748 test images 0.737 → 0.824; 377 frontal 0.780 → 0.868). The often-quoted 0.737 → 0.868 also includes the frontal-only restriction (+0.04), so it overstates the label effect (corrected 2026-09-27). | `_pre_cue_in_scope()` in `src/medxvlm/data/labeler.py`: a leading cue governs the whole clause until a scope-reset token (`is/are/there/seen/…`); word-boundary cue matching. 6 regression cases added to `scripts/validate_labeler.py` (24/24 pass, the original 18 unchanged). Manifest regenerated; **split verified byte-identical** to Phase 3 (uid-level, seed 0), so no leakage was introduced. |
| 3 | **Stage 1 never converged.** Phase 3's val loss was flat across 5 epochs (4.73 → 4.68) while train loss fell — memorising 5.9k pairs. The zero-shot head sits on top of this. | `runs/align/local_on/metrics.csv` | Frozen CXR-pretrained tower (see 4), higher LR on the fresh projection layers, 15 epochs, frontal-only. v2 val loss actually falls: 5.96 → 4.67 by epoch 7. |
| 4 | **Weak visual features for CXR.** BiomedCLIP-initialised ViT-B/16, fine-tuned end-to-end on ~6k images. | — | `VisionEncoder(backbone="xraydinov2")`: `StanfordAIMI/dinov2-base-xray-224` (Apache-2.0, ViT-B/14 self-supervised on chest X-rays — the same tower RadZero uses). Backward-compatible: old checkpoints auto-detected by key layout. |
| 5 | **Half the training images were lateral views**, which carry little signal for most labels and inherit the frontal image's report. | 2,899 of 5,929 train rows | `projections: [Frontal]` in all v2 configs → 3,030 training images. |
| 6 | **Severe class imbalance** (10 of 14 classes under 5% prevalence with the corrected labels). | §1 of this doc | Asymmetric Loss (already in the codebase as an ablation option) instead of plain BCE; early stopping on val macro AUROC; split LR (encoder 1e-5, head 1e-3); cosine schedule. |
| 7 | "XXXX" de-identification tokens in report text (Stage 3 learned to emit them). | `training_summary.md` | Stripped in `scripts/prepare_data.py` (0 remaining). |

## 2. Training runs (all on the corrected labels, frontal views, same split)

Exact commands: `scripts/run_v2_rebuild.sh` (serialized; one GPU job at a time).

| Run | Config | Best val macro AUROC | Best epoch | Wall time |
|---|---|---:|---:|---:|
| Stage 1 v2 (XrayDINOv2 frozen + BiomedBERT, local loss on) | `align_v2.yaml` | val loss 4.67 / recall@1 0.052 | 7 (loss) / 15 (recall) | 36.9 min |
| **Stage 2 v2, image-only, ASL** | `classifier_v2.yaml` | **0.833** | 8 (early-stopped at 14) | 33.8 min |
| Baseline v2: ImageNet ViT-B/16 + linear, BCE | `baseline_v2.yaml --backbone vit_base_patch16_224` | 0.781 | 7 | 25.9 min |
| Baseline v2: XrayDINOv2 + linear, BCE | `baseline_v2.yaml --backbone xraydinov2` | 0.776 | 7 | 26.1 min |
| **Stage 3 v2**, LoRA on Qwen2.5-1.5B, frontal-only, abnormal-oversampled 50% (run 2026-09-19) | `report_gen_v2.yaml` | val loss **1.132** | 3 (early-stopped at 5) | 278.6 min |
| Stage 3 v2b, as v2 but 30% abnormal (2026-09-20) | `report_gen_v2b.yaml` | val loss 1.108 (but fully collapsed — §3.6.4) | 2 (early-stopped at 4) | 131.0 min |
| Stage 2 v2 seeds 1, 2, 3 (2026-09-20) | `classifier_v2.yaml --seed N` | 0.828 / 0.850 / 0.810 | 8 / 9 / 8 | 29.7 / 34.6 / 39.2 min |
| Stage 1 v3 sentence-level multi-positive (2026-09-20) | `align_v3_sentence.yaml` | val loss 2.94 | 10 | 26.1 min |
| Stage 1 v3b finding-sentences only (2026-09-20) | `align_v3b_findings.yaml` | val loss 0.99 (different loss scale) | 5 (early-stopped at 8) | 17.1 min |
| Stage 2 RQ1 encoders: BiomedCLIP / OpenAI CLIP / ImageNet ViT, seed 0 (2026-09-26) | `classifier_v2.yaml --vision-backbone …` | 0.794 / 0.802 / 0.764 | 15 / 18 / 7 | 45.0 / 50.8 / 27.5 min |
| **Stage 3 v2c**, as v2 + findings-conditioned prompt, selection by generation CE F1 (2026-09-21) | `report_gen_v2c.yaml` | val gen CE F1 **0.322** (val loss 1.152); full-val re-check 0.270 | 1 (early-stopped at 3) | 70.4 min (ep 1, batch 4) + 21.3 min (ep 2–3, batch 2×8) — §3.7 |

Stage 2 v2 overfits quickly (train macro AUROC 0.98 by epoch 14 while val slid from 0.83 to 0.79) — early stopping did its job, but ~3k training images is the regime we are in.

**Stage 1 checkpoint-selection note (stated, not hidden):** `train_align.py` selected `best.pt` by mean retrieval recall@1, which on a ~380-pair val set moves by 2–4 images between epochs — noise. It picked epoch 15, the most overfit epoch (val loss 5.97 vs 4.67 at epoch 7). Because the vision tower was **frozen** in Stage 1, this choice cannot affect Stage 2 (verified: Stage 1's saved vision weights are bit-identical to the pretrained XrayDINOv2). Two further Stage 1 variants were run only to test whether the weak zero-shot head (§3.3) was a selection artefact: `--select-by val_loss` (`align_v2/local_on_valloss/`, picks epoch 5, 20.7 min) and additionally `--freeze-text-encoder` (`align_v2/local_on_frozen_text/`, projections only, 0.8M trainable, 21.3 min). Neither helped — see §3.3.

## 3. Held-out test results (frontal test split, n = 377, corrected labels, image-only)

Script: `scripts/eval_v2_classification.py` → `docs/phase5_artifacts/v2/`. RadZero is **re-scored from its Phase 5 saved per-image probabilities** (`external_repro/radzero_on_medxvlm_testsplit_raw.csv`) against the corrected labels on this frontal subset — no re-run, no new inference.

### 3.1 Headline

| System | 8 shared classes | 5 shared classes with ≥10 positives | All 13 valid CheXpert classes |
|---|---:|---:|---:|
| **MedXVLM v2 Stage 2, image-only** | **0.855** | **0.862** | **0.829** |
| RadZero zero-shot (re-scored, identical images/labels) | 0.868 | 0.875 | n/a (no prompts for the other classes) |
| Baseline v2: ImageNet ViT + linear | 0.804 | 0.817 | 0.738 |
| Baseline v2: XrayDINOv2 + linear | 0.777 | 0.795 | 0.734 |
| Phase 3 Stage 2, blind — the "before" | 0.572 | 0.525 | 0.547 |
| Phase 3 Stage 1 zero-shot | 0.669 | 0.708 | 0.562 |
| MedXVLM v2 Stage 1 zero-shot (recall@1-selected, epoch 15) | 0.565 | 0.547 | 0.527 |
| MedXVLM v2 Stage 1 zero-shot (val-loss-selected, epoch 5) | 0.572 | — | 0.504 |
| MedXVLM v2 Stage 1 zero-shot (both towers frozen, projections only) | 0.484 | — | 0.468 |

All rows are apples-to-apples with each other: same 377 images, same ground truth, no text input. "8 shared" = the CheXpert↔RadZero concept matches from Phase 5 (Atelectasis, Cardiomegaly, Pleural Effusion, Pneumonia, Pneumothorax, Lung Opacity, Fracture/Bone Fracture, Edema/Pulmonary Edema).

> **Seed-variance note (added 2026-09-20, §3.6.1):** the 0.855 row is seed 0 of the recipe. Retraining with seeds 1–3 gave 0.833 / 0.839 / 0.819 — **single-model mean 0.837 ± 0.015**; seed 0 was the best draw. The **4-seed ensemble scores 0.868**, equal to RadZero on these images. Quote the mean ± std or the ensemble, not 0.855 alone.

**Reading it:** the image-only classifier went from chance (0.57) to **0.855 (seed 0; 0.837 ± 0.015 over four seeds)**, within **1.3 points of RadZero** — a purpose-built system trained on ~350k image-report pairs versus this project's 3k — on MedXVLM's own held-out data. It beats both same-data baselines by 5–8 points, so the attention-pooling head + ASL contributes on top of the encoder/label/view fixes (the XrayDINOv2 linear baseline isolates that: same encoder, no pipeline, 0.777).

### 3.2 Per class

n+ = true positives among the 377 test images. AUROC on single-digit support is noise and is shown for completeness only.

| Class | n+ | MedXVLM v2 S2 | ImageNet+linear | XrayDINOv2+linear | Phase 3 S2 blind | RadZero |
|---|---:|---:|---:|---:|---:|---:|
| No Finding | 268 | 0.820 | 0.811 | 0.773 | 0.603 | — |
| Cardiomegaly | 36 | 0.939 | 0.928 | 0.871 | 0.627 | **0.964** |
| Lung Opacity | 42 | 0.854 | 0.801 | 0.776 | 0.358 | **0.859** |
| Atelectasis | 36 | 0.851 | 0.771 | 0.787 | 0.580 | **0.883** |
| Pleural Effusion | 15 | 0.787 | 0.743 | 0.678 | 0.640 | **0.833** |
| Edema | 12 | **0.879** | 0.843 | 0.861 | 0.422 | 0.835 |
| Lung Lesion | 9 | 0.653 | 0.442 | 0.517 | 0.476 | — |
| Pneumonia | 8 | 0.771 | 0.748 | 0.797 | 0.466 | **0.911** |
| Fracture | 8 | **0.808** | 0.641 | 0.637 | 0.577 | 0.752 |
| Support Devices | 8 | 0.966 | 0.900 | 0.851 | 0.411 | — |
| Consolidation | 5 | 0.832 | 0.622 | 0.797 | 0.422 | — |
| Pleural Other | 3 | 0.664 | 0.389 | 0.392 | 0.622 | — |
| Pneumothorax | 1 | 0.949 | 0.960 | 0.806 | 0.904 | 0.910 |
| Enlarged Cardiomediastinum | 0 | — | — | — | — | — |

**Support caveat, stated plainly:** removing the false positives left this outpatient test set with genuinely few true cases of several conditions — one pneumothorax, five consolidations, three "pleural other." That is the real prevalence, not a data error, but it means those per-class numbers (and the Phase 5 comparison's Pneumothorax row) should not be read as measurements. The ≥10-positive macro (0.862) is the more trustworthy summary of the shared classes.

### 3.3 Zero-shot head — a negative result (as of 2026-09-17; partially reversed by v3b, §3.6.3: 0.734)

The CLIP-style zero-shot head (image embedding vs. a `"There is {label}."` prompt embedding from the Stage 1 encoders, cosine similarity as the score) is the structurally closest MedXVLM counterpart to RadZero's protocol. **It does not work at this data scale**, and three attempts to rescue it failed:

| Stage 1 variant | What it tested | Zero-shot macro AUROC (8 shared) |
|---|---|---:|
| v2, recall@1-selected (epoch 15, overfit) | the shipped run | 0.565 |
| v2, val-loss-selected (epoch 5) | "was it just a bad checkpoint?" | 0.572 |
| v2, both towers frozen, projections only | "does fine-tuning the text tower on long reports break short prompts?" | 0.484 |
| Phase 3 (BiomedCLIP init, both towers fine-tuned, all views, 5.9k pairs) | the old head, for reference | 0.669 |
| RadZero (353k pairs) | what the same protocol achieves with ~100× the data | 0.868 |

Two hypotheses were tested and **refuted**: (1) that the recall@1 checkpoint selection was to blame — the val-loss-selected checkpoint is equally poor; (2) that fine-tuning BiomedBERT on ~3k long reports drifted it away from short class prompts — freezing it made things worse, not better. (A mid-training dry run of the first variant had shown 0.72 at an early epoch; with a 377-image test set and single-digit positives for several classes, that was one noisy snapshot, not a trend — it is recorded here so it isn't mistaken for a reproducible result.) The Phase 3 head, trained on all 5.9k pairs including laterals with both towers fine-tuned, remains the best zero-shot variant at 0.669 — still 20 points behind RadZero.

**Conclusion:** with ~3k image-report pairs, contrastive alignment here learns enough to *initialise* a supervised head well (Stage 2 v2's 0.855 sits on the frozen XrayDINOv2 tower that Stage 1 passed through unchanged) but not enough to classify from text prompts alone. That is consistent with the literature: RadZero's and CheXzero's zero-shot numbers come from two orders of magnitude more paired data. The project's honest claim is a strong *supervised* image-only classifier, not a zero-shot one. The zero-shot rows above are kept in the results file as data, not as a headline.

### 3.4 Stage 3 v2 — report generation (added 2026-09-19)

**What changed vs. Phase 3** (`configs/report_gen_v2.yaml`, `scripts/train_report.py`): (i) the frozen vision tower is now the **v2 Stage 2 fine-tuned XrayDINOv2** (`checkpoints/classifier_v2/local_on/best.pt`) — the tower that has demonstrably learned this dataset's findings (0.855 test AUROC); Stage 1 v2 froze its tower, so its checkpoint would supply pretrained XrayDINOv2 unchanged; (ii) frontal views only (3,030 train rows, 921 abnormal); (iii) cleaned report text (no "XXXX"); (iv) 8×8 = 64 pooled visual tokens (was 7×7 = 49); (v) **abnormal-study oversampling** — a `WeightedRandomSampler` giving abnormal and normal studies equal total sampling mass, to fight the modal-normal-report collapse; (vi) early stopping on val loss (patience 2). Same decoder, LoRA config (r=16, α=32), effective batch 16, LR 2e-4, greedy decoding, `max_new_tokens=128` as Phase 3. Trainable params 22.0M / frozen 1.63B. Val loss by epoch: 1.157, 1.156, **1.132**, 1.200, 1.339 → `best.pt` = epoch 3; 278.6 min wall clock (~56 min/epoch).

**Evaluation** (`scripts/eval_v2_report_generation.py` → `scripts/eval_v2_score_reports.py` → `docs/phase5_artifacts/v2/report_generation_scores_v2.json`): all three systems on the identical 150 test images from Phase 5 (80 lateral / 70 frontal), scored against the **corrected** labels and cleaned references — so the Phase 5 numbers (old labels) are superseded for all three at once. Clinical efficacy (CE) = `medxvlm.data.labeler` on the generated text vs. the manifest label vector, U-Ones. BLEU-4 (sacrebleu) and ROUGE-L are computed here for the first time in the project.

| System | Subset | CE macro F1 | CE micro F1 | BLEU-4 | ROUGE-L | Distinct outputs | Modal-output share | Mean words |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 3 Stage 3 | frontal 70 | 0.078 | 0.603 | 12.08 | 0.304 | 5 | 74% | 29.6 |
| CheXOne (instruct) | frontal 70 | 0.149 | 0.599 | 10.24 | 0.286 | 36 | 26% | 28.8 |
| **Stage 3 v2** | frontal 70 | **0.152** | 0.575 | 9.76 | 0.280 | 26 | 54% | 35.4 |
| Phase 3 Stage 3 | all 150 | 0.066 | 0.614 | 12.15 | 0.308 | 5 | 71% | 29.7 |
| CheXOne (instruct) | all 150 | 0.107 | 0.479 | 9.61 | 0.278 | 87 | 13% | 29.0 |
| **Stage 3 v2** | all 150 | **0.119** | 0.424 | 7.09 | 0.244 | 61 | 39% | 45.6 |

Per-class CE F1 on the 70 frontals (support in parentheses; Phase 3 / CheXOne / v2): No Finding (46) 0.800 / 0.828 / **0.849**; Cardiomegaly (6) 0.286 / 0.000 / **0.333**; Lung Opacity (9) 0.000 / 0.364 / 0.364 (v2 precision 1.00, recall 0.22); Lung Lesion (4) 0.000 / 0.000 / **0.333**; Pleural Effusion (4) 0.000 / **0.400** / 0.250; Support Devices (1) 0.000 / **0.500** / 0.000; Edema (3), Pneumonia (2), Atelectasis (7), Fracture (3), Pleural Other (1): 0.000 for all three systems.

**What improved.** The collapse is broken: 61 distinct generations out of 150 (Phase 3: 5), and 18 distinct among the 24 abnormal frontal cases. The model now sometimes states findings that are actually present — e.g. reference *"The heart is significantly enlarged. Prominent pulmonary vascularity…"* → v2 *"Cardiomegaly. No infiltrates or effusions…"*; Phase 3 produced the normal boilerplate for every such case. On the clinical-efficacy metric v2 is **level with CheXOne** (0.152 vs 0.149 frontal; 0.119 vs 0.107 all) and roughly double the Phase 3 model.

**What did not improve, or got worse — stated plainly.**
1. **A new failure mode: hallucinated support devices and surgical history.** In 7 of 70 frontal generations (10%) — and 40 of 150 overall (27%; 41% on the lateral views it never trained on) — v2 mentions a catheter, PICC line, tube, surgical clips, sternotomy or lobectomy that the reference does not. Phase 3 did this once; CheXOne never. Example: reference *"…patchy infiltrate has developed in the right lower lobe…"* → v2 *"…tip of the left IJ catheter at the cavoatrial junction. There are small calcified granulomas… There is a large pleural effusion…"*. The cause is visible in the training data: device/surgery words occur in 15% of abnormal frontal reports vs 4% of normal ones, so the balanced sampler taught the decoder "abnormal ⇒ mention a line" as a shortcut, and a vision tower fine-tuned on 14 CheXpert labels (Support Devices: 98 positive studies in the whole corpus) cannot veto it. This is the direct cost of the oversampling that fixed the collapse.
2. **The modal normal report still accounts for 54% of frontal generations** (Phase 3: 74%). Reduced, not eliminated.
3. **Recall on most abnormal classes is still zero** (atelectasis 0/7, edema 0/3, fracture 0/3, pneumonia 0/2). With single-digit positives per class in the test subsample these are not precise estimates, but the direction is unambiguous.
4. **BLEU-4 / ROUGE-L are slightly *lower* than the Phase 3 model's** (9.76 / 0.280 vs 12.08 / 0.304 on frontals). This is the known weakness of n-gram metrics in radiology report generation: the collapsed model's boilerplate happens to match the modal reference phrasing well, so BLEU rewards it. These numbers are reported because the Phase 3 spec asked for them, not because they rank the systems correctly; CE macro F1 and the distinct-output count are the metrics that track what actually matters here. Note also that Phase 3's micro F1 (0.60) is inflated by the same effect — always predicting "No Finding" is right 66% of the time on this test set.
5. **Out-of-distribution on lateral views**: on the 80 laterals v2 produces longer (≈55 words), more varied, more hallucinated text (BLEU 7.09, ROUGE-L 0.244 on all-150). The v2 pipeline is frontal-only end to end; lateral numbers are included only because the Phase 5 comparison set contains them.
6. **A residual labeler false positive was observed while inspecting outputs**: `1367_IM-0237-1001` is labelled Pleural Effusion = 1, but its reference reads *"…no pleural line to suggest pneumothorax or costophrenic blunting to suggest large pleural effusion"* — the negation does not reach across "…to suggest". The 2026-09-17 labeler fix removed the dominant coordinated-list failure (§1) but not every scope pattern. Not fixed here; noted so it is not mistaken for a model error.

**Net:** Stage 3 v2 is a genuine improvement on the failure that mattered (collapse) and matches a 3B-parameter published model on clinical efficacy on this data — but it trades collapse for hallucination, and neither system is close to clinically usable report drafting on ~3k training studies. The honest claim is "comparable to CheXOne-instruct on CE F1, with a documented hallucination rate," not "solved."

### 3.5 External-dataset test — v2 Stage 2 on NIH ChestX-ray14 (added 2026-09-20)

The "MedXVLM on *their* dataset" direction Phase 5 could not run. NIH ChestX-ray14 is one of RadZero's native benchmarks, fully public (Kaggle mirror `nih-chest-xrays/data`, 45 GB), and labelled by NIH's own NLP over its own reports. The v2 Stage 2 checkpoint was run **as-is** (no fine-tuning, no threshold tuning) on the **official patient-disjoint `test_list.txt` split** — 25,596 images, 2,797 patients — so every number is pure domain shift: a different hospital, different scanners, 1024px PNGs downsampled to 224, and a different labeler. Script: `scripts/eval_external_chestxray14.py` → `docs/phase5_artifacts/v2/chestxray14_external_eval_{summary.json,raw.csv}`; 132 s on the GPU.

**Label mapping is deliberately narrow** — only concept-identical pairs: Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion→Effusion, No Finding. Not mapped: Lung Opacity (NIH's *Infiltration* is narrower), Lung Lesion (NIH splits Mass/Nodule), Pleural Other, Fracture, Support Devices, Enlarged Cardiomediastinum (absent from NIH); NIH-only Emphysema/Fibrosis/Hernia/Mass/Nodule/Infiltration/Pleural_Thickening are not predicted by MedXVLM.

| NIH class | n+ (of 25,596) | AUROC |
|---|---:|---:|
| Effusion | 4,658 | 0.785 |
| Edema | 925 | 0.781 |
| Cardiomegaly | 1,069 | 0.765 |
| Atelectasis | 3,279 | 0.708 |
| No Finding | 9,861 | 0.698 |
| Consolidation | 1,815 | 0.690 |
| Pneumonia | 555 | 0.650 |
| Pneumothorax | 2,665 | 0.634 |
| **Macro, 8 mapped classes** | | **0.714** |
| Macro, 7 pathologies (excl. No Finding) | | 0.716 |

**Reading it.** On its home test split the same checkpoint scores 0.855 (8 RadZero-shared classes); on a different hospital's data it drops to **0.714** — a 14-point domain-shift penalty, with every class still clearly above chance and none of the per-class numbers resting on thin support (smallest n+ = 555). Pneumothorax is weakest (0.634): the corrected Indiana training split contains only 23 positive pneumothorax studies (27 in the whole corpus), so this is the class the model had least chance to learn. For context only — **not** a like-for-like comparison: RadZero's paper reports 0.804 on ChestX-ray14 (its own 518px protocol and label set), and supervised models *trained on* ChestX-ray14 typically report ~0.80–0.84 mean AUROC. MedXVLM was trained on 3,030 images from one outpatient site and has never seen this dataset. The honest claim: the classifier generalises to an unseen hospital at 0.71, not at 0.85.

Housekeeping note: the 45 GB download filled the host drive (see `status_report.md` §11 item 16); only the 25,596 test images (~10 GB) were kept after evaluation.

### 3.6 Overnight run, 2026-09-20 — seed ensemble, decoding sweep, and a second zero-shot attempt

All jobs serialized (`scripts/archive/run_overnight_2026-09-20.sh`, `scripts/archive/run_v2b_then_score.sh`, `scripts/archive/run_v3b_after_v2b.sh`); every exit code is in `runs/overnight_2026-09-20*.log`, `runs/v2b_chain.log`, `runs/v3b_chain.log`.

**3.6.1 Seed variance and ensemble — a correction to the headline.** The v2 Stage 2 recipe (`classifier_v2.yaml`, unchanged) was retrained with seeds 1–3 (`--seed`, ~30–39 min each). Held-out frontal test split, image-only (`docs/phase5_artifacts/v2/classifier_seed_ensemble.json`):

| Model | 8 shared classes | 5 shared with ≥10 positives | All 13 | NIH ChestX-ray14 (8 mapped) |
|---|---:|---:|---:|---:|
| seed 0 (the checkpoint quoted in §3.1) | 0.855 | 0.862 | 0.829 | 0.714 |
| seed 1 | 0.833 | 0.850 | 0.792 | 0.701 |
| seed 2 | 0.839 | 0.856 | 0.808 | 0.706 |
| seed 3 | 0.819 | 0.853 | 0.786 | 0.693 |
| **single model, mean ± std (n=4)** | **0.837 ± 0.015** | 0.855 ± 0.005 | 0.804 ± 0.019 | 0.704 ± 0.009 |
| **4-seed probability ensemble** | **0.868** | 0.867 | **0.835** | **0.719** |
| RadZero, same images (reference) | 0.868 | 0.875 | n/a | 0.804 (paper, own protocol — not reproduced) |

Two things follow, stated plainly. (i) **Seed 0 was the best of four draws.** The fair single-model number for this recipe is **0.837 ± 0.015**, not 0.855; every earlier mention of 0.855 should be read as "one seed, upper end of the spread". (ii) **Averaging four seeds recovers it and more: 0.868 on the 8 shared classes — numerically identical to RadZero (0.8684 vs 0.8682) on identical images and labels** — and the ensemble also helps out of distribution (NIH 0.719 vs 0.704 ± 0.009 single). Ensembling is the most reliable gain available at this data scale; it costs 4× inference, which the demo app does not currently use (single seed-0 checkpoint).

**3.6.2 Decoding sweep — much of Stage 3's hallucination was greedy decoding.** Same v2 checkpoint, same 150 images, same scorer (`report_generation_scores_v2_overnight.json`):

| Decoding (v2 checkpoint) | CE macro F1, frontal 70 | CE micro F1 | Distinct / 70 | Modal share | Device hallucination (frontal) | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy (§3.4) | 0.152 | 0.575 | 26 | 54% | 7/70 = 10% | 9.76 | 0.280 |
| **beam = 3, repetition penalty 1.2** | **0.176** | **0.650** | 14 | 61% | **2/70 = 3%** | 8.66 | 0.264 |
| CheXOne instruct, greedy (reference) | 0.149 | 0.599 | 36 | 26% | 0% | 10.24 | 0.286 |

Beam search with a mild repetition penalty removes about two-thirds of the invented devices and raises clinical-efficacy F1 above CheXOne-instruct — at zero training cost — but it also re-concentrates the output (14 distinct vs 26). The collapse/hallucination trade-off is therefore partly a *decoding* knob, not only a training one. The 70-image numbers carry wide error bars (single-digit positives per class); the direction is consistent across the full 150 (CE 0.119 → 0.148).

**3.6.3 Zero-shot, second attempt (v3, sentence-level) — negative result, with the reason measured.** `configs/align_v3_sentence.yaml`: one report sentence per image per epoch (`SentenceAlignmentDataset`), identical sentences masked as mutual positives (`ContrastiveAlignment._multi_positive_infonce_loss`), XrayDINOv2 fine-tuned at LR 1e-5, 12 epochs, val-loss selection (26.1 min). Scoring was also upgraded to prompt-ensembled positive-minus-negative (`load_zero_shot(mode="pos_minus_neg")`, three templates each) and applied to every Stage 1 checkpoint:

| Stage 1 checkpoint | pos-only prompt (Phase 5 protocol) | pos − neg, 3 templates |
|---|---:|---:|
| Phase 3 (report-level, both towers fine-tuned, all views) | 0.669 | 0.642 |
| v2 (report-level, frozen tower, frontal) | 0.565 | 0.604 |
| **v3 (sentence-level, multi-positive, frontal)** | **0.568** | **0.558** |
| RadZero (353k pairs) | 0.868 | — |

The sentence-level recipe did not help, and the scoring change is not a reliable gain either. The training curve shows why the objective barely moved (val loss 3.19 → 2.94 over 12 epochs): counting with the labeler, **only 1,683 of the 16,375 training sentences (10.3%) contain a positive or uncertain finding**; 20% are negations and 70% are boilerplate ("heart size normal", "bony structures intact"). In a batch of 32 sentences, ~3 carry class-discriminative signal; the other 29 are near-synonymous normal statements that an exact-text positive mask cannot equate, so the contrastive task is mostly "which normal phrasing belongs to which normal image" — noise. RadZero sidesteps this with LLM-extracted *finding* sentences and ~100× the images (which would put its finding-sentence count in the tens of thousands versus 1,683 here). **v3b — finding sentences only — confirms the diagnosis and is the best zero-shot head in the project.** `align_v3b_findings.yaml`: same recipe, but only sentences with a positive/uncertain finding mention are used (900 images have one; the other 2,130 normal images all get the single canonical sentence *"No acute cardiopulmonary abnormality."*, making them mutual positives). 8 epochs, early-stopped, best epoch 5, 17.1 min.

| Stage 1 checkpoint | pos-only prompt | pos − neg, 3 templates | shared classes with ≥10 positives (pos-only) |
|---|---:|---:|---:|
| Phase 3 (report-level) | 0.669 | 0.642 | 0.708 |
| v2 (report-level, frozen tower) | 0.565 | 0.604 | 0.547 |
| v3 (all sentences) | 0.568 | 0.558 | — |
| **v3b (finding sentences only)** | **0.734** | 0.605 | **0.693** |
| RadZero | 0.868 | — | 0.875 |

Per class, v3b improves on v2 for every finding (Cardiomegaly 0.63 → 0.74, Atelectasis 0.41 → 0.66, Pneumonia 0.38 → 0.72, Pleural Effusion 0.64 → 0.75, Fracture 0.55 → 0.69); the one regression is "No Finding" (0.34 → 0.23), a negative concept that a single-prompt cosine handles badly. Two further points, both consistent with the mechanism: (a) removing the 90% of sentences that carry no finding is what made the loss learnable — v3's problem was signal density, not the objective; (b) the pos − neg scoring *hurts* v3b (0.605) because it never saw a negated sentence in training — scoring must match the training text distribution, which is also why it helped v2 (trained on full reports full of negations) and not v3b.

**Where this leaves zero-shot:** 0.734 is a real, honestly-obtained prompt-based classifier — well above chance and above every earlier attempt — but 13 points behind RadZero, and below every supervised number in this project (linear baseline 0.777, ensemble 0.868). The remaining gap is what 1,683 finding sentences from 900 images can support versus the tens of thousands RadZero was trained on; further recipe changes are unlikely to close it. The project's classification claim remains the supervised head and its ensemble; the zero-shot head is reported as a characterised secondary result, not a headline.

**3.6.4 Stage 3 v2b (30% abnormal sampling) — total collapse; the oversampling was load-bearing.** After two launch failures (`CUDA driver error: device not ready` at the first backward, traced to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is incompatible with near-cap allocations under WSL2; removed), v2b trained cleanly: 4 epochs, early-stopped, best epoch 2, **val loss 1.108 (lower than v2's 1.132)**, 131 min. Identical recipe to v2 except `abnormal_fraction: 0.3` (the natural rate, i.e. effectively no oversampling). Result on the same 150 images:

| Stage 3 variant (frontal 70) | CE macro F1 | Distinct / 70 | Modal share | Device hallucination | Mean words |
|---|---:|---:|---:|---:|---:|
| Phase 3 (no oversampling, old tower) | 0.078 | 5 | 74% | 1% | 29.6 |
| **v2b — 30% abnormal** | **0.057** | **1** | **100%** | 0% | 17.0 |
| v2 — 50% abnormal, greedy | 0.152 | 26 | 54% | 10% | 35.4 |
| v2 — 50% abnormal, beam 3 + rep. penalty 1.2 | **0.176** | 14 | 61% | **3%** | 33.1 |
| CheXOne instruct (reference) | 0.149 | 36 | 26% | 0% | 28.8 |

v2b emits one sentence — *"The heart is normal in size. The mediastinum is unremarkable. The lungs are clear. No acute disease."* — for **all 150 test images** (beam search adds one variant). Its 0% hallucination is trivial. Two conclusions: (i) the 50% abnormal oversampling in v2 was not a tuning nicety but the thing that broke the collapse — at the natural rate the decoder reverts to the modal normal report; (ii) **teacher-forced validation loss is the wrong model-selection signal for this task**: the fully collapsed v2b has *better* NLL than v2 because it fits the 70%-normal majority more tightly. Any future Stage 3 selection should use a generation-based criterion (distinct-output count or CE F1 on a validation subsample), not NLL. Net: the best Stage 3 configuration in this project is **v2 + beam-3 decoding** — CE macro F1 0.176 vs CheXOne-instruct 0.149 on the same images, 3% hallucination, 61% modal share still open.

### 3.7 Stage 3 v2c — findings-conditioned generation (2026-09-21)

**Why.** Sentence attribution (`docs/explainability_evaluation.md` §3) measured that the Stage 3 decoder draws ~1 % of its attention from the 64 visual tokens and re-uses one glance per image for every sentence: the text is written from the language prior. Text is the channel it uses, so v2c puts the classifier's findings *into the prompt*:

> `Generate a radiology findings report for this chest X-ray. Classifier findings: Cardiomegaly, Pleural Effusion.` (or `… Classifier findings: none flagged.`)

Findings come from the 4-seed Stage 2 ensemble (§3.6.1), cached for all 7,426 images (`scripts/cache_stage2_predictions.py` → `stage2_ensemble_predictions.csv`) and thresholded per class at the val-split F1 optimum (`stage2_ensemble_thresholds.json`; 8 classes with ≥10 val positives, val macro F1 0.566 at those thresholds; the other 6 use the median threshold 0.538). Train and test prompts therefore come from one distribution; during training each flagged finding is dropped with p = 0.1 so the decoder cannot simply copy the list. Everything else is the v2 recipe (frozen Stage 2 tower, 64 visual tokens, LoRA r16, 50 % abnormal oversampling, LR 2e-4). **Checkpoint selection** was changed to a generation criterion — clinical-efficacy macro F1 of greedy outputs on a fixed, class-balanced 48-image val subsample — because §3.6.4 showed val NLL selects the collapsed model.

**Run history (stated in full).** Epoch 1 ran 2026-09-21 07:12–08:23 UTC at batch 4 × accum 4 (70.4 min: train loss 1.429, val loss 1.152, val-subsample gen CE F1 0.322, 14/48 distinct). A WSL restart then killed the job. The first resume (`--resume`, same batch) died at the first backward with `CUDA error: unknown error`; `dmesg` shows `dxgkio_make_resident: Ioctl failed: -12` — host-side GPU out-of-memory, because Windows held ~2.8 GB of the 16.3 GB while training needs ~15.9. The second resume used batch 2 × accum 8 (same effective batch 16) and completed epochs 2–3 in **10.8 and 10.4 min each** — versus 70 min for epoch 1 — then early-stopped:

| Epoch | Batch | Train loss | Val loss | Val-subsample gen CE F1 (48 img) | Distinct / 48 | Full-val gen CE F1 (387 img, post hoc) |
|---|---|---:|---:|---:|---:|---:|
| **1** | 4 × 4 | 1.429 | 1.152 | **0.322** | 14 | **0.270** |
| 2 | 2 × 8 | 0.901 | 1.111 | 0.233 | 14 | (weights not kept) |
| 3 | 2 × 8 | 0.776 | 1.153 | 0.280 | 12 | 0.229 |

`best.pt` = epoch 1 by the pre-registered criterion; re-running the same criterion on the **full** frontal val split (`v2c_full_val_selection.json`) also prefers epoch 1. The 8× speed-up at half the batch size is not a typo: at 15.9/16.3 GB the WSL2 GPU-paravirtualisation layer was paging memory to the host for the whole of epoch 1 (and, in hindsight, probably for v2 and v2b too — 56 and 33 min/epoch). Stage 3 on this machine should run at batch 2 × accum 8.

**Test results** — same 150-image Phase 5 subsample, same scorer (`report_generation_scores_v2c.json`); frontal 70 unless stated. The scorer's macro F1 averages over all 14 classes (three have zero test support and contribute 0). Device hallucination = the §3.4 regex, mention in the generation but not the reference.

| System | CE macro F1 | CE micro F1 | CE macro F1, all 150 | Distinct / 70 | Modal share | Device halluc. | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Classifier's flagged findings as a bare list ("verbalised" ceiling)* | *0.234* | *0.656* | *0.224* | *13* | *0.74* | *0 %* | *0.0* | *0.10* |
| **v2c (epoch 1, official), greedy** | **0.165** | 0.610 | 0.182 | 13 | 0.74 | **0 %** | 9.2 | 0.29 |
| v2c (epoch 1), beam 3 + rep. 1.2 | 0.135 | 0.642 | 0.142 | 12 | 0.74 | 0 % | 8.1 | 0.29 |
| v2c (epoch 1), **findings removed from prompt** (ablation) | 0.057 | 0.590 | 0.057 | **1** | 1.00 | 0 % | 6.7 | 0.29 |
| v2c epoch 3, greedy — *post hoc, not selected* | 0.231 | 0.651 | 0.191 | 13 | 0.74 | 0 % | 8.7 | 0.29 |
| v2c epoch 3, beam — *post hoc* | 0.142 | 0.611 | 0.121 | 11 | 0.74 | 0 % | 10.8 | 0.32 |
| v2 + beam 3 (§3.6.2; the app default) | 0.176 | 0.650 | 0.148 | 14 | 0.61 | 3 % | 8.7 | 0.26 |
| v2 greedy | 0.152 | 0.575 | 0.119 | 26 | 0.54 | 10 % | 9.8 | 0.28 |
| CheXOne-instruct (reference) | 0.149 | 0.599 | 0.107 | 36 | 0.26 | 0 % | 10.2 | 0.29 |
| v2b (§3.6.4) | 0.057 | 0.590 | 0.057 | 1 | 1.00 | 0 % | 6.7 | 0.29 |

Bootstrap over the 70 images (2,000 resamples, macro over supported classes; `report_ce_f1_bootstrap.json`): every system's CE macro F1 has a 95 % CI about ±0.10 wide; v2c-epoch-1 vs v2-beam Δ = −0.01 [−0.13, +0.13]; epoch-3 vs v2-beam Δ = +0.07 [−0.05, +0.20]; the only difference whose CI excludes zero is *verbalised classifier vs v2 beam*, +0.09 [+0.01, +0.16].

**What this says.**

1. **v2c is a verbaliser of the classifier, and nothing else.** With the findings removed from the prompt the same checkpoint emits one sentence for all 150 images (the v2b collapse); with them it reproduces the flagged findings in report prose. Per class it tracks the verbalised ceiling (Cardiomegaly 0.55–0.67, Pleural Effusion 0.40–0.44, Lung Opacity 0.35–0.50 vs ceiling 0.71 / 0.40 / 0.60) and, like the classifier, never produces Lung Lesion, Fracture or Support Devices. The image enters only through the classifier.
2. **The ceiling is low on this test subsample — 0.234 — and that, not the decoder, is the binding constraint.** The ensemble flags a finding on only 18 of the 70 frontal images at val-optimal thresholds; per-class test support is 1–9; the scorer divides by 14 classes. A perfect verbaliser cannot exceed 0.234 here, so the most v2c could ever add over v2-beam (0.176) is ~0.06 — inside the ±0.10 noise band. The val-split number (0.566 macro F1 over 8 supported classes) shows the same classifier looks far better where support exists; the report metric on 70 images simply cannot resolve it.
3. **Hallucination is solved by construction**: 0 % device mentions in every v2c variant, versus 3 % (v2 beam) and 10 % (v2 greedy). The decoder no longer needs the "abnormal ⇒ mention a line" shortcut because the abnormality is named in its prompt. This is the one unambiguous gain.
4. **Greedy beats beam for v2c** (0.165 vs 0.135; 0.231 vs 0.142 at epoch 3) — the opposite of v2. Beam search's length/repetition preferences smooth away the short finding sentences the prompt induces; for a prompt-driven verbaliser there is nothing to de-hallucinate.
5. **Val and test disagree about epochs 1 and 3** (val 0.270 vs 0.229; test 0.165 vs 0.231). With ±0.10 CIs on test and per-class support of 2–4, this is expected noise, not a signal. Epoch 1 is the official checkpoint because it won on the pre-registered criterion *and* on the full val split; the epoch-3 test number is reported so no one has to trust that decision blindly, and it is labelled post hoc because it is.
6. **Decision for the app**: v2 + beam remains the default (v2c-epoch-1 does not beat 0.176), and **v2c is offered as a selectable "findings-conditioned, 0 % hallucination" mode** whose prompt is built live from the ensemble's probabilities and the same thresholds (`MedXVLMPipeline.report_instruction`). The app shows the flagged findings it put into the prompt, so a user can see exactly what the report is a verbalisation of.

**Net for RQ3.** Findings-conditioning removes hallucination and turns the report writer into an honest narrator of the classifier — which is what a two-stage clinical system would want — but on this data it cannot lift the clinical-efficacy metric beyond what the classifier alone scores, and on this 70-image subsample that ceiling is 0.234. Improving the report writer now means improving the classifier's recall on rare findings (data), not the decoder.

### 3.8 Confidence intervals and paired tests for the classification claims (2026-09-26)

`scripts/eval_bootstrap_classification.py` → `classification_bootstrap.json`. Paired bootstrap over the 377 held-out frontal test images (2,000 resamples, seed 0): each resample draws images with replacement once and scores every system on that same resample, so differences between systems are paired. No model is re-run — it uses the saved per-image scores (`v2_classification_raw.csv`, `stage2_ensemble_predictions.csv`). Two macro AUROCs: over the 8 findings shared with RadZero (the headline metric; a class with no positives in a resample is skipped for that resample — Pneumothorax has a single positive image), and over the 5 shared findings with ≥10 positives (Atelectasis, Cardiomegaly, Pleural Effusion, Lung Opacity, Edema), which is defined in every resample.

| System | Macro AUROC, 8 shared | 95% CI | Macro AUROC, 5 with ≥10 positives | 95% CI |
|---|---:|---:|---:|---:|
| RadZero (published) | 0.868 | [0.823, 0.903] | 0.875 | [0.827, 0.916] |
| MedXVLM, 4-model ensemble | 0.868 | [0.827, 0.902] | 0.867 | [0.818, 0.910] |
| MedXVLM, single model (seed 0) | 0.855 | [0.805, 0.888] | 0.862 | [0.820, 0.901] |
| Baseline head, ImageNet ViT | 0.804 | [0.741, 0.843] | 0.817 | [0.766, 0.863] |
| Baseline head, X-ray DINOv2 | 0.777 | [0.731, 0.816] | 0.795 | [0.744, 0.838] |
| Zero-shot prompts (v3b) | 0.734 | [0.656, 0.771] | 0.693 | [0.629, 0.746] |

| Paired difference (a − b) | Δ, 8 shared | 95% CI | CI excludes 0? | Δ, 5 stable [95% CI] |
|---|---:|---:|:---:|---:|
| Ensemble − RadZero | -0.000 | [-0.033, +0.033] | no | -0.008 [-0.031, +0.016] |
| Single model (seed 0) − RadZero | -0.014 | [-0.058, +0.024] | no | -0.013 [-0.039, +0.012] |
| Ensemble − single model (seed 0) | +0.013 | [-0.004, +0.038] | no | +0.005 [-0.015, +0.021] |
| Attention head − baseline head, same encoder (X-ray DINOv2) | +0.078 | [+0.028, +0.119] | yes | +0.067 [+0.032, +0.105] |
| Attention head (X-ray DINOv2) − baseline head (ImageNet ViT) | +0.051 | [+0.011, +0.100] | yes | +0.045 [+0.014, +0.081] |
| Baseline head: ImageNet ViT − X-ray DINOv2 | +0.028 | [-0.022, +0.059] | no | +0.022 [-0.015, +0.059] |
| RadZero − MedXVLM zero-shot | +0.135 | [+0.091, +0.208] | yes | +0.182 [+0.132, +0.237] |

**What this licenses, stated plainly.**
1. **"The ensemble matches RadZero" is now a statistical statement:** Δ = -0.000, 95% CI [-0.033, +0.033]. The two are indistinguishable on these images, and any true gap larger than ~0.03 AUROC in either direction is ruled out at 95%. It is *not* evidence that MedXVLM is better, and on the stable 5-class macro RadZero is nominally ahead (-0.008, CI straddling 0).
2. **Even a single model is not significantly behind RadZero** (Δ -0.014, CI includes 0) — but the individual AUROC intervals are ±0.04–0.05 wide, so this test set cannot resolve differences of a couple of points between good systems. That is also why the seed spread (±0.015, §3.6.1) matters: ranking two single runs on this split is noise.
3. **The per-label attention head's gain over the baseline head is real** (+0.05 to +0.08, both CIs exclude 0) — the project's own architecture contribution survives the test. (The "baseline head" systems fine-tune the whole encoder with a linear layer on the pooled CLS embedding, BCE loss, `configs/baseline_v2.yaml` — they are not frozen linear probes.)
4. **The "ImageNet encoder beats the X-ray encoder" baseline-head result is not significant** (Δ +0.028, CI [-0.022, +0.059]); the controlled encoder comparison for RQ1 is §3.9.
5. **The zero-shot gap to RadZero is real** (+0.135, CI well above 0).

### 3.9 RQ1 — does the image encoder's pretraining matter? (2026-09-26)

**Question (proposal §5):** does domain-specific pretraining of the image encoder — BiomedCLIP rather than general-domain CLIP — give better downstream results at this data scale? The project had never tested this directly: v2 switched to X-ray DINOv2 for practical reasons (§1), and the only encoder evidence was the baseline-head comparison, where ImageNet nominally beat X-ray DINOv2 (0.804 vs 0.777, *not* significant, §3.8).

**Design.** The v2 Stage 2 recipe, unchanged (`configs/classifier_v2.yaml`: image-only per-finding attention head, ASL loss, encoder LR 1e-5 / head LR 1e-3, cosine, early stopping on val macro AUROC, frontal images, seed 0), trained on four ViT-B towers that differ *only* in their starting weights. The three new towers load their public weights directly (`train_classifier.py --vision-backbone`; `scripts/run_rq1_backbones.sh`); the X-ray DINOv2 runs are the existing four seeds (their Stage 1 tower was frozen, so they also start from the public weights). All use the same ImageNet-normalised preprocessing — a slight handicap for the two CLIP towers, whose native normalisation differs, which fine-tuning largely absorbs. Evaluation: `scripts/eval_rq1_backbones.py` (377 held-out frontal test images; NIH ChestX-ray14 test split, 25,596 images, 8 mapped findings; pointing game of each model's own attention maps on the 984 NIH radiologist boxes) and the paired bootstrap of §3.8 (seed-0 runs, same resamples). Budget allowed **one seed per new encoder** (the planned second seed for BiomedCLIP and CLIP was skipped when the first three runs took 124 min instead of ~75).

| Encoder | Pretraining | Seeds | Test AUROC, 8 shared (mean ± sd) | Seed 0 [95% CI] | Test AUROC, 5 stable | NIH AUROC, 8 | Pointing game (984 boxes) | Best epoch(s) | Train time |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **X-ray DINOv2** | domain-specific, self-supervised (chest X-rays) | 4 | 0.837 ± 0.015 | 0.855 [0.805, 0.888] | 0.855 ± 0.005 | 0.704 ± 0.009 | 31.5% ± 6.1 | 8, 8, 10, 12 | 30–39 min |
| **BiomedCLIP** | domain-specific, contrastive (PubMed figures) | 1 | 0.802 | 0.802 [0.764, 0.854] | 0.844 | 0.687 | 26.7% | 15 | 45.0 min |
| **OpenAI CLIP** | general-domain, contrastive (web image-text) | 1 | 0.792 | 0.792 [0.726, 0.834] | 0.824 | 0.667 | 24.3% | 18 | 50.8 min |
| **ImageNet ViT** | general-domain, supervised (ImageNet) | 1 | 0.784 | 0.784 [0.719, 0.821] | 0.798 | 0.633 | 13.3% | 7 | 27.5 min |

Paired differences, seed 0 vs seed 0 (a − b; 2,000 paired resamples):

| Comparison | Δ AUROC, 8 shared [95% CI] | CI excludes 0? | Δ AUROC, 5 stable [95% CI] | CI excludes 0? |
|---|---:|:---:|---:|:---:|
| BiomedCLIP − OpenAI CLIP | +0.010 [-0.025, +0.086] | no | +0.020 [-0.009, +0.049] | no |
| X-ray DINOv2 − BiomedCLIP | +0.053 [-0.006, +0.085] | no | +0.018 [-0.007, +0.048] | no |
| X-ray DINOv2 − OpenAI CLIP | +0.063 [+0.029, +0.105] | **yes** | +0.038 [+0.011, +0.067] | **yes** |
| X-ray DINOv2 − ImageNet ViT | +0.071 [+0.031, +0.119] | **yes** | +0.064 [+0.028, +0.108] | **yes** |
| BiomedCLIP − ImageNet ViT | +0.018 [-0.009, +0.089] | no | +0.046 [+0.014, +0.084] | **yes** |
| ImageNet ViT − OpenAI CLIP | -0.008 [-0.056, +0.039] | no | -0.026 [-0.062, +0.008] | no |

Per finding (seed 0; test positives in brackets — single-digit counts are noisy):

| Finding | X-ray DINOv2 | BiomedCLIP | OpenAI CLIP | ImageNet ViT |
|---|---:|---:|---:|---:|
| Cardiomegaly (36) | 0.94 | 0.94 | 0.89 | 0.82 |
| Lung Opacity (42) | 0.85 | 0.85 | 0.82 | 0.80 |
| Edema (12) | 0.88 | 0.87 | 0.88 | 0.82 |
| Atelectasis (36) | 0.85 | 0.83 | 0.79 | 0.79 |
| Pleural Effusion (15) | 0.79 | 0.74 | 0.74 | 0.76 |
| Pneumonia (8) | 0.77 | 0.87 | 0.70 | 0.78 |
| Fracture (8) | 0.81 | 0.65 | 0.57 | 0.56 |

**Answer, stated plainly.**
1. **The literal RQ1 contrast — BiomedCLIP vs general-domain CLIP — points the hypothesised way but is not resolved.** BiomedCLIP is ahead on every measure (+0.010 test AUROC, +0.020 on the stable 5, +0.019 on NIH, +2.4 points on the pointing game), but with one run each and a seed spread of ±0.015 (§3.6.1), no difference is significant. RQ1 also asked about retrieval, which was not re-tested here.
2. **Domain-specific pretraining matters, and the encoder that clearly pays off is the X-ray-specific self-supervised one.** X-ray DINOv2's *worst* seed (0.819) beats every other encoder's run, and its paired lead over both general-domain towers is significant (+0.06 to +0.07, CIs exclude 0); over BiomedCLIP it is +0.053 with the CI just touching 0. Caveat: seed 0 is X-ray DINOv2's best seed; against its 4-seed mean (0.837) the leads shrink to +0.035 / +0.045 / +0.053 (BiomedCLIP / CLIP / ImageNet).
3. **The domain effect grows out of distribution and is largest for localisation.** On the unseen NIH hospital the ranking is identical and the spread widens (X-ray DINOv2 0.704 → ImageNet 0.633). On the pointing game the ImageNet encoder's attention maps hit the radiologist's box 13% of the time vs 32% (X-ray DINOv2, seeds 23–36%) and 27% (BiomedCLIP) — the classification scores differ by a few points, the heatmap quality by a factor of two. This is a further measured answer to RQ4 (`docs/explainability_evaluation.md`): explainability quality tracks the encoder's pretraining domain more strongly than accuracy does.
4. **The attention head needs a good encoder to help.** With X-ray DINOv2 the per-finding attention head beats the pooled-embedding baseline head by +0.06–0.08 (§3.8); with the ImageNet encoder it does not (0.784 vs the baseline head's 0.804 — within noise, and the two recipes also differ in loss and learning rate). The head pools *patch* tokens, so it can only add value when those patches already encode chest anatomy — which also explains why the pointing game separates encoders more sharply than AUROC.
5. **Where general encoders fall down most is subtle findings:** Fracture 0.56–0.65 for the three non-X-ray encoders vs 0.81 for X-ray DINOv2 (8 test positives, so indicative only).

**Net for RQ1:** yes, the encoder's pretraining domain matters — modestly for in-distribution accuracy, clearly for generalisation to another hospital, and most of all for whether the heatmaps point at the right place. The specific BiomedCLIP-over-CLIP advantage the proposal hypothesised is in the predicted direction but too small to confirm with one run each on 377 test images; the encoder the project actually uses (X-ray DINOv2, self-supervised on chest X-rays) is the best of the four on every measure.

### 3.10 Operating points: why the demo often says "No Finding" (2026-09-27)

Prompted by the demo: on many abnormal test X-rays the classifier flagged nothing. Measured on the 377 frontal test images with the 4-model ensemble (`scripts/compute_operating_points.py` → `stage2_operating_points.json`; thresholds and reference distributions from the **validation** split only; test numbers are for reporting, never for selection):

- At the F1-optimal ("balanced") thresholds the ensemble flags **nothing on 48 of 109 abnormal images** (44%) and falsely flags 29 of 268 normal ones. Median "No Finding" probability is 0.57 on abnormal images vs 0.85 on normal ones.
- The ranking is good (AUROC 0.79–0.95 on the main findings, §3.8) but the absolute probabilities are low: 69% of training studies are normal and each finding has 1–13% prevalence, so F1-optimal thresholds are conservative. Findings with fewer than 10 validation positives (Enlarged Cardiomediastinum, Consolidation, Pneumonia, Pneumothorax, Pleural Other, Fracture) cannot be tuned and are never flagged at the median fallback (test sensitivity 0%).
- A **"sensitive" operating point** (highest threshold that catches ≥80% of validation positives) flags nothing on **18 of 109** abnormal images, at the cost of false alarms on **93 of 268** normal ones. Per finding (threshold / test sensitivity / test specificity):

| Finding | Val + | Test + | Balanced | Sensitive |
|---|---:|---:|---|---|
| Cardiomegaly | 39 | 36 | 0.60 / 58% / 98% | 0.44 / 81% / 92% |
| Lung Opacity | 51 | 42 | 0.57 / 57% / 93% | 0.44 / 81% / 79% |
| Atelectasis | 37 | 36 | 0.51 / 56% / 89% | 0.44 / 69% / 81% |
| Pleural Effusion | 15 | 15 | 0.55 / 47% / 96% | 0.46 / 53% / 94% |
| Edema | 21 | 12 | 0.53 / 25% / 98% | 0.36 / 75% / 92% |
| Lung Lesion | 14 | 9 | 0.49 / 0% / 98% | 0.38 / 56% / 74% |
| Support Devices | 11 | 8 | 0.42 / 50% / 98% | 0.27 / 100% / 81% |

- **A labeler error surfaced in the same review.** Study `3966_IM-2028-1001` is labelled Pleural Effusion = 1, but its report covers a chest *and both knees*; the chest section reads "no … pleural effusion … No acute cardiopulmonary finding" and the positive comes from a **knee-joint** effusion ("limits evaluation for an effusion"). The model's "No Finding" (0.86) is correct there. Reports covering another body part are rare (7 of 3,826 studies, 0.2%; 3 test images), but up to 24 of the 188 effusion-positive reports never mention "pleural" or "costophrenic", so some effusion labels are softer than they appear. Not relabelled (it would shift every result); recorded as a known limitation.
- The demo now offers both operating points, shows model calls as positive/negative (No Finding positive only when nothing else is flagged), shows each probability's percentile among validation X-rays without that finding, and labels the comparison column "Report label (automatic)" rather than "Radiologist".

## 4. What this does and doesn't change about the project's story

- **The headline classification number is now real.** 0.855/0.862 macro AUROC, blind, on held-out data, against corrected labels, within ~1 point of a published state-of-the-art zero-shot system on the same images. This replaces the 0.865 text-fusion figure, which should no longer be quoted as classification accuracy.
- **Most of the recovery is data hygiene, not architecture**: fixing the labeler, dropping lateral views, and using a CXR-pretrained encoder took a linear head from 0.57 to ~0.78–0.80; the pipeline design adds the last ~5–8 points. That is an honest and defensible attribution, and the ablation table above is what supports it.
- **RQ1 answered (§3.9, 2026-09-26):** the image encoder's pretraining domain matters — X-ray DINOv2 beats BiomedCLIP, OpenAI CLIP and ImageNet ViT on test AUROC, on the unseen NIH hospital and, most sharply, on heatmap localisation (32% vs 13% pointing game for ImageNet); the BiomedCLIP-over-CLIP contrast the proposal named is in the predicted direction but not significant with one run each. §3.8 adds 95% CIs: the ensemble equals RadZero (Δ 0.000, CI ±0.033).
- **After the v2c run (§3.7, 2026-09-21):** conditioning the report writer on the classifier's findings gives 0 % hallucination and a faithful verbaliser, but CE F1 0.165 (greedy) vs v2-beam 0.176 — inside a ±0.10 noise band whose ceiling on this 70-image subsample is 0.234 (the classifier flags only 18/70). Stage 3 is now limited by classifier recall on rare findings, not by the decoder. App default stays v2 + beam; v2c is selectable.
- **After the 2026-09-20 overnight run (§3.6):** the honest classification number is **0.837 ± 0.015 (single seed) / 0.868 (4-seed ensemble)** — the ensemble equals RadZero on identical images; the 0.855 quoted earlier was the best seed. The demo app now serves the ensemble. Stage 3's best configuration is v2 + beam-3 decoding (CE F1 0.176, 3% hallucination); v2b showed the 50% abnormal oversampling is essential, and that val NLL selects the collapsed model. Zero-shot rose to 0.734 with finding-sentence training but remains data-scale-limited.
- **Stage 3 was retrained on 2026-09-19 (§3.4).** The boilerplate collapse documented in `cross_dataset_evaluation.md` §5 is broken (5 → 61 distinct outputs on 150 images) and CE macro F1 now matches CheXOne-instruct on the same images (0.152 vs 0.149, frontal). The cost is a 10% (frontal) device/surgery hallucination rate that the Phase 3 model and CheXOne do not have, and the modal normal report still covers 54% of frontal outputs. The next lever is not more epochs (val loss turned up at epoch 4) but a stronger visual signal or a hallucination-aware training objective — out of scope for this rebuild.
- **Phase 3 numbers are historical, not wrong-as-recorded**: `training_summary.md` accurately reports what those runs produced *under their protocol*; that protocol (text fusion, noisy labels, all views) is what this rebuild replaced.

## 5. Files

- Code: `src/medxvlm/data/labeler.py` (negation scope), `src/medxvlm/models/vision_encoder.py` (backbone option), `src/medxvlm/data/dataset.py` (projection filter), `src/medxvlm/models/baseline_classifier.py`, `src/medxvlm/engine/loop_utils.py` (`load_vision_encoder`), `scripts/train_{align,classifier,baseline,report}.py`, `scripts/prepare_data.py`, `scripts/validate_labeler.py`, `scripts/run_v2_rebuild.sh`, `scripts/eval_v2_classification.py`, `scripts/eval_v2_report_generation.py`, `scripts/eval_v2_score_reports.py`, `scripts/eval_external_chestxray14.py`.
- §3.8–3.9: `scripts/eval_bootstrap_classification.py`, `scripts/eval_rq1_backbones.py`, `scripts/run_rq1_backbones.sh`, `scripts/archive/run_rq1_eval_after_training.sh`, `runs/rq1_backbones.log`, `runs/rq1_eval.log`, `checkpoints/classifier_v2/rq1_*_s0/`; results `classification_bootstrap.json`, `rq1_backbones_summary.json`, `rq1_backbones_test_predictions.csv`.
- Configs: `configs/{align,classifier,baseline,report_gen}_v2.yaml`, `report_gen_v2b.yaml`, `report_gen_v2c.yaml`, `align_v3_sentence.yaml`, `align_v3b_findings.yaml`.
- v2c (§3.7): `src/medxvlm/data/findings_prompt.py`, `scripts/cache_stage2_predictions.py`, `scripts/run_v2c_{findings_conditioned,resume,eval_only}.sh`, `runs/v2c_chain.log`, `runs/v2c_resume_chain.log`, `checkpoints/report_gen_v2c/{best,epoch3}.pt`; results `stage2_ensemble_{predictions.csv,thresholds.json}`, `classifier_verbalised_generations.csv`, `medxvlm_stage3_v2c{,_beam3,_nofindings_beam3,_epoch3,_epoch3_beam3}_generations.csv`, `report_generation_scores_v2c.json`, `v2c_full_val_selection.json`, `report_ce_f1_bootstrap.json`.
- Data: `data/processed/manifest.csv` (corrected); `data/processed/manifest_v1_phase3_labels.csv` (the Phase 3 labels, kept for reproducibility).
- Checkpoints/logs: `checkpoints/{align,classifier,baseline,report_gen}_v2/`, `runs/{align,classifier,baseline,report_gen}_v2/`, `runs/v2_rebuild.log`, `runs/train_report_gen_v2.log`.
- Results: `docs/phase5_artifacts/v2/v2_classification_{summary.json,raw.csv}`, `classifier_seed_ensemble.json`, `report_generation_scores_v2_overnight.json`, `medxvlm_stage3_v2{,b}{,_beam3}_generations.csv`, `docs/phase5_artifacts/v2/medxvlm_stage3_v2_generations.csv`, `docs/phase5_artifacts/v2/report_generation_scores_v2.json`, `docs/phase5_artifacts/v2/chestxray14_external_eval_{summary.json,raw.csv}`.
