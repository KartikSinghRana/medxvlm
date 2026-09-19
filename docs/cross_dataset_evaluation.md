# Phase 5 — Cross-Dataset Evaluation: MedXVLM vs. RadZero / CheXOne

Date: 2026-09-06
Builds directly on [`external_repro_feasibility.md`](external_repro_feasibility.md) and [`external_baseline_reproduction.md`](external_baseline_reproduction.md) (Phase 4).

## 1. Direction actually run, and why

**Flipped, for both models — deliberately, not as an afterthought.** Phase 4's feasibility check already established that RadZero's and CheXOne's *other* evaluation datasets (PadChest, CheXpert, MIMIC-CXR, ReXGradient, ReXVQA, Chest ImaGenome) all require either a PhysioNet credentialing process or a separate license/data-use agreement neither obtainable inside this project's timeline. The one dataset both external models evaluate on that is *actually reachable here* is **OpenI / IU X-ray — which is already MedXVLM's home dataset** (RadZero uses it natively for zero-shot classification; CheXOne uses it as an out-of-distribution findings-generation check). So there was no foreign dataset to adapt MedXVLM *to* — the datasets had already converged. Instead, Phase 5's real work was:

1. Restricting the comparison to MedXVLM's own **held-out test split** (`data/processed/manifest.csv`, `split == "test"`, 748 images / 384 studies) — a leak-free subset RadZero and CheXOne have never seen and MedXVLM's own Stage 1/2/3 were never trained on.
2. Running RadZero's and CheXOne's public checkpoints on those exact same images.
3. Running MedXVLM's own Stage 1/2/3 checkpoints on those exact same images, in a form that's actually comparable to what RadZero/CheXOne do (see §3 for why this took real work, not a one-line inference call).

Every result below uses this test split. No adapter was needed for the image data (same files, same directory); the adapter that *was* needed is a label-taxonomy mapping (§2) and, unexpectedly, a fairness fix to MedXVLM's own classification eval protocol (§3).

## 2. Dataset accessibility, re-confirmed (step 19)

Unchanged from Phase 4: RadZero's OpenI test artifacts (`data/OpenI/custom.csv`, `openi_multi_label_text.json`) and CheXOne's plain-`transformers` Quick Start are both usable here; PadChest/CheXpert/ChestXDet10/MS-CXR/SIIM/RSNA (RadZero) and MIMIC-CXR/ReXGradient/ReXVQA/Chest ImaGenome (CheXOne) are not. No change in status since Phase 4; re-verified only to the extent that this phase re-used the exact same public checkpoints and confirmed they still load.

## 3. The adapter: label taxonomies, and a fairness bug this phase found in MedXVLM itself

MedXVLM's Stage 2 classifier and its manifest ground truth use the standard 14-class **CheXpert** vocabulary (`src/medxvlm/models/labels.py`). RadZero's OpenI classifier uses its own 18-class vocabulary derived from Indiana MeSH terms. The overlap, by clinical concept:

| CheXpert-14 (MedXVLM) | RadZero-18 | Comparability |
|---|---|---|
| Atelectasis | Atelectasis | apples-to-apples (identical concept + name) |
| Cardiomegaly | Cardiomegaly | apples-to-apples |
| Pleural Effusion | Pleural Effusion | apples-to-apples |
| Pneumonia | Pneumonia | apples-to-apples |
| Pneumothorax | Pneumothorax | apples-to-apples |
| Lung Opacity | Lung Opacity | apples-to-apples (name identical; each model's underlying labeler defines "opacity" slightly differently, so treat as apples-to-apples-with-caveat) |
| Fracture | Bone Fracture | apples-to-apples (same concept, different name) |
| Edema | Pulmonary Edema | apples-to-apples (same concept, different name) |
| Enlarged Cardiomediastinum, Lung Lesion, Consolidation, Pleural Other, Support Devices, No Finding | — | not comparable — no RadZero counterpart |
| — | Lung Infiltration, Pulmonary Mass, Lung Nodule, Pulmonary Emphysema, Pulmonary Fibrosis, Pleural Thickening, Hernia, Calcified Granuloma, Lung Granuloma | not comparable — no MedXVLM counterpart |

**Ground truth used for all 8 shared classes: MedXVLM's own manifest labels** (from its validated `medxvlm.data.labeler`), not RadZero's MeSH-keyword heuristic from Phase 4 — this is the single, real, non-fabricated ground truth every system in this comparison is scored against, avoiding the label-mapping uncertainty Phase 4 had to carry for RadZero-only runs.

**The bigger finding, which is really the adapter work this phase actually did:** building a fair MedXVLM-side number to put next to RadZero's zero-shot number required first noticing that MedXVLM didn't have one. `scripts/train_classifier.py`'s `ClassifierHead` uses "gated text fusion" — and at both train *and validation* time, the text it fuses is **the paired ground-truth report for that exact image** (see `Stage2Model.forward` in that script, and `ClassifierHead.forward` in `src/medxvlm/models/classifier_head.py`). That means the **0.865 macro AUROC headline number in `docs/training_summary.md` is not a blind image-based prediction** — the model is handed text that already states the findings (e.g. "There is no pleural effusion") as an input feature for predicting whether there's a pleural effusion. This is a legitimate design for some purposes, but it is not the same task RadZero performs (image + a *class-name* prompt, never the ground-truth report), so it cannot be placed next to RadZero's number honestly.

This phase's `scripts/eval_phase5_classification.py` therefore produces **three** MedXVLM numbers on the identical test split, from the identical trained checkpoints, so the difference is purely about what's fed in at inference:

1. **Stage 2, blind** (`ClassifierHead(text_embeds=None)`) — image-only, the fair counterpart to RadZero.
2. **Stage 2, text-fusion** (`ClassifierHead` fed the ground-truth report) — reproduces the training-time protocol/headline number, explicitly **not comparable** to RadZero.
3. **Stage 1 zero-shot** — a CLIP-style zero-shot head built for this phase (no such head existed before) directly from the trained `ContrastiveAlignment` encoders: `"There is {label}."` prompts, cosine similarity as the score — the actual MedXVLM-side structural equivalent of RadZero's protocol.

## 4. Classification comparison (748-image held-out test split, shared ground truth)

| Class | MedXVLM Stage 2 blind [apples-to-apples] | MedXVLM Stage 1 zero-shot [apples-to-apples] | MedXVLM Stage 2 text-fusion [NOT COMPARABLE — see §3] | RadZero zero-shot, same 748 images [apples-to-apples] |
|---|---:|---:|---:|---:|
| Atelectasis | 0.556 | 0.672 | 0.980 | 0.889 |
| Cardiomegaly | 0.582 | 0.753 | 0.880 | 0.941 |
| Pleural Effusion | 0.529 | 0.530 | 0.769 | 0.554 |
| Pneumonia | 0.506 | 0.500 | 0.838 | 0.750 |
| Pneumothorax | 0.471 | 0.456 | 0.784 | 0.512 |
| Lung Opacity | 0.357 | 0.603 | 0.933 | 0.851 |
| Fracture / Bone Fracture | 0.622 | 0.598 | 0.796 | 0.621 |
| Edema / Pulmonary Edema | 0.474 | 0.710 | 0.830 | 0.775 |
| **Macro (8 shared classes)** | **0.512** | **0.603** | **0.851** | **0.737** |
| Macro over all 13 valid CheXpert classes (MedXVLM's own full label set) [not comparable to RadZero — wider label set] | 0.507 | 0.536 | 0.826 | n/a (RadZero has no prompts for the other 5) |

*(MedXVLM's "13 valid" row excludes "Enlarged Cardiomediastinum," which is single-class — all-negative or not-mentioned — in this 748-image test split and has an undefined AUROC; RadZero's 8-class shared macro above uses exactly the 8 rows shown.)*

**Reading this table:**
- **Apples-to-apples**: MedXVLM Stage 2 blind vs. RadZero, and MedXVLM Stage 1 zero-shot vs. RadZero — same images, same ground truth, same task shape (image (+ class prompt) → score, no privileged text). On this basis, **RadZero's zero-shot classifier (0.737 macro) clearly outperforms both real MedXVLM classification paths (0.512 blind, 0.603 zero-shot)** on MedXVLM's own held-out data.
- **Not comparable**: MedXVLM Stage 2 text-fusion (0.851 macro) is included only for completeness and to show where the training-time headline number actually comes from — it is not a fair opponent for RadZero and should not be read as "MedXVLM beats RadZero."
- **Reference only, not reproduced here**: RadZero's own paper (arXiv:2504.07416, Table 5) reports **0.847 mean AUC on the full 7,470-image OpenI set** using its own label mapping — different images (the full set, not this 748-image held-out slice) and a different ground-truth source, so it is cited, not compared cell-for-cell.

## 5. Report generation comparison (150-image subsample of the same test split)

Both models generated a findings section for the identical 150 test-split images (`docs/phase5_artifacts/report_gen_subsample.csv`, seed 0). Scored with MedXVLM's own validated CheXpert-style labeler (`medxvlm.data.labeler.aggregate_report`) run on the generated text, U-Ones policy, compared against the manifest's stored ground truth for that image — the same labeler that built every ground-truth label in this project, reused (not re-derived) as the clinical-efficacy scorer.

| Metric [apples-to-apples: identical 150 images, identical reference reports, identical scorer] | MedXVLM Stage 3 (best.pt, epoch 2) | CheXOne (instruct mode) |
|---|---:|---:|
| Macro precision | 0.131 | 0.094 |
| Macro recall | 0.085 | 0.136 |
| **Macro F1** | **0.075** | **0.105** |
| Micro F1 | 0.411 | 0.384 |

Per-class F1 (both models, same 150 images; support = number of true positives in this subsample):

| Class | Support | MedXVLM F1 | CheXOne F1 |
|---|---:|---:|---:|
| No Finding | 74 | 0.634 | 0.654 |
| Cardiomegaly | 15 | 0.125 | 0.000 |
| Lung Opacity | 21 | 0.000 | 0.319 |
| Pleural Effusion | 24 | 0.286 | 0.211 |
| Support Devices | 2 | 0.000 | 0.286 |
| Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Lung Lesion, Pleural Other, Fracture | 5-16 each | 0.000 (all) | 0.000 (all, except above) |

**This is apples-to-apples** — same images, same reference reports, same scoring code, both models genuinely blind to the reference at generation time. **Honest reading, not glossed over:** neither model does well on this held-out Indiana data by this metric. Both score reasonably on "No Finding" (the majority class in this corpus) and essentially fail to recall almost every specific abnormal finding. Inspecting the raw generations (`docs/phase5_artifacts/medxvlm_stage3_generations.csv`) shows why for MedXVLM specifically: its Stage 3 model, at greedy decoding, frequently emits the **same boilerplate normal-sounding report** regardless of the input image's actual content — e.g. three different test images, one clearly showing rib fractures and aortic tortuosity in its reference report, all produced the identical generated text ("The heart size and pulmonary vascularity appear within normal limits... No evidence of active disease."). This is consistent with the mild overfitting already flagged in `docs/training_summary.md` (epoch 3's val loss regression) and is a genuine limitation surfaced by this evaluation, not a scoring artifact — CheXOne, by contrast, varies its output per image and recalls some findings MedXVLM never does (Lung Opacity, Support Devices) at the cost of an extra false positive on Cardiomegaly.

**Reference only, not reproduced here** (arXiv:2604.00493, CheXOne's own numbers, on its own benchmarks — different datasets, different metrics, cited for context only, not comparable to the table above): findings generation on ReXGradient — 1/RadCliQ 1.116, BertScore 0.483, BLEU 0.229, RadGraph 0.21; reader-study reports "comparable to or better than resident-written reports in 55% of cases." See [`external_baseline_reproduction.md`](external_baseline_reproduction.md) §4 for the full cited table.

**Caveat on the CheXOne run itself:** generated in instruct mode (no reasoning trace) for throughput across 150 images; CheXOne's own paper evaluates primarily in reasoning mode, which the ablation in that paper reports as more accurate (0.9617 vs. 0.9468 on their VQA benchmark) at the cost of latency. This run likely understates CheXOne's real capability on this task.

## 6. Compute cost, for the report's methods section

| Run | Images × classes/tasks | Wall time |
|---|---|---:|
| RadZero zero-shot, MedXVLM test split | 748 × 18 | 32.5 min |
| MedXVLM classification (blind + text-fusion + zero-shot), test split | 748 × 3 modes | ~2 min |
| MedXVLM Stage 3 generation, subsample | 150 | 11.0 min |
| CheXOne generation, subsample | 150 | 11.9 min |

All four ran serially on the single RTX 5080 (see `docs/external_repro_feasibility.md` for why RadZero and MedXVLM's Stage 2/zero-shot are cheap and CheXOne/Stage 3 are not — model size, not task complexity, is what dominates).

## 7. Explicit skip list

- **Direction**: flipped for both models, from the plan's default (MedXVLM adapted onto RadZero's/CheXOne's native benchmark data) — see §1 for why (credentialing).
- RadZero was **not** re-run on its full 7,470-image OpenI set for this phase (that was Phase 4's job); this phase's RadZero numbers are specifically on MedXVLM's 748-image held-out **subset** of that same corpus.
- Only 8 of RadZero's 18 classes and 8 of MedXVLM's 14 classes have a shared-concept counterpart; the rest are reported as "not comparable" per class in §4, not omitted silently.
- MedXVLM's Stage 2 text-fusion number is reported but explicitly marked not comparable to RadZero — see §3 for why the comparison would otherwise be misleading.
- CheXOne generation ran in instruct mode, not its paper's default reasoning mode, for time budget reasons — flagged in §5, not hidden.
- No BLEU/RadGraph/BERTScore-style text-similarity metric was computed for either report-generation system here (only the labeler-based clinical-efficacy F1) — those require reference implementations (RadGraph model, etc.) not set up in this phase; adding them is future work, not claimed here.
- CheXOne's and RadZero's own *reported* numbers throughout this document come from datasets and protocols this project did not reproduce (MIMIC-CXR-based benchmarks, full OpenI, ReXGradient, reader studies) — every such number is labelled "reported by the authors, not reproduced here," matching Phase 4's convention.

## 8. Drop-in table for the Week 7 report

*(This is the condensed version of §4-5, formatted for direct reuse.)*

**Classification, zero-shot / blind, 8 shared CheXpert-taxonomy classes, MedXVLM's held-out Indiana test split (n=748), macro AUROC:**

| System | Macro AUROC | Comparable to the others in this row? |
|---|---:|---|
| RadZero (public checkpoint, zero-shot) | **0.737** | yes — reference point |
| MedXVLM Stage 1 zero-shot (this project) | 0.603 | yes |
| MedXVLM Stage 2, blind (this project) | 0.512 | yes |
| MedXVLM Stage 2, with ground-truth text fusion (this project) | 0.851 | **no** — sees the answer, see §3 |
| RadZero, own reported number, full OpenI (7,470 img, own labels) | 0.847 | reported by the authors, not reproduced here |

**Report generation, clinical-efficacy macro F1, identical 150-image Indiana subsample:**

| System | Macro F1 | Comparable to the other row? |
|---|---:|---|
| CheXOne (public checkpoint, instruct mode) | **0.105** | yes |
| MedXVLM Stage 3 (this project, best checkpoint) | 0.075 | yes |
| CheXOne, own reported numbers, ReXGradient/reader study | 1/RadCliQ 1.116, 55% reports ≥ resident quality | reported by the authors, not reproduced here — different dataset and metric |

## 9. Addendum (2026-09-17): what happened after this comparison

The findings in §3–§5 triggered a full rebuild — documented in [`v2_rebuild.md`](v2_rebuild.md). Two of them changed the ground truth used above:

1. **The labeler had a negation-scope bug** (a leading "no" did not reach the end of coordinated lists, so *"no consolidation, effusion, or pneumothorax"* labelled pneumothorax positive). Fixing it changed study-level prevalence of Pneumothorax 10.2% → 0.8% and Pleural Effusion 19.6% → 4.9%. The Phase 5 tables above were computed against the **old** labels and are kept as the record of that comparison; the ≥10-support rows are the ones that survive the fix meaningfully, and the Pneumothorax row above should be disregarded (it was almost entirely false positives).
2. **RadZero, re-scored against the corrected labels on the frontal test subset (no re-run, same saved predictions), moves from 0.737 → 0.868** — an independent model agreeing much more with the corrected labels is strong evidence the fix was right.

Post-rebuild, on the same frontal held-out images and corrected labels, **image-only** and with no text given to any model:

| System | Macro AUROC, 8 shared classes | Comparable? |
|---|---:|---|
| RadZero zero-shot (re-scored) | 0.868 | yes — reference point |
| **MedXVLM v2 Stage 2, image-only (rebuilt)** | **0.855** | yes |
| MedXVLM Phase 3 Stage 2, blind (the row in §4) | 0.572 (0.512 on all 748 images, old labels) | yes — the "before" |

The direction, comparability labels and skip list of §1–§8 are unchanged; only the classification conclusion is superseded: after fixing the training protocol and the labels, MedXVLM's own image-only classifier is within ~1 point of RadZero on this data rather than 22 points behind.

**Report generation, updated 2026-09-19.** Stage 3 was retrained (`docs/v2_rebuild.md` §3.4) and all three systems were re-scored on the identical 150 images against the corrected labels (`docs/phase5_artifacts/v2/report_generation_scores_v2.json`). The §5 table above (old labels) is superseded by:

| System | CE macro F1, 70 frontal | CE macro F1, all 150 | Distinct outputs / 150 | Device/surgery hallucination rate (frontal) | Comparable? |
|---|---:|---:|---:|---:|---|
| **MedXVLM Stage 3 v2** | **0.152** | **0.119** | 61 | 10% | yes |
| CheXOne (instruct) | 0.149 | 0.107 | 87 | 0% | yes |
| MedXVLM Stage 3, Phase 3 (the §5 row) | 0.078 | 0.066 | 5 | 1% | yes — the "before" |

Conclusion change: MedXVLM's report generator is now level with CheXOne-instruct on clinical efficacy on this data, rather than behind it — but it achieves this with a hallucination failure mode CheXOne does not exhibit, and both remain far from usable (macro F1 ≈ 0.15). BLEU-4 / ROUGE-L were also computed for the first time and slightly favour the *collapsed* Phase 3 model (12.1 / 0.30 vs 9.8 / 0.28), which is why they are not used to rank the systems here.
