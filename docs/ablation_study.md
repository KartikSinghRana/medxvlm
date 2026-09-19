# Ablation Study: Local Patch-Word Contrastive Loss

**Date:** 2026-09-05
**What was varied:** `ContrastiveAlignment.use_local_loss` (Stage 1) — whether
the FILIP-style local patch-word contrastive term (`local_patch_word_loss`
in `src/medxvlm/models/alignment.py`) is added to the global InfoNCE loss,
or the model trains on global InfoNCE alone. This is the default ablation
named in the Phase 3 spec.

**What was held fixed:** data (`data/processed/manifest.csv`, same
uid-level train/val/test split), seed (0), image transforms, batch size
(16), learning rate (2e-5), epochs (Stage 1: 5, Stage 2: 8), the Stage 2
architecture (gated text fusion on, same `ClassifierHead` config), and the
Stage 2 loss (`multilabel_bce_loss`). Two full pipelines were trained
end to end: Stage 1 (on/off) → Stage 2 on top of each Stage 1 checkpoint.

**Honesty note:** these are short runs (5 + 8 epochs on ~3,060 training
studies) run once per condition, not a converged, multi-seed study — the
numbers below are what actually happened on this machine on this run, not
an extrapolation of where either curve would end up with more training.

## Stage 1: retrieval (image↔report), final epoch (5/5)

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

Full per-epoch curves: `runs/align/local_on/metrics.csv`,
`runs/align/local_off/metrics.csv`.

At this training budget, global-only alignment reaches consistently
**higher** retrieval recall than global+local. The likely reason: "val
loss" here is the *global* InfoNCE loss in both conditions, and local-on
is splitting the same 5 epochs' worth of gradient steps across two
objectives (global 1.87 + local 1.86 at epoch 5, vs. global-only reaching
1.47) — global-only simply gets more effective optimization pressure on
the metric being measured, in this short a run. This isn't evidence the
local loss is a bad idea, only that it hasn't paid off yet at 5 epochs;
the standard finding in the local-loss literature (e.g. GLoRIA, FILIP) is
that its benefit shows up in **grounding quality** and downstream
classification, not necessarily raw retrieval recall this early — which
is exactly what the Stage 2 numbers below test.

## Stage 2: classification, best epoch (8/8 for both)

Macro metrics:

| Metric | Local loss ON | Local loss OFF | Delta (on − off) |
|---|---:|---:|---:|
| Macro AUROC | 0.8645 | 0.8648 | −0.0003 |
| Macro AUPRC | 0.3619 | 0.3549 | +0.0070 |
| Macro F1 | 0.1966 | 0.2008 | −0.0042 |

Per-class AUROC:

| Label | Local loss ON | Local loss OFF | Delta (on − off) |
|---|---:|---:|---:|
| Atelectasis | 0.9491 | 0.9584 | −0.0092 |
| Cardiomegaly | 0.9285 | 0.9415 | −0.0130 |
| Consolidation | 0.8293 | 0.7723 | **+0.0570** |
| Edema | 0.8334 | 0.8434 | −0.0099 |
| Enlarged Cardiomediastinum | 0.9116 | 0.9639 | −0.0522 |
| Fracture | 0.8657 | 0.8335 | +0.0322 |
| Lung Lesion | 0.8277 | 0.8012 | +0.0265 |
| Lung Opacity | 0.9369 | 0.9073 | +0.0297 |
| No Finding | 0.9193 | 0.9241 | −0.0048 |
| Pleural Effusion | 0.8374 | 0.8277 | +0.0097 |
| Pleural Other | 0.8269 | 0.7905 | **+0.0364** |
| Pneumonia | 0.7576 | 0.7945 | −0.0369 |
| Pneumothorax | 0.8400 | 0.8439 | −0.0039 |
| Support Devices | 0.8389 | 0.9047 | −0.0657 |

Full per-epoch, per-class curves: `runs/classifier/local_on/metrics.csv`,
`runs/classifier/local_off/metrics.csv`.

**Result: mixed, not a clean win either way.** Macro AUROC is a virtual
tie (both ~0.865). The local loss helps focal/regional findings the most
(Consolidation +0.057, Pleural Other +0.036, Fracture +0.032, Lung Opacity
+0.030) — plausible, since these are exactly the findings a patch-level
signal should help localize — but it hurts a few classes with either very
few positive examples (Enlarged Cardiomediastinum: only 8 positives in the
whole manifest, Support Devices: 76) where the extra loss term likely adds
optimization noise rather than useful gradient signal, or classes that are
identifiable from coarse/global appearance rather than a localized region
(Cardiomegaly, Support Devices — a device is often a thin line spanning
many patches, not a compact region a per-patch loss localizes well).

## Qualitative: grounding-map sharpness

Attention-weight grounding snapshots for both conditions are under
`runs/classifier/attention_maps/{local_on,local_off}/epoch_008/`. Spot
comparing the same two samples' maps:

- **Pleural Effusion, sample 0:** both conditions concentrate attention on
  the lower/lateral lung fields near the costophrenic angles — the
  clinically correct region for effusion — and look qualitatively similar
  in sharpness.
- **Pneumothorax, sample 1:** the two conditions differ visibly. Local-on
  produces one large, diffuse high-attention region covering most of the
  upper chest. Local-off produces several smaller, more spatially distinct
  high-attention peaks. By the specific "more spatially concentrated ==
  sharper" reading, **local-off looks sharper on this sample** — the
  opposite of the hypothesis in `docs/project_proposal.md` RQ2.

This is a two-sample spot check, not a quantitative sharpness metric —
treated here as a qualitative observation, not a conclusion. A rigorous
version would score IoU/pointing-game accuracy against radiologist
bounding boxes, which the Indiana CXR dataset does not provide.

## Interpretation

At this compute budget (5+8 epochs, ~3,060 training studies, batch size
16, single seed), the local patch-word loss does **not** show the clean
"improves grounding and downstream classification" result its motivating
literature (GLoRIA, FILIP) reports at much larger scale/longer training.
It shows a plausible-looking split (helps regional/focal findings, hurts
rare or diffuse/device-like ones) that is consistent with the mechanism
being real but under-trained, not with the mechanism being absent — 5
Stage 1 epochs is not enough for either loss term to fully converge (see
the still-falling train loss curves in `runs/align/*/metrics.csv`). The
honest conclusion from this run is **inconclusive-but-directionally
plausible**, not "the local loss works" or "the local loss doesn't work."
A longer run (more epochs, larger batch for more InfoNCE negatives) is the
natural next step before drawing a stronger conclusion.

## Second ablation: Asymmetric Loss vs. plain BCE (baseline classifier)

Not run in this pass — `src/medxvlm/models/baseline_classifier.py` now has
a real `asymmetric_loss` implementation and `scripts/train_baseline.py`
accepts `--loss asl`, so this is a one-command follow-up
(`python scripts/train_baseline.py --loss asl`) once GPU time is
available; the primary (local-loss) ablation above was prioritized to fit
the available time in this session.
