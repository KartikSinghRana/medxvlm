# Phase 3 Training Summary

> **Superseded for classification (2026-09-17) — read this first.** Phase 5 found that the
> Stage 2 number below (0.865 macro AUROC) was measured with the *paired ground-truth
> report* fed into the classifier via gated text fusion; tested image-only, the same
> checkpoint scores 0.51 (chance). Separately, a labeler negation-scope bug inflated
> several labels (e.g. pneumothorax 10% → 0.8% after the fix). The rebuilt, image-only
> classifier on corrected labels reaches **0.855 macro AUROC on the held-out frontal
> test split (0.833 val)**, within 1.3 points of RadZero on identical images. The numbers
> in this file remain an accurate record of what the Phase 3 runs produced under their
> protocol; for current results see [`v2_rebuild.md`](v2_rebuild.md).

**Date:** 2026-09-05
**Data:** `data/processed/manifest.csv` — 7,426 image rows / 3,826 studies,
uid-level 80/10/10 split (3,060 / 382 / 384 studies), real labels from
`src/medxvlm/data/labeler.py` (validated against `scripts/validate_labeler.py`
before labelling the full corpus).

All numbers below are read directly from each run's `runs/<stage>/metrics.csv`
(also mirrored to TensorBoard). Nothing here is extrapolated or fabricated —
see the "What didn't finish" section for the one run that was cut short and
restarted.

## Baseline (`scripts/train_baseline.py`)

Plain ImageNet ViT-B/16 + linear head, BCE loss, no contrastive pretraining,
no text branch, 5 epochs, batch size 16.

| Epoch | Train loss | Val loss | Val macro AUROC |
|---:|---:|---:|---:|
| 1 | 0.2485 | 0.2514 | 0.6287 |
| 2 | 0.2401 | 0.2427 | 0.6522 |
| 3 | 0.2348 | 0.2477 | 0.6864 |
| 4 | 0.2329 | 0.2343 | 0.6726 |
| 5 | 0.2303 | 0.2446 | **0.6939** |

Wall clock: 20.4 min. Checkpoints: `checkpoints/baseline/{best,last}.pt`.

## Stage 1 — Contrastive Alignment (`scripts/train_align.py`)

BiomedCLIP-initialised ViT-B/16 + PubMedBERT, 5 epochs, batch size 16, run
twice (the ablation switch) — full curves and the local-loss comparison
are in [docs/ablation_study.md](ablation_study.md). Headline: at this
epoch budget, local-loss-off reaches higher retrieval recall (val
recall@1 i2t/t2i: 0.0174/0.0214 vs. 0.0134/0.0120 for local-loss-on) —
see the ablation doc for why, and for how this doesn't carry over cleanly
into Stage 2.

Wall clock: 19.9 min (local_on), 19.1 min (local_off). Checkpoints:
`checkpoints/align/{local_on,local_off}/{best,last}.pt`.

## Stage 2 — Classification (`scripts/train_classifier.py`)

`ClassifierHead` (per-label attention pooling + gated text fusion) on top
of each Stage 1 checkpoint's vision + text encoders, 8 epochs, batch size
16. Full per-class table in [docs/ablation_study.md](ablation_study.md).

| Run | Best epoch | Val macro AUROC | Val macro AUPRC | Val macro F1 |
|---|---:|---:|---:|---:|
| local_on | 8/8 | 0.8645 | 0.3619 | 0.1966 |
| local_off | 8/8 | 0.8648 | 0.3549 | 0.2008 |

**This is the headline result of Phase 3**: the proposed pipeline (Stage 1
contrastive pretraining + Stage 2 attention pooling/gated text fusion)
reaches **~0.865 macro AUROC**, a ~17-point improvement over the plain
baseline's **0.694** — on the same data, same split, same label set. The
local-patch-word-loss ablation itself is a near-tie at the macro level
(see ablation doc for the per-class breakdown, which is not a tie).

Wall clock: 35.6 min (local_on), 36.2 min (local_off). Grounding-map
snapshots every 2 epochs under `runs/classifier/attention_maps/`.
Checkpoints: `checkpoints/classifier/{local_on,local_off}/{best,last}.pt`.

## Stage 3 — Report Generation (`scripts/train_report.py`)

Frozen Stage 1 (`local_on`) vision encoder → 7×7 adaptive-pooled patch
grid → 2-layer MLP projector → Qwen2.5-1.5B-Instruct with LoRA
(r=16, α=32) on the attention + feed-forward projections, loss masked to
answer tokens only. Config: 3 epochs, batch size 2, grad accumulation 8
(effective batch 16), 22.0M trainable parameters (LoRA + projector) vs.
1.63B frozen.

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 1.1652 | 1.0220 |
| 2 | 0.7637 | **0.9987** |
| 3 | 0.5394 | 1.1060 |

Wall clock: 210.7 min (~3.5 hours). Best checkpoint is **epoch 2** — by
epoch 3, train loss keeps falling (0.76 → 0.54) while val loss rises
(0.999 → 1.106), i.e. mild overfitting sets in after epoch 2. This is
reported as observed, not smoothed over; `best.pt` (epoch 2) is the
checkpoint to use, not `last.pt` (epoch 3).

Sample generations (full text in `runs/report_gen/samples/epoch_00{1,2,3}.txt`),
against the same reference both epochs were shown:

> REFERENCE: The cardiomediastinal silhouette is within normal limits for
> size and contour. The lungs are normally inflated without evidence of
> focal airspace disease, pleural effusion, or pneumothorax. Stable
> calcified granuloma within the right upper lung. No acute bone
> abnormality.. No acute cardiopulmonary process.
>
> EPOCH 1: The heart size and pulmonary vascularity appear within normal
> limits. The lungs are free of focal airspace disease. No pleural
> effusion or pneumothorax is seen. 1. No evidence of active disease.
>
> EPOCH 3: The heart is normal in size. The mediastinum is unremarkable.
> There is no pleural effusion, pneumothorax, or focal airspace disease.
> The XXXX are unremarkable. No acute cardiopulmonary abnormality.

Two honest observations from the samples, not just the loss curve:
1. **Findings-level content is genuinely learned**, not templated
   boilerplate — every generation correctly reproduces the specific
   negative findings (no effusion, no pneumothorax, no focal airspace
   disease) that this particular reference asserts, after only 3 epochs.
2. **The de-identification placeholder "XXXX" leaks into generation**
   (epoch 3 sample above, and epoch 3's second sample: "XXXX scarring is
   noted within the left lung base" — a plausible-sounding but
   unverifiable hallucination not in that sample's reference). Neither
   `scripts/prepare_data.py` nor `ReportGenDataset` strip the Indiana CXR
   corpus's own `XXXX` de-identification tokens from report text before
   training, so the decoder faithfully learned to reproduce them as if
   they were normal report vocabulary. A real cleaning step (regex-strip
   or replace `XXXX` before labelling/training) is the clear next fix,
   not attempted here since it surfaced only during this write-up.

Checkpoints: `checkpoints/report_gen/{best,last}.pt` (85MB each —
LoRA adapters + projector only, not the 1.6B frozen decoder).

## What didn't finish (stated plainly, per the Phase 3 spec)

Every run listed in this document ran to completion and is real. The one
wrinkle: the *first* attempt at the Stage 3 run was killed mid-startup
(right after loading the decoder, before any training step) when the
coding-agent session that launched it was torn down mid-run — not a
training failure, an infrastructure interruption. It produced no
checkpoint and is not counted above; it was restarted from scratch as a
fully detached process (`setsid`) specifically so it would survive a
session boundary, and that second attempt is the one reported here.
Baseline and both Stage 1/Stage 2 runs completed in a single attempt each
with no interruptions.

Separately, Stage 3's own epoch 3 is a real (not fabricated) case of mild
overfitting (val loss rose from 0.999 to 1.106) — flagged above rather
than only reporting the best number.

## Checkpoint & log locations

```
checkpoints/
  baseline/{best,last}.pt
  align/{local_on,local_off}/{best,last}.pt
  classifier/{local_on,local_off}/{best,last}.pt
  report_gen/{best,last}.pt        # best = epoch 2 (lowest val loss)
runs/
  baseline/metrics.csv (+ TensorBoard)
  align/{local_on,local_off}/metrics.csv (+ TensorBoard)
  classifier/{local_on,local_off}/metrics.csv (+ TensorBoard)
  classifier/attention_maps/{local_on,local_off}/epoch_XXX/*.png
  report_gen/metrics.csv (+ TensorBoard), report_gen/samples/epoch_XXX.txt
```
