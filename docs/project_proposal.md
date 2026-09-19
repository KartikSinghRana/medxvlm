# MedXVLM: A Multimodal Vision-Language System for Chest X-Ray Understanding

**Project proposal — B.Sc. AI & Data Science major project**

## 1. Problem statement

Chest radiography is the most frequently performed diagnostic imaging
study worldwide, and the narrative report a radiologist writes for each
scan is the artifact that actually drives clinical decisions — not the
image alone. Most deep learning work on chest X-rays, however, treats the
image and the report as separate problems: classification models predict
a fixed label set from pixels only (ChestX-ray14, CheXNet, CheXpert),
while report-generation models are trained end-to-end on image-to-text
pairs without an explicit, reusable, jointly-aligned representation of
what the image and the text have in common. This throws away a source of
supervision that is present in essentially every hospital's radiology
archive for free: the report itself.

MedXVLM is built on the premise that a single, well-aligned image-text
embedding space — learned once via contrastive pretraining — can serve as
the shared backbone for *both* multi-label diagnostic classification and
free-text report generation, rather than training two unrelated models
from scratch. This mirrors how CLIP-style pretraining became the shared
foundation for many downstream vision tasks in the general domain, but
applies it specifically to the vocabulary, visual statistics, and label
structure of chest radiology, and adds a systematic explainability layer
so that both the classifier's and the generator's outputs can be
inspected against the image regions and report phrases that produced
them — a requirement for any model whose output is meant to support (not
replace) a radiologist's judgment.

## 2. Objectives

1. Build a chest X-ray/report contrastive alignment model (Stage 1) that
   produces a shared embedding space usable for both retrieval and as a
   frozen visual backbone for later stages.
2. Build a 14-class CheXpert-style multi-label classifier (Stage 2) on
   top of that shared representation, and measure how much the aligned
   representation improves over a vision-only baseline.
3. Build a lightweight, LoRA-tuned report generator (Stage 3) that
   consumes the same frozen visual backbone through a LLaVA-style visual
   token projection, so all three stages are traceable back to one
   vision encoder rather than three independent models.
4. Instrument every stage with explainability tooling (Grad-CAM,
   attention rollout, text-grounded heatmaps, sentence-level attribution)
   so classification and generation outputs can be visually and textually
   audited, not just scored.
5. Deliver all of the above as clean, modular, typed PyTorch code with a
   reproducible data pipeline — not a single training notebook.

## 3. Proposed architecture (design level)

MedXVLM is deliberately staged: each stage produces an artifact (a
checkpoint or an embedding space) that the next stage consumes, so the
project can be evaluated incrementally rather than needing the entire
system to work before anything can be measured.

### 3.1 Stage 1 — Contrastive alignment

A ViT-B/16 image encoder and a PubMedBERT text encoder, both initialised
from BiomedCLIP weights (rather than general-domain CLIP), are fine-tuned
to map paired chest X-rays and report findings into a shared 512-d
embedding space. The training objective combines a **global** InfoNCE
loss (image embedding vs. report embedding, following CLIP/ConVIRT) with
a **local** patch-word loss (image patch tokens vs. report word tokens,
following GLoRIA) so the model is pushed to align not just
"this image matches this report" but "this region of the image matches
this phrase in the report" — which is what later explainability and
localization depend on.

### 3.2 Stage 2 — Multi-label classification

A 14-class CheXpert-style classification head is trained on top of the
Stage 1 vision tower. Per-label attention pooling lets each of the 14
labels attend to the image regions most relevant to *that* label rather
than sharing one global pooled vector, and a gated fusion mechanism lets
the model optionally incorporate the Stage 1 text encoder's
representation of any available clinical context (e.g. indication text)
without becoming dependent on it at inference time when no text is
available.

### 3.3 Stage 3 — Report generation

Qwen2.5-1.5B-Instruct is fine-tuned with LoRA to generate findings-style
reports conditioned on the image (upgraded from the 0.5B variant originally
scoped here once Phase 2's hardware validation confirmed comfortable VRAM
headroom on the 16 GB RTX 5080 for the larger decoder). Following LLaVA,
the frozen Stage 1 vision tower's patch features are projected into the
language model's token embedding space and spliced in as soft visual
tokens at an `<image>` placeholder in the text prompt, so the language
model treats the image as a sequence of "visual words" it can attend over
like any other token. Keeping the vision tower frozen and only adapting
the language model with LoRA keeps this stage trainable on a single 16 GB
GPU.

### 3.4 Explainability

Rather than being a fourth stage bolted on at the end, explainability is
implemented against whichever stages exist: Grad-CAM and attention
rollout produce visual heatmaps from the classifier and vision encoder,
the Stage 1 local contrastive alignment enables text-grounded heatmaps
(highlighting the image region a specific report phrase corresponds to),
and sentence-level attribution traces which visual tokens most influenced
each generated sentence in Stage 3's output.

## 4. Dataset

MedXVLM uses the **Indiana University Chest X-ray collection** (Open-i),
accessed via the `raddar/chest-xrays-indiana-university` Kaggle mirror:
paired frontal/lateral chest radiographs with de-identified narrative
radiology reports (indication, comparison, findings, and impression
sections) and MeSH-coded diagnostic terms, drawn from the Indiana Network
for Patient Care and released by the NIH's Open-i project.

This was chosen over the two other standard chest X-ray corpora for
architectural, not just practical, reasons:

- **ChestX-ray14 (NIH)** provides images and NLP-mined labels only — it
  has no accompanying report text at all. Since Stage 1's contrastive
  alignment and Stage 3's report generation both require real paired
  reports, ChestX-ray14 cannot support two of this project's three
  stages regardless of compute budget.
- **MIMIC-CXR** does have paired reports and is the dataset most of the
  comparable literature (GLoRIA, BioViL, CheXzero) reports results on,
  but it requires PhysioNet credentialing (CITI human-subjects training
  plus a signed data use agreement tied to an institutional sponsor) and
  is ~377,000 images / ~227,000 studies — an access and scale mismatch
  for a solo, single-GPU B.Sc. project on a fixed timeline.
- **Indiana/Open-i** is freely downloadable with no credentialing, is
  small enough (~7,470 images, ~3,955 reports) to iterate on quickly on a
  single RTX 5080, still has real narrative reports and diagnostic
  labels, and is a recognised evaluation benchmark in this exact
  literature (e.g. ConVIRT reports zero-/few-shot results on it), which
  keeps this project's numbers comparable to prior published work even
  though training happens on a much smaller corpus than MIMIC-CXR.

The tradeoff is accepted explicitly: Indiana/Open-i's small size means
Stage 1 pretraining here is closer to *domain adaptation* of BiomedCLIP's
existing PMC-15M-scale biomedical pretraining than pretraining from
scratch, which is precisely why starting from BiomedCLIP weights (rather
than general CLIP or randomly initialised encoders) is a design decision
and not just a convenience.

## 5. Research questions and hypotheses

The literature review (`docs/literature_review.md`) points to four
concrete, testable questions this project is positioned to answer at
small scale:

**RQ1 — Does domain-specific contrastive initialisation matter at this
scale?** *Hypothesis:* initialising Stage 1 from BiomedCLIP rather than
general-domain CLIP will yield better downstream classification and
retrieval on Indiana/Open-i, because BiomedCLIP has already closed much
of the vocabulary and visual-domain gap that CheXzero and BioViL
identify between natural images/text and radiology.

**RQ2 — Does a local (patch-word) contrastive term improve grounding, not
just accuracy?** *Hypothesis:* following GLoRIA's ablations, adding the
local loss to the global InfoNCE objective will produce Grad-CAM/attention
maps that localise to clinically plausible regions more often than a
global-only (ConVIRT-style) objective, even if global classification
accuracy differs only slightly.

**RQ3 — Can a frozen vision tower plus a LoRA-tuned 1.5B-parameter LLM
produce coherent findings on a narrow, single-modality report style?**
*Hypothesis:* because Indiana/Open-i reports are shorter and more
templated than MIMIC-CXR's, a small LLM with strong visual grounding and
parameter-efficient tuning (per LoRA's finding that low-rank updates
capture task-specific adaptation without full fine-tuning) can reach
reasonable report quality despite being an order of magnitude smaller
than LLaVA-Med's or CheXagent's backbones.

**RQ4 — Does explainability quality track the Stage 1 design choice, not
just downstream task accuracy?** *Hypothesis:* text-grounded heatmap
quality (RQ2) and generation faithfulness (RQ3) are both bottlenecked by
the same Stage 1 alignment quality, so improvements to Stage 1 should be
visible in both downstream stages' explainability outputs, not just their
task metrics. RadZero's 2025 result — that similarity-based cross-attention
over the same kind of contrastive alignment used here yields usable
pixel-level explainability directly, without a separate Grad-CAM pass —
is the closest existing evidence this hypothesis is plausible.

The most directly comparable recent systems are **GLoRIA** (local+global
contrastive alignment on chest X-rays, the direct source of Stage 1's
loss design), **CheXagent** (the closest system in overall scope —
classification and generation from one shared backbone — at roughly two
orders of magnitude more parameters, data, and compute than this project
targets), and **RadZero** (2025; zero-shot alignment with built-in,
similarity-based explainability from a single contrastive representation,
the closest existing target for what Stage 1 is meant to support once
Grad-CAM/attention-rollout/text-grounded heatmaps are layered on top).
MedXVLM's contribution is not a new method relative to these, but an
integration of all three concerns — alignment, classification, and
generation — around one shared, small-footprint vision backbone, trained
end-to-end on hardware and a timeline a single student can actually
complete.

## 6. Week-by-week plan

*(Weeks 1–2 are this Phase 1 submission. Later weeks are a first
scheduling pass and expected to be adjusted as Phase 2+ scoping firms
up.)*

| Week | Focus |
|---|---|
| 1–2 | **Phase 1 (this submission):** repo skeleton, environment validation, toy dataloader smoke test, literature review, proposal. |
| 3–4 | Real data pipeline: clean image/report manifest, train/val/test split (patient-level, no leakage), CheXpert-style rule-based labeler for the 14 target classes. |
| 5–6 | Stage 1 implementation: BiomedCLIP-initialised encoders, global + local contrastive loss, training loop, checkpointing; first retrieval/zero-shot sanity metrics. |
| 7 | Stage 1 evaluation and tuning (RQ1, RQ2 groundwork); freeze the Stage 1 checkpoint used by later stages. |
| 8–9 | Stage 2: classification head, attention pooling, gated text fusion; train and evaluate against a vision-only baseline. |
| 10 | Grad-CAM / attention rollout explainability on Stage 2; qualitative + quantitative localization checks (RQ2). |
| 11–12 | Stage 3: LLaVA-style visual token projection, Qwen2.5-1.5B-Instruct + LoRA fine-tuning, report generation training loop. |
| 13 | Stage 3 evaluation (BLEU/ROUGE + clinical-relevance checks) and sentence attribution explainability (RQ3, RQ4). |
| 14 | Streamlit demo app tying image upload → classification + generated report + explainability overlays together. |
| 15 | Ablations tied to RQ1–RQ4, error analysis, results writeup. |
| 16 | Buffer, final report, and presentation preparation. |
