# Phase 2 Code Review Notes

**Date:** 2026-08-30
**Milestone:** Model build checkpoint — Phase 2, Week 3–4 (Core Architectural
Implementation)

This document is written for the code review, in plain language: what each
new module does and why it's a separate module, what's still stubbed and
why, what `scripts/sanity_check_models.py` actually verifies versus what it
deliberately doesn't, and what comes next. The Phase 2 goal was **wiring
correctness, not training quality** — nothing here has been trained,
evaluated, or tuned.

## What each module does, and why it's separated this way

### `models/labels.py`

Just the 14 CheXpert label names as one constant tuple. Both
`baseline_classifier.py` and `classifier_head.py` import it, so the
baseline and the proposed pipeline can never disagree about label count or
ordering.

### `models/vision_encoder.py` — `VisionEncoder`

Wraps a ViT-B/16. On construction it tries to pull BiomedCLIP's image
tower from the HF Hub (`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`);
if that's unreachable it falls back to a plain ImageNet-pretrained
`vit_base_patch16_224` from `timm` (see the `# TODO` in `_build_backbone`).
Both sources turn out to be the *same* timm architecture under the hood —
BiomedCLIP's own `open_clip_config.json` declares its vision tower as
`timm_model_name: vit_base_patch16_224` — so one forward path
(`forward_features`, split into prefix tokens vs. patch tokens) handles
both, instead of needing two separate code paths per weight source.

`forward()` returns both `patch_tokens` (`[B, 196, 768]`) and a pooled
embedding, not just the pooled one. This is the one design choice every
later milestone depends on: Stage 1's local patch-word loss, Stage 2's
per-label attention pooling, and later Grad-CAM/attention-map
explainability all need the spatial grid, and there's no way to
retrofit it later without changing this module's return type.

### `models/text_encoder.py` — `TextEncoder`

Same pattern as `VisionEncoder`, mirrored for text: tries
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (Microsoft's
current name for the checkpoint historically called "PubMedBERT" — the old
name still resolves via a Hub redirect, but the current name is used
directly to avoid the extra hop), falls back to `bert-base-uncased` if
unreachable. Returns both per-token embeddings and a pooled sentence
embedding (`outputs.pooler_output`), for the same reason `VisionEncoder`
returns patch tokens: the local patch-word loss and later sentence
attribution need word-level representations.

One real bug this module's sanity check caught: the first version pooled
by manually slicing `last_hidden_state[:, 0, :]` (the raw CLS token)
instead of using BERT's own pooler output. That left the pooler's
dense+tanh layer sitting in the model with `requires_grad=True` but never
called, so it received no gradient — exactly the "silently-frozen
parameter" failure mode `sanity_trainer.py` is designed to catch, and it
did, immediately, on the first run. Fixed by using `outputs.pooler_output`
instead.

### `models/alignment.py` — `ContrastiveAlignment` (Stage 1)

Takes a `VisionEncoder` and a `TextEncoder`, projects both encoders'
*pooled* embeddings into a shared 512-d space via one `nn.Linear` per
modality, L2-normalises, and computes a symmetric (image→text and
text→image) InfoNCE loss — standard CLIP-style contrastive alignment. The
projection is a learned linear map rather than requiring the two
encoders' hidden sizes to already match, so the vision and text towers can
vary independently.

`local_patch_word_loss()` is a stubbed module-level function with the
intended signature and a docstring explaining the shape contract
(`patch_tokens`, `token_embeddings`, `attention_mask` → scalar). **Why
stubbed:** it needs a `[B, B, N, L]` cross-modal similarity tensor plus a
padding mask, which is meaningfully harder to get right than the global
loss and orthogonal to whether global alignment works — getting the global
InfoNCE loss numerically correct and stable was the actual goal of this
milestone. It's not called from `forward()`; implementing it is explicitly
next-milestone work (see below).

### `models/classifier_head.py` — `ClassifierHead` (Stage 2)

A learned query per CheXpert label cross-attends over the vision encoder's
patch tokens (`PerLabelAttentionPooling`), producing one pooled vector per
label; a single shared `nn.Linear(embed_dim, 1)` turns each into a logit.
No text fusion yet — there's a `# TODO` marking where gated text fusion
(mixing in `TextEncoder`/`ContrastiveAlignment` output before
classification) belongs. Kept as its own module, separate from
`ContrastiveAlignment`, so the Stage 1 vision tower can be frozen,
fine-tuned, or swapped without touching this head, and so text fusion can
be added later without touching the alignment loss.

The attention weights (`[B, 14, 196]`) are returned alongside the logits —
not used yet, but they're a free per-label localisation map, which is
exactly what the later explainability milestone needs.

### `models/baseline_classifier.py` — `BaselineClassifier`

The literature review's "before" case: a plain ImageNet ViT-B/16 with a
linear multi-label head, no contrastive pretraining, no text branch —
representing the CheXNet/ChestX-ray14-era approach. `multilabel_bce_loss`
is the working loss; `asymmetric_loss_stub` is a documented `NotImplementedError`
stub for label-imbalance handling (Ben-Baruch et al., 2021), deferred
because it needs real (imbalanced) label distributions to be worth
implementing — synthetic random targets in this milestone's sanity checks
are balanced by construction, so ASL would have nothing meaningful to
correct yet.

### `engine/sanity_trainer.py` — `run_sanity_check`

One model-agnostic overfit-one-batch loop (AdamW, gradient-norm clipping,
50 steps by default), used identically by the baseline, Stage 1, and Stage
2 checks — each just supplies its own `forward_and_loss_fn`. Checks, in
order: the loss is finite every step; every parameter with
`requires_grad=True` actually received a gradient (nothing silently
disconnected from the graph); the final loss is lower than the first.

Note the second check is "received *a* gradient," not "received a
*nonzero*" one — an exactly-zero gradient can be a legitimate transient
value (e.g. a dead ReLU unit on one particular batch), so treating that as
a hard failure would be over-strict; `param.grad is None` is what actually
indicates a wiring bug (frozen or disconnected), and that's what's checked.

### `scripts/sanity_check_models.py`

Runs four checks and prints one PASS/FAIL summary line each: **data
loading** (real Indiana CXR data if `data/raw` is populated, else
synthetic — both count as PASS, since Phase 2 only requires "whatever
data is available"), **baseline forward/backward**, **Stage 1
forward/backward**, **Stage 2 forward/backward**. Classification targets
are always synthetic random multi-label vectors, real image/report data or
not — there is no real CheXpert-style labeler yet (that's separate,
earlier-numbered Phase 2 work not covered by this checkpoint).

Defaults to **randomly-initialised weights**, not BiomedCLIP/PubMedBERT/
ImageNet (`--pretrained` opts in). This was a deliberate choice, not
laziness: this phase checks wiring, not representation quality, and the
default path should run in seconds without needing network access. Both
paths are verified working — `--pretrained` was run against the real
cached checkpoints on this machine and passes too.

Two more decisions worth flagging for review:

- **Each check reseeds `torch.manual_seed(0)` independently**, not just
  once at the top of `main()`. The first version only seeded once, so
  whichever check ran last inherited whatever RNG state the earlier checks
  happened to consume — a fresh attention head could land on an unlucky
  random init purely because of *how many* random numbers prior checks
  drew, not because of an actual bug. That made the script flaky and its
  failures non-reproducible. Reseeding per check makes every run
  deterministic and isolates each component's result from the others'.
- **Stage 1's sanity check freezes both encoder backbones** (only in this
  script — `ContrastiveAlignment` itself has no such flag, since real
  Stage 1 training needs to fine-tune both encoders). Empirically, jointly
  training two from-scratch ~100M-parameter transformers via a
  batch-of-8 InfoNCE loss doesn't reliably move the loss within 50 steps
  at any learning rate tried (1e-3 through 1e-2, with and without
  gradient clipping — the loss sits at `ln(8) ≈ 2.079`, i.e. the
  random-guessing point). That's a training-schedule problem (warmup,
  effective batch size), not a wiring problem, and squarely out of scope
  for this milestone. Freezing isolates exactly what this check is meant
  to verify — that `vision_proj` / `text_proj` / `logit_scale` and the
  InfoNCE loss are wired correctly — while the backbones' own gradient
  wiring is already exercised by the Stage 2 check, which trains a fresh
  head on top of the same `VisionEncoder`.

## What the sanity checks verify — and what they don't

**Verified:**
- Every module's forward pass produces the documented output shapes.
- Backward passes run without NaN/Inf.
- Every trainable parameter that's supposed to receive a gradient does.
- Loss decreases on a repeated batch (the model *can* learn something,
  i.e. it's not dead code).
- Both the random-init and real-pretrained-weight code paths work.

**Not verified, on purpose:**
- No claim about training quality, convergence speed, or final
  performance on real data — 50 steps on 8–16 examples is a wiring probe,
  not a training run.
- No evaluation metrics (AUROC, F1, etc.) — there's no held-out set, and
  no real CheXpert-style labels yet, only synthetic random targets.
- No hyperparameter tuning — the learning rate, gradient clip norm, and
  step count were chosen to make the *wiring check* reliable and
  deterministic, not to reflect what real training will use.
- Stage 1's local patch-word loss and Stage 2's text fusion are not
  implemented — they're documented stubs (see above), not silently
  missing.

## Immediate next milestone

Per the Phase 2 plan: build the real CheXpert-style labeler and
train/val/test manifest over the Indiana CXR reports (the dataset-pipeline
items this checkpoint didn't cover), then move on to real training runs
for Stage 1 and Stage 2 with actual labels in place of the synthetic
targets used here. Stage 1's local patch-word loss and Stage 2's gated
text fusion — both stubbed above — are natural follow-ups once global
alignment and plain classification are confirmed to train properly on
real data. Stage 3 (report generation) and the explainability channels
remain untouched, as scoped.
