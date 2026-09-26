# Explainability evaluation — do the heatmaps mean anything?

Date: 2026-09-21. Closes the proposal's Week-10/13 items (Grad-CAM / attention rollout, text-grounded heatmaps, sentence attribution, and a *quantitative* localisation check) and gives RQ2 and RQ4 their first measured answers. Every number here comes from `scripts/eval_grounding_chestxray14.py` and `scripts/eval_sentence_attribution.py`, whose outputs are in `docs/phase5_artifacts/v2/grounding_chestxray14_{summary.json,raw.csv}`, `sentence_attribution_{summary.json,raw.csv}` and the `grounding_examples/`, `sentence_attribution_examples/` folders. Nothing is trained here; all runs are inference over existing checkpoints.

## 1. Why NIH boxes, and what was compared

Indiana/Open-i has no pixel-level annotations, so the project's own held-out split cannot say whether a heatmap is *right*. NIH ChestX-ray14 ships **984 radiologist-drawn bounding boxes on 880 test images for 8 findings** (Wang et al., 2017). Those images come from a hospital the models never saw (the same split behind the 0.719 external AUROC in `v2_rebuild.md` §3.5), so this is a strict test: out-of-distribution images *and* a localisation criterion.

Box label → MedXVLM head label: Atelectasis, Cardiomegaly, Pneumonia, Pneumothorax as-is; Effusion → Pleural Effusion; Mass and Nodule → Lung Lesion (the CheXpert definition of that class); Infiltrate → Lung Opacity (**approximate**).

Methods scored (all in `src/medxvlm/explain/saliency.py`; every map is one scalar per patch on the model's 16×16 or 14×14 grid, bilinearly upsampled to 224×224):

| Method | What it is | Class-specific? |
|---|---|---|
| `stage2_attn_*` | Stage 2's per-label attention-pooling weights — the heatmap the app has always shown | yes |
| `stage2_gradcam_*` | Grad-CAM on the final patch tokens (channel weights = mean gradient of the label logit; ReLU of the weighted sum) | yes |
| `rollout_stage2_tower` | Attention rollout (Abnar & Zuidema 2020) from CLS through all 12 DINOv2 layers of the fine-tuned tower | **no** — one map per image |
| `text_v3b_findings` | Cosine of each projected patch to the prompt "There is *label*." in the Stage 1 space of the 0.734 zero-shot head | yes |
| `text_v2_report` | same, report-level v2 alignment | yes |
| `text_phase3_local_on` / `_off` | same, Phase 3 BiomedCLIP alignment with / without the local patch-word loss — **the RQ2 pair** | yes |
| `baseline_center` | always point at the image centre | — |
| `baseline_random_expected` | expected hit rate of a uniformly random point = box area / image area | — |

`_seed0` = the single checkpoint quoted in earlier docs; `_ensemble` = the four seeds' maps averaged, matching what the app serves.

Metrics per box: **pointing game** (is the map's hottest pixel inside the box; Zhang et al. 2018), **mass ratio** (share of min-max-normalised map mass inside the box ÷ the box's share of the image; 1.0 = uniform map), **area-matched IoU** (IoU between the box and the map's top-*k* pixels with *k* = box area). 95% CIs are Wilson intervals over boxes.

## 2. Results — pointing game on 984 boxes

| Method | Hit rate (95% CI) | Macro over 8 findings | Macro over 7 (excl. Cardiomegaly) | Mass ratio | Area-matched IoU |
|---|---:|---:|---:|---:|---:|
| **Stage 2 attention pooling, 4-seed ensemble** | **41.2 % [38.1, 44.3]** | **36.8 %** | **28.7 %** | **2.68** | **0.183** |
| Stage 2 attention pooling, seed 0 | 33.7 % [30.9, 36.8] | 28.2 % | 20.2 % | 2.33 | 0.127 |
| *baseline: image centre* | *21.5 % [19.1, 24.2]* | *19.3 %* | *8.2 %* | – | – |
| Stage 2 Grad-CAM, ensemble | 19.0 % [16.7, 21.6] | 16.9 % | 11.2 % | 1.96 | 0.081 |
| Text-grounded, v3b (finding sentences) | 11.3 % [9.5, 13.4] | 10.9 % | 9.2 % | 1.25 | 0.088 |
| Stage 2 Grad-CAM, seed 0 | 9.9 % [8.1, 11.9] | 8.2 % | 6.7 % | 1.91 | 0.051 |
| Text-grounded, v2 (report level) | 8.7 % [7.1, 10.7] | 7.7 % | 3.9 % | 1.05 | 0.060 |
| *baseline: random point (expected)* | *7.5 %* | *7.1 %* | *5.6 %* | *1.00* | – |
| Attention rollout (class-agnostic) | 6.4 % [5.0, 8.1] | 5.6 % | 2.2 % | 1.19 | 0.066 |
| Text-grounded, Phase 3 **with** local loss | 5.3 % [4.1, 6.9] | 5.6 % | 5.3 % | 0.95 | 0.034 |
| Text-grounded, Phase 3 **without** local loss | 3.8 % [2.7, 5.1] | 3.9 % | 3.4 % | 0.96 | 0.027 |

Per finding, ensemble attention pooling (n boxes / hit rate / random / centre):

| Finding | n | Attention ens. | Grad-CAM ens. | Text v3b | Random | Centre |
|---|---:|---:|---:|---:|---:|---:|
| Atelectasis | 180 | **33.3 %** | 12.8 % | 1.7 % | 3.2 % | 6.1 % |
| Cardiomegaly | 146 | 93.8 % | 56.8 % | 23.3 % | 17.6 % | **97.3 %** |
| Effusion | 153 | **41.8 %** | 13.1 % | 14.4 % | 6.9 % | 2.6 % |
| Infiltrate (≈ Lung Opacity) | 123 | **61.8 %** | 32.5 % | 13.0 % | 9.7 % | 22.8 % |
| Mass (→ Lung Lesion) | 85 | 12.9 % | 8.2 % | 10.6 % | 4.2 % | 10.6 % |
| Nodule (→ Lung Lesion) | 79 | 7.6 % | 0.0 % | 1.3 % | 0.5 % | 0.0 % |
| Pneumonia | 120 | **39.2 %** | 10.8 % | 15.0 % | 9.4 % | 14.2 % |
| Pneumothorax | 98 | 4.1 % | 1.0 % | 8.2 % | 5.2 % | 1.0 % |

Example overlays (white box = radiologist; panels = attention ens. / Grad-CAM ens. / rollout / text v3b): `docs/phase5_artifacts/v2/grounding_examples/*.jpg`, two per finding.

### What this says, plainly

1. **The classifier's own attention maps localise real findings.** 41 % of the time the hottest pixel is inside the radiologist's box, against 7.5 % for a random point and 21.5 % for the centre — and the centre baseline is inflated by Cardiomegaly, whose boxes cover the middle of every image (97 % centre hit). Excluding Cardiomegaly the gap is 28.7 % vs 8.2 % centre / 5.6 % random. For Atelectasis, Effusion, Infiltrate and Pneumonia the maps are 5–10× better than chance. This is the first validated explainability claim in the project; until now the heatmaps were "spot-checked qualitatively".
2. **Ensembling helps localisation as much as it helps AUROC**: 33.7 % → 41.2 % overall, and the per-class table shows why — the single seed had 0 % on Mass and Pneumothorax and 9 % on Pneumonia; averaging four seeds' maps removes one seed's idiosyncrasies.
3. **Where the classifier is blind, the maps are blind too.** Pneumothorax (4 %, at chance) and Nodule (8 %; boxes are 0.5 % of the image) match the classifier's weakest external AUROCs (0.634 pneumothorax) and its training data (0.8 % pneumothorax prevalence after the label fix). A heatmap cannot explain a decision the model is not making.
4. **Grad-CAM is worse than the attention weights on this architecture** (19 % vs 41 %), and is the noisiest method across seeds (10 % single → 19 % ensemble). The attention-pooling weights *are* the decision mechanism — Grad-CAM on the last tokens is a gradient approximation of it. The proposal asked for Grad-CAM; it was implemented and it is the inferior view, which is itself a useful finding.
5. **Class-agnostic saliency (rollout) is at chance** (6.4 %). Generic "what is salient in this X-ray" does not localise a named finding; the per-label queries are what does.
6. **RQ2 — does the local patch-word loss improve grounding? Not measurably at this scale.** Phase 3 with the local loss points inside the box 5.3 % of the time, without it 3.8 %; both are *below* the random-point baseline (7.5 %) with overlapping-to-adjacent CIs, and both have a mass ratio below 1 (their maps are, if anything, anti-correlated with the box). The hypothesis was that the local term would give clinically plausible localisation "even if global accuracy differs only slightly"; on 3,000 studies it gives none. The direction of the difference (on > off) is consistent with the hypothesis; the magnitude is not usable.
7. **RQ4 — does explainability quality track the encoder and its alignment? Yes. Most strongly the encoder's pretraining domain** (added 2026-09-26, `v2_rebuild.md` §3.9): the same classifier's attention maps hit the box 32% ± 6% of the time on X-ray DINOv2 (4 seeds), 27% on BiomedCLIP, 24% on OpenAI CLIP and 13% on ImageNet ViT, while their test AUROCs span only 0.78–0.84, so localisation separates encoders far more sharply than accuracy does. **For the Stage 1 alignment the link is weak, and appears only once the alignment learned finding sentences.** Text-grounded pointing goes Phase 3 (0.669 zero-shot AUROC) 5.3 % → v2 (0.565) 8.7 % → v3b (0.734) 11.3 %, i.e. the one alignment that carries real finding signal is the only one whose patch similarities beat a random point (CI [9.5, 13.4] vs 7.5). It is still far below the supervised attention maps: at this data scale the alignment's patch-level similarity is not a substitute for a supervised head, which is the same conclusion as the classification comparison (0.734 vs 0.868).

## 3. Sentence attribution — how much does the report writer look at the image?

`scripts/eval_sentence_attribution.py` re-runs the decoder teacher-forced over each generated report with attention weights exposed (eager attention, one extra pass ≈ 0.2 s) and, for every sentence, averages its tokens' attention onto the 64 visual-token positions over all 28 layers × 12 heads. Two numbers per sentence: the **visual share** — attention mass on the visual tokens ÷ total, reported with the sequence's first position (the attention sink every decoder-only LM has) removed — and the 8×8 **visual map** itself. Beam-3 decoding, the project default. For scale: the 64 visual tokens are roughly half of the positions a report token can attend to, so a decoder that used the image and the text equally would show a share near 0.5; the prompt's own ~30 text tokens receive about 0.10.

### 3.1 Visual share on the 377 frontal Indiana test images (2,032 generated sentences)

| Sentence kind (rule-based labeler on the sentence) | n | Visual share, sink-corrected | raw | Attention on prompt text | Map entropy (1 = uniform) |
|---|---:|---:|---:|---:|---:|
| Names a finding (POSITIVE / UNCERTAIN mention) | 100 | **0.027** | 0.014 | 0.100 | 0.88 |
| Negation only ("No pneumothorax…") | 459 | 0.013 | 0.007 | 0.099 | 0.93 |
| Other ("Heart size is normal.", boilerplate) | 1473 | 0.012 | 0.006 | 0.112 | 0.92 |
| **All sentences** | 2,032 | **0.013** (90th percentile 0.020) | | | |

**The generator attends to the image about 1 % of the time.** Sentences that assert a finding look twice as hard (2.7 %) as boilerplate and negations (1.2–1.3 %) — the difference is real and in the right direction — but 2.7 % is still an order of magnitude below the prompt text and 20× below parity. The maps are also nearly uniform (normalised entropy 0.88–0.93).

**One glance per image, reused for every sentence.** On 60 test reports (800 sentence pairs) the 64-dim visual maps of two *different sentences from the same report* correlate at **0.91** (median 0.93); mean maps of two *different images* correlate at 0.60. So the visual attention is image-dependent, but essentially not sentence-dependent: the decoder does not re-consult different regions when it moves from the heart to the lungs to the bones. The example panels in `docs/phase5_artifacts/v2/sentence_attribution_examples/indiana_*.jpg` show this directly — five sentences, five near-identical heatmaps.

### 3.2 When a sentence names the boxed finding, does its map point at the box? (NIH, 310 images / 320 boxes, 40 per finding, seed 0)

Out of distribution the generator is even more collapsed — 90 distinct reports for 310 images, and *"There are opacities in the left lower lobe … which may represent atelectasis or infiltrate"* for 67 of them regardless of the image (an OOD hallucination pattern worth knowing about). Only **97 sentences** name the finding of the box on their image; for those:

| Finding | Boxes | Sentences naming it | Pointing-game hit | Centre baseline |
|---|---:|---:|---:|---:|
| Atelectasis | 40 | 17 | 23.5 % | 5.9 % |
| Cardiomegaly | 40 | 28 | 89.3 % | 100.0 % |
| Effusion | 40 | 7 | 42.9 % | 0.0 % |
| Infiltrate | 40 | 40 | 50.0 % | 5.0 % |
| Mass | 40 | 0 | — | — |
| Nodule | 40 | 2 | 0.0 % | 0.0 % |
| Pneumonia | 40 | 1 | 100.0 % | 0.0 % |
| Pneumothorax | 40 | 2 | 0.0 % | 0.0 % |
| **All** | 320 | **97** | **54.6 %** [45, 64] | 32.0 % |
| All except Cardiomegaly | 280 | 69 | 40.6 % | 4.3 % |

Where the generator does name the finding, the glance it took is on the finding far more often than chance (mass ratio 2.8); Infiltrate 50 % and Atelectasis 24 % against a centre baseline of 5–6 %. Combined with §3.1 this says the *visual encoder side* of Stage 3 carries localised information about the finding (as it should — it is the frozen Stage 2 tower whose attention maps score 41 % in §2), but the *decoder* draws on it only faintly and identically for every sentence. That is the mechanism behind the 61 % modal-report share and the v2b collapse in `v2_rebuild.md` §3.6.4: the text is written mostly from the language prior, with the image acting as a weak global bias rather than a per-sentence source.

### 3.3 What follows for Stage 3

The measured next lever was to move the finding signal into the channel the decoder actually uses — text. **This was run as Stage 3 v2c** (`v2_rebuild.md` §3.7, 2026-09-21): the classifier ensemble's flagged findings are written into the prompt and checkpoints are selected by a generation metric. Outcome: device hallucination 0 % (v2 beam: 3 %), CE macro F1 0.165 vs 0.176 — within a ±0.10 bootstrap band whose ceiling on the 70-image subsample is 0.234 (the classifier flags only 18/70 images). Removing the findings from the prompt collapses the same checkpoint to one output, which is the attribution result of §3.1 seen from the other side: the decoder writes what its prompt says and looks at the image only through the classifier. Stronger visual coupling (more visual tokens, unfrozen projector with a grounding auxiliary loss) remains the untested longer-term alternative.

RQ3 restated with this evidence: a frozen tower + LoRA-tuned 1.5B decoder does produce coherent, report-style findings on this data (the collapse is broken, CE F1 0.176 ≥ CheXOne), but "strong visual grounding" — the premise of the hypothesis — is not what the trained model does; its outputs are coherent *because* they are mostly language-prior.

## 4. What was not done

- No box annotations exist for Indiana, so the localisation numbers are out-of-distribution by construction; in-distribution grounding could be better or worse.
- Grad-CAM was run on the final patch tokens only (no layer sweep); attention rollout without head-importance weighting; text grounding with a single prompt (the pos-only protocol). Each is the standard form, not a tuned one.
- The Infiltrate → Lung Opacity mapping is approximate; Mass/Nodule → Lung Lesion follows CheXpert's definition but the classifier's Lung Lesion AUROC is not validated externally.
- No human reader study; a pointing game is a proxy for "is this region relevant", not for "would a radiologist find this overlay useful".

## 5. Files

- `src/medxvlm/explain/saliency.py` — Grad-CAM, rollout, `TextGrounder`, upsampling helpers.
- `src/medxvlm/explain/sentence_attribution.py` — teacher-forced per-sentence visual attention.
- `src/medxvlm/inference.py` — `MedXVLMPipeline.heatmaps(image, labels, method)` and `.attribute_report(image, text)`; the app exposes both.
- `scripts/eval_grounding_chestxray14.py`, `scripts/eval_sentence_attribution.py` — the evaluations.
- `docs/phase5_artifacts/v2/grounding_chestxray14_summary.json` (+ `_raw.csv`, `grounding_examples/`), `sentence_attribution_summary.json` (+ `_raw.csv`, `sentence_attribution_examples/`).
