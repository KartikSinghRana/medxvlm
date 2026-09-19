# Phase 4 — External Baseline Reproduction: RadZero and CheXOne

Date: 2026-09-06
See [`external_repro_feasibility.md`](external_repro_feasibility.md) for the pre-setup feasibility check this phase started from.

## 1. What this document is (and isn't)

RadZero and CheXOne solve different problems — RadZero does zero-shot classification/grounding/segmentation from a small contrastive vision-language model; CheXOne does reasoning-enabled report generation and VQA from a 3B-parameter instruction-tuned VLM. There is no meaningful shared metric between them, so this document does **not** invent one. Instead:

- **RadZero**: we reproduced a real number ourselves, on our own hardware, using their public weights and their own OpenI test data.
- **CheXOne**: we confirmed the released weights actually run on our hardware and produce coherent, clinically-formatted output (a qualitative pipeline check), but did **not** reproduce its quantitative benchmarks — the required data (MIMIC-CXR via PhysioNet credentialing) was not obtainable in this phase. Every CheXOne number below is the authors' own, cited from arXiv:2604.00493, and labelled **"reported by the authors, not reproduced here."**

## 2. Environments

Two isolated Python 3.11 venvs under `external_repro/`, one per repo, neither merged into MedXVLM's own environment:

- `external_repro/radzero_venv` — cloned from [github.com/deepnoid-ai/RadZero](https://github.com/deepnoid-ai/RadZero) @ `656ae5f1af3f106e96c95542ce3ee5c0ee8777fc`.
- `external_repro/chexone_venv` — cloned from [github.com/YBZh/CheXOne](https://github.com/YBZh/CheXOne) @ `bed20f6a75b2a5dd84cac0b03e7c937f0a2b601a`.

**Deviation from each repo's pinned `requirements.txt` (documented, not silent):** this machine's GPU is an RTX 5080 (Blackwell, `sm_120`), which needs a torch build newer than either repo's pin (RadZero pins `torch==2.2.1`+cu121; that combination has no `sm_120` kernels and would fail at runtime). We installed current PyPI releases instead of the pinned versions:

```bash
# RadZero — minimal inference deps, not the full pinned training requirements.txt
python3 -m venv external_repro/radzero_venv
source external_repro/radzero_venv/bin/activate
pip install torch transformers open_clip_torch peft opencv-python-headless timm accelerate scikit-learn pillow pandas huggingface_hub pydicom
pip install "transformers==4.46.3"   # transformers 5.x changed a config-diffing contract RadZero's custom config class violates; 4.46.3 is compatible
```

```bash
# CheXOne — the README's own plain-transformers Quick Start path
python3 -m venv external_repro/chexone_venv
source external_repro/chexone_venv/bin/activate
pip install torch torchvision accelerate pillow qwen-vl-utils
pip install "git+https://github.com/huggingface/transformers"   # per CheXOne README, for Qwen2_5_VLForConditionalGeneration
```

Resulting versions: `torch==2.14.0+cu130` (both venvs), `transformers==4.46.3` (RadZero venv), `transformers==5.17.0.dev0` (CheXOne venv, built from source as instructed).

## 3. RadZero — reproduced zero-shot classification on OpenI

**Weights:** [`Deepnoid/RadZero`](https://huggingface.co/Deepnoid/RadZero) on Hugging Face, revision `1e6e7d6295a13aa0a4a7e1c1e2bb36b9ef4e02e8`, loaded via `AutoModel.from_pretrained(..., trust_remote_code=True)`. 209.4M parameters, fp32. Peak GPU memory: well under 1GB.

**Data:** RadZero's own OpenI test split, shipped in its HF repo at `data/OpenI/` — `custom.csv` (7,470 rows of Indiana MeSH annotations) and `openi_multi_label_text.json` (the 18 class prompts, e.g. `"There is Atelectasis"`). The images are the same Kaggle "chest-xrays-indiana-university" source already present locally at `medxvlm/data/raw/images/images_normalized/` (confirmed by exact filename match — same 7,470 images).

**Exact command run:**
```bash
cd external_repro
source radzero_venv/bin/activate
python3 run_radzero_openi_eval.py --n 1000 --seed 0 --out-prefix radzero_openi_eval
```
(`external_repro/run_radzero_openi_eval.py` in this repo.)

**Ground-truth caveat (documented, not hidden):** RadZero's shipped `custom.csv` contains only free-text Indiana MeSH annotations (`labels_major`, `labels_automatic`), not a ready 18-class label matrix. The original CARZero preprocessing code that RadZero's own `data/README.md` credits for this mapping (`github.com/laihaoran/CARZero`) was not obtained for this phase. We built ground truth with our own keyword-matching heuristic against the MeSH strings (see `matches()` in the eval script) — a best-effort, fully documented substitute, not the authors' exact label-mapping code. **This means our AUROC is a reproduction of RadZero's model on RadZero's own data, evaluated against our own label mapping — not a byte-exact replication of their published number.**

**Compute-cost caveat:** each (image, class) pair costs ~110–130ms on this GPU (measured), because RadZero's `compute_logits` path recomputes the full vision forward pass per call rather than caching image embeddings across the 18 class prompts. The full 7,470-image × 18-class grid would take roughly 4+ hours single-GPU; we instead evaluated a **fixed random subsample of 1,000 images (seed=0)**, disclosed here rather than presented as a full-dataset number.

**Results (reproduced by us, n=1,000 images, seed=0, wall time 2,647s ≈ 44 min):**

| Class | Positives (of 1,000) | AUROC |
|---|---:|---:|
| Hernia | 15 | 0.995 |
| Pulmonary Mass | 4 | 0.975 |
| Cardiomegaly | 81 | 0.953 |
| Pleural Effusion | 46 | 0.923 |
| Pulmonary Emphysema | 32 | 0.926 |
| Pulmonary Edema | 22 | 0.905 |
| Pulmonary Fibrosis | 5 | 0.910 |
| Pneumothorax | 8 | 0.899 |
| Pneumonia | 20 | 0.886 |
| Lung Infiltration | 25 | 0.856 |
| Atelectasis | 85 | 0.853 |
| Lung Opacity | 101 | 0.804 |
| Bone Fracture | 22 | 0.796 |
| Pleural Thickening | 15 | 0.758 |
| Calcified Granuloma | 71 | 0.732 |
| Lung Granuloma | 57 | 0.705 |
| Lung Nodule | 23 | 0.616 |
| Abnormal Lesion | 0 | excluded — zero positives in this sample, cannot compute AUROC |
| **Macro AUROC (17 valid classes)** | | **0.852** |

Full per-image predictions: `external_repro/radzero_openi_eval_n1000_seed0.csv`. Summary: `external_repro/radzero_openi_eval_n1000_seed0_summary.json`.

**Paper-reported number, for reference (arXiv:2504.07416, Table 5):** RadZero (518px, the full model) reports **mean AUC = 0.847** on OpenI ("OI" column) using their own CARZero-derived label mapping and the full 7,470-image test set — labelled **reported by the authors, not reproduced here**, since it uses their exact label mapping and full dataset, neither of which we replicated exactly. Our independently-measured macro AUROC of **0.852** on a 1,000-image subsample with our own keyword-based labels lands within ~0.5 points of their reported 0.847 — a genuinely independent measurement that happens to corroborate their result, not a copy of it. Per-class spread is wide (0.616–0.995): rarer/coarser categories (Lung Nodule, Lung/Calcified Granuloma) score lower, plausibly reflecting both harder visual discrimination and noisier keyword-derived labels for those specific classes (e.g. "Lung Granuloma" vs. "Calcified Granuloma" overlap in our heuristic, see §3's caveat).

## 4. CheXOne — qualitative pipeline confirmation (no benchmark reproduction)

**Weights:** [`StanfordAIMI/CheXOne`](https://huggingface.co/StanfordAIMI/CheXOne) on Hugging Face, revision `0c350e6852ea08f9d9baf3b7595c1a10d4849927` — Qwen2.5-VL-3B-Instruct backbone, post-trained with SFT + GRPO. bf16, ~8GB on disk.

**Exact command run:**
```bash
cd external_repro
source chexone_venv/bin/activate
python3 run_chexone_demo.py "medxvlm/data/raw/images/images_normalized/1000_IM-0003-1001.dcm.png"
```
(`external_repro/run_chexone_demo.py`, adapted directly from the README's "Quick Start" `transformers` code — not the `swift infer`/vLLM fast-inference path, which requires the ms-swift CLI and a `rexvqa-test-reason.json` test file that is not shipped and must be built from PhysioNet-credentialed MIMIC-CXR images plus gated CheXInstruct-v2/CheXReason annotations — access we do not have.)

**Measured on this hardware:**
- Load time: 175s (mostly download; ~7s once cached).
- Peak GPU memory: **7.51GB allocated** — comfortably inside the 16GB RTX 5080, confirming the feasibility check's estimate.
- Generation: 20.9s / 257 tokens for a full reasoning-mode trace on the demo image; 4.5s / 35 tokens for instruct-mode on a local Indiana/Open-i image (out-of-protocol spot check, not a benchmark).

**Sample output (reasoning mode, CheXOne's own demo image):**
> **Final Answer:** The cardiac silhouette size is normal. The mediastinal contours are normal. There is persistent diffuse increase in interstitial opacities seen. There is no focal consolidation or pneumothorax. There is no pleural effusion. The lungs are clear.

**Sample output (instruct mode, one local Indiana/Open-i image, clearly out-of-protocol — not part of any official benchmark):**
> The cardiomediastinal silhouette is normal. There is no focal consolidation or pneumothorax. The lungs are clear. There is no pleural effusion.

This confirms the pipeline genuinely works end-to-end on our hardware and produces coherent, radiology-report-formatted text with an explicit reasoning trace. **We did not compute BLEU/RadGraph/etc. against any reference report for this image** — that would require the ReXRank/CheXbench evaluation harness and reference annotations we don't have, and doing so on one unvalidated sample would not be a meaningful metric regardless.

**Paper-reported numbers, for reference (arXiv:2604.00493) — reported by the authors, not reproduced here:**

| Task | Metric | CheXOne | Best baseline reported |
|---|---|---|---|
| VQA — Presence assessment (ReXVQA) | Accuracy | 0.947 | ChestX-Reasoner 0.603 |
| VQA — Anatomical localization (ReXVQA) | Accuracy | 0.931 | ChestX-Reasoner 0.651 |
| VQA — Negation detection (ReXVQA) | Accuracy | 0.988 | CheXagent 0.811 |
| VQA — Differential diagnosis (ReXVQA) | Accuracy | 0.951 | ChestX-Reasoner 0.758 |
| VQA — Geometric reasoning (ReXVQA) | Accuracy | 0.883 | ChestX-Reasoner 0.614 |
| View classification (MIMIC-CXR) | Accuracy | 0.998 | CheXagent (tied) |
| Temporal classification (Chest ImaGenome) | Accuracy | 0.646 | CheXagent 0.604 |
| Findings generation (ReXGradient, ReXRank protocol) | 1/RadCliQ / BertScore / BLEU / RadGraph / RaTEScore / SembScore | 1.116 / 0.483 / 0.229 / 0.21 / 0.535 / 0.498 | SOTA among evaluated models (paper does not give single best-baseline number in main text) |
| Progression generation (MIMIC-CXR) | same 6 metrics | 0.937 / 0.530 / 0.142 / 0.239 / 0.543 / 0.580 | SOTA among evaluated models |
| Phrase grounding | mIoU / mAP | 0.608 / 0.835 | not stated in extracted main text |
| Reasoning factuality (RadGraph-XL entity overlap) | Score | 0.173 | ChestX-Reasoner "comparable" |
| Reader study: report quality vs. residents (attending-rated) | mean Likert [-10,10] | CheXOne 4.80 vs. resident 5.15 (not significant, p>0.1) | — |
| Reader study: reports comparable/better than residents | % of cases | 55% | — |
| Reader study: resident time-to-final-report | seconds | 64.9±34.2 (CheXOne-assisted) vs. 178.6±116.6 (from scratch) | — |

IU X-ray (i.e. the same Indiana/Open-i dataset RadZero uses) is one of CheXOne's four findings-generation benchmarks, used as an out-of-distribution check — but the paper's main text only gives exact metric values for the ReXGradient dataset ("detailed results ... IU Xray ... provided in the Supplementary Material," which we did not have access to). We do not report an IU X-ray number for CheXOne because we could not verify one from the source we had.

## 5. What each model is actually good at (qualitative)

- **RadZero** is a small (~200M-parameter), fast, zero-shot contrastive alignment model. Its strength is *built-in, similarity-based explainability*: every classification decision comes with a pixel-level VL similarity map for free, with no separate Grad-CAM pass, and it generalizes to open-vocabulary text prompts without retraining. It does not generate free text, and its classification signal is a single similarity probability per (image, phrase) pair — useful for screening/localization, not for narrative reporting.
- **CheXOne** is a 3B-parameter instruction-tuned foundation model. Its strength is *reasoning-enabled report generation and VQA*: it produces full findings sections with an explicit, inspectable reasoning trace, handles negation and differential diagnosis, and in a reader study its drafts were rated comparable to resident-written reports by attendings. It has no built-in pixel-level grounding map analogous to RadZero's (its "visual grounding" evaluation is box/IoU-based, from a different mechanism), and it is ~15x larger.
- These are complementary, not competing, systems: RadZero for fast, explainable zero-shot screening/localization; CheXOne for narrative reporting and complex reasoning at foundation-model cost.

## 6. Explicit skip list

- **RadZero** — not run: PadChest, CheXpert, ChestXray14, ChestXDet10 (grounding), MS-CXR, SIIM, RSNA. Reason: these need separate downloads, some gated (CheXpert requires a Stanford license form; PadChest requires a BIMCV data request). Only OpenI, already on disk for this project, was run.
- **RadZero** — full 7,470-image OpenI test set not run: 4+ hour single-GPU cost for this phase's time budget; ran a disclosed 1,000-image random subsample instead.
- **RadZero** — exact CARZero label-mapping code for OpenI not obtained; used our own documented keyword-matching heuristic instead (see §3).
- **CheXOne** — no quantitative benchmark reproduced at all (ReXVQA, ReXRank findings/progression generation, phrase/abnormality grounding, reasoning factuality, reader study). Reason: all require either PhysioNet-credentialed MIMIC-CXR access, Google-Drive-gated CheXInstruct-v2/CheXReason annotations, or a multi-radiologist reader study — none obtainable inside this phase. All CheXOne numbers in §4 are cited from arXiv:2604.00493, not reproduced.
- **CheXOne** — GRPO/SFT training code not exercised (not needed; only inference was in scope for this phase).

## 7. Files

- `external_repro/run_radzero_openi_eval.py` — RadZero OpenI zero-shot classification eval script (this phase's own code, not part of either external repo).
- `external_repro/run_chexone_demo.py` — CheXOne qualitative Quick Start runner.
- `external_repro/radzero_openi_eval_n1000_seed0.csv` / `_summary.json` — raw per-image predictions and summary metrics from the RadZero run.
- `external_repro/radzero_venv/`, `external_repro/chexone_venv/` — isolated environments (not committed; see their `pip freeze` for exact resolved versions if needed).
