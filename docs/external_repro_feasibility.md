# Phase 4 — Feasibility Check: RadZero and CheXOne

Date: 2026-09-06
Purpose: decide, *before* any environment setup, whether each external repo can actually be run here, and on what.

## Summary verdict

| | RadZero | CheXOne |
|---|---|---|
| Code public? | Yes — [github.com/deepnoid-ai/RadZero](https://github.com/deepnoid-ai/RadZero) | Yes — [github.com/YBZh/CheXOne](https://github.com/YBZh/CheXOne) |
| Weights public? | Yes — [Deepnoid/RadZero](https://huggingface.co/Deepnoid/RadZero) on Hugging Face | Yes — [StanfordAIMI/CheXOne](https://huggingface.co/StanfordAIMI/CheXOne) on Hugging Face |
| Model size | ~0.2B params (DINOv2-base vision encoder + all-mpnet-base-v2 text encoder), fp32 | ~3B params (Qwen2.5-VL-3B-Instruct backbone), bf16 recommended |
| Fits in 16GB VRAM? | Yes, trivially (fp32 weights are well under 1GB; this could run on CPU) | Yes — bf16 weights ≈ 6GB, comfortably inside the 16GB RTX 5080 (14GB currently free) |
| Native quantitative eval reproducible here? | **Yes** — zero-shot classification on OpenI/Indiana, which we already have on disk (`data/raw/`, 3,851 reports / 7,466 images) | **No, not the paper's benchmark numbers** — see below |
| Native qualitative run reproducible here? | Yes (also grounding/segmentation, but those need MS-CXR/SIIM/RSNA which we have not downloaded) | Yes — single-image report generation via the README's plain `transformers` Quick Start |
| License | CC BY-NC 4.0 (non-commercial — fine for a B.Sc. project) | Apache 2.0 for the framework; model/data follow original sources |

Neither repo is blocked outright. The originally anticipated risk profile inverts for CheXOne: weights and code are public and the base model is small enough to run, but the paper's own **benchmark protocol** requires data we cannot get in the time available for this phase (see below). That is the actual constraint, not GPU memory.

## RadZero — detail

- Repo: code + a `requirements.txt`/`pyproject.toml`, MIT-adjacent-style research repo (license file present, marked "Other"/CC BY-NC 4.0 on the model card).
- Weights: `Deepnoid/RadZero` on HF, loadable via plain `transformers` (`AutoModel`, `trust_remote_code=True`). No training required to reproduce zero-shot numbers.
- Native eval tasks and datasets, per `exp/cxr_pt/inference/inference.py` and the README: zero-shot **classification** on OpenI, PadChest, CheXpert, ChestXray14, ChestXDet10; **grounding** on ChestXDet10, MS-CXR; **segmentation** on SIIM, RSNA.
- OpenI *is* the Indiana University / Open-i dataset already sitting in `medxvlm/data/raw/` for this project — fully public, no credentialing needed. This is the one native benchmark we can run end-to-end without new data access. PadChest/CheXpert/MS-CXR/SIIM/RSNA would need separate downloads (some gated, e.g. CheXpert requires a Stanford license form) — out of scope for this phase given the OpenI path already gives a real, native number.
- GPU requirement: not stated explicitly in the README; the model is small enough (~0.2B params, fp32) that it is a non-issue on a 16GB card.
- Plan: **run.** Set up an isolated venv, use HF weights directly, run their classification inference path against OpenI-format inputs (either their own preprocessed OpenI split from `data/` on the HF repo, if available, or our local Indiana data reshaped to the input format their `Inference` class expects — `image_path`/`text_path` per dataset).

## CheXOne — detail

- Repo: a fork of the ModelScope **ms-swift** training/serving framework with CheXOne-specific assets under `examples/train/chexone/`. This is a large, general-purpose LLM fine-tuning framework, not a small standalone script — expect more surface area and more places for version friction than RadZero.
- Weights: `StanfordAIMI/CheXOne` on HF, a post-trained **Qwen2.5-VL-3B-Instruct** checkpoint. The README ships a complete, minimal `transformers`-only Quick Start (no `swift`/`vllm` dependency required) that loads the model and generates a report for one image — this we can run as-is.
- GPU requirement: not stated explicitly; 3B params in bf16 ≈ 6GB weights, comfortably within 16GB even accounting for KV cache and the vision tower. Not a blocker.
- **The actual blocker is data, not compute or code availability.** CheXOne's own quantitative benchmarks (CheXbench / RexVQA-style report-generation and reasoning evaluation, referenced in `examples/train/chexone/prepare_testing/3_inference_code/CheXOne.sh`) require:
  1. A test JSON (e.g. `rexvqa-test-reason.json`) that is **not included in the repo** — it must be built from the CheXInstruct-v2/CheXReason text annotations (Google-Drive-hosted, access requested separately) combined with CXR images from their source datasets, which for the reasoning/report-generation splits is predominantly **MIMIC-CXR** — a PhysioNet-credentialed dataset requiring a data-use agreement and a multi-day approval process we cannot complete inside this phase.
  2. No standalone metric-computation script is shipped for CheXbench in the repo (only `swift/plugin/metric.py`, a generic ms-swift training-time metric hook, and ad-hoc reader-study collection scripts under `Reader-Study/`) — so even with the data, reproducing their headline numbers exactly would mean partially re-deriving their scoring methodology.
- Plan: **run the Quick Start inference qualitatively** (load the public HF weights, generate a report for a sample image — their own demo image and, separately, one image from our local Indiana/Open-i set, clearly labelled as an out-of-protocol spot-check, not a benchmark run) to confirm the pipeline genuinely works end-to-end on our hardware. **Do not attempt to reproduce CheXOne's quantitative benchmark table.** For every CheXOne number in the final comparison, cite the paper's own reported figure, labelled "reported by the authors, not reproduced here."

## What this means for the rest of Phase 4

- Step 16 (isolated environments): build venvs for both — both are worth setting up since both can actually execute.
- Step 17 (native benchmark): RadZero → real zero-shot classification numbers on OpenI, reproduced here. CheXOne → qualitative pipeline confirmation only, no numeric benchmark reproduction.
- Step 18 (comparison): the table will carry a mix of reproduced (RadZero, OpenI zero-shot classification) and cited-only (CheXOne, all of CheXbench/reasoning/report-quality numbers) figures, explicitly labelled per the hard constraint against fabricating numbers.
