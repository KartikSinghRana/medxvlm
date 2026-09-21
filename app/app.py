"""MedXVLM demo: image upload -> classification + attention overlays + generated report.

The proposal's Week-14 deliverable. This file is deliberately a thin view over
`medxvlm.inference.MedXVLMPipeline`; no model code lives here.

Run from the repo root:
    PYTHONPATH=src streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.inference import REPORT_CHECKPOINTS, MedXVLMPipeline  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images" / "images_normalized"
ARTIFACTS = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"

# Classes whose held-out test support is single-digit after the label fix
# (docs/v2_rebuild.md §3.2): their AUROCs are not meaningful, and the app says so.
THIN_SUPPORT = {"Pneumothorax": 1, "Pleural Other": 3, "Consolidation": 5, "Enlarged Cardiomediastinum": 0}

st.set_page_config(page_title="MedXVLM demo", page_icon="🩻", layout="wide")


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading the classifier…")
def get_pipeline() -> MedXVLMPipeline:
    return MedXVLMPipeline()


@st.cache_data
def load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(MANIFEST)
    return df[(df.split == "test") & (df.projection == "Frontal")].reset_index(drop=True)


@st.cache_data
def load_model_card() -> dict:
    card: dict = {}
    cls = ARTIFACTS / "v2_classification_summary.json"
    rep = ARTIFACTS / "report_generation_scores_v2.json"
    nih = ARTIFACTS / "chestxray14_external_eval_summary.json"
    if cls.exists():
        s = json.load(open(cls))["systems"]
        card["home_auroc_8"] = s["v2_stage2_image_only"]["macro_auroc_8_shared_classes"]
        card["home_auroc_13"] = s["v2_stage2_image_only"]["macro_auroc_all_valid_classes"]
        card["radzero_auroc_8"] = s["radzero_zero_shot_rescored"]["macro_auroc_8_shared_classes"]
        card["per_class_auroc"] = s["v2_stage2_image_only"]["per_class_auroc"]
        card["per_class_support"] = s["v2_stage2_image_only"]["per_class_positive_support"]
    if rep.exists():
        s = json.load(open(rep))["systems"]
        card["ce_f1_v2"] = s["v2_stage3"]["frontal_70"]["clinical_efficacy"]["macro_f1"]
        card["ce_f1_chexone"] = s["chexone_instruct"]["frontal_70"]["clinical_efficacy"]["macro_f1"]
        card["distinct_v2"] = s["v2_stage3"]["all_150"]["collapse"]["n_distinct"]
    if nih.exists():
        card["nih_auroc_8"] = json.load(open(nih))["macro_auroc_8_mapped_classes"]
    grounding = ARTIFACTS / "grounding_chestxray14_summary.json"
    if grounding.exists():
        g = json.load(open(grounding))
        pm = g["per_method"]
        card["grounding"] = {
            "n_boxes": g["n_boxes"],
            "attention": pm["stage2_attn_ensemble"]["overall"]["hit_rate"],
            "gradcam": pm["stage2_gradcam_ensemble"]["overall"]["hit_rate"],
            "text": pm["text_v3b_findings"]["overall"]["hit_rate"],
            "random": pm["baseline_random_expected"]["overall"]["hit_rate"],
            "center": pm["baseline_center"]["overall"]["hit_rate"],
        }
    v2c = ARTIFACTS / "report_generation_scores_v2c.json"
    if v2c.exists():
        s = json.load(open(v2c))["systems"]
        card["ce_f1_v2c"] = s["v2c_stage3"]["frontal_70"]["clinical_efficacy"]["macro_f1"]
        card["ce_f1_v2_beam"] = s["v2_stage3_beam3"]["frontal_70"]["clinical_efficacy"]["macro_f1"]
        card["ce_f1_ceiling"] = s["classifier_verbalised"]["frontal_70"]["clinical_efficacy"]["macro_f1"]
    attribution = ARTIFACTS / "sentence_attribution_summary.json"
    if attribution.exists():
        card["attribution"] = json.load(open(attribution)).get("indiana_frontal_test", {})
    return card


def true_labels(row: pd.Series) -> dict[str, str]:
    """Raw labeler value -> display string (1 positive, 0 negative, -1 uncertain, NaN not mentioned)."""
    out = {}
    for label in CHEXPERT_LABELS:
        v = row[label]
        out[label] = "not mentioned" if pd.isna(v) else {1.0: "positive", 0.0: "negative", -1.0: "uncertain"}[float(v)]
    return out


def _pct(card: dict, key: str) -> str:
    v = card.get("grounding", {}).get(key)
    return "—" if v is None else f"{v * 100:.0f}%"


# ---------------------------------------------------------------- sidebar
pipeline = get_pipeline()
card = load_model_card()

with st.sidebar:
    st.header("Model")
    n_models = pipeline.classifier_meta.get("n_models", 1)
    st.caption(
        f"Device: `{pipeline.device}` · classifier: **{n_models}-seed ensemble** of `classifier_v2` "
        f"({pipeline.classifier_meta['backbone']}, image-only) — 0.868 macro AUROC vs 0.837 ± 0.015 for a single seed."
        if n_models > 1 else
        f"Device: `{pipeline.device}` · classifier `{pipeline.classifier_checkpoint.relative_to(REPO_ROOT)}` "
        f"({pipeline.classifier_meta['backbone']}, image-only)"
    )
    if pipeline.classifier_meta.get("skipped"):
        st.caption(f"Ensemble members not found: {pipeline.classifier_meta['skipped']}")
    enable_report = st.toggle(
        "Enable report generation",
        value=False,
        help="Loads the Qwen2.5-1.5B LoRA decoder (~7 GB VRAM). Off by default to keep the GPU free.",
    )
    REPORT_MODEL_LABELS = {
        "v2": f"v2 — writes from the image (CE F1 {card.get('ce_f1_v2_beam', 0.176):.3f} beam, ~3% invented devices)",
        "v2c": f"v2c — findings-conditioned (CE F1 {card.get('ce_f1_v2c', 0.165):.3f} greedy, 0% invented devices)",
    }
    report_model = st.radio("Report model", list(REPORT_MODEL_LABELS), format_func=REPORT_MODEL_LABELS.get, disabled=not enable_report,
                            help="v2c puts the classifier ensemble's flagged findings into the prompt, so the report is a verbalisation of the "
                                 "classifier: no invented devices, but it cannot know more than the classifier does (docs/v2_rebuild.md §3.7).")
    wanted = REPORT_CHECKPOINTS[report_model]
    if enable_report and (pipeline.report_generator is None or pipeline.report_checkpoint != wanted):
        if not wanted.exists():
            st.warning(f"Checkpoint missing: {wanted.relative_to(REPO_ROOT)}")
        else:
            with st.spinner(f"Loading report generator {report_model} (≈10 s)…"):
                pipeline.load_report_generator(wanted)
            st.session_state.pop("last_report", None)
    if not enable_report and pipeline.report_generator is not None:
        pipeline.unload_report_generator()
    if pipeline.report_generator is not None:
        m = pipeline.report_meta
        st.caption(f"Generator: `{m['decoder']}` + LoRA, epoch {m['epoch']}, {m['visual_tokens']} visual tokens, on `{m['vision_checkpoint']}`"
                   + (", findings-conditioned prompt." if m.get("findings_conditioning") else "."))
    if pipeline.device.type == "cuda":
        st.caption(f"GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    with st.expander("Model card — measured numbers", expanded=False):
        if card:
            st.markdown(
                f"""
**Classification (image-only, held-out Indiana test split, 377 frontal images)**
- Macro AUROC, 8 classes shared with RadZero: single seed **{card.get('home_auroc_8', float('nan')):.3f}** (0.837 ± 0.015 over 4 seeds); **4-seed ensemble 0.868** — RadZero on the same images: {card.get('radzero_auroc_8', float('nan')):.3f}
- Macro AUROC, all 13 measurable classes: **{card.get('home_auroc_13', float('nan')):.3f}**
- On an unseen hospital (NIH ChestX-ray14 official test split, 25,596 images): **{card.get('nih_auroc_8', float('nan')):.3f}**

**Report generation (70 frontal test images)**
- Clinical-efficacy macro F1: **{card.get('ce_f1_v2', float('nan')):.3f}** (CheXOne-instruct on the same images: {card.get('ce_f1_chexone', float('nan')):.3f})
- Distinct outputs over 150 test images: {card.get('distinct_v2', '?')} (the previous model produced 5)
- Known failure: mentions a device or surgery that is not there in ~10% of frontal cases with greedy decoding, ~3% with beam search (the app default)
- **v2c, findings-conditioned** (classifier findings written into the prompt): CE F1 {card.get('ce_f1_v2c', float('nan')):.3f} greedy, **0% hallucination**; a perfect verbaliser of the classifier would score {card.get('ce_f1_ceiling', float('nan')):.3f} on these 70 images, so the two report models are within noise of each other (±0.10) — v2c's gain is honesty, not recall
- Zero-shot (prompt-based) classification: best variant 0.734 macro AUROC vs RadZero 0.868 — data-scale-limited, not used by this app

**Explainability (pointing game on {card.get('grounding', {}).get('n_boxes', '—')} radiologist boxes, NIH ChestX-ray14)**
- Heatmap argmax inside the box: attention pooling **{_pct(card, 'attention')}**, Grad-CAM {_pct(card, 'gradcam')}, text-grounded {_pct(card, 'text')} — random point {_pct(card, 'random')}, image centre {_pct(card, 'center')}
- Generated sentences put only ~{card.get('attribution', {}).get('visual_share_excl_sink_mean_all', float('nan')) * 100:.0f}% of their attention on the image tokens (sentence attribution below) — a measured reason the report writer is weak

Sources: `docs/v2_rebuild.md`, `docs/status_report.md`, `docs/explainability_evaluation.md`.
"""
            )
        else:
            st.info("Evaluation artefacts not found under docs/phase5_artifacts/v2/.")


# ---------------------------------------------------------------- header
st.title("MedXVLM — chest X-ray findings demo")
st.error(
    "**Research prototype, not a medical device.** Trained on ~3,000 outpatient X-rays from one site. "
    "Outputs are for demonstrating the pipeline, not for clinical use.",
    icon="⚠️",
)

# ---------------------------------------------------------------- input
manifest = load_manifest()
image: Image.Image | None = None
row: pd.Series | None = None

# why a radio, not st.tabs: Streamlit renders every tab's widgets on each run,
# so a selectbox in a second tab always has a value and would silently override
# an uploaded file. One explicit source makes the input unambiguous.
sources = ["Held-out test image", "Upload an X-ray"] if not manifest.empty else ["Upload an X-ray"]
source = st.radio("Input source", sources, horizontal=True)

if source == "Upload an X-ray":
    up = st.file_uploader(
        "Frontal chest X-ray (PNG/JPG). Lateral views are out of distribution for this model.",
        type=["png", "jpg", "jpeg"],
    )
    if up is not None:
        image = Image.open(up)
else:
    only_abnormal = st.checkbox("Abnormal studies only", value=True)
    pool = manifest[manifest["No Finding"] != 1.0] if only_abnormal else manifest
    choice = st.selectbox(f"{len(pool)} frontal test images (never used in training)", pool.filename.tolist())
    if choice:
        row = pool[pool.filename == choice].iloc[0]
        path = IMAGES_DIR / choice
        if path.exists():
            image = Image.open(path)
        else:
            st.warning(f"Image file missing: {path}")

if image is None:
    st.info("Upload an image to run the model.")
    st.stop()

# ---------------------------------------------------------------- classification
result = pipeline.classify(image)
ranked = result.ranked()

left, right = st.columns([1, 1.2])
with left:
    st.subheader("Input")
    st.image(image, width="stretch")
    if row is not None:
        with st.expander("Reference report (ground truth for this image)", expanded=False):
            st.write(row.report_text)

with right:
    st.subheader("Findings — probabilities")
    st.caption(
        "Per-label sigmoid outputs of the image-only classifier. Shown as probabilities, not decisions: "
        "no operating threshold has been chosen or validated."
    )
    df = pd.DataFrame(ranked, columns=["label", "probability"])
    df["thin test support"] = df.label.map(lambda l: l in THIN_SUPPORT)
    if row is not None:
        truth = true_labels(row)
        df["ground truth"] = df.label.map(truth)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition(alt.datum["thin test support"], alt.value("#b6bac2"), alt.value("#2f6fed")),
            tooltip=[c for c in df.columns],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, width="stretch")
    thin = ", ".join(f"{k} (n={v})" for k, v in THIN_SUPPORT.items())
    st.caption(f"Grey bars: classes with too few positive test cases to have a meaningful validated score — {thin}.")
    if row is not None:
        st.dataframe(df.set_index("label")[["probability", "ground truth"]].style.format({"probability": "{:.3f}"}), width="stretch", height=300)

# ---------------------------------------------------------------- explainability
st.subheader("Where the model looked")
g = card.get("grounding", {})
METHOD_LABELS = {
    "attention": f"Attention pooling (classifier's own weights) — {_pct(card, 'attention')} pointing-game accuracy" if g else "Attention pooling (classifier's own weights)",
    "gradcam": f"Grad-CAM on the patch tokens — {_pct(card, 'gradcam')}" if g else "Grad-CAM on the patch tokens",
    "text": f"Text-grounded (Stage 1 patch–prompt similarity) — {_pct(card, 'text')}" if g else "Text-grounded (Stage 1 patch–prompt similarity)",
}
method = st.radio("Heatmap method", list(METHOD_LABELS), format_func=METHOD_LABELS.get, horizontal=True)
if g:
    st.caption(
        f"Accuracies are the share of {g['n_boxes']} radiologist-drawn boxes (NIH ChestX-ray14, a hospital the model never saw) whose "
        f"finding the heatmap's hottest pixel lands inside. Baselines on the same boxes: a random point {_pct(card, 'random')}, "
        f"always the image centre {_pct(card, 'center')}. Details: `docs/explainability_evaluation.md`."
    )
else:
    st.caption("Heatmaps have not been validated against radiologist annotations in this build.")
default_labels = [l for l, _ in ranked[:3]]
chosen = st.multiselect("Labels to visualise", CHEXPERT_LABELS, default=default_labels)
if chosen:
    try:
        overlays = pipeline.heatmaps(image, chosen, method=method)
    except FileNotFoundError as exc:
        st.warning(f"Checkpoint for this method is missing: {exc}")
        overlays = {}
    cols = st.columns(min(len(chosen), 4)) if overlays else []
    for i, label in enumerate(l for l in chosen if l in overlays):
        with cols[i % len(cols)]:
            st.image(overlays[label], caption=f"{label} — p = {result.probabilities[label]:.2f}", width="stretch")
    if method == "text":
        st.caption("Text-grounded maps come from the zero-shot alignment head (0.734 AUROC), not the classifier whose probabilities are shown above.")

# ---------------------------------------------------------------- report generation
st.subheader("Generated findings")
if not enable_report:
    st.info("Turn on **Enable report generation** in the sidebar to load the decoder (~7 GB VRAM).")
else:
    max_new = st.slider("Max new tokens", 32, 256, 128, 32)
    is_v2c = bool(pipeline.report_meta.get("findings_conditioning"))
    decoding_options = ["Beam search (3 beams, repetition penalty 1.2)", "Greedy"]
    decoding_choice = st.radio(
        "Decoding", decoding_options, index=1 if is_v2c else 0, horizontal=True,
        help="v2: beam search raised clinical-efficacy F1 0.152 → 0.176 and cut invented devices 10% → 3%. "
             "v2c: greedy scores higher (0.165 vs 0.135) — beam smooths away the short finding sentences the prompt induces.",
    )
    decoding = {"num_beams": 3, "repetition_penalty": 1.2} if decoding_choice.startswith("Beam") else {"num_beams": 1}
    if st.button("Generate findings", type="primary"):
        with st.spinner("Generating (≈5–15 s on GPU; minutes on CPU)…"):
            report = pipeline.generate_report(image, max_new_tokens=max_new, decoding=decoding)
        st.session_state["last_report"] = report
    report = st.session_state.get("last_report")
    if report is not None:
        gen_col, ref_col = st.columns(2) if row is not None else (st.container(), None)
        with gen_col:
            st.markdown("**Model**")
            st.write(report.text)
            if pipeline.report_meta.get("findings_conditioning"):
                st.caption("Findings the classifier put into the prompt: **"
                           + (", ".join(report.flagged_findings) if report.flagged_findings else "none flagged")
                           + "** — the report is a verbalisation of these; it cannot contain findings the classifier missed.")
            for flag in report.flags:
                st.warning(flag, icon="⚠️")
        if ref_col is not None:
            with ref_col:
                st.markdown("**Reference (radiologist)**")
                st.write(row.report_text)
        st.caption(
            f"Decoding {report.decoding}, max {report.max_new_tokens} new tokens, checkpoint `{report.checkpoint}`. "
            "Measured on held-out data: clinical-efficacy F1 0.176 with beam search (CheXOne-instruct: 0.149), "
            "0.152 greedy — low in absolute terms: most specific abnormal findings are still missed; invented devices "
            "appear in ~3% (beam) / ~10% (greedy) of outputs."
        )
        with st.expander("Sentence attribution — which image regions each sentence attended to", expanded=False):
            st.caption(
                "One extra decoder pass over the generated text with attention weights exposed. Each panel is the sentence's "
                "attention over the 64 visual tokens (8×8 grid); *vis* is the share of that sentence's attention that went to the "
                "image at all (attention-sink corrected). Low shares mean the sentence came from the language prior, not the image."
            )
            explanations = pipeline.attribute_report(image, report.text, instruction=report.instruction)
            if explanations:
                cols = st.columns(min(len(explanations), 4))
                for i, e in enumerate(explanations):
                    with cols[i % len(cols)]:
                        st.image(e.overlay, caption=f"vis = {e.visual_share:.2f} — {e.sentence}", width="stretch")
