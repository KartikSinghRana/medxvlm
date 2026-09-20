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

from medxvlm.inference import MedXVLMPipeline  # noqa: E402
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
    return card


def true_labels(row: pd.Series) -> dict[str, str]:
    """Raw labeler value -> display string (1 positive, 0 negative, -1 uncertain, NaN not mentioned)."""
    out = {}
    for label in CHEXPERT_LABELS:
        v = row[label]
        out[label] = "not mentioned" if pd.isna(v) else {1.0: "positive", 0.0: "negative", -1.0: "uncertain"}[float(v)]
    return out


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
    if enable_report and pipeline.report_generator is None:
        with st.spinner("Loading the report generator (≈10 s)…"):
            pipeline.load_report_generator()
    if not enable_report and pipeline.report_generator is not None:
        pipeline.unload_report_generator()
    if pipeline.report_generator is not None:
        m = pipeline.report_meta
        st.caption(f"Generator: `{m['decoder']}` + LoRA, epoch {m['epoch']}, {m['visual_tokens']} visual tokens, on `{m['vision_checkpoint']}`.")
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
- Zero-shot (prompt-based) classification: best variant 0.734 macro AUROC vs RadZero 0.868 — data-scale-limited, not used by this app

Sources: `docs/v2_rebuild.md`, `docs/status_report.md`.
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
st.subheader("Where the classifier looked")
st.caption(
    "Each label has its own learned attention query over the image patches (per-label attention pooling); the heatmap "
    "is that label's attention weights, upsampled. This is the model's own pooling weights — not Grad-CAM — and it has "
    "only been spot-checked qualitatively, not validated against radiologist annotations."
)
default_labels = [l for l, _ in ranked[:3]]
chosen = st.multiselect("Labels to visualise", CHEXPERT_LABELS, default=default_labels)
if chosen:
    cols = st.columns(min(len(chosen), 4))
    for i, label in enumerate(chosen):
        with cols[i % len(cols)]:
            st.image(result.heatmap(label), caption=f"{label} — p = {result.probabilities[label]:.2f}", width="stretch")

# ---------------------------------------------------------------- report generation
st.subheader("Generated findings")
if not enable_report:
    st.info("Turn on **Enable report generation** in the sidebar to load the decoder (~7 GB VRAM).")
else:
    max_new = st.slider("Max new tokens", 32, 256, 128, 32)
    decoding_choice = st.radio(
        "Decoding", ["Beam search (3 beams, repetition penalty 1.2) — recommended", "Greedy"], horizontal=True,
        help="On held-out data beam search raised clinical-efficacy F1 0.152 → 0.176 and cut invented devices 10% → 3%, "
             "at the cost of less varied output.",
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
