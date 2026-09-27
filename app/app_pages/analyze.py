"""Analyze page: pick or upload a frontal chest X-ray -> findings, heatmaps, drafted report."""

from __future__ import annotations

import io
import random

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from common import (
    ACCENT,
    IMAGES_DIR,
    LABELS,
    classify,
    get_pipeline,
    heatmaps,
    image_key,
    load_model_card,
    load_test_images,
    load_operating_points,
    load_thresholds,
    percentile_vs_negatives,
    pct,
    true_label,
)
from medxvlm.inference import REPORT_CHECKPOINTS

card = load_model_card()
thresholds = load_thresholds()
operating_points = load_operating_points()
test_images = load_test_images()
grounding = card.get("grounding", {})

HEATMAP_METHODS = {"attention": "Attention", "gradcam": "Grad-CAM", "text": "Text-grounded"}
REPORT_MODELS = {"v2": "v2", "v2c": "v2c"}
DECODINGS = {"beam": {"num_beams": 3, "repetition_penalty": 1.2}, "greedy": {"num_beams": 1}}
DECODING_NAMES = {"beam": "Beam search", "greedy": "Greedy"}


def _pool(only_abnormal: bool) -> list[str]:
    df = test_images[test_images["No Finding"] != 1.0] if only_abnormal else test_images
    return df.filename.tolist()


def _pick_random() -> None:
    pool = _pool(st.session_state.get("only_abnormal", True))
    if pool:
        st.session_state["test_image"] = random.choice(pool)


def _sync_decoding() -> None:
    # Measured best decoding per model (docs/v2_rebuild.md §3.6.2, §3.7): beam for v2, greedy for v2c.
    st.session_state["decoding"] = "greedy" if st.session_state.get("report_model") == "v2c" else "beam"


def _positives(row: pd.Series) -> list[str]:
    return [l for l in LABELS if l != "No Finding" and row[l] == 1.0]


def _field(label: str, value: str) -> None:
    """A small caption label over a value -- the page's one pattern for summary facts."""
    with st.container(gap=None):
        st.caption(label)
        st.markdown(value)


# ---------------------------------------------------------------- header (renders before any model work)
st.caption("MedXVLM · research prototype, not a medical device")
st.title("Chest X-ray analysis")
st.markdown(
    "Choose a held-out test image or upload a frontal chest X-ray. Four classifiers score 14 findings and show "
    "where they looked; a fine-tuned language model drafts the findings paragraph."
)
st.space("small")

pipeline = get_pipeline()

with st.sidebar:
    st.space("medium")
    meta = pipeline.classifier_meta
    st.caption("Runtime")
    lines = [f"{meta.get('n_models', 1)} classifiers · {meta['backbone']} · {pipeline.device}"]
    if pipeline.report_generator is not None:
        lines.append(f"Report model {'v2c' if pipeline.report_meta.get('findings_conditioning') else 'v2'} loaded")
    else:
        lines.append("Report model loads on first use")
    if pipeline.device.type == "cuda":
        lines.append(f"{torch.cuda.memory_allocated() / 1e9:.1f} GB GPU memory in use")
    st.markdown("  \n".join(lines))
    if meta.get("skipped"):
        st.caption(f"Missing ensemble members: {meta['skipped']}")
    st.space("small")
    st.caption(
        "Trained on about 3,000 outpatient X-rays from one hospital. Outputs demonstrate the pipeline and "
        "must not be used for clinical decisions."
    )
    st.caption(
        "Test-set X-rays: Indiana University Chest X-ray Collection (Open-i, U.S. National Library of Medicine), "
        "CC BY-NC-ND 4.0; shown for non-commercial research demonstration. Uploaded images are processed in memory "
        "and not stored."
    )

# ---------------------------------------------------------------- input + findings
image: Image.Image | None = None
key: str | None = None
row: pd.Series | None = None

left, right = st.columns([8, 16], gap="medium")

with left, st.container(border=True):
    st.subheader("X-ray")
    sources = {"test": "Test set", "upload": "Upload"}
    if test_images.empty:
        sources.pop("test")
    source = st.segmented_control(
        "Source", list(sources), format_func=sources.get, default=next(iter(sources)), required=True,
        key="source", label_visibility="collapsed",
    )
    if source == "upload":
        up = st.file_uploader(
            "Frontal chest X-ray", type=["png", "jpg", "jpeg"],
            help="Lateral views are out of distribution: every model here was trained on frontal views only.",
        )
        if up is not None:
            data = up.getvalue()
            image, key = Image.open(io.BytesIO(data)).convert("RGB"), image_key(data)
            st.image(image, caption=up.name, width="stretch")
    else:
        with st.container(horizontal=True, vertical_alignment="center"):
            only_abnormal = st.toggle("Abnormal only", value=True, key="only_abnormal")
            st.button("Random", icon=":material/shuffle:", on_click=_pick_random, type="tertiary")
        pool = _pool(only_abnormal)
        if st.session_state.get("test_image") not in pool:
            st.session_state["test_image"] = pool[0] if pool else None
        findings_by_file = {r.filename: _positives(r) for _, r in test_images.iterrows()}
        choice = st.selectbox(
            f"{len(pool)} held-out images, never used in training", pool, key="test_image",
            format_func=lambda f: f"{f}  —  {', '.join(findings_by_file[f]) or 'normal'}",
        )
        if choice:
            row = test_images[test_images.filename == choice].iloc[0]
            path = IMAGES_DIR / choice
            if path.exists():
                data = path.read_bytes()
                image, key = Image.open(io.BytesIO(data)).convert("RGB"), image_key(data)
                st.image(image, width="stretch")
                with st.expander("Radiologist's report"):
                    st.write(row.report_text)
            else:
                st.warning(f"Image file missing: {path}")

with right, st.container(border=True, height="stretch"):
    st.subheader("Findings")
    if image is None:
        st.caption("Upload a frontal chest X-ray to run the model.")
        st.stop()

    probs = classify(key, pipeline, image)
    ranked = sorted(probs, key=probs.get, reverse=True)

    op = operating_points.get("per_class", {})
    summary = operating_points.get("test_summary", {})
    modes = {"balanced": "Balanced", "sensitive": "Sensitive"}
    mode_help = "How readily a finding is flagged. Both thresholds were chosen on the validation split."
    if summary:
        b, sv = summary["balanced"], summary["sensitive"]
        mode_help += (
            f"  \n**Balanced** (maximises F1): on the {b['abnormal_images']} abnormal test X-rays it flags nothing on "
            f"{b['abnormal_images_with_no_flag']}, and it falsely flags {b['normal_images_flagged']} of {b['normal_images']} normal ones."
            f"  \n**Sensitive** (catches ~80% of each finding in validation): nothing flagged on {sv['abnormal_images_with_no_flag']} "
            f"abnormal X-rays, but {sv['normal_images_flagged']} of {sv['normal_images']} normal X-rays get a false alarm."
        )
    mode = st.segmented_control("Operating point", list(modes), format_func=modes.get, default="balanced",
                                required=True, key="operating_point", help=mode_help)

    def threshold(label: str) -> float | None:
        if label in op:
            return op[label]["thresholds"][mode]
        return thresholds[label]["threshold"] if label in thresholds else None

    flagged = [l for l in ranked if l != "No Finding" and threshold(l) is not None and probs[l] >= threshold(l)]

    def model_call(label: str) -> str:
        t = threshold(label)
        if t is None:
            return "—"
        if label == "No Finding":  # only claim "no finding" when nothing else is flagged
            return "positive" if probs[label] >= t and not flagged else "negative"
        return "positive" if probs[label] >= t else "negative"

    with st.container(horizontal=True, gap="large"):
        _field("Model flags", ", ".join(f"{l} ({probs[l]:.0%})" for l in flagged) or "Nothing above threshold")
        if row is not None:
            _field("Report label (automatic)", ", ".join(_positives(row)) or "Normal")

    table = pd.DataFrame({
        "Finding": LABELS,
        "Probability": [probs[l] * 100 for l in LABELS],
        "Percentile": [percentile_vs_negatives(probs[l], op[l]["val_negative_percentiles"]) if l in op else None for l in LABELS],
        "Threshold": [threshold(l) * 100 if threshold(l) is not None else None for l in LABELS],
        "Model": [model_call(l) for l in LABELS],
    })
    if row is not None:
        table["Report label"] = [true_label(row[l], l) for l in LABELS]
    table = table.sort_values("Probability", ascending=False)
    st.dataframe(
        table, hide_index=True, height="content",
        column_config={
            "Finding": st.column_config.TextColumn("Finding", width=180),
            "Probability": st.column_config.ProgressColumn(
                "Probability", min_value=0, max_value=100, format="%.0f%%", width=120, color=ACCENT,
            ),
            "Percentile": st.column_config.NumberColumn(
                "Percentile", format="%d", width=80,
                help="How unusual this probability is: 90 means higher than 90% of validation X-rays that do not have "
                     "this finding. Useful for rare findings, whose probabilities stay low even when present.",
            ),
            "Threshold": st.column_config.NumberColumn(
                "Threshold", format="%.0f%%", width=78,
                help="The cut-off for the selected operating point, chosen on the validation split.",
            ),
            "Model": st.column_config.TextColumn(
                "Model", width=72,
                help="Positive at or above the threshold, negative below it. No Finding is positive only when nothing else is flagged.",
            ),
            "Report label": st.column_config.TextColumn(
                "Report label", width=104,
                help="Extracted automatically from the radiologist's report by the project's rule-based labeler, which "
                     "can be wrong. 'stated absent' = the report rules it out; 'not mentioned' is scored as negative.",
            ),
        },
    )
    untuned = [l for l in LABELS if l in op and not op[l]["validated"]]
    notes = []
    if untuned:
        notes.append("Thresholds could not be tuned for " + ", ".join(untuned)
                     + " (fewer than 10 validation examples), so their yes/no calls are unreliable: use the percentile instead.")
    notes.append("Report labels come from an automatic reading of the report and are occasionally wrong "
                 "(e.g. a knee-joint effusion in a combined chest-and-knee report read as pleural effusion).")
    st.caption(" ".join(notes))

# ---------------------------------------------------------------- explainability
with st.container(border=True):
    st.subheader("Where the model looked")
    controls = st.columns([2, 3], gap="medium", vertical_alignment="bottom")
    method = controls[0].segmented_control(
        "Method", list(HEATMAP_METHODS), format_func=HEATMAP_METHODS.get, default="attention", required=True,
        key="heatmap_method",
    )
    chosen = controls[1].multiselect(
        "Findings", ranked, default=ranked[:3], key=f"heatmap_labels_{key}",
        format_func=lambda l: f"{l} ({probs[l]:.0%})", placeholder="Choose findings to map",
    )
    if grounding:
        st.caption(
            f"Validated on {card['grounding_n_boxes']} radiologist-drawn boxes from a hospital the model never saw: "
            f"the hottest point falls inside the box {pct(grounding['attention'])} of the time for attention maps, "
            f"{pct(grounding['gradcam'])} for Grad-CAM and {pct(grounding['text'])} for text-grounded maps "
            f"(a random point: {pct(grounding['random'], 1)})."
            + (" Text-grounded maps come from the zero-shot alignment model, not the classifiers above." if method == "text" else "")
        )
    if chosen:
        try:
            with st.spinner("Rendering heatmaps…"):
                overlays = heatmaps(key, method, tuple(chosen), pipeline, image)
        except FileNotFoundError as exc:
            st.warning(f"The checkpoint for this method is missing: {exc}")
            overlays = {}
        labels = [l for l in chosen if l in overlays]
        for start in range(0, len(labels), 4):
            for col, label in zip(st.columns(4), labels[start:start + 4]):
                col.image(overlays[label], caption=f"{label} — {probs[label]:.0%}", width="stretch")

# ---------------------------------------------------------------- report
with st.container(border=True):
    st.subheader("Findings report")
    ce = card.get("report_systems", {})
    hal = card.get("hallucination", {})
    st.caption(
        "Qwen2.5-1.5B with LoRA adapters. v2 writes from the image; v2c is first told the classifiers' flagged "
        "findings: in testing it invented no devices, but it cannot report anything they missed. Loads on first use (about 7 GB of GPU memory)."
    )
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        model = st.segmented_control(
            "Model", list(REPORT_MODELS), format_func=REPORT_MODELS.get, default="v2", required=True,
            key="report_model", on_change=_sync_decoding,
            help=f"Held-out clinical-efficacy F1: v2 {ce.get('MedXVLM v2, beam search', 0):.3f}, "
                 f"v2c {ce.get('MedXVLM v2c, findings-conditioned', 0):.3f}. Invented devices: "
                 f"v2 {pct(hal.get('MedXVLM v2, beam search'))}, v2c {pct(hal.get('MedXVLM v2c, findings-conditioned'))}.",
        )
        st.session_state.setdefault("decoding", "beam")  # set here (not via default=) because _sync_decoding writes it
        decoding = st.segmented_control(
            "Decoding", list(DECODINGS), format_func=DECODING_NAMES.get, required=True, key="decoding",
            help="Measured best: beam search for v2 (fewer invented devices), greedy for v2c.",
        )
        with st.popover("Options", icon=":material/tune:"):
            max_new = st.slider("Max new tokens", 32, 256, 128, 32, key="max_new_tokens")
            st.button("Unload report model", on_click=pipeline.unload_report_generator,
                      disabled=pipeline.report_generator is None, help="Frees about 7 GB of GPU memory.")
        st.space("stretch")
        generate = st.button("Generate report", type="primary", key="generate")

    if generate:
        checkpoint = REPORT_CHECKPOINTS[model]
        if not checkpoint.exists():
            st.error(f"Checkpoint missing: {checkpoint.name} for report model {model}.")
        else:
            with st.status("Writing the report…") as status:
                if pipeline.report_generator is None or pipeline.report_checkpoint != checkpoint:
                    status.update(label=f"Loading report model {model}…")
                    pipeline.load_report_generator(checkpoint)
                status.update(label="Generating…")
                report = pipeline.generate_report(image, max_new_tokens=max_new, decoding=DECODINGS[decoding])
                status.update(label="Report ready", state="complete")
            st.session_state["report"] = {"image": key, "model": model, "decoding": decoding, "result": report}

    saved = st.session_state.get("report")
    if not saved or saved["image"] != key:
        st.caption("No report for this image yet.")
    else:
        report = saved["result"]
        columns = st.columns(2, gap="medium") if row is not None else [st.container()]
        with columns[0].container(border=True, height="stretch"):
            st.caption(f"MedXVLM {saved['model']}, {DECODING_NAMES[saved['decoding']].lower()}")
            st.write(report.text)
            if saved["model"] == "v2c":
                st.caption("Findings given to the writer: " + (", ".join(report.flagged_findings) or "none"))
            for flag in report.flags:
                st.warning(flag)
        if row is not None:
            with columns[1].container(border=True, height="stretch"):
                st.caption("Radiologist")
                st.write(row.report_text)

        if st.toggle("Show sentence attention", key="explain_sentences",
                     help="Re-reads the report with attention weights exposed and maps each sentence's attention over "
                          "the 64 image tokens (8×8 grid)."):
            cache_key = (key, saved["model"], report.text)
            if st.session_state.get("attribution", {}).get("key") != cache_key:
                with st.spinner("Tracing attention…"):
                    checkpoint = REPORT_CHECKPOINTS[saved["model"]]
                    if pipeline.report_generator is None or pipeline.report_checkpoint != checkpoint:
                        pipeline.load_report_generator(checkpoint)
                    st.session_state["attribution"] = {
                        "key": cache_key,
                        "items": pipeline.attribute_report(image, report.text, instruction=report.instruction, style="focus"),
                    }
            items = st.session_state["attribution"]["items"]
            share = card.get("visual_attention_share")
            st.caption(
                "Each map shows where one sentence looked. The percentage is how much of that sentence's attention "
                "went to the image at all"
                + (f"; across the test set it averages {pct(share, 1)}, so the writer relies mostly on language." if share else ".")
            )
            for start in range(0, len(items), 4):
                for col, item in zip(st.columns(4), items[start:start + 4]):
                    col.image(item.overlay, caption=f"{item.visual_share:.0%} — {item.sentence}", width="stretch")
