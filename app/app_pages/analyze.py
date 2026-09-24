"""Analyze page: pick or upload a frontal chest X-ray -> findings, heatmaps, drafted report."""

from __future__ import annotations

import io
import random

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from common import (
    IMAGES_DIR,
    LABELS,
    classify,
    get_pipeline,
    heatmaps,
    image_key,
    load_model_card,
    load_test_images,
    load_thresholds,
    pct,
    true_label,
)
from medxvlm.inference import REPORT_CHECKPOINTS

card = load_model_card()
thresholds = load_thresholds()
test_images = load_test_images()
grounding = card.get("grounding", {})

HEATMAP_METHODS = {
    "attention": ("Attention", ":material/center_focus_strong:"),
    "gradcam": ("Grad-CAM", ":material/layers:"),
    "text": ("Text-grounded", ":material/text_fields:"),
}
REPORT_MODELS = {"v2": "v2 · from the image", "v2c": "v2c · findings-conditioned"}
DECODINGS = {"beam": {"num_beams": 3, "repetition_penalty": 1.2}, "greedy": {"num_beams": 1}}


def _pick_random() -> None:
    pool = _pool(st.session_state.get("only_abnormal", True))
    if pool:
        st.session_state["test_image"] = random.choice(pool)


def _pool(only_abnormal: bool) -> list[str]:
    df = test_images[test_images["No Finding"] != 1.0] if only_abnormal else test_images
    return df.filename.tolist()


def _sync_decoding() -> None:
    # Measured best decoding per model (docs/v2_rebuild.md §3.6.2, §3.7): beam for v2, greedy for v2c.
    st.session_state["decoding"] = "greedy" if st.session_state.get("report_model") == "v2c" else "beam"


def _positives(row: pd.Series) -> list[str]:
    return [l for l in LABELS if l != "No Finding" and row[l] == 1.0]


# ---------------------------------------------------------------- header (renders before any model work)
st.title("Chest X-ray analysis")
st.caption(
    "Pick a held-out test image or upload a frontal chest X-ray. A 4-model classifier ensemble scores 14 findings "
    "and shows where it looked; a LoRA-tuned Qwen2.5-1.5B drafts the findings paragraph."
)
badges = []
if "ensemble_auroc" in card:
    badges.append(f":green-badge[:material/verified: AUROC {card['ensemble_auroc']:.3f} · on par with RadZero]")
if grounding:
    badges.append(f":blue-badge[:material/my_location: Heatmaps land on the radiologist's box {pct(grounding['attention'])} of the time]")
badges.append(":orange-badge[:material/science: Research prototype · not a medical device]")
st.markdown(" ".join(badges))

pipeline = get_pipeline()

with st.sidebar:
    st.markdown("**:material/memory: Runtime**")
    meta = pipeline.classifier_meta
    st.caption(f"{meta.get('n_models', 1)}-model classifier ensemble · {meta['backbone']} · `{pipeline.device}`")
    if meta.get("skipped"):
        st.caption(f"Missing ensemble members: {meta['skipped']}")
    if pipeline.report_generator is not None:
        loaded = "v2c" if pipeline.report_meta.get("findings_conditioning") else "v2"
        st.caption(f"Report model **{loaded}** loaded (Qwen2.5-1.5B + LoRA)")
    else:
        st.caption("Report model not loaded; it loads on first use.")
    if pipeline.device.type == "cuda":
        st.caption(f"GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    st.space("small")
    st.caption(
        ":material/info: Trained on ~3,000 outpatient X-rays from one hospital (Indiana University). "
        "Not a medical device: outputs demonstrate the pipeline and must not guide care."
    )

# ---------------------------------------------------------------- input + findings
image: Image.Image | None = None
key: str | None = None
row: pd.Series | None = None

left, right = st.columns([9, 15], gap="medium")

with left, st.container(border=True):
    st.subheader(":material/radiology: X-ray")
    sources = {"test": ":material/folder_open: Test set", "upload": ":material/upload: Upload"}
    if test_images.empty:
        sources.pop("test")
    source = st.segmented_control(
        "Source", list(sources), format_func=sources.get, default=next(iter(sources)), required=True,
        key="source", label_visibility="collapsed",
    )
    if source == "upload":
        up = st.file_uploader(
            "Frontal chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"],
            help="Lateral views are out of distribution: every model here was trained on frontal views only.",
        )
        if up is not None:
            data = up.getvalue()
            image, key = Image.open(io.BytesIO(data)).convert("RGB"), image_key(data)
            st.image(image, caption=up.name, width="stretch")
    else:
        with st.container(horizontal=True, vertical_alignment="center"):
            only_abnormal = st.toggle("Abnormal studies only", value=True, key="only_abnormal")
            st.button("Random", icon=":material/shuffle:", on_click=_pick_random, type="tertiary")
        pool = _pool(only_abnormal)
        if st.session_state.get("test_image") not in pool:
            st.session_state["test_image"] = pool[0] if pool else None
        findings_by_file = {r.filename: _positives(r) for _, r in test_images.iterrows()}
        choice = st.selectbox(
            f"{len(pool)} held-out images · never seen in training", pool, key="test_image",
            format_func=lambda f: f"{f} · {', '.join(findings_by_file[f]) or 'normal'}",
        )
        if choice:
            row = test_images[test_images.filename == choice].iloc[0]
            path = IMAGES_DIR / choice
            if path.exists():
                data = path.read_bytes()
                image, key = Image.open(io.BytesIO(data)).convert("RGB"), image_key(data)
                st.image(image, width="stretch")
                with st.expander("Radiologist's report", icon=":material/description:"):
                    st.write(row.report_text)
            else:
                st.warning(f"Image file missing: {path}", icon=":material/broken_image:")

with right, st.container(border=True, height="stretch"):
    st.subheader(":material/clinical_notes: Findings")
    if image is None:
        st.caption(":material/image_search: Upload a frontal chest X-ray to run the model.")
        st.stop()

    probs = classify(key, pipeline, image)
    ranked = sorted(probs, key=probs.get, reverse=True)
    flagged = sorted(
        (l for l in LABELS if l != "No Finding" and l in thresholds and probs[l] >= thresholds[l]["threshold"]),
        key=lambda l: -probs[l],
    )
    summary = [f":orange-badge[{l} · {probs[l]:.0%}]" for l in flagged] or [":green-badge[:material/check_circle: No finding above its threshold]"]
    st.markdown("**Model** &nbsp; " + " ".join(summary))
    if row is not None:
        truth = [f":blue-badge[{l}]" for l in _positives(row)] or [":gray-badge[normal]"]
        st.markdown("**Radiologist** &nbsp; " + " ".join(truth))

    table = pd.DataFrame({
        "Finding": LABELS,
        "Probability": [probs[l] * 100 for l in LABELS],
        "Threshold": [thresholds[l]["threshold"] * 100 if l in thresholds else None for l in LABELS],
        "Flagged": [l in thresholds and probs[l] >= thresholds[l]["threshold"] for l in LABELS],
    })
    if row is not None:
        table["Radiologist"] = [true_label(row[l]) for l in LABELS]
    table = table.sort_values("Probability", ascending=False)
    st.dataframe(
        table, hide_index=True, height="content",
        column_config={
            "Finding": st.column_config.TextColumn("Finding", width=190),
            "Probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=100, format="%.0f%%", width=130),
            "Threshold": st.column_config.NumberColumn(
                "Threshold", format="%.0f%%", width=85,
                help="Operating point chosen on the validation split to maximise F1. Classes with fewer than 10 "
                     "validation positives use the median threshold instead.",
            ),
            "Flagged": st.column_config.CheckboxColumn("Flagged", width=70, help="Probability at or above the class threshold."),
            "Radiologist": st.column_config.TextColumn("Radiologist", width=95, help="Label extracted from the radiologist's report."),
        },
    )
    support = card.get("per_class_support", {})
    thin = [f"{l} ({support[l]})" for l in LABELS if l in support and support[l] < 10]
    if thin:
        st.caption(":material/info: Too few positive test cases for a meaningful validated score: " + ", ".join(thin) + ".")

# ---------------------------------------------------------------- explainability
with st.container(border=True):
    st.subheader(":material/my_location: Where the model looked")
    method = st.segmented_control(
        "Heatmap method", list(HEATMAP_METHODS), default="attention", required=True, key="heatmap_method",
        format_func=lambda m: f"{HEATMAP_METHODS[m][1]} {HEATMAP_METHODS[m][0]}" + (f" · {pct(grounding[m])}" if grounding else ""),
    )
    top3 = ranked[:3]
    chosen = st.pills(
        "Findings to map", ranked, selection_mode="multi", default=top3, key=f"heatmap_labels_{key}",
        format_func=lambda l: f"{l} {probs[l]:.0%}",
    )
    if grounding:
        st.caption(
            f"Percentages are validated accuracy: how often the map's hottest pixel falls inside a radiologist-drawn box, "
            f"over {card['grounding_n_boxes']} boxes from a hospital the model never saw (NIH ChestX-ray14). "
            f"A random point scores {pct(grounding['random'], 1)}; always pointing at the centre, {pct(grounding['center'])}."
            + (" Text-grounded maps come from the zero-shot alignment model, not the classifier above." if method == "text" else "")
        )
    if chosen:
        try:
            with st.spinner("Rendering heatmaps…"):
                overlays = heatmaps(key, method, tuple(chosen), pipeline, image)
        except FileNotFoundError as exc:
            st.warning(f"The checkpoint for this method is missing: {exc}", icon=":material/warning:")
            overlays = {}
        labels = [l for l in chosen if l in overlays]
        for start in range(0, len(labels), 4):
            for col, label in zip(st.columns(4), labels[start:start + 4]):
                col.image(overlays[label], caption=f"{label} · p = {probs[label]:.2f}", width="stretch")
    else:
        st.caption("Select one or more findings to map.")

# ---------------------------------------------------------------- report
with st.container(border=True):
    st.subheader(":material/summarize: Findings report")
    st.caption(
        "Qwen2.5-1.5B with LoRA adapters writes a findings paragraph. The model loads on first use "
        "(~7 GB of GPU memory, ~10 s)."
    )
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        model = st.segmented_control(
            "Report model", list(REPORT_MODELS), format_func=REPORT_MODELS.get, default="v2", required=True,
            key="report_model", on_change=_sync_decoding,
            help=f"**v2** writes from the image: clinical-efficacy F1 {card.get('report_systems', {}).get('MedXVLM v2, beam search', 0):.3f}, "
                 f"invents a device in ~{pct(card.get('hallucination', {}).get('MedXVLM v2, beam search'))} of cases.  \n"
                 f"**v2c** is told the classifier's flagged findings first: F1 {card.get('report_systems', {}).get('MedXVLM v2c, findings-conditioned', 0):.3f}, "
                 f"{pct(card.get('hallucination', {}).get('MedXVLM v2c, findings-conditioned'))} invented devices, but it cannot report "
                 "anything the classifier missed.",
        )
        st.session_state.setdefault("decoding", "beam")  # set here (not via default=) because _sync_decoding writes it
        decoding = st.segmented_control(
            "Decoding", list(DECODINGS), format_func={"beam": "Beam search", "greedy": "Greedy"}.get,
            required=True, key="decoding",
            help="Measured best: beam search for v2 (fewer invented devices), greedy for v2c.",
        )
        with st.popover("Options", icon=":material/tune:"):
            max_new = st.slider("Max new tokens", 32, 256, 128, 32, key="max_new_tokens")
            st.button("Unload report model", icon=":material/eject:", on_click=pipeline.unload_report_generator,
                      disabled=pipeline.report_generator is None, help="Frees ~7 GB of GPU memory.")
        generate = st.button("Generate report", type="primary", icon=":material/auto_awesome:", key="generate")

    if generate:
        checkpoint = REPORT_CHECKPOINTS[model]
        if not checkpoint.exists():
            st.error(f"Checkpoint missing: {checkpoint.name} for report model {model}.", icon=":material/error:")
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
        st.caption(":material/edit_note: No report for this image yet. Choose a model and select **Generate report**.")
    else:
        report = saved["result"]
        columns = st.columns(2) if row is not None else [st.container()]
        with columns[0].container(border=True, height="stretch"):
            st.markdown(f"**:material/auto_awesome: MedXVLM {saved['model']}** · {'beam search' if saved['decoding'] == 'beam' else 'greedy'}")
            st.write(report.text)
            if saved["model"] == "v2c":
                prompt = " ".join(f":orange-badge[{f}]" for f in report.flagged_findings) or ":gray-badge[none flagged]"
                st.markdown("Findings given to the writer: " + prompt)
            for flag in report.flags:
                st.warning(flag, icon=":material/warning:")
        if row is not None:
            with columns[1].container(border=True, height="stretch"):
                st.markdown("**:material/stethoscope: Radiologist**")
                st.write(row.report_text)
        ce = card.get("report_systems", {})
        st.caption(
            f"Held-out clinical-efficacy F1: v2 {ce.get('MedXVLM v2, beam search', 0):.3f}, "
            f"v2c {ce.get('MedXVLM v2c, findings-conditioned', 0):.3f}, CheXOne {ce.get('CheXOne (published, 3B)', 0):.3f}. "
            "Low in absolute terms: most specific abnormal findings are still missed."
        )

        if st.toggle("Explain each sentence", key="explain_sentences",
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
                        "items": pipeline.attribute_report(image, report.text, instruction=report.instruction),
                    }
            items = st.session_state["attribution"]["items"]
            share = card.get("visual_attention_share")
            st.caption(
                "**vis** is the share of the sentence's attention that went to the image at all (attention-sink corrected)."
                + (f" Across the test set it averages {pct(share, 1)}: the writer relies mostly on language, not the image." if share else "")
            )
            for start in range(0, len(items), 4):
                for col, item in zip(st.columns(4), items[start:start + 4]):
                    col.image(item.overlay, caption=f"vis {item.visual_share:.2f} · {item.sentence}", width="stretch")
