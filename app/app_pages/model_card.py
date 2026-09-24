"""Model card: every measured number behind the demo, with the comparisons that give it meaning."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common import ACCENT, load_model_card, pct

card = load_model_card()
INK, MUTED, REFERENCE = "#18181b", "#71717a", "#d4d4d8"


def ranking_chart(values: dict[str, float | None], *, fmt: str, domain: tuple[float, float], axis_fmt: str | None = None,
                  baseline: float = 0.0, rules: dict[str, float] | None = None) -> alt.Chart:
    """Horizontal bars, MedXVLM systems in the accent colour, references in grey, optional dashed reference lines."""
    df = pd.DataFrame({"System": list(values), "Score": list(values.values())}).dropna()
    df["Group"] = ["MedXVLM" if s.startswith("MedXVLM") else "Reference" for s in df.System]
    base = alt.Chart(df).encode(
        y=alt.Y("System:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=280, labelFontSize=13, labelColor=INK, domain=False, ticks=False, labelPadding=10)),
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=18).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=list(domain)), title=None,
                axis=alt.Axis(format=axis_fmt or fmt, grid=True, gridColor="#f4f4f5", tickCount=5, domain=False, ticks=False,
                              labelColor=MUTED)),
        x2=alt.datum(baseline),
        color=alt.Color("Group:N", scale=alt.Scale(domain=["MedXVLM", "Reference"], range=[ACCENT, REFERENCE]), legend=None),
        tooltip=["System", alt.Tooltip("Score:Q", format=fmt)],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12, fontWeight=500, color=INK).encode(
        x="Score:Q", text=alt.Text("Score:Q", format=fmt),
    )
    chart = bars + labels
    if rules:
        rdf = pd.DataFrame({"Label": list(rules), "Score": list(rules.values())})
        line = alt.Chart(rdf).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="Score:Q")
        text = alt.Chart(rdf).mark_text(align="left", dx=5, dy=-6, fontSize=11, color=MUTED).encode(
            x="Score:Q", y=alt.value(0), text="Label:N",
        )
        chart = chart + line + text
    return chart.properties(height=max(36 * len(df), 120)).configure_view(strokeWidth=0)


st.caption("MedXVLM · evaluation")
st.title("Model card")
st.markdown(
    "Every number here was measured on held-out data the models never trained on, and is read directly from the "
    "evaluation files in `docs/phase5_artifacts/v2/`."
)
st.space("small")

if not card:
    st.warning("Evaluation artefacts not found under docs/phase5_artifacts/v2/.", icon=":material/warning:")
    st.stop()

classification, reporting, explainability, limitations = st.tabs(["Classification", "Report writing", "Explainability", "Limitations"])

with classification:
    # Round first so each delta equals the difference of the numbers actually shown.
    ens, single, std = (round(card[k], 3) for k in ("ensemble_auroc", "single_auroc", "single_auroc_std"))
    nih = round(card["nih_ensemble_auroc"], 3) if card.get("nih_ensemble_auroc") else None
    with st.container(horizontal=True):
        st.metric("Ensemble AUROC", f"{ens:.3f}", f"{ens - single:+.3f}", delta_description="vs one model", border=True, height=128,
                  help="Probability average of 4 classifiers trained with different random seeds. Macro AUROC over the 8 findings shared with RadZero.")
        st.metric("One model", f"{single:.3f} ± {std:.3f}", border=True, height=128, help="Mean ± standard deviation over the 4 seeds.")
        st.metric("RadZero, same images", f"{card['radzero_auroc']:.3f}", border=True, height=128,
                  help="Published model (Deepnoid), run from its public weights on the identical images and labels.")
        if nih is not None:
            st.metric("Unseen hospital", f"{nih:.3f}", f"{nih - ens:+.3f}",
                      delta_description="vs home test set", border=True, height=128,
                      help=f"NIH ChestX-ray14 official test split ({card.get('nih_n_images', 0):,} images), 8 mapped findings.")
    with st.container(border=True):
        st.markdown("**Macro AUROC**")
        st.caption(f"8 findings shared with RadZero, {card.get('n_test_images', 377)} held-out frontal X-rays")
        st.altair_chart(ranking_chart(card["auroc_systems"], fmt=".3f", axis_fmt=".2f", domain=(0.5, 0.92), baseline=0.5), width="stretch")
        st.caption("AUROC 0.5 is chance, 1.0 is perfect. Linear probes use the same data and image encoder, isolating what the "
                   "per-label attention head adds; zero-shot prompts use no labels at all.")

with reporting:
    ce = card.get("report_systems", {})
    hal = card.get("hallucination", {})
    v2, v2c, chexone = (round(ce[k], 3) for k in ("MedXVLM v2, beam search", "MedXVLM v2c, findings-conditioned", "CheXOne (published, 3B)"))
    with st.container(horizontal=True):
        st.metric("v2, beam search", f"{v2:.3f}", f"{v2 - chexone:+.3f}", delta_description="vs CheXOne", border=True, height=128,
                  help="Clinical-efficacy macro F1: findings extracted from the generated text vs the radiologist's labels, 70 frontal test images.")
        st.metric("v2c, findings-conditioned", f"{v2c:.3f}", f"{v2c - v2:+.3f}", delta_description="vs v2", border=True, height=128,
                  help="Told the classifier's flagged findings before writing. Within noise of v2 (95% CIs are about ±0.10 on 70 images).")
        st.metric("Ceiling", f"{card['report_ceiling']:.3f}", border=True, height=128,
                  help="The classifier's flagged findings written out as a bare list. The classifier flags a finding on only "
                       "18 of the 70 images, which caps any report built on it.")
        st.metric("Invented devices", f"{pct(hal.get('MedXVLM v2, beam search'))} → {pct(hal.get('MedXVLM v2c, findings-conditioned'))}",
                  delta_description="v2 → v2c", border=True,
                  help="Share of frontal reports mentioning a catheter, line, clip or surgery the radiologist did not report.")
    with st.container(border=True):
        st.markdown("**Clinical-efficacy F1**")
        st.caption("Findings extracted from each generated report, scored against the radiologist's labels on 70 held-out frontal X-rays")
        st.altair_chart(ranking_chart(ce, fmt=".3f", axis_fmt=".2f", domain=(0, 0.26), rules={"ceiling": card["report_ceiling"]}), width="stretch")
        st.caption("Scores are low in absolute terms for every system: most specific abnormal findings are still missed. "
                   "Differences under ~0.10 are within measurement noise on 70 images.")
    with st.container(border=True):
        st.markdown("**Invented devices or surgery**")
        st.caption("Share of reports mentioning a line, tube, clip or operation the radiologist did not report")
        st.altair_chart(ranking_chart(hal, fmt=".0%", domain=(0, 0.12)), width="stretch")

with explainability:
    g = card.get("grounding", {})
    if g:
        with st.container(horizontal=True):
            st.metric("Attention maps", pct(g["attention"]), f"{(g['attention'] - g['random']) * 100:+.0f} pts",
                      delta_description="vs a random point", border=True, height=128,
                      help="Share of radiologist-drawn boxes where the heatmap's hottest pixel falls inside. Classifier's own per-finding attention, 4-model average.")
            st.metric("Grad-CAM", pct(g["gradcam"]), border=True, height=128, help="Gradient-based maps on the same classifier.")
            st.metric("Text-grounded", pct(g["text"]), border=True, height=128, help="Patch-to-prompt similarity in the zero-shot alignment model.")
            if card.get("visual_attention_share") is not None:
                st.metric("Report writer's image attention", pct(card["visual_attention_share"], 1), border=True, height=128,
                          help="Share of each generated sentence's attention that goes to the 64 image tokens (sink-corrected), averaged over the test set.")
        with st.container(border=True):
            st.markdown("**Pointing game**")
            st.caption(f"How often the heatmap's hottest point falls inside a radiologist-drawn box — {card['grounding_n_boxes']} boxes, NIH ChestX-ray14, never seen in training")
            methods = {
                "MedXVLM attention maps": g["attention"],
                "MedXVLM Grad-CAM": g["gradcam"],
                "MedXVLM text-grounded": g["text"],
                "MedXVLM attention rollout": g["rollout"],
                "Always the image centre": g["center"],
            }
            st.altair_chart(ranking_chart(methods, fmt=".0%", domain=(0, 0.5), rules={"random point": g["random"]}), width="stretch")
            st.caption("The centre baseline is inflated by enlarged-heart boxes, which sit in the middle of every image. "
                       "Where the classifier is weak (pneumothorax, small nodules), its heatmaps are no better than chance.")

with limitations:
    st.markdown(
        """
- **One hospital, small data.** About 3,000 outpatient studies from Indiana University. On an unseen hospital's
  images the classifier drops from 0.868 to 0.719 AUROC.
- **Rare findings are unreliable.** After label cleaning, pneumothorax has 1 positive test case, pleural other 3,
  consolidation 5. Their individual scores are not meaningful; the app greys them out and shows the counts.
- **The report writer mostly paraphrases.** It gives the same normal paragraph for about 6 in 10 images and misses
  most specific abnormal findings. The findings-conditioned v2c removes invented devices but cannot report anything
  the classifier missed.
- **Heatmaps are validated out of distribution only.** The Indiana data has no radiologist boxes; the pointing-game
  numbers come from NIH ChestX-ray14.
- **Not a medical device.** This is a research prototype for a B.Sc. project; nothing it outputs should guide care.
"""
    )
