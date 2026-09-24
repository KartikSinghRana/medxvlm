"""Shared, cached resources for the MedXVLM demo pages.

Everything expensive (the classifier ensemble, per-image classification and
heatmaps, the evaluation artefacts behind the model card) is loaded here once
and cached, so the page scripts in `app_pages/` stay thin views.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.inference import MedXVLMPipeline  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images" / "images_normalized"
ARTIFACTS = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"
CHEXONE_GENERATIONS = REPO_ROOT.parent / "external_repro" / "chexone_testsplit_generations.csv"

# Same device/surgery regex as docs/v2_rebuild.md §3.4 (mention in the generation, absent from the reference).
DEVICE_RE = re.compile(r"catheter|picc|\btube|\bline\b|clip|wire|pacemaker|lobectomy|sternotomy|\bport\b", re.I)


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading the 4-model classifier ensemble…")
def get_pipeline() -> MedXVLMPipeline:
    return MedXVLMPipeline()


def image_key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


@st.cache_data(max_entries=64, show_spinner=False)
def classify(key: str, _pipeline: MedXVLMPipeline, _image) -> dict[str, float]:
    """Ensemble probability per finding for one image, cached by image content hash.

    # why plain floats, not the ClassificationResult: cache_data pickles its return value, and a
    # hot reload of medxvlm.inference changes the class identity, which makes pickling fail.
    """
    return dict(_pipeline.classify(_image).probabilities)


@st.cache_data(max_entries=64, show_spinner=False)
def heatmaps(key: str, method: str, labels: tuple[str, ...], _pipeline: MedXVLMPipeline, _image) -> dict:
    return _pipeline.heatmaps(_image, list(labels), method=method)


# ---------------------------------------------------------------- data
@st.cache_data
def load_test_images() -> pd.DataFrame:
    """Held-out frontal test images (never used in training), with their reference reports and labels."""
    if not MANIFEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(MANIFEST)
    return df[(df.split == "test") & (df.projection == "Frontal")].reset_index(drop=True)


@st.cache_data
def load_thresholds() -> dict:
    """Per-class operating thresholds chosen on the validation split (F1-optimal), with fallback flags."""
    path = ARTIFACTS / "stage2_ensemble_thresholds.json"
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {label: {"threshold": data["thresholds"][label], "validated": not data["per_class"][label]["fallback"]}
            for label in data["thresholds"]}


def _load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.load(open(path)) if path.exists() else None


def _hallucination_rate(csv: Path) -> float | None:
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    if "projection" not in d.columns:  # the CheXOne file has no projection column; take it from the manifest
        d = d.merge(pd.read_csv(MANIFEST, usecols=["filename", "projection"]), on="filename", how="left")
    d = d[d.projection == "Frontal"]
    hits = [bool(DEVICE_RE.search(str(g))) and not bool(DEVICE_RE.search(str(r))) for g, r in zip(d.generated, d.reference)]
    return sum(hits) / len(hits) if hits else None


@st.cache_data
def load_model_card() -> dict:
    """Every measured number the app displays, read from the evaluation artefacts (nothing hard-coded)."""
    card: dict = {}
    if cls := _load("v2_classification_summary.json"):
        s = cls["systems"]
        card["per_class_support"] = s["v2_stage2_image_only"]["per_class_positive_support"]
        card["auroc_systems"] = {
            "Linear probe, ImageNet ViT": s["v2_baseline_imagenet_vit"]["macro_auroc_8_shared_classes"],
            "Linear probe, X-ray DINOv2": s["v2_baseline_xraydinov2_linear"]["macro_auroc_8_shared_classes"],
            "Zero-shot prompts (v3b)": s["v3b_stage1_findings_zero_shot"]["macro_auroc_8_shared_classes"],
            "RadZero (published)": s["radzero_zero_shot_rescored"]["macro_auroc_8_shared_classes"],
        }
        card["radzero_auroc"] = s["radzero_zero_shot_rescored"]["macro_auroc_8_shared_classes"]
    if ens := _load("classifier_seed_ensemble.json"):
        card["single_auroc"], card["single_auroc_std"] = ens["seed_mean_std"]["macro_auroc_8_shared"]
        card["ensemble_auroc"] = ens["ensemble"]["macro_auroc_8_shared_classes"]
        card["n_test_images"] = ens["n_frontal_test_images"]
        nih = ens.get("nih_chestxray14", {})
        card["nih_ensemble_auroc"] = nih.get("ensemble_macro_auroc_8")
        card["nih_single_auroc"] = (nih.get("seed_mean_std") or [None])[0]
        card["nih_n_images"] = nih.get("n_images")
        card.setdefault("auroc_systems", {})
        card["auroc_systems"]["MedXVLM, single model"] = card["single_auroc"]
        card["auroc_systems"]["MedXVLM, 4-model ensemble"] = card["ensemble_auroc"]
    if rep := _load("report_generation_scores_v2c.json"):
        s = rep["systems"]
        f1 = lambda name: s[name]["frontal_70"]["clinical_efficacy"]["macro_f1"]  # noqa: E731
        card["report_systems"] = {
            "MedXVLM v2, beam search": f1("v2_stage3_beam3"),
            "MedXVLM v2, greedy": f1("v2_stage3"),
            "MedXVLM v2c, findings-conditioned": f1("v2c_stage3"),
            "CheXOne (published, 3B)": f1("chexone_instruct"),
            "Phase 3 model (collapsed)": f1("phase3_stage3"),
        }
        card["report_ceiling"] = f1("classifier_verbalised")
        card["hallucination"] = {
            "MedXVLM v2, beam search": _hallucination_rate(ARTIFACTS / "medxvlm_stage3_v2_beam3_generations.csv"),
            "MedXVLM v2, greedy": _hallucination_rate(ARTIFACTS / "medxvlm_stage3_v2_generations.csv"),
            "MedXVLM v2c, findings-conditioned": _hallucination_rate(ARTIFACTS / "medxvlm_stage3_v2c_generations.csv"),
            "CheXOne (published, 3B)": _hallucination_rate(CHEXONE_GENERATIONS),
        }
    if g := _load("grounding_chestxray14_summary.json"):
        pm = g["per_method"]
        hit = lambda name: pm[name]["overall"]["hit_rate"]  # noqa: E731
        card["grounding_n_boxes"] = g["n_boxes"]
        card["grounding"] = {
            "attention": hit("stage2_attn_ensemble"),
            "gradcam": hit("stage2_gradcam_ensemble"),
            "text": hit("text_v3b_findings"),
            "rollout": hit("rollout_stage2_tower"),
            "center": hit("baseline_center"),
            "random": hit("baseline_random_expected"),
        }
    if a := _load("sentence_attribution_summary.json"):
        card["visual_attention_share"] = a.get("indiana_frontal_test", {}).get("visual_share_excl_sink_mean_all")
    return card


def true_label(value) -> str:
    """Raw labeler value -> display string (1 positive, 0 negative, -1 uncertain, NaN not mentioned)."""
    if pd.isna(value):
        return "—"
    return {1.0: "positive", 0.0: "negative", -1.0: "uncertain"}[float(value)]


def pct(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


LABELS = CHEXPERT_LABELS
