#!/usr/bin/env python3
"""Post-rebuild evaluation: every classifier on the SAME frontal-view test split,
against the SAME corrected labels, image-only.

Systems scored (all blind -- no report text is ever given to any model):
  - MedXVLM v2 Stage 2 (image-only head on the v2 Stage 1 XrayDINOv2 encoder)
  - MedXVLM v2 Stage 1 zero-shot ("There is {label}." cosine similarity)
  - v2 baselines: ImageNet ViT-B/16 + linear, XrayDINOv2 + linear
  - Phase 3 Stage 2 (BiomedCLIP init) run blind, and Phase 3 Stage 1 zero-shot,
    on this same test set / labels -- the honest "before" for the rebuild
  - RadZero zero-shot, re-scored from its saved per-image probabilities
    (external_repro/radzero_on_medxvlm_testsplit_raw.csv) -- no re-run needed

Ground truth: data/processed/manifest.csv after the labeler negation-scope
fix, U-Ones policy, frontal rows of split == "test".

Usage:
    python scripts/eval_v2_classification.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import ClassificationDataset  # noqa: E402
from medxvlm.engine.loop_utils import load_checkpoint, load_stage1_encoders  # noqa: E402
from medxvlm.models.alignment import SHARED_EMBED_DIM  # noqa: E402
from medxvlm.models.baseline_classifier import BaselineClassifier  # noqa: E402
from medxvlm.models.classifier_head import ClassifierHead  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402
from medxvlm.models.vision_encoder import VisionEncoder, infer_backbone_from_state_dict  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"
RADZERO_RAW = Path("/home/kartik/DNN Project 2/external_repro/radzero_on_medxvlm_testsplit_raw.csv")

CKPT = {
    "v2_stage2_image_only": REPO_ROOT / "checkpoints/classifier_v2/local_on/best.pt",
    "v2_stage1": REPO_ROOT / "checkpoints/align_v2/local_on/best.pt",
    "v2_stage1_valloss": REPO_ROOT / "checkpoints/align_v2/local_on_valloss/best.pt",
    "v3_stage1_sentence": REPO_ROOT / "checkpoints/align_v3/sentence/best.pt",
    "v2_stage1_frozen_text": REPO_ROOT / "checkpoints/align_v2/local_on_frozen_text/best.pt",
    "v2_baseline_imagenet_vit": REPO_ROOT / "checkpoints/baseline_v2/vit_base_patch16_224/best.pt",
    "v2_baseline_xraydinov2_linear": REPO_ROOT / "checkpoints/baseline_v2/xraydinov2/best.pt",
    "phase3_stage2": REPO_ROOT / "checkpoints/classifier/local_on/best.pt",
    "phase3_stage1": REPO_ROOT / "checkpoints/align/local_on/best.pt",
}

# RadZero class name -> CheXpert-14 column (same 8 shared concepts as Phase 5).
RADZERO_TO_MEDXVLM = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Pleural Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax", "Lung Opacity": "Lung Opacity",
    "Bone Fracture": "Fracture", "Pulmonary Edema": "Edema",
}
SHARED_CLASSES = list(RADZERO_TO_MEDXVLM.values())
ZERO_SHOT_PROMPTS = {l: f"There is {l}." if l != "No Finding" else "There is no acute cardiopulmonary abnormality." for l in CHEXPERT_LABELS}

# Prompt ensembling + positive-vs-negative contrast (CheXzero/RadZero-style scoring).
# score = mean_over_templates( sim(image, positive prompt) - sim(image, negative prompt) )
POS_TEMPLATES = ("There is {}.", "{} is present.", "Findings consistent with {}.")
NEG_TEMPLATES = ("There is no {}.", "No {}.", "No evidence of {}.")
NO_FINDING_POS = ("No acute cardiopulmonary abnormality.", "The lungs are clear.", "Normal chest radiograph.")
NO_FINDING_NEG = ("Acute cardiopulmonary abnormality.", "There is an abnormality.", "Abnormal chest radiograph.")


def _prompt_sets(label: str) -> tuple[list[str], list[str]]:
    if label == "No Finding":
        return list(NO_FINDING_POS), list(NO_FINDING_NEG)
    name = label.lower()
    return [t.format(name) for t in POS_TEMPLATES], [t.format(name) for t in NEG_TEMPLATES]


def load_stage2_image_only(path, device):
    """Stage2Model saved by train_classifier.py: vision_encoder.* + head.* (+ text_encoder.* if fused).

    Only the vision tower and head are needed to score image-only; the text
    encoder (absent entirely in v2 image-only checkpoints) is never built.
    """
    state = load_checkpoint(path)["model_state_dict"]
    vision_encoder = VisionEncoder(pretrained=False, backbone=infer_backbone_from_state_dict(state)).to(device)
    vision_encoder.load_state_dict({k[len("vision_encoder."):]: v for k, v in state.items() if k.startswith("vision_encoder.")})
    head_state = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    text_dim = head_state["text_proj.weight"].shape[1] if "text_proj.weight" in head_state else None
    head = ClassifierHead(embed_dim=vision_encoder.embed_dim, text_dim=text_dim).to(device)
    head.load_state_dict(head_state)
    vision_encoder.eval(); head.eval()

    def score(pixel_values):
        return torch.sigmoid(head(vision_encoder(pixel_values).patch_tokens, text_embeds=None).logits)
    return score


def load_zero_shot(path, device, mode: str = "pos_only"):
    """Zero-shot scorer from a Stage 1 checkpoint.

    mode="pos_only": cosine to one "There is {label}." prompt (the Phase 5 protocol).
    mode="pos_minus_neg": mean over templates of sim(pos) - sim(neg) -- prompt
    ensembling with an explicit negative, which removes the shared "what a chest
    X-ray sentence looks like" component that dominates single-prompt cosine.
    """
    vision_encoder, text_encoder = load_stage1_encoders(path, device)
    state = load_checkpoint(path)["model_state_dict"]
    vision_proj = torch.nn.Linear(vision_encoder.embed_dim, SHARED_EMBED_DIM).to(device)
    text_proj = torch.nn.Linear(text_encoder.embed_dim, SHARED_EMBED_DIM).to(device)
    vision_proj.load_state_dict({k[len("vision_proj."):]: v for k, v in state.items() if k.startswith("vision_proj.")})
    text_proj.load_state_dict({k[len("text_proj."):]: v for k, v in state.items() if k.startswith("text_proj.")})
    for m in (vision_encoder, text_encoder, vision_proj, text_proj):
        m.eval()

    def embed(texts: list[str]) -> torch.Tensor:
        enc = text_encoder.tokenize(texts, max_length=64)
        enc = {k: v.to(device) for k, v in enc.items()}
        return F.normalize(text_proj(text_encoder(enc["input_ids"], enc["attention_mask"]).pooled), dim=-1)

    with torch.no_grad():
        if mode == "pos_only":
            pos = embed([ZERO_SHOT_PROMPTS[l] for l in CHEXPERT_LABELS])  # [14, D]
            neg = None
        elif mode == "pos_minus_neg":
            pos_sets, neg_sets = zip(*(_prompt_sets(l) for l in CHEXPERT_LABELS))
            n_t = len(pos_sets[0])
            pos = embed([p for ps in pos_sets for p in ps]).view(len(CHEXPERT_LABELS), n_t, -1)  # [14, T, D]
            neg = embed([p for ps in neg_sets for p in ps]).view(len(CHEXPERT_LABELS), n_t, -1)
        else:
            raise ValueError(mode)

    def score(pixel_values):
        img = F.normalize(vision_proj(vision_encoder(pixel_values).pooled), dim=-1)  # [B, D]
        if neg is None:
            return img @ pos.t()
        sim_pos = torch.einsum("bd,ctd->bct", img, pos).mean(-1)  # [B, 14]
        sim_neg = torch.einsum("bd,ctd->bct", img, neg).mean(-1)
        return sim_pos - sim_neg
    return score


def load_baseline(path, device):
    ckpt = load_checkpoint(path)
    backbone = ckpt.get("config", {}).get("backbone", "vit_base_patch16_224")
    model = BaselineClassifier(backbone_name=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return lambda pv: torch.sigmoid(model(pv).logits)


def per_class_auroc(scores: np.ndarray, targets: np.ndarray, classes: list[str]) -> dict:
    out = {}
    for i, c in enumerate(classes):
        y = targets[:, i]
        out[c] = float(roc_auc_score(y, scores[:, i])) if len(np.unique(y)) > 1 else None
    return out


MIN_SUPPORT = 10  # AUROC on a handful of positives is noise; report a support-filtered macro too


def summarize(pc: dict, support: dict) -> dict:
    valid_all = [v for v in pc.values() if v is not None]
    valid_shared = [pc[c] for c in SHARED_CLASSES if pc.get(c) is not None]
    well_supported = [c for c in pc if pc[c] is not None and support[c] >= MIN_SUPPORT]
    ws_shared = [c for c in well_supported if c in SHARED_CLASSES]
    return {
        "per_class_auroc": pc,
        "per_class_positive_support": {c: int(support[c]) for c in pc},
        "macro_auroc_all_valid_classes": float(np.mean(valid_all)) if valid_all else None,
        "n_valid_classes": len(valid_all),
        "macro_auroc_8_shared_classes": float(np.mean(valid_shared)) if valid_shared else None,
        "n_valid_shared": len(valid_shared),
        f"macro_auroc_classes_with_ge{MIN_SUPPORT}_positives": float(np.mean([pc[c] for c in well_supported])) if well_supported else None,
        f"classes_with_ge{MIN_SUPPORT}_positives": well_supported,
        f"macro_auroc_shared_classes_with_ge{MIN_SUPPORT}_positives": float(np.mean([pc[c] for c in ws_shared])) if ws_shared else None,
        f"shared_classes_with_ge{MIN_SUPPORT}_positives": ws_shared,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = ClassificationDataset(MANIFEST, "test", projections=("Frontal",))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])))
    filenames = ds.rows["filename"].tolist()
    targets = torch.cat([t for _, t in loader]).numpy().astype(int)
    support = {c: int(targets[:, i].sum()) for i, c in enumerate(CHEXPERT_LABELS)}
    print(f"[info] frontal test images: {len(ds)} | positives per class: "
          + ", ".join(f"{c}={int(targets[:, i].sum())}" for i, c in enumerate(CHEXPERT_LABELS)), flush=True)

    systems = {}
    loaders = {
        "v2_stage2_image_only": load_stage2_image_only,
        "v2_stage1_zero_shot": lambda p, d: load_zero_shot(p, d),
        "v2_stage1_zero_shot_valloss_selected": lambda p, d: load_zero_shot(p, d),
        "v2_stage1_zero_shot_posneg": lambda p, d: load_zero_shot(p, d, mode="pos_minus_neg"),
        "phase3_stage1_zero_shot_posneg": lambda p, d: load_zero_shot(p, d, mode="pos_minus_neg"),
        "v3_stage1_sentence_zero_shot": lambda p, d: load_zero_shot(p, d),
        "v3_stage1_sentence_zero_shot_posneg": lambda p, d: load_zero_shot(p, d, mode="pos_minus_neg"),
        "v2_stage1_zero_shot_frozen_text": lambda p, d: load_zero_shot(p, d),
        "v2_baseline_imagenet_vit": load_baseline,
        "v2_baseline_xraydinov2_linear": load_baseline,
        "phase3_stage2_blind": load_stage2_image_only,
        "phase3_stage1_zero_shot": lambda p, d: load_zero_shot(p, d),
    }
    ckpt_for = {
        "v2_stage2_image_only": CKPT["v2_stage2_image_only"], "v2_stage1_zero_shot": CKPT["v2_stage1"],
        "v2_stage1_zero_shot_valloss_selected": CKPT["v2_stage1_valloss"],
        "v2_stage1_zero_shot_posneg": CKPT["v2_stage1"],
        "phase3_stage1_zero_shot_posneg": CKPT["phase3_stage1"],
        "v3_stage1_sentence_zero_shot": CKPT["v3_stage1_sentence"],
        "v3_stage1_sentence_zero_shot_posneg": CKPT["v3_stage1_sentence"],
        "v2_stage1_zero_shot_frozen_text": CKPT["v2_stage1_frozen_text"],
        "v2_baseline_imagenet_vit": CKPT["v2_baseline_imagenet_vit"], "v2_baseline_xraydinov2_linear": CKPT["v2_baseline_xraydinov2_linear"],
        "phase3_stage2_blind": CKPT["phase3_stage2"], "phase3_stage1_zero_shot": CKPT["phase3_stage1"],
    }
    raw = pd.DataFrame({"filename": filenames})
    for i, c in enumerate(CHEXPERT_LABELS):
        raw[f"target_{c}"] = targets[:, i]

    for name, build in loaders.items():
        path = ckpt_for[name]
        if not path.exists():
            print(f"[skip] {name}: {path} missing", flush=True)
            continue
        score_fn = build(path, device)
        scores = []
        with torch.no_grad():
            for pv, _ in loader:
                scores.append(score_fn(pv.to(device)).cpu())
        scores = torch.cat(scores).numpy()
        pc = per_class_auroc(scores, targets, list(CHEXPERT_LABELS))
        systems[name] = summarize(pc, support)
        for i, c in enumerate(CHEXPERT_LABELS):
            raw[f"{name}_{c}"] = scores[:, i]
        print(f"[done] {name}: macro(all)={systems[name]['macro_auroc_all_valid_classes']:.3f} "
              f"macro(8 shared)={systems[name]['macro_auroc_8_shared_classes']:.3f}", flush=True)
        torch.cuda.empty_cache()

    # RadZero: re-score saved probabilities on the frontal subset with the corrected labels.
    rz = pd.read_csv(RADZERO_RAW).set_index("filename").loc[filenames]
    rz_scores = np.stack([rz[rc].values for rc in RADZERO_TO_MEDXVLM], axis=1)
    rz_targets = np.stack([targets[:, CHEXPERT_LABELS.index(mc)] for mc in RADZERO_TO_MEDXVLM.values()], axis=1)
    pc = per_class_auroc(rz_scores, rz_targets, SHARED_CLASSES)
    systems["radzero_zero_shot_rescored"] = summarize(pc, support)
    systems["radzero_zero_shot_rescored"]["note"] = "re-scored from Phase 5 saved per-image probabilities; no re-run"
    for rc, mc in RADZERO_TO_MEDXVLM.items():
        raw[f"radzero_{mc}"] = rz[rc].values
    print(f"[done] radzero_zero_shot_rescored: macro(8 shared)={systems['radzero_zero_shot_rescored']['macro_auroc_8_shared_classes']:.3f}")

    out = {"n_frontal_test_images": len(ds), "shared_classes": SHARED_CLASSES, "systems": systems}
    (OUT_DIR / "v2_classification_summary.json").write_text(json.dumps(out, indent=2))
    raw.to_csv(OUT_DIR / "v2_classification_raw.csv", index=False)
    print(f"[done] wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
