#!/usr/bin/env python3
"""Phase 5: classification inference on MedXVLM's own held-out test split.

Produces three MedXVLM prediction sets, all on the *same* 748-image test
split (`data/processed/manifest.csv`, split == "test", never seen during
Stage 1/2/3 training), so they can be directly compared to each other and
to RadZero's zero-shot predictions on the identical images
(`eval_radzero_on_medxvlm_testsplit.py` in `external_repro/`):

1. **Stage 2, blind** -- `ClassifierHead(text_embeds=None)`. Image-only,
   exactly what RadZero's zero-shot classifier also is. This is the fair,
   apples-to-apples number.
2. **Stage 2, text-fusion** -- `ClassifierHead` fed the *paired ground-truth
   report's* text embedding, matching exactly what `train_classifier.py`
   trains and validates against (the 0.865 macro AUROC headline number in
   docs/training_summary.md). This is NOT a blind prediction -- the model
   is given text that already states the findings -- so it is reported
   for completeness but flagged as not comparable to RadZero.
3. **Stage 1 zero-shot** -- a CLIP-style zero-shot head built fresh here
   (no separate checkpoint exists for this; it reuses the trained Stage 1
   `ContrastiveAlignment` encoders' `encode_image`/`encode_text`), scoring
   each image against a `"There is {label}"` prompt per CheXpert-14 class,
   cosine similarity as the score. This is the MedXVLM-side counterpart to
   RadZero's own zero-shot protocol -- same task shape, different model.

Usage:
    python scripts/eval_phase5_classification.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import ClassificationDataset, labels_to_training_target  # noqa: E402
from medxvlm.engine.loop_utils import load_checkpoint, load_stage1_encoders  # noqa: E402
from medxvlm.models.classifier_head import ClassifierHead  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402
from medxvlm.models.text_encoder import TextEncoder  # noqa: E402

STAGE1_CHECKPOINT = REPO_ROOT / "checkpoints" / "align" / "local_on" / "best.pt"
STAGE2_CHECKPOINT = REPO_ROOT / "checkpoints" / "classifier" / "local_on" / "best.pt"
MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts"

ZERO_SHOT_PROMPTS = {label: f"There is {label}." if label != "No Finding" else "There is no acute cardiopulmonary abnormality." for label in CHEXPERT_LABELS}


def build_stage2_model(device):
    vision_encoder, text_encoder = load_stage1_encoders(STAGE1_CHECKPOINT, device)
    head = ClassifierHead(embed_dim=vision_encoder.embed_dim, text_dim=text_encoder.embed_dim).to(device)

    state = load_checkpoint(STAGE2_CHECKPOINT)["model_state_dict"]
    vision_encoder.load_state_dict({k[len("vision_encoder."):]: v for k, v in state.items() if k.startswith("vision_encoder.")})
    text_encoder.load_state_dict({k[len("text_encoder."):]: v for k, v in state.items() if k.startswith("text_encoder.")})
    head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})

    vision_encoder.eval().to(device)
    text_encoder.eval().to(device)
    head.eval().to(device)
    return vision_encoder, text_encoder, head


def build_alignment_encoders(device):
    """Stage 1 encoders + their projection heads, for the zero-shot path.

    `load_stage1_encoders` only restores VisionEncoder/TextEncoder, not the
    ContrastiveAlignment module's `vision_proj`/`text_proj`/`logit_scale`,
    so those three are loaded here directly from the same checkpoint.
    """
    from medxvlm.models.alignment import SHARED_EMBED_DIM

    vision_encoder, text_encoder = load_stage1_encoders(STAGE1_CHECKPOINT, device)
    state = load_checkpoint(STAGE1_CHECKPOINT)["model_state_dict"]

    vision_proj = torch.nn.Linear(vision_encoder.embed_dim, SHARED_EMBED_DIM).to(device)
    text_proj = torch.nn.Linear(text_encoder.embed_dim, SHARED_EMBED_DIM).to(device)
    vision_proj.load_state_dict({k[len("vision_proj."):]: v for k, v in state.items() if k.startswith("vision_proj.")})
    text_proj.load_state_dict({k[len("text_proj."):]: v for k, v in state.items() if k.startswith("text_proj.")})

    vision_encoder.eval()
    text_encoder.eval()
    vision_proj.eval()
    text_proj.eval()
    return vision_encoder, text_encoder, vision_proj, text_proj


def macro_and_per_class_auroc(probs: torch.Tensor, targets: torch.Tensor) -> dict:
    probs_np, targets_np = probs.numpy(), targets.numpy().astype(int)
    per_class, valid = {}, []
    for i, label in enumerate(CHEXPERT_LABELS):
        y_true, y_score = targets_np[:, i], probs_np[:, i]
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
            per_class[label] = auc
            valid.append(auc)
        else:
            per_class[label] = None
    return {"per_class_auroc": per_class, "macro_auroc": sum(valid) / len(valid) if valid else None, "n_valid": len(valid)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ClassificationDataset(MANIFEST, "test")
    print(f"[info] test split: {len(dataset)} images")

    def collate(batch):
        images, targets, reports = zip(*batch)
        return torch.stack(images), torch.stack(targets), list(reports)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

    vision_encoder, text_encoder, head = build_stage2_model(device)
    zs_vision_encoder, zs_text_encoder, vision_proj, text_proj = build_alignment_encoders(device)

    # Precompute zero-shot text embeddings, one per CheXpert class, once.
    with torch.no_grad():
        prompts = [ZERO_SHOT_PROMPTS[label] for label in CHEXPERT_LABELS]
        encoded = zs_text_encoder.tokenize(prompts, max_length=64)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        text_pooled = zs_text_encoder(encoded["input_ids"], encoded["attention_mask"]).pooled
        class_text_embeds = F.normalize(text_proj(text_pooled), dim=-1)  # [14, 512]

    all_targets = []
    all_blind_probs, all_fused_probs, all_zs_scores = [], [], []
    filenames = []

    with torch.no_grad():
        row_ptr = 0
        for pixel_values, targets, reports in loader:
            pixel_values = pixel_values.to(device)
            batch_filenames = dataset.rows.iloc[row_ptr: row_ptr + len(targets)]["filename"].tolist()
            row_ptr += len(targets)
            filenames.extend(batch_filenames)

            patch_tokens = vision_encoder(pixel_values).patch_tokens
            blind_logits = head(patch_tokens, text_embeds=None).logits
            all_blind_probs.append(torch.sigmoid(blind_logits).cpu())

            encoded_reports = text_encoder.tokenize(list(reports), max_length=256)
            encoded_reports = {k: v.to(device) for k, v in encoded_reports.items()}
            report_text_embeds = text_encoder(encoded_reports["input_ids"], encoded_reports["attention_mask"]).pooled
            fused_logits = head(patch_tokens, text_embeds=report_text_embeds).logits
            all_fused_probs.append(torch.sigmoid(fused_logits).cpu())

            zs_pooled = zs_vision_encoder(pixel_values).pooled
            image_embeds = F.normalize(vision_proj(zs_pooled), dim=-1)  # [B, 512]
            zs_logits = image_embeds @ class_text_embeds.t()  # [B, 14] cosine similarity in [-1, 1]
            all_zs_scores.append(zs_logits.cpu())

            all_targets.append(targets)

    targets = torch.cat(all_targets)
    blind_probs = torch.cat(all_blind_probs)
    fused_probs = torch.cat(all_fused_probs)
    zs_scores = torch.cat(all_zs_scores)

    results = {
        "n_images": len(dataset),
        "stage2_blind_image_only": macro_and_per_class_auroc(blind_probs, targets),
        "stage2_text_fusion_ground_truth_report": macro_and_per_class_auroc(fused_probs, targets),
        "stage1_zero_shot_cosine_similarity": macro_and_per_class_auroc(zs_scores, targets),
    }

    with open(OUT_DIR / "medxvlm_testsplit_classification_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame({"filename": filenames})
    for i, label in enumerate(CHEXPERT_LABELS):
        df[f"target_{label}"] = targets[:, i].numpy()
        df[f"blind_{label}"] = blind_probs[:, i].numpy()
        df[f"fused_{label}"] = fused_probs[:, i].numpy()
        df[f"zeroshot_{label}"] = zs_scores[:, i].numpy()
    df.to_csv(OUT_DIR / "medxvlm_testsplit_classification_raw.csv", index=False)

    print(json.dumps(results, indent=2))
    print(f"[done] wrote results to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
