#!/usr/bin/env python3
"""External-dataset test: MedXVLM's v2 image-only classifier on NIH ChestX-ray14.

This is the "MedXVLM on *their* dataset" direction Phase 5 could not run:
ChestX-ray14 is one of RadZero's native benchmarks (RadZero paper Table 5:
CXR14 AUC 0.804, 518px, authors' labels), is fully public (no credentialing),
and its labels overlap CheXpert-14 on a handful of concepts. Nothing here is
trained -- the v2 Stage 2 checkpoint is run as-is on the OFFICIAL test split
(`test_list.txt`, patient-disjoint), so every number is a pure domain-shift
measurement: a different hospital, different scanners, different labeler
(NIH's NLP over its own reports), and 1024px PNGs downsampled to 224.

Label mapping (CheXpert-14 head output -> ChestX-ray14 ground-truth column),
deliberately narrow -- only concept-identical pairs:
    Cardiomegaly      -> Cardiomegaly
    Edema             -> Edema
    Consolidation     -> Consolidation
    Pneumonia         -> Pneumonia
    Atelectasis       -> Atelectasis
    Pneumothorax      -> Pneumothorax
    Pleural Effusion  -> Effusion
    No Finding        -> "No Finding" (NIH's own label string)
Not mapped (no clean counterpart): Lung Opacity (NIH has Infiltration, a
narrower concept), Lung Lesion (NIH splits Mass/Nodule), Pleural Other (NIH
Pleural_Thickening is one sub-type), Fracture, Support Devices, Enlarged
Cardiomediastinum (absent from NIH). Emphysema/Fibrosis/Hernia (NIH-only)
are not predicted by MedXVLM. NIH provides no uncertainty labels, so no
U-Ones policy applies; view filter = PA/AP frontal only (all NIH images are
frontal, but the column is kept for the record).

Usage:
    python scripts/eval_external_chestxray14.py [--nih-root <dir>] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import build_transforms  # noqa: E402
from medxvlm.engine.loop_utils import load_checkpoint  # noqa: E402
from medxvlm.models.classifier_head import ClassifierHead  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402
from medxvlm.models.vision_encoder import VisionEncoder, infer_backbone_from_state_dict  # noqa: E402

CKPT = REPO_ROOT / "checkpoints" / "classifier_v2" / "local_on" / "best.pt"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"
LABEL_MAP = {  # MedXVLM head label -> NIH label string
    "Cardiomegaly": "Cardiomegaly", "Edema": "Edema", "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia", "Atelectasis": "Atelectasis", "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion", "No Finding": "No Finding",
}
NIH_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
              "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"]


def find_nih_root(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "nih-chest-xrays" / "data" / "versions"
    versions = sorted(cache.glob("*")) if cache.exists() else []
    if not versions:
        raise SystemExit(f"NIH ChestX-ray14 not found under {cache}; pass --nih-root")
    return versions[-1]


class NihDataset(Dataset):
    def __init__(self, df: pd.DataFrame, index: dict[str, Path]):
        self.df = df.reset_index(drop=True)
        self.index = index
        self.transform = build_transforms("test")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        name = self.df.loc[i, "Image Index"]
        return self.transform(Image.open(self.index[name]).convert("RGB")), i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-root", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test only: evaluate first N test images.")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    root = find_nih_root(args.nih_root)
    entry = pd.read_csv(next(root.rglob("Data_Entry_2017*.csv")))
    test_list = set(next(root.rglob("test_list.txt")).read_text().split())
    index = {p.name: p for p in root.rglob("*.png")}
    df = entry[entry["Image Index"].isin(test_list)].copy()
    missing = [n for n in df["Image Index"] if n not in index]
    assert not missing, f"{len(missing)} test images listed but not found on disk (download incomplete?)"
    for lab in NIH_LABELS:
        df[lab] = df["Finding Labels"].str.split("|").apply(lambda xs: int(lab in xs))
    if args.limit:
        df = df.head(args.limit)
    print(f"[info] NIH root {root} | official test split: {len(df)} images, {df['Patient ID'].nunique()} patients | "
          f"views: {df['View Position'].value_counts().to_dict()}", flush=True)
    print("[info] positives:", {LABEL_MAP[k]: int(df[LABEL_MAP[k]].sum()) for k in LABEL_MAP}, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(CKPT)["model_state_dict"]
    vision = VisionEncoder(pretrained=False, backbone=infer_backbone_from_state_dict(state)).to(device)
    vision.load_state_dict({k[len("vision_encoder."):]: v for k, v in state.items() if k.startswith("vision_encoder.")})
    head = ClassifierHead(embed_dim=vision.embed_dim, text_dim=None).to(device)
    head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})
    vision.eval(); head.eval()

    loader = DataLoader(NihDataset(df, index), batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
    probs = np.zeros((len(df), len(CHEXPERT_LABELS)), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b, (pixel_values, idx) in enumerate(loader):
            out = head(vision(pixel_values.to(device, non_blocking=True)).patch_tokens, text_embeds=None)
            probs[idx.numpy()] = torch.sigmoid(out.logits).cpu().numpy()
            if (b + 1) % 50 == 0 or (b + 1) == len(loader):
                done = min((b + 1) * args.batch_size, len(df))
                print(f"[progress] {done}/{len(df)} {time.time()-t0:.0f}s", flush=True)

    per_class = {}
    for i, cls in enumerate(CHEXPERT_LABELS):
        if cls not in LABEL_MAP:
            continue
        y = df[LABEL_MAP[cls]].to_numpy()
        per_class[LABEL_MAP[cls]] = {
            "medxvlm_head": cls, "n_pos": int(y.sum()),
            "auroc": float(roc_auc_score(y, probs[:, i])) if len(np.unique(y)) > 1 else None,
        }
    valid = [v["auroc"] for v in per_class.values() if v["auroc"] is not None]
    pathology_only = [v["auroc"] for k, v in per_class.items() if k != "No Finding" and v["auroc"] is not None]
    summary = {
        "dataset": "NIH ChestX-ray14, official test_list.txt split", "n_images": int(len(df)), "n_patients": int(df["Patient ID"].nunique()),
        "checkpoint": str(CKPT.relative_to(REPO_ROOT)), "trained_on": "Indiana/Open-i frontal, 3,030 images -- ChestX-ray14 never seen",
        "label_map": LABEL_MAP, "per_class": per_class,
        "macro_auroc_8_mapped_classes": float(np.mean(valid)), "macro_auroc_7_pathologies_excl_no_finding": float(np.mean(pathology_only)),
        "reference_radzero_paper_cxr14_auc": {"value": 0.804, "note": "arXiv:2504.07416 Table 5, RadZero 518px, authors' own label set/protocol -- reported by the authors, NOT reproduced here, not the same class subset"},
        "elapsed_sec": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chestxray14_external_eval_summary.json").write_text(json.dumps(summary, indent=2))
    raw = pd.DataFrame(probs, columns=[f"prob_{c}" for c in CHEXPERT_LABELS]); raw.insert(0, "Image Index", df["Image Index"].values)
    for lab in NIH_LABELS:
        raw[f"nih_{lab}"] = df[lab].values
    raw.to_csv(OUT_DIR / "chestxray14_external_eval_raw.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("label_map",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
