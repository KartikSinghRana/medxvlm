#!/usr/bin/env python3
"""Seed ensemble of the v2 image-only classifier: per-seed spread and the averaged model.

Loads every `checkpoints/classifier_v2/<run>/best.pt` given on the command
line, scores each on the frontal held-out test split (and optionally the NIH
ChestX-ray14 test split), then averages their probabilities. Reports:
  - per-seed macro AUROC (8 RadZero-shared classes / all 13) -> mean +- std,
    i.e. the error bar every single-seed number in this project has lacked;
  - the ensemble's macro AUROC, per class.

Usage:
    PYTHONPATH=src python scripts/eval_classifier_ensemble.py \\
        checkpoints/classifier_v2/local_on/best.pt checkpoints/classifier_v2/seed1/best.pt ... [--nih]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_v2_classification import MIN_SUPPORT, SHARED_CLASSES, load_stage2_image_only, per_class_auroc, summarize  # noqa: E402
from medxvlm.data.dataset import ClassificationDataset  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"


def score_loader(score_fn, loader, device) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for pv, *_ in loader:
            outs.append(score_fn(pv.to(device)).cpu())
    return torch.cat(outs).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=lambda p: Path(p).resolve())
    ap.add_argument("--nih", action="store_true", help="Also evaluate on the NIH ChestX-ray14 test split (needs eval_external_chestxray14 data).")
    ap.add_argument("--tag", default="seed_ensemble")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = ClassificationDataset(MANIFEST, "test", projections=("Frontal",))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])))
    targets = torch.cat([t for _, t in loader]).numpy().astype(int)
    support = {c: int(targets[:, i].sum()) for i, c in enumerate(CHEXPERT_LABELS)}

    per_seed, probs_all = {}, []
    for ck in args.checkpoints:
        if not ck.exists():
            print(f"[skip] {ck} missing"); continue
        probs = score_loader(load_stage2_image_only(ck, device), loader, device)
        probs_all.append(probs)
        per_seed[str(ck.relative_to(REPO_ROOT))] = summarize(per_class_auroc(probs, targets, list(CHEXPERT_LABELS)), support)
        s = per_seed[str(ck.relative_to(REPO_ROOT))]
        print(f"[seed] {ck.parent.name:12s} 8-shared={s['macro_auroc_8_shared_classes']:.4f} all13={s['macro_auroc_all_valid_classes']:.4f}", flush=True)
        torch.cuda.empty_cache()

    assert len(probs_all) >= 2, "need at least two checkpoints to ensemble"
    ens = np.mean(probs_all, axis=0)
    ens_summary = summarize(per_class_auroc(ens, targets, list(CHEXPERT_LABELS)), support)
    m8 = np.array([v["macro_auroc_8_shared_classes"] for v in per_seed.values()])
    m13 = np.array([v["macro_auroc_all_valid_classes"] for v in per_seed.values()])
    result = {
        "n_models": len(probs_all), "n_frontal_test_images": len(ds),
        "per_seed": per_seed,
        "seed_mean_std": {"macro_auroc_8_shared": [float(m8.mean()), float(m8.std(ddof=1))],
                          "macro_auroc_all13": [float(m13.mean()), float(m13.std(ddof=1))]},
        "ensemble": ens_summary,
    }
    print(f"[seeds] 8-shared {m8.mean():.4f} +- {m8.std(ddof=1):.4f} | all13 {m13.mean():.4f} +- {m13.std(ddof=1):.4f}")
    print(f"[ensemble n={len(probs_all)}] 8-shared={ens_summary['macro_auroc_8_shared_classes']:.4f} "
          f"shared>=10={ens_summary['macro_auroc_shared_classes_with_ge10_positives']:.4f} all13={ens_summary['macro_auroc_all_valid_classes']:.4f}")

    if args.nih:
        from eval_external_chestxray14 import LABEL_MAP, NIH_LABELS, NihDataset, find_nih_root

        root = find_nih_root(None)
        entry = pd.read_csv(next(root.rglob("Data_Entry_2017*.csv")))
        test_list = set(next(root.rglob("test_list.txt")).read_text().split())
        index = {p.name: p for p in root.rglob("*.png")}
        df = entry[entry["Image Index"].isin(test_list)].copy()
        df = df[df["Image Index"].isin(index)]
        for lab in NIH_LABELS:
            df[lab] = df["Finding Labels"].str.split("|").apply(lambda xs: int(lab in xs))
        nih_loader = DataLoader(NihDataset(df, index), batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
        nih_probs = []
        for ck in args.checkpoints:
            if not ck.exists():
                continue
            fn = load_stage2_image_only(ck, device)
            arr = np.zeros((len(df), len(CHEXPERT_LABELS)), dtype=np.float32)
            with torch.no_grad():
                for pv, idx in nih_loader:
                    arr[idx.numpy()] = fn(pv.to(device, non_blocking=True)).cpu().numpy()
            nih_probs.append(arr)
            torch.cuda.empty_cache()
        nih_ens = np.mean(nih_probs, axis=0)
        per_class = {}
        for i, cls in enumerate(CHEXPERT_LABELS):
            if cls in LABEL_MAP:
                y = df[LABEL_MAP[cls]].to_numpy()
                per_class[LABEL_MAP[cls]] = float(roc_auc_score(y, nih_ens[:, i]))
        per_seed_nih = []
        for arr in nih_probs:
            vals = [roc_auc_score(df[LABEL_MAP[c]].to_numpy(), arr[:, CHEXPERT_LABELS.index(c)]) for c in LABEL_MAP]
            per_seed_nih.append(float(np.mean(vals)))
        result["nih_chestxray14"] = {
            "n_images": int(len(df)), "per_seed_macro_auroc_8": per_seed_nih,
            "seed_mean_std": [float(np.mean(per_seed_nih)), float(np.std(per_seed_nih, ddof=1))],
            "ensemble_macro_auroc_8": float(np.mean(list(per_class.values()))), "ensemble_per_class": per_class,
        }
        print(f"[NIH] seeds {np.mean(per_seed_nih):.4f} +- {np.std(per_seed_nih, ddof=1):.4f} | ensemble {result['nih_chestxray14']['ensemble_macro_auroc_8']:.4f}")

    out = OUT_DIR / f"classifier_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
