#!/usr/bin/env python3
"""Phase 5: score generated reports with MedXVLM's own validated CheXpert-style
labeler (medxvlm.data.labeler), the same code that built every ground-truth
label in data/processed/manifest.csv. This is the "clinical efficacy" (CE)
protocol: run the labeler on the generated text, apply the same U-Ones
policy training uses, and compare against the manifest's own stored ground
truth for that exact image -- not a re-derivation from the reference text,
so labeler noise on the (unedited) reference text never enters the score.

Scores both MedXVLM's Stage 3 generations and CheXOne's generations from
the identical 150-image subsample (see docs/cross_dataset_evaluation.md).

Usage:
    python scripts/eval_phase5_score_reports.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.labeler import aggregate_report  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
ARTIFACTS = REPO_ROOT / "docs" / "phase5_artifacts"
MEDXVLM_GENERATIONS = ARTIFACTS / "medxvlm_stage3_generations.csv"
CHEXONE_GENERATIONS_EXTERNAL = Path("/home/kartik/DNN Project 2/external_repro/chexone_testsplit_generations.csv")


def labels_dict_to_binary_vector(labels: dict[str, float]) -> np.ndarray:
    """Same U-Ones policy as medxvlm.data.dataset.labels_to_training_target."""
    values = np.array([labels[name] for name in CHEXPERT_LABELS], dtype=float)
    values = np.where(np.isnan(values), 0.0, values)
    values = np.where(values == -1.0, 1.0, values)
    return values.astype(int)


def score_generations(gen_df: pd.DataFrame, ground_truth_df: pd.DataFrame, system_name: str) -> dict:
    merged = gen_df.merge(ground_truth_df, on="filename", how="inner", validate="one_to_one")
    assert len(merged) == len(gen_df), f"{system_name}: {len(gen_df) - len(merged)} generations had no manifest match"

    pred_matrix, true_matrix = [], []
    for _, row in merged.iterrows():
        pred_labels = aggregate_report(row["generated"])
        pred_matrix.append(labels_dict_to_binary_vector(pred_labels))
        true_matrix.append(row[list(CHEXPERT_LABELS)].to_numpy(dtype=int))
    pred_matrix = np.stack(pred_matrix)
    true_matrix = np.stack(true_matrix)

    per_class = {}
    f1s, precisions, recalls = [], [], []
    for i, label in enumerate(CHEXPERT_LABELS):
        y_true, y_pred = true_matrix[:, i], pred_matrix[:, i]
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        per_class[label] = {"precision": p, "recall": r, "f1": f1, "support_positive": int(y_true.sum())}
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    micro_f1 = f1_score(true_matrix.flatten(), pred_matrix.flatten(), zero_division=0)
    return {
        "system": system_name,
        "n_reports": len(merged),
        "per_class": per_class,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "micro_f1": float(micro_f1),
    }


def main() -> int:
    manifest = pd.read_csv(MANIFEST)
    ground_truth = manifest[["filename", *CHEXPERT_LABELS]].copy()
    for label in CHEXPERT_LABELS:
        ground_truth[label] = ground_truth[label].fillna(0.0).replace(-1.0, 1.0).astype(int)

    results = {}

    medxvlm_gens = pd.read_csv(MEDXVLM_GENERATIONS)
    results["medxvlm_stage3"] = score_generations(medxvlm_gens, ground_truth, "MedXVLM Stage 3")

    if CHEXONE_GENERATIONS_EXTERNAL.exists():
        chexone_gens = pd.read_csv(CHEXONE_GENERATIONS_EXTERNAL)
        results["chexone"] = score_generations(chexone_gens, ground_truth, "CheXOne")
    else:
        print(f"[warn] {CHEXONE_GENERATIONS_EXTERNAL} not found yet -- skipping CheXOne scoring")

    out_path = ARTIFACTS / "report_generation_clinical_efficacy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    for name, r in results.items():
        print(f"\n=== {r['system']} (n={r['n_reports']}) ===")
        print(f"macro P={r['macro_precision']:.3f} R={r['macro_recall']:.3f} F1={r['macro_f1']:.3f} | micro F1={r['micro_f1']:.3f}")
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
