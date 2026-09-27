#!/usr/bin/env python3
"""Operating points and reference distributions for the classifier ensemble's yes/no calls.

Two operating points per finding, both chosen on the frontal VALIDATION split only:
  - "balanced":  the F1-optimal threshold (already in stage2_ensemble_thresholds.json);
  - "sensitive": the highest threshold that still catches >= 80% of validation positives.
Findings with fewer than 10 validation positives cannot be tuned reliably; they keep the
median fallback threshold in both modes and are marked `validated: false`.

Also stores, per finding, percentiles of the ensemble probability among validation images
that do NOT have that finding, so the app can say "higher than 90% of X-rays without it".

The test-split sensitivity/specificity of each mode is recorded for display only (it is never
used to choose anything). Output: docs/phase5_artifacts/v2/stage2_operating_points.json.

Usage:
    python scripts/compute_operating_points.py [--target-sensitivity 0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

V2 = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"
MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
MIN_VAL_POSITIVES = 10


def split(manifest: pd.DataFrame, pred: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = manifest[(manifest.split == name) & (manifest.projection == "Frontal")].set_index("filename")
    labels = rows[list(CHEXPERT_LABELS)].apply(lambda c: c.isin([1.0, -1.0])).astype(int)  # U-Ones, as in training
    return pred.loc[rows.index], labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-sensitivity", type=float, default=0.8)
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST)
    pred = pd.read_csv(V2 / "stage2_ensemble_predictions.csv").set_index("filename")
    f1 = json.load(open(V2 / "stage2_ensemble_thresholds.json"))["per_class"]
    pv, yv = split(manifest, pred, "val")
    pt, yt = split(manifest, pred, "test")

    out: dict = {"target_sensitivity": args.target_sensitivity, "min_val_positives": MIN_VAL_POSITIVES,
                 "source": "4-seed classifier_v2 ensemble; thresholds and percentiles from the frontal validation split",
                 "per_class": {}}
    for label in CHEXPERT_LABELS:
        n_val = int(yv[label].sum())
        validated = n_val >= MIN_VAL_POSITIVES
        balanced = float(f1[label]["threshold"])
        if validated:
            _fpr, tpr, thr = roc_curve(yv[label], pv[label])
            sensitive = float(thr[int(np.argmax(tpr >= args.target_sensitivity))])
        else:
            sensitive = balanced
        negatives = np.sort(pv.loc[yv[label] == 0, label].to_numpy())
        entry = {"validated": validated, "val_positives": n_val,
                 "thresholds": {"balanced": balanced, "sensitive": sensitive},
                 "val_negative_percentiles": [float(np.percentile(negatives, q)) for q in range(101)]}
        y, p = yt[label], pt[label]
        if 0 < y.sum() < len(y):
            entry["test"] = {mode: {"sensitivity": float((p[y == 1] >= t).mean()), "specificity": float((p[y == 0] < t).mean())}
                             for mode, t in entry["thresholds"].items()}
            entry["test"]["positives"] = int(y.sum())
        out["per_class"][label] = entry

    findings = [l for l in CHEXPERT_LABELS if l != "No Finding"]
    abnormal = yt["No Finding"] == 0
    summary = {}
    for mode in ("balanced", "sensitive"):
        flags = pd.DataFrame({l: pt[l] >= out["per_class"][l]["thresholds"][mode] for l in findings})
        summary[mode] = {"abnormal_images_with_no_flag": int((~flags[abnormal].any(axis=1)).sum()),
                         "abnormal_images": int(abnormal.sum()),
                         "normal_images_flagged": int(flags[~abnormal].any(axis=1).sum()),
                         "normal_images": int((~abnormal).sum())}
    out["test_summary"] = summary
    (V2 / "stage2_operating_points.json").write_text(json.dumps(out, indent=2))
    for mode, s in summary.items():
        print(f"{mode:9s}: abnormal with no flag {s['abnormal_images_with_no_flag']}/{s['abnormal_images']}, "
              f"normal falsely flagged {s['normal_images_flagged']}/{s['normal_images']}")
    print(f"[done] {V2 / 'stage2_operating_points.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
