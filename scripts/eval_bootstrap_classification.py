#!/usr/bin/env python3
"""95% confidence intervals for the classification AUROCs, and paired tests between systems.

Every system is scored on the same 377 held-out frontal test images, so the
bootstrap resamples *images* once per replicate and scores all systems on
that same resample -- differences between systems are paired, which is what
makes "ensemble vs RadZero" a fair test rather than two overlapping intervals.

Two macro AUROCs are reported:
  - over the 8 findings shared with RadZero (the headline metric; classes with
    no positives in a resample are skipped for that resample -- Pneumothorax
    has a single positive image, so it is absent from ~37% of resamples);
  - over the 5 shared findings with >= 10 test positives (stable in every
    resample; the more trustworthy of the two).

Inputs (no model is run): `v2_classification_raw.csv` (targets, single-seed
model, baselines, zero-shot, RadZero), `stage2_ensemble_predictions.csv` (the
4-model ensemble), and, if present, `rq1_backbones_test_predictions.csv`
(the RQ1 encoder comparison, written by eval_rq1_backbones.py).

Usage:
    python scripts/eval_bootstrap_classification.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
V2 = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"
SHARED = ["Atelectasis", "Cardiomegaly", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Lung Opacity", "Fracture", "Edema"]
MIN_SUPPORT = 10

RAW_SYSTEMS = {  # display name -> column prefix in v2_classification_raw.csv
    "MedXVLM, single model (seed 0)": "v2_stage2_image_only",
    "RadZero (published)": "radzero",
    "Baseline head, ImageNet ViT": "v2_baseline_imagenet_vit",
    "Baseline head, X-ray DINOv2": "v2_baseline_xraydinov2_linear",
    "Zero-shot prompts (v3b)": "v3b_stage1_findings_zero_shot",
}
PAIRS = [  # (a, b): report a - b
    ("MedXVLM, 4-model ensemble", "RadZero (published)"),
    ("MedXVLM, single model (seed 0)", "RadZero (published)"),
    ("MedXVLM, 4-model ensemble", "MedXVLM, single model (seed 0)"),
    ("MedXVLM, single model (seed 0)", "Baseline head, X-ray DINOv2"),
    ("MedXVLM, single model (seed 0)", "Baseline head, ImageNet ViT"),
    ("Baseline head, ImageNet ViT", "Baseline head, X-ray DINOv2"),
    ("RadZero (published)", "Zero-shot prompts (v3b)"),
]


def macro(scores: dict[str, np.ndarray], y: dict[str, np.ndarray], classes: list[str], idx: np.ndarray) -> float:
    vals = []
    for c in classes:
        yc = y[c][idx]
        if 0 < yc.sum() < len(yc):
            vals.append(roc_auc_score(yc, scores[c][idx]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = pd.read_csv(V2 / "v2_classification_raw.csv")
    y = {c: raw[f"target_{c}"].to_numpy().astype(int) for c in SHARED}
    stable = [c for c in SHARED if y[c].sum() >= MIN_SUPPORT]

    systems: dict[str, dict[str, np.ndarray]] = {}
    for name, prefix in RAW_SYSTEMS.items():
        systems[name] = {c: raw[f"{prefix}_{c}"].to_numpy() for c in SHARED}
    ens = pd.read_csv(V2 / "stage2_ensemble_predictions.csv").set_index("filename").loc[raw.filename]
    systems["MedXVLM, 4-model ensemble"] = {c: ens[c].to_numpy() for c in SHARED}
    pairs = list(PAIRS)
    rq1_path = V2 / "rq1_backbones_test_predictions.csv"
    if rq1_path.exists():
        rq1 = pd.read_csv(rq1_path).set_index("filename").loc[raw.filename]
        for name in sorted({col.split("__")[0] for col in rq1.columns}):
            systems[name] = {c: rq1[f"{name}__{c}"].to_numpy() for c in SHARED}
        names = set(systems)
        for a, b in [("RQ1 BiomedCLIP", "RQ1 OpenAI CLIP"), ("RQ1 X-ray DINOv2", "RQ1 BiomedCLIP"),
                     ("RQ1 X-ray DINOv2", "RQ1 OpenAI CLIP"), ("RQ1 X-ray DINOv2", "RQ1 ImageNet ViT"),
                     ("RQ1 ImageNet ViT", "RQ1 OpenAI CLIP"), ("RQ1 BiomedCLIP", "RQ1 ImageNet ViT")]:
            if a in names and b in names:
                pairs.append((a, b))

    n = len(raw)
    all_idx = np.arange(n)
    rng = np.random.RandomState(args.seed)
    boots = {name: {"8": [], "5": []} for name in systems}
    for _ in range(args.n_boot):
        idx = rng.randint(0, n, n)
        for name, sc in systems.items():
            boots[name]["8"].append(macro(sc, y, SHARED, idx))
            boots[name]["5"].append(macro(sc, y, stable, idx))
    boots = {name: {k: np.array(v) for k, v in d.items()} for name, d in boots.items()}

    def ci(a: np.ndarray) -> list[float]:
        a = a[~np.isnan(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = {
        "n_images": n, "n_boot": args.n_boot, "shared_classes": SHARED,
        "stable_classes": stable, "positives": {c: int(y[c].sum()) for c in SHARED},
        "systems": {}, "paired_differences": {},
    }
    for name, sc in systems.items():
        out["systems"][name] = {
            "macro_auroc_8": macro(sc, y, SHARED, all_idx), "ci95_8": ci(boots[name]["8"]),
            "macro_auroc_5_stable": macro(sc, y, stable, all_idx), "ci95_5_stable": ci(boots[name]["5"]),
        }
    for a, b in pairs:
        entry = {}
        for k in ("8", "5"):
            d = boots[a][k] - boots[b][k]
            d = d[~np.isnan(d)]
            point = out["systems"][a]["macro_auroc_8" if k == "8" else "macro_auroc_5_stable"] - \
                out["systems"][b]["macro_auroc_8" if k == "8" else "macro_auroc_5_stable"]
            entry[k] = {"diff": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "p_a_better": float((d > 0).mean())}
        out["paired_differences"][f"{a} − {b}"] = entry

    (V2 / "classification_bootstrap.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{'system':36s} {'AUROC-8':>8s} {'95% CI':>16s}   {'AUROC-5':>8s} {'95% CI':>16s}")
    for name, s in sorted(out["systems"].items(), key=lambda kv: -kv[1]["macro_auroc_8"]):
        print(f"{name:36s} {s['macro_auroc_8']:8.3f} [{s['ci95_8'][0]:.3f}, {s['ci95_8'][1]:.3f}]   "
              f"{s['macro_auroc_5_stable']:8.3f} [{s['ci95_5_stable'][0]:.3f}, {s['ci95_5_stable'][1]:.3f}]")
    print("\npaired differences (a − b):")
    for pair, e in out["paired_differences"].items():
        print(f"  {pair:66s} 8-class {e['8']['diff']:+.3f} [{e['8']['ci95'][0]:+.3f}, {e['8']['ci95'][1]:+.3f}] "
              f"P(a>b)={e['8']['p_a_better']:.2f} | 5-class {e['5']['diff']:+.3f} [{e['5']['ci95'][0]:+.3f}, {e['5']['ci95'][1]:+.3f}]")
    print(f"[done] wrote {V2 / 'classification_bootstrap.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
