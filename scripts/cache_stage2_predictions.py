#!/usr/bin/env python3
"""Cache the 4-seed Stage 2 ensemble's probabilities for every manifest image + per-class thresholds.

Feeds the findings-conditioned Stage 3 (v2c): the same cached predictions are
used to build training prompts and test prompts, so the decoder sees one
prompt distribution. Thresholds maximise F1 per class on the *frontal val*
split (classes with fewer than `--min-val-positives` positives fall back to the
median threshold of the well-supported classes). Also writes, for the 150-image
Phase 5 report subsample, a "classifier verbalised" pseudo-report CSV -- the
flagged findings as a bare list -- so the scorer can show what CE F1 a report
that merely names the classifier's findings would get (the ceiling a faithful
verbaliser can reach).

Usage:
    PYTHONPATH=src python scripts/cache_stage2_predictions.py
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
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import build_transforms, labels_to_training_target  # noqa: E402
from medxvlm.data.findings_prompt import LISTABLE_LABELS, findings_phrase  # noqa: E402
from medxvlm.inference import MedXVLMPipeline  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images" / "images_normalized"
SUBSAMPLE = REPO_ROOT / "docs" / "phase5_artifacts" / "report_gen_subsample.csv"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"


class ManifestImages(Dataset):
    def __init__(self, filenames: list[str]):
        self.filenames = filenames
        self.transform = build_transforms("test")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        return self.transform(Image.open(IMAGES_DIR / self.filenames[i]).convert("RGB")), i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-val-positives", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = MedXVLMPipeline(device=device)
    manifest = pd.read_csv(MANIFEST)
    if args.limit:
        manifest = manifest.head(args.limit)
    loader = DataLoader(ManifestImages(manifest.filename.tolist()), batch_size=args.batch_size, num_workers=6, pin_memory=True)
    probs = np.zeros((len(manifest), len(CHEXPERT_LABELS)), dtype=np.float32)
    with torch.no_grad():
        for pv, idx in loader:
            pv = pv.to(device, non_blocking=True)
            acc = None
            for vision, head, _ in pipe.classifiers:
                p = torch.sigmoid(head(vision(pv).patch_tokens, text_embeds=None).logits)
                acc = p if acc is None else acc + p
            probs[idx.numpy()] = (acc / len(pipe.classifiers)).cpu().numpy()
    pred = manifest[["filename", "uid", "split", "projection"]].copy()
    for j, l in enumerate(CHEXPERT_LABELS):
        pred[l] = probs[:, j]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_DIR / "stage2_ensemble_predictions.csv", index=False)
    print(f"[cache] {len(pred)} images scored by {len(pipe.classifiers)} models in {time.time() - t0:.0f}s", flush=True)

    # thresholds on frontal val
    val = manifest[(manifest.split == "val") & (manifest.projection == "Frontal")]
    yv = np.stack([labels_to_training_target(r).numpy() for _, r in val.iterrows()]).astype(int)
    pv_ = probs[val.index.to_numpy()]
    thresholds, report = {}, {}
    good = []
    for j, l in enumerate(CHEXPERT_LABELS):
        n_pos = int(yv[:, j].sum())
        if n_pos >= args.min_val_positives:
            prec, rec, thr = precision_recall_curve(yv[:, j], pv_[:, j])
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-8, None)
            k = int(np.nanargmax(f1[:-1]))
            thresholds[l] = float(thr[k]); good.append(float(thr[k]))
            report[l] = {"val_positives": n_pos, "threshold": float(thr[k]), "val_f1_at_threshold": float(f1[k]), "fallback": False}
        else:
            report[l] = {"val_positives": n_pos, "fallback": True}
    fallback = float(np.median(good))
    for l in CHEXPERT_LABELS:
        if report[l]["fallback"]:
            thresholds[l] = fallback
            report[l]["threshold"] = fallback
    # val macro F1 of the thresholded classifier (well-supported classes) for reference
    yhat = (pv_ >= np.array([thresholds[l] for l in CHEXPERT_LABELS])).astype(int)
    supported = [j for j, l in enumerate(CHEXPERT_LABELS) if not report[l]["fallback"]]
    macro = float(np.mean([f1_score(yv[:, j], yhat[:, j], zero_division=0) for j in supported]))
    out = {"source": "4-seed classifier_v2 ensemble", "split_for_thresholds": "val, Frontal", "n_val_frontal": int(len(val)),
           "min_val_positives": args.min_val_positives, "fallback_threshold": fallback, "thresholds": thresholds, "per_class": report,
           "val_macro_f1_supported_classes": macro, "n_supported_classes": len(supported)}
    (OUT_DIR / "stage2_ensemble_thresholds.json").write_text(json.dumps(out, indent=2))
    print(f"[thresholds] fallback={fallback:.3f}; val macro F1 over {len(supported)} supported classes = {macro:.3f}")
    for l in CHEXPERT_LABELS:
        r = report[l]
        print(f"   {l:26s} pos={r['val_positives']:3d} thr={r['threshold']:.3f}" + ("" if r["fallback"] else f" F1={r['val_f1_at_threshold']:.3f}"))

    # verbalised-classifier pseudo-reports for the 150-image subsample
    if SUBSAMPLE.exists() and not args.limit:
        sub = pd.read_csv(SUBSAMPLE)[["filename"]].merge(pred, on="filename", how="left")
        ref = manifest.set_index("filename").report_text
        def verbalise(row):
            flagged = [l for l in LISTABLE_LABELS if row[l] >= thresholds[l]]
            flagged.sort(key=lambda l: -row[l])
            return (". ".join(flagged) + ".") if flagged else "No acute cardiopulmonary abnormality."
        vb = pd.DataFrame({"filename": sub.filename, "projection": sub.projection, "reference": sub.filename.map(ref),
                           "generated": sub.apply(verbalise, axis=1)})
        vb.to_csv(OUT_DIR / "classifier_verbalised_generations.csv", index=False)
        print(f"[verbalised] wrote {len(vb)} pseudo-reports ({vb.generated.nunique()} distinct); flagged-per-image mean {sub.apply(lambda r: sum(r[l] >= thresholds[l] for l in LISTABLE_LABELS), axis=1).mean():.2f}")
    print(f"[done] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
