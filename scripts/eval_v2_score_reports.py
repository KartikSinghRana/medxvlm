#!/usr/bin/env python3
"""Score every report-generation system on the same 150 test images, same labels.

Systems (each a CSV with `filename, generated`):
  - Phase 3 Stage 3 (docs/phase5_artifacts/medxvlm_stage3_generations.csv)
  - CheXOne, instruct mode (external_repro/chexone_testsplit_generations.csv)
  - v2 Stage 3 (docs/phase5_artifacts/v2/medxvlm_stage3_v2_generations.csv)

Ground truth: the CURRENT manifest (corrected labels, cleaned text) -- so the
Phase 5 numbers computed against the old labels are superseded here for all
three systems at once, on equal footing.

Metrics:
  - Clinical efficacy (CE): medxvlm.data.labeler on the generated text, U-Ones,
    vs. the manifest's stored label vector -> per-class P/R/F1, macro/micro F1.
  - BLEU-4 (sacrebleu corpus BLEU) and ROUGE-L F (rouge_score, mean over
    reports) against the cleaned reference text -- first time these are
    computed in this project.
  - Collapse diagnostics: number of distinct generations, share of the single
    most frequent generation, mean generated length in words.
Everything is reported on the full 150 and on the 70-image frontal subset.

Usage:
    python scripts/eval_v2_score_reports.py
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
ART = REPO_ROOT / "docs" / "phase5_artifacts"
SYSTEMS = {
    "phase3_stage3": ART / "medxvlm_stage3_generations.csv",
    "chexone_instruct": Path("/home/kartik/DNN Project 2/external_repro/chexone_testsplit_generations.csv"),
    "v2_stage3": ART / "v2" / "medxvlm_stage3_v2_generations.csv",
}
OUT = ART / "v2" / "report_generation_scores_v2.json"


def to_binary(labels: dict[str, float]) -> np.ndarray:
    v = np.array([labels[n] for n in CHEXPERT_LABELS], dtype=float)
    v = np.where(np.isnan(v), 0.0, v)
    v = np.where(v == -1.0, 1.0, v)
    return v.astype(int)


def clinical_efficacy(gen: pd.Series, truth: np.ndarray) -> dict:
    pred = np.stack([to_binary(aggregate_report(t)) for t in gen])
    per_class, f1s, ps, rs = {}, [], [], []
    for i, label in enumerate(CHEXPERT_LABELS):
        y, yhat = truth[:, i], pred[:, i]
        p = precision_score(y, yhat, zero_division=0); r = recall_score(y, yhat, zero_division=0); f = f1_score(y, yhat, zero_division=0)
        per_class[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support_positive": int(y.sum())}
        ps.append(p); rs.append(r); f1s.append(f)
    return {
        "macro_precision": float(np.mean(ps)), "macro_recall": float(np.mean(rs)), "macro_f1": float(np.mean(f1s)),
        "micro_f1": float(f1_score(truth.flatten(), pred.flatten(), zero_division=0)),
        "per_class": per_class,
    }


def text_metrics(gen: list[str], ref: list[str]) -> dict:
    import sacrebleu
    from rouge_score import rouge_scorer

    bleu = sacrebleu.corpus_bleu(gen, [ref])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [scorer.score(r, g)["rougeL"].fmeasure for g, r in zip(gen, ref)]
    return {"bleu4": bleu.score, "bleu_note": "sacrebleu corpus_bleu, default 13a tokenizer, 0-100 scale", "rougeL_f": float(np.mean(rl))}


def collapse_stats(gen: pd.Series) -> dict:
    vc = gen.value_counts()
    return {
        "n": int(len(gen)),
        "n_distinct": int(gen.nunique()),
        "top1_share": float(vc.iloc[0] / len(gen)),
        "top1_text": str(vc.index[0])[:200],
        "mean_words": float(gen.str.split().str.len().mean()),
    }


def score(df: pd.DataFrame, truth_cols: list[str], tag: str) -> dict:
    truth = df[truth_cols].to_numpy(dtype=int)
    return {
        "subset": tag, "n": int(len(df)),
        "clinical_efficacy": clinical_efficacy(df.generated, truth),
        "text": text_metrics(df.generated.tolist(), df.reference.tolist()),
        "collapse": collapse_stats(df.generated),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PATH", help="Additional generation CSVs to score.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    for item in args.extra:
        name, _, path = item.partition("=")
        SYSTEMS[name] = Path(path)

    man = pd.read_csv(MANIFEST)
    truth_cols = list(CHEXPERT_LABELS)
    for c in truth_cols:
        man[c] = man[c].fillna(0.0).replace(-1.0, 1.0).astype(int)
    man = man[["filename", "projection", "report_text", *truth_cols]].rename(columns={"report_text": "reference"})

    results = {"ground_truth": "data/processed/manifest.csv (corrected labels, cleaned text)", "systems": {}}
    for name, path in SYSTEMS.items():
        if not path.exists():
            print(f"[skip] {name}: {path} missing"); continue
        gen = pd.read_csv(path)[["filename", "generated"]].fillna({"generated": ""})
        df = gen.merge(man, on="filename", how="inner", validate="one_to_one")
        assert len(df) == len(gen), f"{name}: {len(gen) - len(df)} rows missing from manifest"
        frontal = df[df.projection == "Frontal"]
        results["systems"][name] = {
            "source": str(path),
            "all_150": score(df, truth_cols, "all (80 lateral + 70 frontal)"),
            "frontal_70": score(frontal, truth_cols, "frontal only"),
        }
        for k in ("all_150", "frontal_70"):
            r = results["systems"][name][k]
            print(f"{name:18s} {k:11s} n={r['n']:3d} | CE macroF1={r['clinical_efficacy']['macro_f1']:.3f} microF1={r['clinical_efficacy']['micro_f1']:.3f} "
                  f"| BLEU4={r['text']['bleu4']:.2f} ROUGE-L={r['text']['rougeL_f']:.3f} | distinct={r['collapse']['n_distinct']} top1={r['collapse']['top1_share']:.2f} words={r['collapse']['mean_words']:.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
