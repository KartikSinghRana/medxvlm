#!/usr/bin/env python3
"""Build the real training manifest: `data/processed/manifest.csv`.

Replaces Phase 1/2's toy dataloader with the real pipeline: downloads the
dataset if needed, merges image and report metadata, cleans report text,
runs the rule-based CheXpert-style labeler (`medxvlm.data.labeler`) once
per study (uid), and writes a single manifest with a `split` column from a
uid-level (not row-level) train/val/test split.

# why uid-level, not row-level: a study's frontal and lateral images share
# one report, so splitting by image row would put the same report's text
# on both sides of a split -- a uid *is* the patient/study unit here.

Refuses to run the labeler over the full corpus unless
`scripts/validate_labeler.py`'s assertion suite passes first.

Usage:
    python scripts/prepare_data.py [--data-root data/raw] [--out data/processed/manifest.csv]
        [--train-frac 0.8] [--val-frac 0.1] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPORTS_CSV_NAME = "indiana_reports.csv"
PROJECTIONS_CSV_NAME = "indiana_projections.csv"
IMAGES_DIR_NAME = "images_normalized"
REPORT_TEXT_COLUMNS = ("findings", "impression")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed" / "manifest.csv")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-labeler-validation",
        action="store_true",
        help="Skip scripts/validate_labeler.py before labelling the full corpus (not recommended).",
    )
    return parser.parse_args()


def _ensure_raw_data(data_root: Path) -> None:
    if list(data_root.rglob(REPORTS_CSV_NAME)):
        return
    print(f"No raw data found under {data_root}; downloading...")
    from download_dataset import download_indiana_cxr

    download_indiana_cxr(data_root)


def _find_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find '{name}' under {root}.")
    return matches[0]


def _find_images_dir(root: Path) -> Path:
    matches = [p for p in root.rglob(IMAGES_DIR_NAME) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find an '{IMAGES_DIR_NAME}' directory under {root}.")
    return matches[0]


_DEID_TOKEN_RE = re.compile(r"X{3,}")  # no word boundary: the token is sometimes glued to a digit ("4XXXX")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:])")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _clean_report_text(text: str) -> str:
    """Strip the Indiana corpus's 'XXXX' de-identification placeholders.

    # why: left in, Stage 3 learned to emit "XXXX" as if it were vocabulary
    # (docs/training_summary.md), and the tokens pad out negation windows
    # in the labeler. They carry no information, so drop them and tidy the
    # punctuation spacing that removal leaves behind.
    """
    text = _DEID_TOKEN_RE.sub("", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _report_text(report_row: pd.Series) -> str:
    """Join findings + impression, skipping missing (NaN) sections, then clean.

    Mirrors `medxvlm.data.toy_dataset._report_text` -- duplicated rather
    than imported since the toy dataset is a frozen Phase 1 smoke test, not
    a shared utility.
    """
    parts = [str(report_row[col]).strip() for col in REPORT_TEXT_COLUMNS if pd.notna(report_row.get(col))]
    return _clean_report_text(" ".join(part for part in parts if part))


def _run_labeler_validation() -> None:
    import validate_labeler

    if validate_labeler.main() != 0:
        raise RuntimeError(
            "scripts/validate_labeler.py failed -- refusing to label the full corpus "
            "with an unvalidated labeler. Fix the failing assertion(s) first."
        )


def _assign_splits(uids: list[int], train_frac: float, val_frac: float, seed: int) -> dict[int, str]:
    shuffled = sorted(uids)  # sort first so the shuffle is reproducible regardless of input order
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    split_of: dict[int, str] = {}
    for i, uid in enumerate(shuffled):
        if i < n_train:
            split_of[uid] = "train"
        elif i < n_train + n_val:
            split_of[uid] = "val"
        else:
            split_of[uid] = "test"
    return split_of


def build_manifest(data_root: Path, train_frac: float, val_frac: float, seed: int) -> pd.DataFrame:
    from medxvlm.data.labeler import aggregate_report
    from medxvlm.models.labels import CHEXPERT_LABELS

    reports = pd.read_csv(_find_file(data_root, REPORTS_CSV_NAME)).set_index("uid")
    projections = pd.read_csv(_find_file(data_root, PROJECTIONS_CSV_NAME))
    images_dir = _find_images_dir(data_root)

    report_text_by_uid: dict[int, str] = {}
    labels_by_uid: dict[int, dict[str, float]] = {}
    for uid, row in reports.iterrows():
        text = _report_text(row)
        if not text:
            continue
        report_text_by_uid[uid] = text
        labels_by_uid[uid] = aggregate_report(text)

    split_of = _assign_splits(list(report_text_by_uid.keys()), train_frac, val_frac, seed)

    rows: list[dict] = []
    for proj_row in projections.itertuples(index=False):
        uid = proj_row.uid
        image_path = images_dir / str(proj_row.filename)
        if uid not in report_text_by_uid or not image_path.exists():
            continue
        row = {
            "uid": uid,
            "filename": proj_row.filename,
            "projection": proj_row.projection,
            "report_text": report_text_by_uid[uid],
            "split": split_of[uid],
        }
        row.update(labels_by_uid[uid])
        rows.append(row)

    manifest = pd.DataFrame(rows, columns=["uid", "filename", "projection", "report_text", "split", *CHEXPERT_LABELS])
    return manifest


def main() -> int:
    args = parse_args()
    _ensure_raw_data(args.data_root)

    if args.skip_labeler_validation:
        print("WARNING: skipping labeler validation (--skip-labeler-validation).")
    else:
        print("Validating labeler against scripts/validate_labeler.py before labelling the full corpus...")
        _run_labeler_validation()

    print(f"Building manifest from {args.data_root} ...")
    manifest = build_manifest(args.data_root, args.train_frac, args.val_frac, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    n_studies = manifest["uid"].nunique()
    split_counts = manifest.drop_duplicates("uid")["split"].value_counts().to_dict()
    print(f"Wrote {len(manifest)} image rows across {n_studies} studies to {args.out}")
    print(f"Study-level split: {split_counts}")
    no_finding_rate = manifest["No Finding"].mean()
    print(f"'No Finding' positive rate: {no_finding_rate:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
