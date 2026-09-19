#!/usr/bin/env python3
"""Smoke-test the raw Indiana CXR download via ToyChestXrayDataset.

Instantiates the toy dataset, pulls a handful of samples, and prints each
image's shape/mode plus a truncated report string. This only confirms the
raw download under data/raw/ is readable end to end — it is not a data
quality check (no missing-value analysis, no label validation); that is
Phase 2 work once the real manifest and labeler exist.

Usage:
    python scripts/sanity_check_data.py [--root data/raw] [--n 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPORT_PREVIEW_CHARS = 120


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Directory containing the raw downloaded dataset (default: data/raw).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of samples to pull and print (default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Imported lazily so --help works even if the package isn't installed yet.
    from medxvlm.data.toy_dataset import ToyChestXrayDataset

    dataset = ToyChestXrayDataset(args.root)
    n_total = len(dataset)
    print(f"Loaded {n_total} (image, report) pairs from {args.root}")
    if n_total == 0:
        print("ERROR: dataset is empty.", file=sys.stderr)
        return 1

    n_samples = min(args.n, n_total)
    for i in range(n_samples):
        image, report = dataset[i]
        shape = np.array(image).shape
        preview = report[:REPORT_PREVIEW_CHARS]
        ellipsis = "..." if len(report) > REPORT_PREVIEW_CHARS else ""
        print(f"[{i}] image shape={shape} mode={image.mode}")
        print(f"[{i}] report: {preview}{ellipsis}")

    print(f"\nOK: {n_samples} samples loaded from {args.root} with no errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
