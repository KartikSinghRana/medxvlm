#!/usr/bin/env python3
"""Download the Indiana University Chest X-ray collection into data/raw/.

Pulls the "raddar/chest-xrays-indiana-university" dataset from Kaggle via
kagglehub. Some environments can fetch this dataset anonymously; others
require a Kaggle API token. On any download failure this script prints
manual-download fallback instructions rather than leaving a bare
stack trace.

Usage:
    python scripts/download_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

DATASET_HANDLE = "raddar/chest-xrays-indiana-university"
DATASET_URL = "https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university"

MANUAL_DOWNLOAD_INSTRUCTIONS = """
Automatic download via kagglehub failed:
    {error}

This usually means Kaggle API credentials are missing or invalid. To fix:

Option A -- configure Kaggle API credentials (recommended):
  1. Sign in at https://www.kaggle.com and go to Account -> Create New API Token.
     This downloads a `kaggle.json` file with your username and key.
  2. Place it at ~/.kaggle/kaggle.json and restrict its permissions:
         mkdir -p ~/.kaggle
         mv /path/to/kaggle.json ~/.kaggle/kaggle.json
         chmod 600 ~/.kaggle/kaggle.json
     (Or set the KAGGLE_USERNAME and KAGGLE_KEY environment variables.)
  3. Re-run: python scripts/download_dataset.py

Option B -- manual download (no API credentials needed):
  1. Visit {dataset_url} in a browser and click "Download".
  2. Unzip the archive.
  3. Move its contents directly into: {target_dir}
"""


def download_indiana_cxr(target_dir: Path) -> Path:
    """Download the Indiana University CXR dataset into ``target_dir``.

    Args:
        target_dir: Directory the dataset files should end up in
            (created if it does not exist).

    Returns:
        The path kagglehub wrote the dataset files to.

    Raises:
        RuntimeError: if the kagglehub download fails, with manual-download
            fallback instructions attached to the message.
    """
    import kagglehub

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_path = kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(target_dir))
    except Exception as exc:
        raise RuntimeError(
            MANUAL_DOWNLOAD_INSTRUCTIONS.format(
                error=exc, dataset_url=DATASET_URL, target_dir=target_dir
            )
        ) from exc
    return Path(downloaded_path)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    target_dir = repo_root / "data" / "raw"

    print(f"Downloading '{DATASET_HANDLE}' into {target_dir} ...")
    try:
        downloaded_path = download_indiana_cxr(target_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    n_files = sum(1 for p in downloaded_path.rglob("*") if p.is_file())
    print(f"Done. Dataset ready at: {downloaded_path} ({n_files} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
