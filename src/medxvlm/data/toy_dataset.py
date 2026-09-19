"""Smoke-test dataset over the raw Indiana University CXR download.

This is deliberately not the real training pipeline: there is no image
cleaning, no CheXpert-style labelling, no train/val/test split, and no
transforms. Its only job is to confirm that whatever
`scripts/download_dataset.py` placed under `data/raw/` can be read back
as (image, report) pairs end to end. The real manifest and labeler are
Phase 2 work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

REPORTS_CSV_NAME = "indiana_reports.csv"
PROJECTIONS_CSV_NAME = "indiana_projections.csv"
IMAGES_DIR_NAME = "images_normalized"
REPORT_TEXT_COLUMNS = ("findings", "impression")


@dataclass(frozen=True)
class ToyExample:
    """One (image path, report text) pairing discovered under a root dir."""

    image_path: Path
    report_text: str


def _find_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{name}' under {root}. Did you run "
            "scripts/download_dataset.py first?"
        )
    return matches[0]


def _find_images_dir(root: Path) -> Path:
    matches = [p for p in root.rglob(IMAGES_DIR_NAME) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find an '{IMAGES_DIR_NAME}' directory under {root}. "
            "Did you run scripts/download_dataset.py first?"
        )
    return matches[0]


def _report_text(report_row: pd.Series) -> str:
    """Join the findings + impression sections of one report row.

    Either section may be missing (NaN) for a given report; missing
    sections are skipped rather than turned into the literal string
    "nan". No further cleaning is applied — de-identification
    placeholders (e.g. "XXXX") are left as-is.
    """
    parts = [
        str(report_row[col]).strip()
        for col in REPORT_TEXT_COLUMNS
        if pd.notna(report_row.get(col))
    ]
    return " ".join(part for part in parts if part)


def _load_examples(root: Path) -> list[ToyExample]:
    reports = pd.read_csv(_find_file(root, REPORTS_CSV_NAME)).set_index("uid")
    projections = pd.read_csv(_find_file(root, PROJECTIONS_CSV_NAME))
    images_dir = _find_images_dir(root)

    examples: list[ToyExample] = []
    for row in projections.itertuples(index=False):
        image_path = images_dir / str(row.filename)
        if not image_path.exists() or row.uid not in reports.index:
            continue
        report_text = _report_text(reports.loc[row.uid])
        if report_text:
            examples.append(ToyExample(image_path, report_text))
    return examples


class ToyChestXrayDataset(Dataset):
    """Minimal (image, report) reader over the raw Indiana CXR download.

    Joins `indiana_projections.csv` (image filename -> uid) with
    `indiana_reports.csv` (uid -> findings/impression text) and returns
    each resulting pair as `(PIL.Image, str)`. No filtering beyond
    skipping rows whose image file or report text is missing, no
    labelling, and no transforms — this exists purely to smoke-test that
    the raw download is readable end to end.

    Args:
        root: Directory containing the raw, unprocessed dataset files
            (typically `data/raw/`).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._examples = _load_examples(self.root)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        example = self._examples[index]
        image = Image.open(example.image_path).convert("RGB")
        return image, example.report_text
