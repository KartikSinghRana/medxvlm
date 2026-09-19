"""Real training datasets over `data/processed/manifest.csv`.

Replaces `toy_dataset.py` (kept as-is, still used by
`scripts/sanity_check_models.py`) for actual training: real labels from
`medxvlm.data.labeler`, a uid-level train/val/test split already baked
into the manifest by `scripts/prepare_data.py`, and a real augmentation
pipeline.

# why no horizontal flip, ever: it mirrors cardiac laterality and
# invalidates every left/right statement in the paired report (e.g. "small
# left pleural effusion" would no longer match a flipped image).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from medxvlm.models.labels import CHEXPERT_LABELS
from medxvlm.models.vision_encoder import IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_REPORT_INSTRUCTION = "Generate a radiology findings report for this chest X-ray."


class CLAHETransform:
    """Contrast-Limited Adaptive Histogram Equalization on the grayscale channel.

    # why: chest X-rays are effectively single-channel; CLAHE improves local
    # contrast (e.g. subtle opacities) without the global-histogram
    # over-amplification plain histogram equalization causes. Applied
    # before Resize/ToTensor since it operates on 8-bit pixel values.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Image.Image) -> Image.Image:
        import cv2

        gray = np.array(image.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        equalized = clahe.apply(gray)
        rgb = np.stack([equalized] * 3, axis=-1)
        return Image.fromarray(rgb)


def build_transforms(split: str, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Train gets mild, laterality-preserving augmentation; val/test do not.

    Train: CLAHE -> mild rotation -> brightness jitter -> resize/tensor/
    normalize -> random erasing. Val/test: CLAHE -> resize/tensor/normalize
    only (deterministic preprocessing, no augmentation).
    """
    common_tail = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if split == "train":
        return transforms.Compose(
            [
                CLAHETransform(),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.2),
                *common_tail,
                transforms.RandomErasing(p=0.25),
            ]
        )
    return transforms.Compose([CLAHETransform(), *common_tail])


def _load_split(manifest_path: str | Path, split: str, projections: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Rows of one split, optionally restricted to given view(s) (e.g. ``("Frontal",)``).

    # why a projection filter: the Indiana set is ~50% lateral views, which
    # carry little signal for most CheXpert labels -- training on them
    # halves the useful data and adds label noise (the lateral image
    # inherits the frontal's report). Phase 5's rebuild trains frontal-only.
    """
    manifest = pd.read_csv(manifest_path)
    if split not in set(manifest["split"]):
        raise ValueError(f"split={split!r} not found in {manifest_path} (has {sorted(set(manifest['split']))})")
    rows = manifest[manifest["split"] == split]
    if projections is not None:
        rows = rows[rows["projection"].isin(projections)]
    return rows.reset_index(drop=True)


@lru_cache(maxsize=8)
def _image_index(data_root: Path) -> dict[str, Path]:
    """Filename -> full path, built once per `data_root` and cached.

    # why: `data_root` holds thousands of images; resolving each
    # `__getitem__` call with a fresh `rglob` walk (as the Phase 1 toy
    # dataset does, fine for a handful of smoke-test samples) would make a
    # real multi-epoch training run prohibitively slow. One walk, cached.
    """
    return {p.name: p for p in data_root.rglob("*.png")}


def _resolve_image_path(manifest_path: str | Path, filename: str) -> Path:
    manifest_dir = Path(manifest_path).resolve().parent
    data_root = manifest_dir.parent / "raw"
    index = _image_index(data_root)
    try:
        return index[filename]
    except KeyError:
        raise FileNotFoundError(f"Could not find image '{filename}' under {data_root}") from None


def labels_to_training_target(row: pd.Series, uncertain_to_positive: bool = True) -> torch.Tensor:
    """Map the labeler's {1, 0, -1, NaN} values to a dense {0, 1} training target.

    # why here, not in labeler.py: keeps the labeler's raw output
    # (including uncertainty and "not mentioned") inspectable for the
    # assertion suite and manual review, and isolates the training-time
    # policy choice (U-Ones: uncertain -> positive, matching CheXpert's
    # best-performing overall policy) to the one place it's applied.
    """
    values = row[list(CHEXPERT_LABELS)].to_numpy(dtype=float)
    values = np.where(np.isnan(values), 0.0, values)
    fill = 1.0 if uncertain_to_positive else 0.0
    values = np.where(values == -1.0, fill, values)
    return torch.from_numpy(values.astype(np.float32))


class AlignmentDataset(Dataset):
    """Stage 1: (image, report text) pairs for contrastive alignment."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: transforms.Compose | None = None,
        projections: tuple[str, ...] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.rows = _load_split(manifest_path, split, projections)
        self.transform = transform or build_transforms(split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows.iloc[index]
        image_path = _resolve_image_path(self.manifest_path, row["filename"])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), row["report_text"]


class ClassificationDataset(Dataset):
    """Stage 2 / baseline: (image, 14-dim label vector, report text) triples.

    `report_text` is included (not just the image) because Stage 2's
    classifier head optionally fuses a text embedding of the paired report
    into classification (gated text fusion); the baseline simply ignores it.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: transforms.Compose | None = None,
        uncertain_to_positive: bool = True,
        projections: tuple[str, ...] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.rows = _load_split(manifest_path, split, projections)
        self.transform = transform or build_transforms(split)
        self.uncertain_to_positive = uncertain_to_positive

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows.iloc[index]
        image_path = _resolve_image_path(self.manifest_path, row["filename"])
        image = Image.open(image_path).convert("RGB")
        target = labels_to_training_target(row, self.uncertain_to_positive)
        return self.transform(image), target, row["report_text"]


class ReportGenDataset(Dataset):
    """Stage 3: (image, instruction, target report text) triples."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: transforms.Compose | None = None,
        instruction: str = DEFAULT_REPORT_INSTRUCTION,
        projections: tuple[str, ...] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.rows = _load_split(manifest_path, split, projections)
        self.transform = transform or build_transforms(split)
        self.instruction = instruction

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.rows.iloc[index]
        image_path = _resolve_image_path(self.manifest_path, row["filename"])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.instruction, row["report_text"]
