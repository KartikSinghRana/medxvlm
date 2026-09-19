"""Shared CheXpert-style label vocabulary for both the baseline and proposed models.

Kept as one constant tuple so the baseline classifier and the Stage 2
classifier head can never disagree about label count or ordering.
"""

from __future__ import annotations

CHEXPERT_LABELS: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

NUM_CHEXPERT_LABELS: int = len(CHEXPERT_LABELS)
