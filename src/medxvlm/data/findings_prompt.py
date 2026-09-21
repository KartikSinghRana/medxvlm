"""Findings-conditioned instructions for Stage 3 (v2c).

Sentence attribution (docs/explainability_evaluation.md §3) showed the Stage 3
decoder draws ~1% of its attention from the visual tokens and writes mostly
from the language prior. The channel it does use is text -- so the
classifier's predicted findings are put *into the instruction*:

    "Generate a radiology findings report for this chest X-ray.
     Classifier findings: Cardiomegaly, Pleural Effusion."

The same lookup serves training and inference (the 4-seed Stage 2 ensemble's
probabilities cached for every image, thresholded per class on the val split),
so train and test prompts come from the same distribution. During training
each flagged finding is dropped with probability `dropout` so the decoder
learns to tolerate the classifier's misses rather than to copy the list.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from medxvlm.models.labels import CHEXPERT_LABELS

DEFAULT_REPORT_INSTRUCTION = "Generate a radiology findings report for this chest X-ray."
FINDINGS_SUFFIX = " Classifier findings: {findings}."
NONE_FLAGGED = "none flagged"
# "No Finding" is the absence of findings, not a finding to list.
LISTABLE_LABELS = tuple(l for l in CHEXPERT_LABELS if l != "No Finding")


def findings_phrase(flagged: list[str]) -> str:
    return ", ".join(flagged) if flagged else NONE_FLAGGED


def findings_instruction(flagged: list[str], base: str = DEFAULT_REPORT_INSTRUCTION) -> str:
    return base + FINDINGS_SUFFIX.format(findings=findings_phrase(flagged))


class FindingsLookup:
    """filename -> flagged finding names, from a cached prediction CSV + per-class thresholds."""

    def __init__(self, predictions_csv: str | Path, thresholds_json: str | Path, dropout: float = 0.0, seed: int = 0) -> None:
        df = pd.read_csv(predictions_csv)
        self.probs = {row.filename: {l: float(getattr(row, l.replace(" ", "_"))) for l in LISTABLE_LABELS}
                      for row in df.rename(columns=lambda c: c.replace(" ", "_")).itertuples(index=False)}
        self.thresholds = json.load(open(thresholds_json))["thresholds"]
        self.dropout = dropout
        self._rng = random.Random(seed)

    def flagged(self, filename: str, training: bool = False) -> list[str]:
        probs = self.probs[filename]
        names = [l for l in LISTABLE_LABELS if probs[l] >= self.thresholds[l]]
        names.sort(key=lambda l: -probs[l])
        if training and self.dropout > 0:
            names = [l for l in names if self._rng.random() >= self.dropout]
        return names

    def instruction(self, filename: str, training: bool = False) -> str:
        return findings_instruction(self.flagged(filename, training=training))
