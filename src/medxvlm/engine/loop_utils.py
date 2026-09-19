"""Shared real-training-loop plumbing: seeding, checkpointing, dual logging.

Used identically by all four `scripts/train_*.py` entry points so none of
them re-implements checkpoint bookkeeping or logging setup -- this is a
different, real training loop from `sanity_trainer.py`'s overfit-one-batch
wiring check, not a patch on top of it (see docs/code_review_notes.md).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply any non-`None` CLI overrides on top.

    # why: epoch count / batch size / etc. must be config-driven, not
    # hardcoded in each train_*.py body, while still letting a one-off run
    # tweak a single value from the CLI without editing the YAML file.
    """
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: nn.Module, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_stage1_encoders(checkpoint_path: str | Path, device: torch.device):
    """Load a trained Stage 1 `ContrastiveAlignment` checkpoint's two encoders.

    Returns fresh `VisionEncoder`/`TextEncoder` instances with the trained
    weights loaded in -- built with `pretrained=False` (skips the network
    download) since both backbones are architecturally identical regardless
    of which weight source built them (see vision_encoder.py/text_encoder.py
    docstrings), so the checkpoint's state dict loads in cleanly either way.
    """
    from medxvlm.models.text_encoder import TextEncoder
    from medxvlm.models.vision_encoder import VisionEncoder, infer_backbone_from_state_dict

    state = load_checkpoint(checkpoint_path)["model_state_dict"]
    vision_encoder = VisionEncoder(pretrained=False, backbone=infer_backbone_from_state_dict(state))
    text_encoder = TextEncoder(pretrained=False)
    vision_encoder.load_state_dict({k[len("vision_encoder.") :]: v for k, v in state.items() if k.startswith("vision_encoder.")})
    text_encoder.load_state_dict({k[len("text_encoder.") :]: v for k, v in state.items() if k.startswith("text_encoder.")})
    return vision_encoder.to(device), text_encoder.to(device)


def load_vision_encoder(checkpoint_path: str | Path, device: torch.device):
    """Load only the `VisionEncoder` from any checkpoint that contains one.

    Works for Stage 1 (`ContrastiveAlignment`) and Stage 2 (`Stage2Model`)
    checkpoints alike -- the v2 image-only Stage 2 checkpoint has no text
    encoder at all, so `load_stage1_encoders` cannot read it, but its
    fine-tuned vision tower is exactly what Stage 3 wants to sit on.
    """
    from medxvlm.models.vision_encoder import VisionEncoder, infer_backbone_from_state_dict

    state = load_checkpoint(checkpoint_path)["model_state_dict"]
    vision_encoder = VisionEncoder(pretrained=False, backbone=infer_backbone_from_state_dict(state))
    vision_encoder.load_state_dict({k[len("vision_encoder.") :]: v for k, v in state.items() if k.startswith("vision_encoder.")})
    return vision_encoder.to(device)


class EpochLogger:
    """Writes each epoch's scalars to both TensorBoard and a plain CSV.

    # why both: TensorBoard for interactive inspection during/after a run;
    # the CSV is what docs/ablation_study.md and docs/training_summary.md
    # quote numbers from directly, without needing to parse event files.
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "metrics.csv"
        self._rows: list[dict[str, Any]] = []

        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, epoch)
        self._rows.append({"epoch": epoch, **metrics})
        self._write_csv()

    def _write_csv(self) -> None:
        fieldnames: list[str] = []
        for row in self._rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

    def close(self) -> None:
        self._writer.close()
