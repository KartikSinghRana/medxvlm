"""Grounding-heatmap snapshots from Stage 2's per-label attention weights.

# why attention weights, not a separate backprop Grad-CAM: ClassifierHead
# already returns a per-label attention map over patches
# (`ClassifierOutput.attention_weights`, `[B, num_labels, num_patches]`) --
# docs/code_review_notes.md flagged this as "exactly what the later
# explainability milestone needs". Reusing it is a deliberate scope call:
# it is this project's Grad-CAM/"segmentation map" equivalent, not a
# separate gradient-based CAM.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from medxvlm.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def _denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    pixels = (image.detach().cpu() * std + mean).clamp(0, 1)
    return (pixels.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _overlay_heatmap(image_rgb: np.ndarray, attention: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    grid = int(attention.shape[-1] ** 0.5)
    heatmap = attention.detach().cpu().reshape(grid, grid).numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    height, width = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap_uint8, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.cvtColor(cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = image_rgb.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha
    return overlay.astype(np.uint8)


def save_grounding_snapshots(
    images: torch.Tensor,
    attention_weights: torch.Tensor,
    label_indices_per_sample: list[list[int]],
    label_names: tuple[str, ...],
    out_dir: str | Path,
    epoch: int,
    max_samples: int = 4,
) -> list[Path]:
    """Save a handful of per-label grounding-heatmap overlays for one epoch.

    Args:
        images: `[B, 3, H, W]`, normalised the same way the model saw them.
        attention_weights: `[B, num_labels, num_patches]`.
        label_indices_per_sample: which label indices to render for each of
            the first `max_samples` images (e.g. the labels the model
            predicted positive for that sample).
        out_dir: base directory; snapshots are written under
            `out_dir/epoch_{epoch:03d}/`.
    """
    epoch_dir = Path(out_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for sample_idx in range(min(max_samples, images.shape[0])):
        image_rgb = _denormalize(images[sample_idx])
        for label_idx in label_indices_per_sample[sample_idx]:
            overlay = _overlay_heatmap(image_rgb, attention_weights[sample_idx, label_idx])
            label_name = label_names[label_idx].replace(" ", "_")
            out_path = epoch_dir / f"sample{sample_idx}_{label_name}.png"
            Image.fromarray(overlay).save(out_path)
            saved.append(out_path)
    return saved
