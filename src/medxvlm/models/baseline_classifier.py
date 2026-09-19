"""Baseline: a plain supervised ViT-B/16 + linear multi-label head.

Represents the CheXNet / ChestX-ray14-era supervised approach from
`docs/literature_review.md` — no contrastive pretraining, no text branch —
used as the point of comparison for the proposed Stage 1 + 2 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

from medxvlm.models.labels import NUM_CHEXPERT_LABELS

DEFAULT_BACKBONE_NAME = "vit_base_patch16_224"


@dataclass
class BaselineOutput:
    """Output of the baseline classifier."""

    logits: torch.Tensor  # [batch, num_labels]


class BaselineClassifier(nn.Module):
    """Plain supervised ViT-B/16 + linear multi-label head, no text branch.

    # why: exists purely as the "before" side of the project's central
    # comparison — a single global-pooled embedding through a linear head,
    # with none of Stage 1/2's contrastive pretraining or per-label
    # attention — so later evaluation can attribute any gain to those
    # specific design choices rather than to backbone capacity alone.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE_NAME,
        num_labels: int = NUM_CHEXPERT_LABELS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "xraydinov2":
            # Same CXR-pretrained tower as the v2 pipeline, so the v2
            # baseline-vs-pipeline comparison isolates the pipeline design
            # rather than crediting it with the encoder's pretraining.
            from medxvlm.models.vision_encoder import VisionEncoder

            self.backbone = VisionEncoder(pretrained=pretrained, backbone="xraydinov2")
            num_features = self.backbone.embed_dim
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features
        self.head = nn.Linear(num_features, num_labels)

    def forward(self, pixel_values: torch.Tensor) -> BaselineOutput:
        if self.backbone_name == "xraydinov2":
            pooled_features = self.backbone(pixel_values).pooled
        else:
            pooled_features = self.backbone(pixel_values)  # [B, D] (num_classes=0 -> pre-logit features)
        logits = self.head(pooled_features)
        return BaselineOutput(logits=logits)


def multilabel_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard multi-label loss: independent BCE per class, no class weighting."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Asymmetric Loss (Ben-Baruch et al., 2021) for multi-label class imbalance.

    CheXpert-style labels are heavily skewed toward negatives per class
    (most images don't show most findings); ASL down-weights easy
    negatives via `gamma_neg > gamma_pos` (focal-style focusing, applied
    asymmetrically) and "probability shifting" (`clip`), which hard-zeroes
    the loss for negatives the model is already confident about, rather
    than treating every class with the same BCE weight.

    Used as the ablation alternative to `multilabel_bce_loss` -- everything
    else about the run (data, seed, epochs, architecture) held fixed.
    """
    p = torch.sigmoid(logits)
    p_pos = p.clamp(min=eps, max=1 - eps)
    p_neg = (p - clip).clamp(min=0, max=1 - eps) if clip > 0 else p.clamp(max=1 - eps)

    loss_pos = targets * torch.log(p_pos) * (1 - p_pos).pow(gamma_pos)
    loss_neg = (1 - targets) * torch.log(1 - p_neg) * p_neg.pow(gamma_neg)
    return -(loss_pos + loss_neg).sum(dim=-1).mean()
