"""Vision encoder: a ViT-B image tower that exposes patch tokens, with two backbones.

- ``"biomedclip"`` (Phase 1-3 default): timm ViT-B/16 initialised from
  BiomedCLIP's image tower where reachable, plain ImageNet weights otherwise.
- ``"xraydinov2"``: ``StanfordAIMI/dinov2-base-xray-224`` (Apache-2.0), a
  DINOv2 ViT-B/14 self-supervised on chest X-rays -- the same encoder
  RadZero uses. Added after Phase 5 showed the BiomedCLIP-initialised
  tower could not produce a usable image-only classifier on ~3k frontal
  training images (docs/cross_dataset_evaluation.md).

Two general-domain towers were added for the RQ1 encoder comparison
(docs/v2_rebuild.md §3.8), both timm ViT-B/16:

- ``"clip_openai"``: OpenAI CLIP's image tower (general-domain contrastive).
- ``"imagenet_vit"``: ImageNet-21k -> 1k supervised ViT-B/16.

All expose the same contract (768-d patch tokens + pooled CLS), so every
downstream head is backbone-agnostic; only the patch-grid side length
differs (14 vs 16), and every consumer already derives that from the
token count rather than a constant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

logger = logging.getLogger(__name__)

BIOMEDCLIP_HUB_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
FALLBACK_TIMM_MODEL_NAME = "vit_base_patch16_224"
CLIP_OPENAI_TIMM_MODEL_NAME = "vit_base_patch16_clip_224.openai"
XRAYDINOV2_HUB_ID = "StanfordAIMI/dinov2-base-xray-224"
IMAGE_SIZE = 224
PATCH_SIZE = 16  # biomedclip / ImageNet ViT-B/16; xraydinov2 is patch 14 (see patch_grid_size)

BackboneName = Literal["biomedclip", "xraydinov2", "clip_openai", "imagenet_vit"]
TIMM_BACKBONES = ("biomedclip", "clip_openai", "imagenet_vit")
DEFAULT_BACKBONE: BackboneName = "biomedclip"


@dataclass
class VisionEncoderOutput:
    """Both the spatial patch grid and a pooled global embedding.

    Attributes:
        patch_tokens: `[batch, num_patches, embed_dim]`, prefix tokens
            (e.g. CLS) stripped out.
        pooled: `[batch, embed_dim]`, the mean of the prefix token(s).
    """

    patch_tokens: torch.Tensor
    pooled: torch.Tensor


def _load_biomedclip_trunk() -> nn.Module:
    """Pull BiomedCLIP's image tower from the HF Hub and return its timm trunk."""
    import open_clip

    model, _preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_HUB_ID)
    trunk = model.visual.trunk
    del model
    return trunk


def _load_fallback_trunk(pretrained: bool) -> nn.Module:
    import timm

    return timm.create_model(FALLBACK_TIMM_MODEL_NAME, pretrained=pretrained, num_classes=0)


def _load_xraydinov2(pretrained: bool) -> nn.Module:
    from transformers import Dinov2Config, Dinov2Model

    if pretrained:
        return Dinov2Model.from_pretrained(XRAYDINOV2_HUB_ID)
    # Architecture only (no download): ViT-B/14 at 224px -> 16x16 patches + CLS,
    # matching the pretrained checkpoint's position-embedding table.
    config = Dinov2Config(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        mlp_ratio=4, patch_size=14, image_size=IMAGE_SIZE,
    )
    return Dinov2Model(config)


class VisionEncoder(nn.Module):
    """ViT-B image tower that exposes patch tokens, not just a pooled embedding.

    # why: Stage 1's local patch-word loss and later Grad-CAM / attention-map
    # explainability both need the spatial patch grid, so returning only a
    # pooled CLS vector (as a plain classifier backbone would) is not
    # enough — every consumer of this class gets patch_tokens for free.
    """

    def __init__(self, pretrained: bool = True, backbone: BackboneName = DEFAULT_BACKBONE) -> None:
        super().__init__()
        self.backbone_name: BackboneName = backbone
        if backbone == "xraydinov2":
            self.backbone = _load_xraydinov2(pretrained)
            self.used_biomedclip_weights = False
            self.embed_dim: int = self.backbone.config.hidden_size
            self.num_prefix_tokens: int = 1 + getattr(self.backbone.config, "num_register_tokens", 0)
            self.patch_size: int = self.backbone.config.patch_size
        elif backbone in TIMM_BACKBONES:
            if backbone == "biomedclip":
                self.backbone, self.used_biomedclip_weights = self._build_biomedclip_backbone(pretrained)
            else:
                import timm

                name = CLIP_OPENAI_TIMM_MODEL_NAME if backbone == "clip_openai" else FALLBACK_TIMM_MODEL_NAME
                self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
                self.used_biomedclip_weights = False
            self.embed_dim = self.backbone.num_features
            self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)
            self.patch_size = PATCH_SIZE
        else:
            raise ValueError(f"Unknown vision backbone {backbone!r}")

    @staticmethod
    def _build_biomedclip_backbone(pretrained: bool) -> tuple[nn.Module, bool]:
        if not pretrained:
            return _load_fallback_trunk(pretrained=False), False
        try:
            return _load_biomedclip_trunk(), True
        except Exception as exc:  # noqa: BLE001 - any hub/network failure should fall back, not crash
            logger.warning(
                "Could not load BiomedCLIP vision weights (%s); falling back to ImageNet ViT-B/16.",
                exc,
            )
            return _load_fallback_trunk(pretrained=True), False

    def forward(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        if self.backbone_name == "xraydinov2":
            tokens = self.backbone(pixel_values=pixel_values).last_hidden_state  # [B, 1 + N, D]
        else:
            tokens = self.backbone.forward_features(pixel_values)  # [B, prefix + N, D]
        if self.num_prefix_tokens > 0:
            pooled = tokens[:, : self.num_prefix_tokens, :].mean(dim=1)
            patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        else:
            patch_tokens = tokens
            pooled = tokens.mean(dim=1)
        return VisionEncoderOutput(patch_tokens=patch_tokens, pooled=pooled)

    @property
    def patch_grid_size(self) -> int:
        """Side length of the square patch grid (14 for ViT-B/16, 16 for ViT-B/14 at 224px)."""
        return IMAGE_SIZE // self.patch_size


def infer_backbone_from_state_dict(state_dict: dict, prefix: str = "vision_encoder.") -> BackboneName:
    """Which backbone a saved VisionEncoder state dict came from.

    # why: Phase 3 checkpoints predate the `backbone` argument and carry no
    # marker, so loaders sniff the key layout instead -- HF DINOv2 keys start
    # with `backbone.embeddings.`, timm ViT keys with `backbone.patch_embed.`.
    """
    # biomedclip and imagenet_vit share one timm architecture, so "biomedclip" loads either;
    # the CLIP tower differs (pre-norm before the blocks), so it needs its own builder.
    if prefix + "backbone.norm_pre.weight" in state_dict:
        return "clip_openai"
    for key in state_dict:
        if key.startswith(prefix + "backbone.embeddings."):
            return "xraydinov2"
        if key.startswith(prefix + "backbone.patch_embed."):
            return "biomedclip"
    raise ValueError(f"No VisionEncoder backbone keys found under prefix {prefix!r}")
