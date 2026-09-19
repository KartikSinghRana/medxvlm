"""Stage 1: contrastive alignment of the vision and text encoders.

Projects both encoders' pooled embeddings into a shared 512-d space,
L2-normalises them, and computes a symmetric (image->text and
text->image) InfoNCE loss over the batch, CLIP-style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from medxvlm.models.text_encoder import TextEncoder
from medxvlm.models.vision_encoder import VisionEncoder

SHARED_EMBED_DIM = 512
LOGIT_SCALE_INIT = math.log(1 / 0.07)  # CLIP's standard init: exp(.) starts at 1/0.07


@dataclass
class AlignmentOutput:
    """Shared-space embeddings and the Stage 1 loss for one batch.

    `global_loss`/`local_loss` are populated whenever their term is
    computed (local is `None` when `use_local_loss=False`) so training
    scripts can log each component separately -- this is the ablation
    study's on/off switch (see `ContrastiveAlignment.use_local_loss`).
    """

    image_embeds: torch.Tensor  # [B, SHARED_EMBED_DIM], L2-normalised
    text_embeds: torch.Tensor  # [B, SHARED_EMBED_DIM], L2-normalised
    loss: torch.Tensor  # scalar
    global_loss: torch.Tensor | None = None
    local_loss: torch.Tensor | None = None


class ContrastiveAlignment(nn.Module):
    """Stage 1: projects vision + text encoders into a shared space and aligns them.

    # why: a learned linear projection per modality — rather than requiring
    # the two encoders' native hidden sizes to already match — lets
    # VisionEncoder and TextEncoder vary independently (different backbone,
    # different hidden size) while still landing in one shared space that
    # Stage 2 and later grounding/explainability can rely on.
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        text_encoder: TextEncoder,
        embed_dim: int = SHARED_EMBED_DIM,
        use_local_loss: bool = False,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_proj = nn.Linear(vision_encoder.embed_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(LOGIT_SCALE_INIT))
        # why a constructor flag, not always-on: this is the ablation
        # study's on/off switch (local patch-word loss vs global-only),
        # everything else about the run held fixed -- see
        # docs/ablation_study.md and scripts/train_align.py.
        self.use_local_loss = use_local_loss

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision_encoder(pixel_values).pooled
        return F.normalize(self.vision_proj(pooled), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.text_encoder(input_ids, attention_mask).pooled
        return F.normalize(self.text_proj(pooled), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> AlignmentOutput:
        vision_out = self.vision_encoder(pixel_values)
        text_out = self.text_encoder(input_ids, attention_mask)
        image_embeds = F.normalize(self.vision_proj(vision_out.pooled), dim=-1)
        text_embeds = F.normalize(self.text_proj(text_out.pooled), dim=-1)

        global_loss = self._global_infonce_loss(image_embeds, text_embeds)
        local_loss = None
        loss = global_loss
        if self.use_local_loss:
            local_loss = local_patch_word_loss(
                vision_out.patch_tokens,
                text_out.token_embeddings,
                attention_mask,
                self.vision_proj,
                self.text_proj,
                self.logit_scale,
            )
            loss = global_loss + local_loss

        return AlignmentOutput(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            loss=loss,
            global_loss=global_loss,
            local_loss=local_loss,
        )

    def _global_infonce_loss(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE: each sample's own image/text pair is the positive."""
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        targets = torch.arange(image_embeds.shape[0], device=image_embeds.device)
        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_text, targets)
        return (loss_i2t + loss_t2i) / 2


def local_patch_word_loss(
    patch_tokens: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_proj: nn.Module,
    text_proj: nn.Module,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Fine-grained patch-word contrastive loss (FILIP-style).

    Projects `patch_tokens` (`[B, N, D_v]`) and `token_embeddings`
    (`[B, L, D_t]`) through the same `vision_proj`/`text_proj` used for the
    global loss (both plain `nn.Linear`, so they apply per-token
    unchanged), L2-normalises, and builds the full `[B, B, N, L]`
    patch-word cosine-similarity tensor. For each (image i, text j) pair,
    every word is matched to its most similar patch (masked by
    `attention_mask` so padding contributes nothing), then averaged over
    real words -- this max-over-patches/mean-over-words reduction is the
    standard FILIP token-level aggregation. The resulting `[B, B]` local
    logit matrix is scored with the same symmetric InfoNCE as the global
    loss, using the same learned `logit_scale`.
    """
    proj_patches = F.normalize(vision_proj(patch_tokens), dim=-1)  # [B, N, D]
    proj_words = F.normalize(text_proj(token_embeddings), dim=-1)  # [B, L, D]

    # [B_img, N, D] x [B_txt, L, D] -> [B_img, B_txt, N, L]
    sim = torch.einsum("ind,jld->ijnl", proj_patches, proj_words)

    # Mask out padding words before the max-over-patches reduction: a
    # padded word must never win as anyone's "most similar patch".
    word_mask = attention_mask.bool()  # [B_txt, L]
    sim = sim.masked_fill(~word_mask[None, :, None, :], float("-inf"))

    max_sim_per_word = sim.max(dim=2).values  # [B_img, B_txt, L]
    max_sim_per_word = max_sim_per_word.masked_fill(~word_mask[None, :, :], 0.0)
    valid_words_per_text = word_mask.sum(dim=-1).clamp(min=1)  # [B_txt]
    local_logits_per_image = max_sim_per_word.sum(dim=-1) / valid_words_per_text[None, :]  # [B_img, B_txt]

    logit_scale_value = logit_scale.exp()
    logits_per_image = logit_scale_value * local_logits_per_image
    logits_per_text = logits_per_image.t()
    targets = torch.arange(patch_tokens.shape[0], device=patch_tokens.device)
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    return (loss_i2t + loss_t2i) / 2
