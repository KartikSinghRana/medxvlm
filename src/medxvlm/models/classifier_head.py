"""Stage 2: per-label attention pooling classifier over vision patch tokens.

No text fusion yet — this module only consumes `VisionEncoder` patch
tokens. Gated text fusion (mixing in `TextEncoder`/`ContrastiveAlignment`
embeddings before classification) is the next milestone; see the # TODO
below and `docs/code_review_notes.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from medxvlm.models.labels import NUM_CHEXPERT_LABELS

DEFAULT_NUM_ATTENTION_HEADS = 8


@dataclass
class ClassifierOutput:
    """Per-label logits and the attention map each logit was pooled from.

    Attributes:
        logits: `[batch, num_labels]`.
        attention_weights: `[batch, num_labels, num_patches]` — how much
            each label's query attended to each patch. Free localisation
            signal for later explainability work.
    """

    logits: torch.Tensor
    attention_weights: torch.Tensor


class PerLabelAttentionPooling(nn.Module):
    """A learned query per class attends over patch tokens to pool a label-specific vector.

    # why: a single global-pooled image embedding forces every one of the
    # 14 CheXpert labels to share the same spatial summary, even though
    # e.g. cardiomegaly and pneumothorax live in different regions of the
    # image; a learned query per label lets each class attend to whichever
    # patches are relevant to it instead.
    """

    def __init__(
        self,
        embed_dim: int,
        num_labels: int = NUM_CHEXPERT_LABELS,
        num_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.label_queries = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = patch_tokens.shape[0]
        queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, D]
        pooled, attn_weights = self.attention(
            queries, patch_tokens, patch_tokens, need_weights=True, average_attn_weights=True
        )
        pooled = self.norm(pooled)
        return pooled, attn_weights  # [B, L, D], [B, L, N]


class ClassifierHead(nn.Module):
    """Stage 2: 14-class multi-label CheXpert-style classifier over vision patch tokens.

    # why: kept separate from ContrastiveAlignment so the Stage 1 vision
    # tower can be frozen, fine-tuned, or swapped independently of this
    # head, and so gated text fusion (the next milestone) can be added
    # here without touching the alignment loss.
    """

    def __init__(
        self,
        embed_dim: int,
        num_labels: int = NUM_CHEXPERT_LABELS,
        num_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
        text_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling = PerLabelAttentionPooling(embed_dim, num_labels, num_heads)
        # Shared (not per-label) weight: each label already gets its own
        # pooled vector from `pooling`, so one Linear(D, 1) applied to all
        # of them is sufficient and keeps the head parameter-light.
        self.classifier = nn.Linear(embed_dim, 1)

        # Gated text fusion: an optional text embedding (e.g. the paired
        # report's Stage 1 text encoding) is projected to embed_dim and
        # FiLM-gated against the per-label pooled visual vector before
        # classification. Only allocated when text_dim is given, so the
        # image-only path (forward(text_embeds=None)) stays exactly the
        # Phase 2 behaviour scripts/sanity_check_models.py already exercises.
        self.text_proj: nn.Linear | None = None
        self.gate: nn.Linear | None = None
        if text_dim is not None:
            self.text_proj = nn.Linear(text_dim, embed_dim)
            self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, patch_tokens: torch.Tensor, text_embeds: torch.Tensor | None = None) -> ClassifierOutput:
        pooled, attn_weights = self.pooling(patch_tokens)  # [B, L, D], [B, L, N]

        fused = pooled
        if text_embeds is not None:
            if self.text_proj is None or self.gate is None:
                raise ValueError("text_embeds was given but this ClassifierHead was built without text_dim")
            num_labels = pooled.shape[1]
            text_context = self.text_proj(text_embeds).unsqueeze(1).expand(-1, num_labels, -1)  # [B, L, D]
            gate_weight = torch.sigmoid(self.gate(torch.cat([pooled, text_context], dim=-1)))  # [B, L, D]
            fused = gate_weight * pooled + (1 - gate_weight) * text_context

        logits = self.classifier(fused).squeeze(-1)  # [B, L]
        return ClassifierOutput(logits=logits, attention_weights=attn_weights)
