"""Text encoder: PubMedBERT tower, falling back to bert-base-uncased.

Mirrors `vision_encoder.py`'s BiomedCLIP-or-fallback pattern: try the
domain-pretrained checkpoint first, fall back to a general-domain model
with the same architecture family if the Hub is unreachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)

# "PubMedBERT" was Microsoft's original name for this checkpoint; the HF repo
# itself was later renamed to BiomedBERT (same weights). The old PubMedBERT
# id still resolves via a Hub redirect, but using the current id directly
# avoids that extra hop and matches how BiomedCLIP's own text tower is named.
PUBMEDBERT_MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
FALLBACK_MODEL_ID = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 256


@dataclass
class TextEncoderOutput:
    """Per-token embeddings and a pooled sentence embedding.

    Attributes:
        token_embeddings: `[batch, seq_len, embed_dim]`.
        pooled: `[batch, embed_dim]`, the CLS token's embedding.
    """

    token_embeddings: torch.Tensor
    pooled: torch.Tensor


class TextEncoder(nn.Module):
    """PubMedBERT text tower that exposes per-token embeddings, not just CLS.

    # why: mirrors VisionEncoder's patch_tokens design — Stage 1's local
    # patch-word loss and later sentence-attribution explainability both
    # need word-level representations, so token_embeddings is returned
    # alongside the pooled sentence vector rather than discarded.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.model, self.tokenizer, self.used_pubmedbert_weights = self._build(pretrained)
        self.embed_dim: int = self.model.config.hidden_size

    @staticmethod
    def _build(pretrained: bool):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if pretrained:
            try:
                tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL_ID)
                model = AutoModel.from_pretrained(PUBMEDBERT_MODEL_ID)
                return model, tokenizer, True
            except Exception as exc:  # noqa: BLE001 - any hub/network failure should fall back, not crash
                logger.warning(
                    "Could not load PubMedBERT weights (%s); falling back to bert-base-uncased.",
                    exc,
                )
                # TODO: swap in PubMedBERT weights if the checkpoint isn't reachable in this environment.

        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)
        if pretrained:
            model = AutoModel.from_pretrained(FALLBACK_MODEL_ID)
        else:
            # Untrained weights on purpose: Phase 2 sanity checks care about
            # wiring correctness, not representation quality, so skip the
            # pretrained-weight download entirely for speed.
            config = AutoConfig.from_pretrained(FALLBACK_MODEL_ID)
            model = AutoModel.from_config(config)
        return model, tokenizer, False

    def tokenize(self, texts: list[str], max_length: int = DEFAULT_MAX_LENGTH) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TextEncoderOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # why: outputs.pooler_output (dense + tanh over the CLS token) is used
        # instead of a plain outputs.last_hidden_state[:, 0, :] slice — BERT's
        # pooler is a real trainable layer, and slicing around it would leave
        # it permanently disconnected from the loss (caught by
        # scripts/sanity_check_models.py's every-parameter-has-a-gradient check).
        return TextEncoderOutput(token_embeddings=outputs.last_hidden_state, pooled=outputs.pooler_output)
