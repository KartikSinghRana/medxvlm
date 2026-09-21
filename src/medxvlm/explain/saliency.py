"""Label- and text-conditioned saliency maps over the vision patch grid.

Four ways of asking "where is the model looking?", all returning one scalar
per patch so they can be overlaid with `attention_maps.overlay_attention`
and scored against bounding boxes with `scripts/eval_grounding_chestxray14.py`:

- `attention_pooling_map`  -- Stage 2's own per-label attention weights (free;
  what the app has shown since day one).
- `gradcam_map`            -- Grad-CAM on the final patch tokens: channel weights
  are the mean gradient of the label logit, the map is ReLU of the weighted sum.
- `attention_rollout`      -- Abnar & Zuidema (2020) rollout of the ViT's own
  self-attention from CLS to every patch. Class-agnostic: one map per image.
- `text_grounded_map`      -- Stage 1 patch-word similarity: cosine between each
  projected patch token and a projected prompt in the shared 512-d space (the
  local-loss view of the alignment, and RadZero's explainability mechanism).

# why patch tokens and not the input pixels for Grad-CAM: on a ViT the last
# block's tokens are the only spatial feature map, and it is also what the
# classifier head consumes, so gradients there are the attribution of the
# head's decision. Working at the 16x16 (or 14x14) grid keeps every method on
# the same footing for the bounding-box comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from medxvlm.models.alignment import ContrastiveAlignment
from medxvlm.models.classifier_head import ClassifierHead
from medxvlm.models.vision_encoder import VisionEncoder


@torch.no_grad()
def attention_pooling_map(vision: VisionEncoder, head: ClassifierHead, pixel_values: torch.Tensor) -> torch.Tensor:
    """Stage 2 per-label attention weights, `[B, num_labels, N]` (rows sum to 1)."""
    return head(vision(pixel_values).patch_tokens, text_embeds=None).attention_weights


def gradcam_map(
    vision: VisionEncoder, head: ClassifierHead, pixel_values: torch.Tensor, label_indices: list[int]
) -> torch.Tensor:
    """Grad-CAM on the final patch tokens for each label in `label_indices` -> `[B, len(label_indices), N]`.

    alpha_c = mean_n dL_label/dA[n, c];  cam_n = ReLU(sum_c alpha_c * A[n, c]).
    """
    was_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        patch_tokens = vision(pixel_values).patch_tokens.detach().requires_grad_(True)  # [B, N, D]
        logits = head(patch_tokens, text_embeds=None).logits  # [B, L]
        maps = []
        for j, label_idx in enumerate(label_indices):
            grads = torch.autograd.grad(logits[:, label_idx].sum(), patch_tokens, retain_graph=j < len(label_indices) - 1)[0]
            alpha = grads.mean(dim=1, keepdim=True)  # [B, 1, D]
            maps.append(F.relu((alpha * patch_tokens).sum(dim=-1)))  # [B, N]
        return torch.stack(maps, dim=1).detach()
    finally:
        torch.set_grad_enabled(was_grad)


@torch.no_grad()
def attention_rollout(vision: VisionEncoder, pixel_values: torch.Tensor, discard_ratio: float = 0.0) -> torch.Tensor:
    """Attention rollout from the CLS token to each patch, `[B, N]` (class-agnostic).

    Only implemented for the HF DINOv2 backbone (needs `output_attentions`, so
    the module is switched to eager attention on first use -- numerically the
    same forward, just slower).
    """
    if vision.backbone_name != "xraydinov2":
        raise NotImplementedError("attention_rollout is implemented for the xraydinov2 backbone only")
    backbone = vision.backbone
    if backbone.config._attn_implementation != "eager":
        backbone.set_attn_implementation("eager")
    attentions = backbone(pixel_values=pixel_values, output_attentions=True).attentions  # L x [B, H, T, T]
    batch, _, tokens, _ = attentions[0].shape
    eye = torch.eye(tokens, device=pixel_values.device).unsqueeze(0)
    rollout = eye.expand(batch, -1, -1).clone()
    for attn in attentions:
        a = attn.mean(dim=1)  # [B, T, T]
        if discard_ratio > 0:
            flat = a.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            idx = flat.topk(k, dim=1, largest=False).indices
            flat.scatter_(1, idx, 0.0)
            a = flat.view_as(a)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    prefix = vision.num_prefix_tokens
    return rollout[:, 0, prefix:]  # CLS row, patch columns -> [B, N]


class TextGrounder:
    """Prompt -> patch similarity maps from a trained Stage 1 alignment model."""

    def __init__(self, model: ContrastiveAlignment, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device
        self.grid = model.vision_encoder.patch_grid_size

    @torch.no_grad()
    def embed_prompts(self, prompts: list[str]) -> torch.Tensor:
        enc = self.model.text_encoder.tokenize(prompts, max_length=64)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.model.encode_text(enc["input_ids"], enc["attention_mask"])  # [P, 512], unit norm

    @torch.no_grad()
    def patch_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model.vision_encoder(pixel_values).patch_tokens  # [B, N, D]
        return F.normalize(self.model.vision_proj(tokens), dim=-1)  # [B, N, 512]

    @torch.no_grad()
    def map(self, pixel_values: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of every patch to every prompt -> `[B, P, N]`."""
        return torch.einsum("bnd,pd->bpn", self.patch_embeds(pixel_values), prompt_embeds)


def text_grounded_map(model: ContrastiveAlignment, pixel_values: torch.Tensor, prompts: list[str]) -> torch.Tensor:
    """One-shot convenience wrapper around `TextGrounder` -> `[B, len(prompts), N]`."""
    grounder = TextGrounder(model, pixel_values.device)
    return grounder.map(pixel_values, grounder.embed_prompts(prompts))


def to_grid(patch_map: torch.Tensor | np.ndarray) -> np.ndarray:
    """`[N]` -> `[g, g]` float32 numpy."""
    arr = patch_map.detach().float().cpu().numpy() if isinstance(patch_map, torch.Tensor) else np.asarray(patch_map, dtype=np.float32)
    g = int(round(arr.size**0.5))
    return arr.reshape(g, g)


def upsample_map(patch_map: torch.Tensor | np.ndarray, size: int = 224) -> np.ndarray:
    """Bilinear upsample a patch-grid map to `size x size` (float32 numpy, raw scale)."""
    grid = torch.from_numpy(to_grid(patch_map))[None, None]
    return F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
