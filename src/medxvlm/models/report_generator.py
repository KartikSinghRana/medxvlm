"""Stage 3: LLaVA-style report generation off the frozen Stage 1 vision tower.

Pools the frozen `VisionEncoder`'s patch tokens down to a short grid of
visual tokens, projects them into a small instruction-tuned decoder's
embedding space with a 2-layer MLP, and splices them into the decoder's
input sequence at a literal `<image>` placeholder -- the LLaVA recipe,
scaled down to a 1.5B decoder with LoRA rather than full fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from medxvlm.models.vision_encoder import VisionEncoder

DEFAULT_DECODER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IMAGE_TOKEN = "<image>"
DEFAULT_POOLED_GRID_SIZE = 7  # 7x7 = 49 visual tokens, LLaVA-style compression from 14x14
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
IGNORE_INDEX = -100


@dataclass
class ReportGenBatch:
    """A ready-to-forward batch: spliced visual+text embeddings, mask, labels."""

    inputs_embeds: torch.Tensor  # [B, T, H]
    attention_mask: torch.Tensor  # [B, T]
    labels: torch.Tensor  # [B, T], IGNORE_INDEX everywhere except answer+eos tokens


class ReportGenerator(nn.Module):
    """Stage 3: frozen vision tower -> projector -> LoRA-adapted causal LM.

    # why the vision encoder is passed in, not built here: it must be the
    # *trained* Stage 1 tower (loaded from a ContrastiveAlignment
    # checkpoint by the training script), not a freshly-initialised one --
    # this module only owns the projection + splicing + decoder, matching
    # "one responsibility per module".
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        decoder_name: str = DEFAULT_DECODER_NAME,
        pooled_grid_size: int = DEFAULT_POOLED_GRID_SIZE,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()

        self.pooled_grid_size = pooled_grid_size

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        # why float32, not the checkpoint's native bfloat16: keeps every
        # tensor in this module (frozen vision encoder, projector, LoRA
        # adapters, decoder) in one dtype, avoiding dtype-mismatch errors
        # at the projector/embedding splice point -- simplicity over the
        # (here unneeded) memory savings of mixed precision.
        base_decoder = AutoModelForCausalLM.from_pretrained(decoder_name, dtype=torch.float32)
        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            base_decoder.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(DEFAULT_LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        )
        self.decoder = get_peft_model(base_decoder, lora_config)

        decoder_hidden = base_decoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.vision_encoder.embed_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
        )

    @property
    def num_visual_tokens(self) -> int:
        return self.pooled_grid_size**2

    def _build_prompt(self, instruction: str) -> str:
        messages = [{"role": "user", "content": f"{IMAGE_TOKEN}\n{instruction}"}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _pool_patches(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """2-D average-pool the 14x14 patch grid down to `pooled_grid_size`^2 tokens."""
        batch, num_patches, dim = patch_tokens.shape
        grid = int(num_patches**0.5)
        spatial = patch_tokens.transpose(1, 2).reshape(batch, dim, grid, grid)
        pooled = F.adaptive_avg_pool2d(spatial, (self.pooled_grid_size, self.pooled_grid_size))
        return pooled.flatten(2).transpose(1, 2)  # [B, pooled_grid_size**2, D]

    def _visual_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            patch_tokens = self.vision_encoder(pixel_values).patch_tokens
        pooled = self._pool_patches(patch_tokens)
        return self.projector(pooled)  # [B, num_visual_tokens, decoder_hidden]

    def _split_at_image_token(self, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_positions = (prompt_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        if image_positions.numel() == 0:
            raise ValueError(f"Prompt has no '{IMAGE_TOKEN}' token: {self.tokenizer.decode(prompt_ids)!r}")
        image_pos = image_positions[0].item()
        return prompt_ids[:image_pos], prompt_ids[image_pos + 1 :]

    def prepare_batch(
        self, pixel_values: torch.Tensor, instructions: list[str], answers: list[str]
    ) -> ReportGenBatch:
        """Build `inputs_embeds`/`attention_mask`/`labels` for one training batch.

        Each example's visual tokens are spliced in at its own `<image>`
        placeholder; `labels` is `IGNORE_INDEX` over the whole image+prompt
        span so the loss is masked to answer tokens only, per the Phase 3
        spec.
        """
        device = pixel_values.device
        visual_embeds_all = self._visual_embeds(pixel_values)  # [B, V, H]
        embed_layer = self.decoder.get_input_embeddings()

        sequences: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for i, (instruction, answer) in enumerate(zip(instructions, answers)):
            prompt_ids = self.tokenizer(
                self._build_prompt(instruction), add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)
            answer_ids = self.tokenizer(
                answer + self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)
            prefix_ids, suffix_ids = self._split_at_image_token(prompt_ids)

            seq_embeds = torch.cat(
                [
                    embed_layer(prefix_ids),
                    visual_embeds_all[i],
                    embed_layer(suffix_ids),
                    embed_layer(answer_ids),
                ],
                dim=0,
            )
            seq_labels = torch.cat(
                [
                    torch.full((prefix_ids.shape[0],), IGNORE_INDEX, device=device, dtype=torch.long),
                    torch.full((self.num_visual_tokens,), IGNORE_INDEX, device=device, dtype=torch.long),
                    torch.full((suffix_ids.shape[0],), IGNORE_INDEX, device=device, dtype=torch.long),
                    answer_ids,
                ]
            )
            sequences.append(seq_embeds)
            labels_list.append(seq_labels)

        max_len = max(seq.shape[0] for seq in sequences)
        hidden = sequences[0].shape[-1]
        batch_embeds = torch.zeros(len(sequences), max_len, hidden, device=device, dtype=sequences[0].dtype)
        batch_labels = torch.full((len(sequences), max_len), IGNORE_INDEX, device=device, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_len, device=device, dtype=torch.long)
        for i, (seq, lab) in enumerate(zip(sequences, labels_list)):
            length = seq.shape[0]
            batch_embeds[i, :length] = seq
            batch_labels[i, :length] = lab
            attention_mask[i, :length] = 1

        return ReportGenBatch(inputs_embeds=batch_embeds, attention_mask=attention_mask, labels=batch_labels)

    def forward(self, pixel_values: torch.Tensor, instructions: list[str], answers: list[str]):
        batch = self.prepare_batch(pixel_values, instructions, answers)
        return self.decoder(
            inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask, labels=batch.labels
        )

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, instruction: str, max_new_tokens: int = 128) -> list[str]:
        """Greedy-decode a report per image, given one shared instruction for the batch."""
        was_training = self.training
        self.eval()
        device = pixel_values.device
        batch_size = pixel_values.shape[0]

        visual_embeds_all = self._visual_embeds(pixel_values)  # [B, V, H]
        embed_layer = self.decoder.get_input_embeddings()
        prompt_ids = self.tokenizer(
            self._build_prompt(instruction), add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(device)
        prefix_ids, suffix_ids = self._split_at_image_token(prompt_ids)
        prefix_embeds = embed_layer(prefix_ids).unsqueeze(0).expand(batch_size, -1, -1)
        suffix_embeds = embed_layer(suffix_ids).unsqueeze(0).expand(batch_size, -1, -1)

        inputs_embeds = torch.cat([prefix_embeds, visual_embeds_all, suffix_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if was_training:
            self.train()
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
