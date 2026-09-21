"""Sentence-level visual attribution for the Stage 3 report generator.

For a generated report, answers "which part of the image was each sentence
looking at, and how much was it looking at the image at all?" by re-running
the decoder teacher-forced on [prompt | 64 visual tokens | prompt | report]
with attention weights exposed, then averaging, for every token of a
sentence, its attention onto the 64 visual-token positions (over all layers
and heads). The per-sentence `visual_share` -- attention mass on visual tokens
divided by attention mass on everything -- is a cheap faithfulness signal: a
sentence with a tiny visual share was written from the language prior, not
the image.

# why teacher-forced re-scoring and not attention captured during generation:
# beam search re-orders and prunes hypotheses, so per-step attentions do not
# line up with the final text; a second forward pass over the final text is
# exact and costs one decoder call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from medxvlm.models.report_generator import ReportGenerator

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SentenceAttribution:
    sentence: str
    token_count: int
    visual_map: torch.Tensor  # [num_visual_tokens], sums to 1, CPU
    visual_share: float  # attention mass on visual tokens / total attention mass (0-1)
    visual_share_excl_sink: float  # same, with the sequence's first position (the attention sink) removed
    text_share_prompt: float  # attention mass on the prompt's own text tokens, for scale


def split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Sentences in order with their (start, end) character offsets in `text`."""
    spans, pos = [], 0
    for part in _SENTENCE_SPLIT.split(text):
        if not part.strip():
            pos += len(part)
            continue
        start = text.index(part, pos)
        spans.append((part.strip(), start, start + len(part)))
        pos = start + len(part)
    return spans


def _ensure_eager(model: ReportGenerator) -> None:
    base = model.decoder.get_base_model()
    if base.config._attn_implementation != "eager":
        base.set_attn_implementation("eager")


@torch.no_grad()
def attribute_sentences(
    model: ReportGenerator, pixel_values: torch.Tensor, instruction: str, report: str, layers: slice | None = None
) -> list[SentenceAttribution]:
    """Per-sentence visual attention maps for one image (`pixel_values`: `[1, 3, H, W]`) and its `report`."""
    assert pixel_values.shape[0] == 1, "one image at a time"
    _ensure_eager(model)
    model.eval()
    device = pixel_values.device
    batch = model.prepare_batch(pixel_values, [instruction], [report])
    out = model.decoder(inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask, output_attentions=True)
    attentions = out.attentions[layers] if layers is not None else out.attentions  # L x [1, H, T, T]
    attn = torch.stack([a[0].mean(dim=0) for a in attentions]).mean(dim=0)  # [T, T], mean over layers & heads

    # Positions: prefix | visual | suffix | answer(+eos). prepare_batch lays them out in that order.
    prompt_ids = model.tokenizer(model._build_prompt(instruction), add_special_tokens=False, return_tensors="pt").input_ids[0]
    prefix_ids, _suffix_ids = model._split_at_image_token(prompt_ids.to(device))
    v0 = prefix_ids.shape[0]
    v1 = v0 + model.num_visual_tokens
    enc = model.tokenizer(report + model.tokenizer.eos_token, add_special_tokens=False, return_offsets_mapping=True)
    answer_len = len(enc.input_ids)
    total_len = int(batch.attention_mask[0].sum())
    a0 = total_len - answer_len
    offsets = enc.offset_mapping

    results = []
    for sentence, start, end in split_sentences_with_spans(report):
        tok_rows = [a0 + i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
        if not tok_rows:
            continue
        rows = attn[tok_rows]  # [n_tok, T]
        visual = rows[:, v0:v1].mean(dim=0)  # [V]
        total = rows.sum(dim=1).mean().clamp(min=1e-8)
        visual_mass = rows[:, v0:v1].sum(dim=1).mean()
        sink_mass = rows[:, 0].mean()
        prompt_text_mass = rows[:, 1:v0].sum(dim=1).mean() + rows[:, v1:a0].sum(dim=1).mean()
        results.append(SentenceAttribution(
            sentence=sentence, token_count=len(tok_rows),
            visual_map=(visual / visual.sum().clamp(min=1e-8)).cpu(),
            visual_share=float(visual_mass / total),
            visual_share_excl_sink=float(visual_mass / (total - sink_mass).clamp(min=1e-8)),
            text_share_prompt=float(prompt_text_mass / total),
        ))
    return results
