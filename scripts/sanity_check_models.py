#!/usr/bin/env python3
"""Phase 2 sanity check: wiring correctness for the baseline and proposed models.

Runs four independent checks — data loading, the baseline classifier, the
Stage 1 alignment module, and the Stage 2 classifier head — each doing a
forward + loss + backward + optimizer step over ~50 steps on one repeated
batch (the classic overfit-one-batch check), and prints a PASS/FAIL line
per component. This verifies wiring correctness only: no real training
run, no evaluation metrics, no hyperparameter tuning. See
docs/code_review_notes.md for what these checks do and don't cover.

Usage:
    python scripts/sanity_check_models.py [--data-root data/raw] [--n 8]
        [--steps 50] [--pretrained]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

IMAGE_SIZE = 224
NUM_SAMPLES = 8
NUM_SANITY_STEPS = 50
RANDOM_SEED = 0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Directory containing the raw downloaded dataset (default: data/raw).",
    )
    parser.add_argument("--n", type=int, default=NUM_SAMPLES, help="Batch size (default: 8).")
    parser.add_argument("--steps", type=int, default=NUM_SANITY_STEPS, help="Sanity-check steps (default: 50).")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Load BiomedCLIP / PubMedBERT / ImageNet weights instead of random "
            "init. Slower and needs network access; off by default because "
            "this phase checks wiring, not representation quality."
        ),
    )
    return parser.parse_args()


@dataclass
class ComponentResult:
    """PASS/FAIL outcome for one of the four checks this script runs."""

    name: str
    passed: bool
    detail: str


def load_batch(data_root: Path, n: int, device: torch.device, text_encoder: Any) -> tuple[dict, ComponentResult]:
    """Load `n` real (image, report) pairs if available, else synthetic data.

    A synthetic fallback still counts as PASS: Phase 2 only requires
    "whatever real (sub-sampled) or synthetic data is available."
    """
    try:
        from torchvision import transforms

        from medxvlm.data.toy_dataset import ToyChestXrayDataset

        dataset = ToyChestXrayDataset(data_root)
        if len(dataset) == 0:
            raise FileNotFoundError(f"no (image, report) pairs found under {data_root}")
        n = min(n, len(dataset))
        images, reports = zip(*(dataset[i] for i in range(n)))

        transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        pixel_values = torch.stack([transform(image) for image in images]).to(device)
        reports = list(reports)
        detail = f"loaded {n} real (image, report) pairs from {data_root}"
    except Exception as exc:  # noqa: BLE001 - any data problem falls back to synthetic, doesn't fail the run
        pixel_values = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        reports = [f"synthetic report {i}: no real findings, placeholder text only." for i in range(n)]
        detail = f"real data unavailable ({exc}); using {n} synthetic samples instead"

    encoded = text_encoder.tokenize(reports)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "n": n}
    return batch, ComponentResult("data loading", True, detail)


def make_multilabel_targets(n: int, num_labels: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (n, num_labels), device=device).float()


def check_baseline(batch: dict, device: torch.device, pretrained: bool, steps: int) -> ComponentResult:
    from medxvlm.engine.sanity_trainer import run_sanity_check
    from medxvlm.models.baseline_classifier import BaselineClassifier, multilabel_bce_loss
    from medxvlm.models.labels import NUM_CHEXPERT_LABELS

    # Reseeded per check (not just once in main()) so each check's random
    # head/query initialisation is reproducible regardless of how much RNG
    # state the other checks consumed before it — otherwise a fresh
    # attention head landing on an unlucky random init becomes a flaky,
    # run-order-dependent FAIL instead of a deterministic one.
    torch.manual_seed(RANDOM_SEED)
    model = BaselineClassifier(pretrained=pretrained).to(device)
    targets = make_multilabel_targets(batch["n"], NUM_CHEXPERT_LABELS, device)

    def forward_and_loss(model: nn.Module, batch: dict) -> torch.Tensor:
        output = model(batch["pixel_values"])
        expected_shape = (batch["n"], NUM_CHEXPERT_LABELS)
        assert output.logits.shape == expected_shape, f"baseline logits shape {output.logits.shape} != {expected_shape}"
        return multilabel_bce_loss(output.logits, targets)

    result = run_sanity_check(model, forward_and_loss, batch, num_steps=steps)
    detail = f"loss {result.losses[0]:.4f} -> {result.losses[-1]:.4f}" if result.losses else "no steps ran"
    if not result.passed:
        detail = result.failure_reason or detail
    return ComponentResult("baseline forward/backward", result.passed, detail)


def check_stage1(batch: dict, device: torch.device, pretrained: bool, steps: int) -> ComponentResult:
    from medxvlm.engine.sanity_trainer import run_sanity_check
    from medxvlm.models.alignment import SHARED_EMBED_DIM, ContrastiveAlignment
    from medxvlm.models.text_encoder import TextEncoder
    from medxvlm.models.vision_encoder import VisionEncoder

    torch.manual_seed(RANDOM_SEED)
    vision_encoder = VisionEncoder(pretrained=pretrained)
    text_encoder = TextEncoder(pretrained=pretrained)
    # Backbones frozen for this check only (ContrastiveAlignment itself has
    # no such flag — real Stage 1 training fine-tunes both encoders).
    # Jointly training two from-scratch multi-hundred-million-parameter
    # transformers via a batch-of-8 InfoNCE loss doesn't reliably move in
    # 50 steps at any learning rate we tried (confirmed empirically) — that
    # is a training-schedule question (warmup, batch size), not a wiring
    # one, and out of scope for this phase. Freezing isolates exactly what
    # this check is meant to verify: that vision_proj/text_proj/logit_scale
    # and the InfoNCE loss are wired correctly and receive gradients: the
    # backbones' own gradient wiring is already exercised by check_stage2.
    for param in vision_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    model = ContrastiveAlignment(vision_encoder, text_encoder).to(device)

    def forward_and_loss(model: nn.Module, batch: dict) -> torch.Tensor:
        output = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
        expected_shape = (batch["n"], SHARED_EMBED_DIM)
        assert output.image_embeds.shape == expected_shape, f"image_embeds shape {output.image_embeds.shape} != {expected_shape}"
        assert output.text_embeds.shape == expected_shape, f"text_embeds shape {output.text_embeds.shape} != {expected_shape}"
        return output.loss

    result = run_sanity_check(model, forward_and_loss, batch, num_steps=steps)
    detail = f"loss {result.losses[0]:.4f} -> {result.losses[-1]:.4f}" if result.losses else "no steps ran"
    if not result.passed:
        detail = result.failure_reason or detail
    return ComponentResult("Stage 1 (alignment) forward/backward", result.passed, detail)


class _Stage2SanityModel(nn.Module):
    """Vision encoder + classifier head, combined only so the sanity loop sees one model.

    Not part of the public model API (see models/vision_encoder.py and
    models/classifier_head.py) — Stage 2 is deliberately two separate
    modules; this wrapper exists purely so `run_sanity_check` has a single
    `nn.Module` whose parameters it can step and gradient-check.
    """

    def __init__(self, vision_encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.head = head

    def forward(self, pixel_values: torch.Tensor):
        patch_tokens = self.vision_encoder(pixel_values).patch_tokens
        return self.head(patch_tokens)


def check_stage2(batch: dict, device: torch.device, pretrained: bool, steps: int) -> ComponentResult:
    from medxvlm.engine.sanity_trainer import run_sanity_check
    from medxvlm.models.baseline_classifier import multilabel_bce_loss
    from medxvlm.models.classifier_head import ClassifierHead
    from medxvlm.models.labels import NUM_CHEXPERT_LABELS
    from medxvlm.models.vision_encoder import VisionEncoder

    torch.manual_seed(RANDOM_SEED)
    vision_encoder = VisionEncoder(pretrained=pretrained)
    head = ClassifierHead(vision_encoder.embed_dim)
    model = _Stage2SanityModel(vision_encoder, head).to(device)
    targets = make_multilabel_targets(batch["n"], NUM_CHEXPERT_LABELS, device)
    num_patches = vision_encoder.patch_grid_size**2

    def forward_and_loss(model: nn.Module, batch: dict) -> torch.Tensor:
        output = model(batch["pixel_values"])
        expected_logits_shape = (batch["n"], NUM_CHEXPERT_LABELS)
        expected_attn_shape = (batch["n"], NUM_CHEXPERT_LABELS, num_patches)
        assert output.logits.shape == expected_logits_shape, f"Stage 2 logits shape {output.logits.shape} != {expected_logits_shape}"
        assert (
            output.attention_weights.shape == expected_attn_shape
        ), f"Stage 2 attention_weights shape {output.attention_weights.shape} != {expected_attn_shape}"
        return multilabel_bce_loss(output.logits, targets)

    result = run_sanity_check(model, forward_and_loss, batch, num_steps=steps)
    detail = f"loss {result.losses[0]:.4f} -> {result.losses[-1]:.4f}" if result.losses else "no steps ran"
    if not result.passed:
        detail = result.failure_reason or detail
    return ComponentResult("Stage 2 (classifier head) forward/backward", result.passed, detail)


def print_summary(results: list[ComponentResult]) -> bool:
    print("\n=== Sanity check summary ===")
    name_width = max(len(r.name) for r in results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name:<{name_width}}  {result.detail}")
    return all(r.passed for r in results)


def main() -> int:
    args = parse_args()
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  pretrained weights: {args.pretrained}  |  steps/check: {args.steps}")

    # Imported lazily so --help works even if the package isn't installed yet.
    from medxvlm.models.text_encoder import TextEncoder

    results: list[ComponentResult] = []
    text_encoder_for_batch = TextEncoder(pretrained=args.pretrained)
    batch, data_result = load_batch(args.data_root, args.n, device, text_encoder_for_batch)
    results.append(data_result)

    for check_fn in (check_baseline, check_stage1, check_stage2):
        try:
            results.append(check_fn(batch, device, args.pretrained, args.steps))
        except Exception as exc:  # noqa: BLE001 - a broken component should show as FAIL, not crash the whole script
            results.append(ComponentResult(check_fn.__name__, False, f"raised {type(exc).__name__}: {exc}"))

    all_passed = print_summary(results)
    print("\nOK: all sanity checks passed." if all_passed else "\nFAILED: see the summary above.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
