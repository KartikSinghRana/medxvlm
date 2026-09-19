"""Minimal sanity-check training loop: overfit one repeated batch.

No checkpointing, no logging framework, no validation split — this only
answers "is this model wired correctly," via the classic overfit-one-batch
check, not "does this model train well."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

DEFAULT_NUM_STEPS = 50
DEFAULT_LR = 1e-3
GRAD_CLIP_MAX_NORM = 1.0

ForwardAndLossFn = Callable[[nn.Module, Any], torch.Tensor]


@dataclass
class SanityCheckResult:
    """Outcome of one `run_sanity_check` call.

    Attributes:
        losses: loss value at every step, in order.
        passed: whether all wiring checks (finite loss, every trainable
            parameter received a gradient, loss decreased) held.
        failure_reason: human-readable cause of the first failure, if any.
    """

    losses: list[float]
    passed: bool
    failure_reason: str | None = None


def run_sanity_check(
    model: nn.Module,
    forward_and_loss_fn: ForwardAndLossFn,
    batch: Any,
    num_steps: int = DEFAULT_NUM_STEPS,
    lr: float = DEFAULT_LR,
) -> SanityCheckResult:
    """Overfit `model` on one repeated `batch` for `num_steps` and check wiring.

    # why: separated from any specific model class so the same loop can
    # drive the baseline classifier, Stage 1 alignment, or Stage 2 head
    # interchangeably — each just supplies its own `forward_and_loss_fn` —
    # rather than duplicating the step/assert logic three times.

    Verifies three things, in order, and stops at the first failure:
    1. the loss is finite at every step (no NaN/Inf),
    2. every trainable parameter receives a gradient (nothing silently
       frozen or disconnected from the graph),
    3. the loss at the final step is lower than at the first step (the
       classic overfit-one-batch check).

    Gradients are clipped to `GRAD_CLIP_MAX_NORM` before each optimizer
    step — this is a real training-loop concern, not just a check, since
    without it a moderately high sanity-check learning rate can blow up a
    randomly-initialised transformer backbone into producing the very
    NaNs this function is checking for.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = forward_and_loss_fn(model, batch)

        if not torch.isfinite(loss):
            return SanityCheckResult(losses, False, f"non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                return SanityCheckResult(
                    losses, False, f"parameter '{name}' has requires_grad=True but received no gradient"
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        optimizer.step()
        losses.append(loss.item())

    if losses[-1] >= losses[0]:
        return SanityCheckResult(
            losses, False, f"loss did not decrease over {num_steps} steps: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    return SanityCheckResult(losses, True)
