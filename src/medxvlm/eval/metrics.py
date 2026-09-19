"""Evaluation metrics shared by the classification and retrieval training scripts.

Per-class + macro AUROC/AUPRC/F1 for the 14-label multi-label
classification task (Stage 2 and the baseline), and Recall@k for Stage 1's
image<->text retrieval.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from medxvlm.models.labels import CHEXPERT_LABELS


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Per-class + macro AUROC/AUPRC/F1 over one (typically val/test) split.

    A class with only one label value present (all-0 or all-1 in this
    split) has an undefined AUROC/AUPRC -- skipped from that class's score
    and from the macro average, rather than raising, since small-split
    single-class columns are expected with 14 imbalanced labels.
    """
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    targets_np = targets.numpy().astype(int)

    per_class: dict[str, dict[str, float]] = {}
    aurocs, auprcs, f1s = [], [], []
    for i, label in enumerate(CHEXPERT_LABELS):
        y_true, y_prob, y_pred = targets_np[:, i], probs[:, i], preds[:, i]
        entry: dict[str, float] = {}
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            entry["auroc"], entry["auprc"] = auroc, auprc
            aurocs.append(auroc)
            auprcs.append(auprc)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        entry["f1"] = f1
        f1s.append(f1)
        per_class[label] = entry

    metrics: dict[str, float] = {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "macro_f1": float(np.mean(f1s)),
    }
    for label, entry in per_class.items():
        for metric_name, value in entry.items():
            metrics[f"{label}/{metric_name}"] = value
    return metrics


def retrieval_recall_at_k(
    image_embeds: torch.Tensor, text_embeds: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Recall@k for both directions over one val/test split's full similarity matrix.

    `image_embeds`/`text_embeds` are assumed L2-normalised and row `i` of
    each is the matching pair, as produced by `ContrastiveAlignment`.
    """
    similarity = image_embeds @ text_embeds.t()  # [N, N]
    n = similarity.shape[0]
    targets = torch.arange(n)

    metrics: dict[str, float] = {}
    for direction, sim in (("i2t", similarity), ("t2i", similarity.t())):
        ranks = sim.argsort(dim=-1, descending=True)
        for k in ks:
            k_eff = min(k, n)
            hit = (ranks[:, :k_eff] == targets.unsqueeze(1)).any(dim=1)
            metrics[f"recall@{k}_{direction}"] = hit.float().mean().item()
    return metrics
