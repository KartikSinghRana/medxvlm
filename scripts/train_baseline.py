#!/usr/bin/env python3
"""Real training run: the baseline classifier (Phase 2's "before" case).

Config-driven (configs/baseline.yaml) -- epochs, batch size, and lr are
never hardcoded here. Checkpoints best (by val macro AUROC) + last, logs
train/val loss and per-class + macro AUROC/AUPRC/F1 to TensorBoard + CSV
every epoch via `EpochLogger`.

Usage:
    python scripts/train_baseline.py [--config configs/baseline.yaml]
        [--epochs N] [--batch-size N] [--loss bce|asl]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import ClassificationDataset  # noqa: E402
from medxvlm.engine.loop_utils import EpochLogger, load_config, save_checkpoint, set_seed  # noqa: E402
from medxvlm.eval.metrics import classification_metrics  # noqa: E402
from medxvlm.models.baseline_classifier import BaselineClassifier, asymmetric_loss, multilabel_bce_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--loss", choices=["bce", "asl"], default=None)
    parser.add_argument("--backbone", type=str, default=None, help="timm model name, or 'xraydinov2'")
    parser.add_argument("--run-name", type=str, default=None, help="Subdirectory under output_dir/log_dir.")
    return parser.parse_args()


def run_epoch(model, loader, device, loss_fn, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    all_logits, all_targets = [], []

    for pixel_values, targets, _reports in loader:
        pixel_values, targets = pixel_values.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            output = model(pixel_values)
            loss = loss_fn(output.logits, targets)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(output.logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_targets))
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main() -> int:
    args = parse_args()
    config = load_config(args.config, {"epochs": args.epochs, "batch_size": args.batch_size, "loss": args.loss, "backbone": args.backbone})
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    projections = tuple(config["projections"]) if config.get("projections") else None
    manifest = REPO_ROOT / config["manifest"]
    train_loader = DataLoader(
        ClassificationDataset(manifest, "train", projections=projections),
        batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        ClassificationDataset(manifest, "val", projections=projections),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
    )

    backbone = config.get("backbone", "vit_base_patch16_224")
    model = BaselineClassifier(backbone_name=backbone, pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = asymmetric_loss if config["loss"] == "asl" else multilabel_bce_loss
    run_name = args.run_name or backbone
    print(f"Device: {device} | backbone: {backbone} | projections: {projections} | loss: {config['loss']} | epochs: {config['epochs']} | batch_size: {config['batch_size']} | train imgs: {len(train_loader.dataset)}")

    logger = EpochLogger(REPO_ROOT / config["log_dir"] / run_name)
    output_dir = REPO_ROOT / config["output_dir"] / run_name
    best_auroc = -1.0
    start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_metrics = run_epoch(model, val_loader, device, loss_fn)

        payload = {f"train/{k}": v for k, v in train_metrics.items()}
        payload.update({f"val/{k}": v for k, v in val_metrics.items()})
        logger.log(epoch, payload)
        print(
            f"[epoch {epoch}/{config['epochs']}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_auroc={val_metrics['macro_auroc']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, {"epoch": epoch, "config": config})
        if val_metrics["macro_auroc"] > best_auroc:
            best_auroc = val_metrics["macro_auroc"]
            save_checkpoint(
                output_dir / "best.pt", model, {"epoch": epoch, "config": config, "val_macro_auroc": best_auroc}
            )

    logger.close()
    print(f"Done in {(time.time() - start) / 60:.1f} min. Best val macro AUROC: {best_auroc:.4f}. Checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
