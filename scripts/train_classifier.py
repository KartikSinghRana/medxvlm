#!/usr/bin/env python3
"""Real training run: Stage 2 classifier head with gated text fusion.

Config-driven (configs/classifier.yaml). Loads its vision + text encoders
from a trained Stage 1 checkpoint (`--stage1-checkpoint`); running this
script once against each of `train_align.py`'s `local_on`/`local_off`
checkpoints is the ablation study's downstream comparison (see
docs/ablation_study.md). Also saves per-label attention-weight grounding
snapshots (this project's Grad-CAM/"segmentation map" equivalent) every
few epochs.

Usage:
    python scripts/train_classifier.py --stage1-checkpoint checkpoints/align/local_on/best.pt
        [--config configs/classifier.yaml] [--epochs N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import ClassificationDataset  # noqa: E402
from medxvlm.engine.loop_utils import EpochLogger, load_config, load_stage1_encoders, save_checkpoint, set_seed  # noqa: E402
from medxvlm.eval.metrics import classification_metrics  # noqa: E402
from medxvlm.explain.attention_maps import save_grounding_snapshots  # noqa: E402
from medxvlm.models.baseline_classifier import asymmetric_loss, multilabel_bce_loss  # noqa: E402
from medxvlm.models.classifier_head import ClassifierHead  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402
from medxvlm.models.text_encoder import TextEncoder  # noqa: E402
from medxvlm.models.vision_encoder import VisionEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "classifier.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Subdirectory under output_dir/log_dir (defaults to the stage1 checkpoint's parent dir name).")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed (used for the seed ensemble).")
    parser.add_argument("--vision-backbone", type=str, default=None,
                        help="Build the vision tower from pretrained weights (biomedclip | clip_openai | imagenet_vit | xraydinov2) "
                             "instead of a Stage 1 checkpoint -- the RQ1 encoder comparison. Image-only configs only.")
    return parser.parse_args()


class Stage2Model(nn.Module):
    """Vision encoder + (optional) text encoder + ClassifierHead, combined for one training loop.

    `use_text_fusion=False` is the image-only configuration: the paired
    report is never encoded, so the classifier can only learn from pixels.
    # why this exists: Phase 5 (docs/cross_dataset_evaluation.md) showed the
    # text-fusion classifier had learned to read the answer out of the
    # report text it was given, and scored at chance when tested image-only.
    """

    def __init__(self, vision_encoder, text_encoder: TextEncoder | None, head: ClassifierHead, use_text_fusion: bool) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.head = head
        self.use_text_fusion = use_text_fusion

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        patch_tokens = self.vision_encoder(pixel_values).patch_tokens
        text_embeds = None
        if self.use_text_fusion:
            text_embeds = self.text_encoder(input_ids, attention_mask).pooled
        return self.head(patch_tokens, text_embeds)


def make_collate(text_encoder: TextEncoder | None, max_length: int):
    def collate(batch):
        images, targets, reports = zip(*batch)
        pixel_values = torch.stack(images)
        targets = torch.stack(targets)
        if text_encoder is None:  # image-only run without a text tower: placeholders, never read
            dummy = torch.zeros(len(images), 1, dtype=torch.long)
            return pixel_values, targets, dummy, dummy
        encoded = text_encoder.tokenize(list(reports), max_length=max_length)
        return pixel_values, targets, encoded["input_ids"], encoded["attention_mask"]

    return collate


def run_epoch(model: Stage2Model, loader, device, loss_fn, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    all_logits, all_targets = [], []

    for pixel_values, targets, input_ids, attention_mask in loader:
        pixel_values, targets = pixel_values.to(device), targets.to(device)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.set_grad_enabled(is_train):
            output = model(pixel_values, input_ids, attention_mask)
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


def save_epoch_snapshot(model: Stage2Model, loader, device, snapshot_dir: Path, epoch: int, num_samples: int) -> None:
    model.eval()
    pixel_values, _targets, input_ids, attention_mask = next(iter(loader))
    pixel_values = pixel_values[:num_samples].to(device)
    input_ids, attention_mask = input_ids[:num_samples].to(device), attention_mask[:num_samples].to(device)
    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask)
    probs = torch.sigmoid(output.logits)
    label_indices_per_sample = []
    for row in probs:
        top = torch.topk(row, k=min(3, row.shape[0])).indices.tolist()
        label_indices_per_sample.append(top)
    save_grounding_snapshots(pixel_values, output.attention_weights, label_indices_per_sample, CHEXPERT_LABELS, snapshot_dir, epoch, max_samples=num_samples)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                                       "vision_backbone": args.vision_backbone})
    vision_backbone = config.get("vision_backbone")
    stage1_checkpoint = None if vision_backbone else (args.stage1_checkpoint or (REPO_ROOT / config["stage1_checkpoint"]))
    run_name = args.run_name or (vision_backbone or Path(stage1_checkpoint).parent.name)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_text_fusion = config.get("use_text_fusion", True)
    if vision_backbone:
        # RQ1: same head/recipe, tower straight from its public pretrained weights (no Stage 1, no text tower).
        assert not use_text_fusion, "--vision-backbone is for image-only configs"
        vision_encoder, text_encoder = VisionEncoder(pretrained=True, backbone=vision_backbone).to(device), None
    else:
        vision_encoder, text_encoder = load_stage1_encoders(stage1_checkpoint, device)
        for param in text_encoder.parameters():
            param.requires_grad = False
    if config["freeze_vision_encoder"]:
        for param in vision_encoder.parameters():
            param.requires_grad = False

    head = ClassifierHead(
        embed_dim=vision_encoder.embed_dim,
        text_dim=text_encoder.embed_dim if use_text_fusion else None,
    ).to(device)
    model = Stage2Model(vision_encoder, text_encoder if use_text_fusion else None, head, use_text_fusion).to(device)

    projections = tuple(config["projections"]) if config.get("projections") else None
    collate = make_collate(text_encoder, config["max_text_length"])
    manifest = REPO_ROOT / config["manifest"]
    train_loader = DataLoader(
        ClassificationDataset(manifest, "train", projections=projections),
        batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],
        drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        ClassificationDataset(manifest, "val", projections=projections),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=collate,
    )

    # Split LR: the freshly initialised head learns fast; the pretrained
    # encoder gets a much smaller step so fine-tuning cannot wreck it on a
    # small dataset (config `backbone_lr`, defaults to `lr`).
    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.get("backbone_lr", config["lr"])},
            {"params": head_params, "lr": config["lr"]},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    loss_fn = asymmetric_loss if config.get("loss", "bce") == "asl" else multilabel_bce_loss
    patience = config.get("early_stopping_patience", 0)
    print(
        f"Device: {device} | {'vision_backbone: ' + vision_backbone if vision_backbone else 'stage1_checkpoint: ' + str(stage1_checkpoint)} | run: {run_name} | "
        f"text_fusion: {use_text_fusion} | projections: {projections} | loss: {config.get('loss', 'bce')} | "
        f"epochs: {config['epochs']} | train imgs: {len(train_loader.dataset)}"
    )

    output_dir = REPO_ROOT / config["output_dir"] / run_name
    log_dir = REPO_ROOT / config["log_dir"] / run_name
    snapshot_dir = REPO_ROOT / config["attention_maps_dir"] / run_name
    logger = EpochLogger(log_dir)
    best_auroc = -1.0
    start = time.time()

    epochs_since_best = 0
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_metrics = run_epoch(model, val_loader, device, loss_fn)
        scheduler.step()

        payload = {f"train/{k}": v for k, v in train_metrics.items()}
        payload.update({f"val/{k}": v for k, v in val_metrics.items()})
        logger.log(epoch, payload)
        print(
            f"[epoch {epoch}/{config['epochs']}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_auroc={val_metrics['macro_auroc']:.4f}"
        )

        if epoch % config["attention_snapshot_every"] == 0 or epoch == config["epochs"]:
            save_epoch_snapshot(model, val_loader, device, snapshot_dir, epoch, config["attention_snapshot_samples"])

        save_checkpoint(output_dir / "last.pt", model, {"epoch": epoch, "config": config})
        if val_metrics["macro_auroc"] > best_auroc:
            best_auroc = val_metrics["macro_auroc"]
            epochs_since_best = 0
            save_checkpoint(output_dir / "best.pt", model, {"epoch": epoch, "config": config, "val_macro_auroc": best_auroc})
        else:
            epochs_since_best += 1
            if patience and epochs_since_best >= patience:
                print(f"Early stopping: no val macro AUROC improvement for {patience} epochs.")
                break

    logger.close()
    print(f"Done in {(time.time() - start) / 60:.1f} min. Best val macro AUROC: {best_auroc:.4f}. Checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
