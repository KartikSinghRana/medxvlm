#!/usr/bin/env python3
"""Real training run: Stage 1 contrastive alignment.

Config-driven (configs/align.yaml). `--use-local-loss`/`--no-use-local-loss`
is the ablation study's on/off switch (see docs/ablation_study.md) --
everything else about the run stays fixed between the two invocations.
Runs with the flag on and off write to separate `local_on`/`local_off`
subdirectories so neither run's checkpoints/logs clobber the other's.

Usage:
    python scripts/train_align.py --use-local-loss [--config configs/align.yaml]
        [--epochs N] [--batch-size N]
    python scripts/train_align.py --no-use-local-loss
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

from medxvlm.data.dataset import AlignmentDataset, SentenceAlignmentDataset  # noqa: E402
from medxvlm.engine.loop_utils import EpochLogger, load_config, save_checkpoint, set_seed  # noqa: E402
from medxvlm.eval.metrics import retrieval_recall_at_k  # noqa: E402
from medxvlm.models.alignment import ContrastiveAlignment  # noqa: E402
from medxvlm.models.text_encoder import TextEncoder  # noqa: E402
from medxvlm.models.vision_encoder import VisionEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "align.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-local-loss", dest="use_local_loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--select-by", choices=["recall1", "val_loss"], default=None,
                        help="Checkpoint selection metric. 'recall1' (Phase 3 default) is noisy on a ~380-pair val set; 'val_loss' is not.")
    parser.add_argument("--run-name", type=str, default=None, help="Subdirectory under output_dir/log_dir (default: local_on/local_off).")
    parser.add_argument("--freeze-text-encoder", dest="freeze_text_encoder", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().replace(",", " ").split())


def make_collate(text_encoder: TextEncoder, max_length: int, multi_positive: bool = False):
    """Collate (image, text) pairs; with `multi_positive`, also a `[B, B]` mask of identical texts.

    # why the mask: at sentence level, the same sentence ("The lungs are clear")
    # recurs many times per batch; those pairs are positives, not negatives.
    """

    def collate(batch):
        images, texts = zip(*batch)
        pixel_values = torch.stack(images)
        encoded = text_encoder.tokenize(list(texts), max_length=max_length)
        positive_mask = None
        if multi_positive:
            keys = [_normalise_text(t) for t in texts]
            positive_mask = torch.tensor([[a == b for b in keys] for a in keys], dtype=torch.bool)
        return pixel_values, encoded["input_ids"], encoded["attention_mask"], positive_mask

    return collate


def run_epoch(model: ContrastiveAlignment, loader, device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_global = total_local = 0.0
    n_batches = n_local_batches = 0
    all_image_embeds, all_text_embeds = [], []

    for pixel_values, input_ids, attention_mask, positive_mask in loader:
        pixel_values = pixel_values.to(device)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.set_grad_enabled(is_train):
            output = model(pixel_values, input_ids, attention_mask, positive_mask=positive_mask)
        if is_train:
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

        total_loss += output.loss.item()
        total_global += output.global_loss.item()
        if output.local_loss is not None:
            total_local += output.local_loss.item()
            n_local_batches += 1
        n_batches += 1
        all_image_embeds.append(output.image_embeds.detach().cpu())
        all_text_embeds.append(output.text_embeds.detach().cpu())

    metrics = {"loss": total_loss / n_batches, "global_loss": total_global / n_batches}
    if n_local_batches:
        metrics["local_loss"] = total_local / n_local_batches
    if not is_train:
        metrics.update(retrieval_recall_at_k(torch.cat(all_image_embeds), torch.cat(all_text_embeds)))
    return metrics


def main() -> int:
    args = parse_args()
    config = load_config(args.config, {"epochs": args.epochs, "batch_size": args.batch_size, "use_local_loss": args.use_local_loss, "select_by": args.select_by, "freeze_text_encoder": args.freeze_text_encoder})
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    select_by = config.get("select_by", "recall1")

    variant = args.run_name or ("local_on" if config["use_local_loss"] else "local_off")
    output_dir = REPO_ROOT / config["output_dir"] / variant
    log_dir = REPO_ROOT / config["log_dir"] / variant

    vision_backbone = config.get("vision_backbone", "biomedclip")
    text_encoder = TextEncoder(pretrained=config["pretrained"]).to(device)
    vision_encoder = VisionEncoder(pretrained=config["pretrained"], backbone=vision_backbone).to(device)
    model = ContrastiveAlignment(vision_encoder, text_encoder, use_local_loss=config["use_local_loss"]).to(device)

    # why optionally freeze the image tower: with only ~3k frontal training
    # pairs, full fine-tuning overfit in Phase 3 (train loss fell, val loss
    # flat -- docs/training_summary.md). Freezing keeps the CXR-pretrained
    # features intact and lets Stage 1 learn the *projection* into the
    # shared space, which is all Stage 2 / the zero-shot head consume.
    if config.get("freeze_vision_encoder", False):
        for param in vision_encoder.parameters():
            param.requires_grad = False
    # why optionally freeze the text tower too: fine-tuning BiomedBERT on ~3k
    # long reports drifts it away from short class prompts ("There is X."),
    # which is what the zero-shot head queries with -- v2 zero-shot got
    # *worse* as Stage 1 trained. Frozen towers => only the two projections
    # + temperature learn (~1.2M params), so the pretrained text space is kept.
    if config.get("freeze_text_encoder", False):
        for param in text_encoder.parameters():
            param.requires_grad = False

    projections = tuple(config["projections"]) if config.get("projections") else None
    text_unit = config.get("text_unit", "report")
    if text_unit not in ("report", "sentence"):
        raise ValueError(f"text_unit must be 'report' or 'sentence', got {text_unit!r}")
    dataset_cls = SentenceAlignmentDataset if text_unit == "sentence" else AlignmentDataset
    collate = make_collate(text_encoder, config["max_text_length"], multi_positive=(text_unit == "sentence"))
    manifest = REPO_ROOT / config["manifest"]
    train_loader = DataLoader(
        dataset_cls(manifest, "train", projections=projections),
        batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],
        drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        dataset_cls(manifest, "val", projections=projections),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=collate,
    )

    # Projection layers + temperature are freshly initialised, so they get a
    # higher LR than the pretrained encoders (config `proj_lr`, defaults to `lr`).
    proj_params = list(model.vision_proj.parameters()) + list(model.text_proj.parameters()) + [model.logit_scale]
    proj_ids = {id(p) for p in proj_params}
    encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in proj_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config["lr"]},
            {"params": proj_params, "lr": config.get("proj_lr", config["lr"])},
        ],
        weight_decay=config["weight_decay"],
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Device: {device} | backbone: {vision_backbone} | projections: {projections} | text_unit: {text_unit} | "
        f"freeze_vision: {config.get('freeze_vision_encoder', False)} | freeze_text: {config.get('freeze_text_encoder', False)} | use_local_loss: {config['use_local_loss']} | "
        f"epochs: {config['epochs']} | batch_size: {config['batch_size']} | train pairs: {len(train_loader.dataset)} | "
        f"trainable params: {n_trainable/1e6:.1f}M"
    )

    logger = EpochLogger(log_dir)
    best_recall1 = -1.0
    best_val_loss = float("inf")
    patience = config.get("early_stopping_patience", 0)
    epochs_since_best = 0
    start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)

        payload = {f"train/{k}": v for k, v in train_metrics.items()}
        payload.update({f"val/{k}": v for k, v in val_metrics.items()})
        logger.log(epoch, payload)
        mean_recall1 = (val_metrics["recall@1_i2t"] + val_metrics["recall@1_t2i"]) / 2
        print(
            f"[epoch {epoch}/{config['epochs']}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_recall@1(mean)={mean_recall1:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, {"epoch": epoch, "config": config})
        if select_by == "val_loss":
            improved = val_metrics["loss"] < best_val_loss
        else:
            improved = mean_recall1 > best_recall1
        if improved:
            best_recall1 = max(best_recall1, mean_recall1)
            best_val_loss = min(best_val_loss, val_metrics["loss"])
            epochs_since_best = 0
            save_checkpoint(output_dir / "best.pt", model, {"epoch": epoch, "config": config, "val_recall@1_mean": mean_recall1, "val_loss": val_metrics["loss"], "select_by": select_by})
        else:
            epochs_since_best += 1
            if patience and epochs_since_best >= patience:
                print(f"Early stopping: no {select_by} improvement for {patience} epochs.")
                break

    logger.close()
    print(f"Done in {(time.time() - start) / 60:.1f} min. select_by={select_by} | best val recall@1 (mean): {best_recall1:.4f} | best val loss: {best_val_loss:.4f}. Checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
