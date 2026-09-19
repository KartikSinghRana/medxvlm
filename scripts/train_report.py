#!/usr/bin/env python3
"""Real training run: Stage 3 report generation (LoRA on the projected-image decoder).

Config-driven (configs/report_gen.yaml). Loads its frozen vision encoder
from a trained Stage 1 checkpoint; only the projector MLP and LoRA
adapters are optimized. Saves a handful of generated report samples
alongside their reference report every `sample_every` epochs, and
checkpoints only the trainable parameters (projector + LoRA), not the
~1.6B frozen decoder weights.

Usage:
    python scripts/train_report.py --stage1-checkpoint checkpoints/align/local_on/best.pt
        [--config configs/report_gen.yaml] [--epochs N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import ReportGenDataset  # noqa: E402
from medxvlm.engine.loop_utils import EpochLogger, load_config, load_vision_encoder, set_seed  # noqa: E402
from medxvlm.models.report_generator import ReportGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "report_gen.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Smoke-test only: stop each train epoch after N batches.")
    return parser.parse_args()


def collate(batch):
    images, instructions, reports = zip(*batch)
    return torch.stack(images), list(instructions), list(reports)


def save_trainable_checkpoint(path: Path, model: ReportGenerator, extra: dict) -> None:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state_dict": state, **extra}, path)


def make_balanced_sampler(dataset: ReportGenDataset, abnormal_fraction: float = 0.5) -> WeightedRandomSampler:
    """Sample abnormal studies with total probability `abnormal_fraction`.

    # why: ~70% of Indiana reports are normal, and with LoRA + a frozen tower
    # the decoder's cheapest solution is to emit the modal normal report for
    # every image (Phase 5 measured 106/150 identical generations). Giving
    # abnormal studies a fixed share of the sampling mass forces the loss to
    # pay for finding-specific content without changing the dataset or labels.
    # 0.5 (v2) broke the collapse but taught a device-mention shortcut
    # (docs/v2_rebuild.md §3.4); the natural rate is ~0.30, so values between
    # trade collapse against hallucination.
    """
    normal = (dataset.rows["No Finding"] == 1.0).to_numpy()
    n_normal, n_abnormal = int(normal.sum()), int((~normal).sum())
    weights = torch.where(torch.from_numpy(normal), (1 - abnormal_fraction) / n_normal, abnormal_fraction / n_abnormal).double()
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def run_train_epoch(model: ReportGenerator, loader, device, optimizer, grad_accum_steps: int, limit_batches: int | None = None) -> float:
    model.train()
    total_loss, n_steps = 0.0, 0
    optimizer.zero_grad()
    for step, (pixel_values, instructions, reports) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        pixel_values = pixel_values.to(device)
        output = model(pixel_values, instructions, reports)
        loss = output.loss / grad_accum_steps
        loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += output.loss.item()
        n_steps += 1
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def run_val_epoch(model: ReportGenerator, loader, device) -> float:
    model.eval()
    total_loss, n_steps = 0.0, 0
    for pixel_values, instructions, reports in loader:
        pixel_values = pixel_values.to(device)
        output = model(pixel_values, instructions, reports)
        total_loss += output.loss.item()
        n_steps += 1
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def save_samples(model: ReportGenerator, loader, device, samples_dir: Path, epoch: int, num_samples: int, max_new_tokens: int) -> None:
    pixel_values, instructions, reports = next(iter(loader))
    pixel_values = pixel_values[:num_samples].to(device)
    generated = model.generate(pixel_values, instructions[0], max_new_tokens=max_new_tokens)

    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"epoch_{epoch:03d}.txt"
    with open(out_path, "w") as f:
        for i, (gen, ref) in enumerate(zip(generated, reports[:num_samples])):
            f.write(f"=== sample {i} ===\nREFERENCE: {ref}\nGENERATED: {gen}\n\n")
    print(f"  wrote {num_samples} generation samples to {out_path}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config, {"epochs": args.epochs, "batch_size": args.batch_size})
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # `vision_checkpoint` may be any checkpoint holding a VisionEncoder (Stage 1
    # or Stage 2); `stage1_checkpoint` is kept as the legacy key/flag name.
    vision_checkpoint = args.stage1_checkpoint or (REPO_ROOT / config.get("vision_checkpoint", config.get("stage1_checkpoint")))
    vision_encoder = load_vision_encoder(vision_checkpoint, device)

    model = ReportGenerator(
        vision_encoder,
        decoder_name=config["decoder_name"],
        pooled_grid_size=config["pooled_grid_size"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    ).to(device)

    manifest = REPO_ROOT / config["manifest"]
    projections = tuple(config["projections"]) if config.get("projections") else None
    train_dataset = ReportGenDataset(manifest, "train", projections=projections)
    sampler = make_balanced_sampler(train_dataset, config.get("abnormal_fraction", 0.5)) if config.get("oversample_abnormal", False) else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"], shuffle=sampler is None, sampler=sampler, num_workers=config["num_workers"],
        drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        ReportGenDataset(manifest, "val", projections=projections),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=collate,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    n_abnormal = int((train_dataset.rows["No Finding"] != 1.0).sum())
    print(
        f"Device: {device} | vision_checkpoint: {vision_checkpoint} | backbone: {vision_encoder.backbone_name} | "
        f"decoder: {config['decoder_name']} | visual tokens: {model.num_visual_tokens} | projections: {projections} | "
        f"train rows: {len(train_dataset)} ({n_abnormal} abnormal) | oversample_abnormal: {sampler is not None} (fraction {config.get('abnormal_fraction', 0.5) if sampler is not None else 'n/a'}) | "
        f"batch {config['batch_size']} x accum {config['grad_accum_steps']} | "
        f"trainable params: {n_trainable:,} | frozen params: {n_frozen:,}"
    )

    output_dir = REPO_ROOT / config["output_dir"]
    logger = EpochLogger(REPO_ROOT / config["log_dir"])
    samples_dir = REPO_ROOT / config["samples_dir"]
    best_val_loss = float("inf")
    patience = config.get("early_stopping_patience", 0)
    epochs_since_best = 0
    start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_loss = run_train_epoch(model, train_loader, device, optimizer, config["grad_accum_steps"], args.limit_train_batches)
        val_loss = run_val_epoch(model, val_loader, device)
        logger.log(epoch, {"train/loss": train_loss, "val/loss": val_loss})
        print(f"[epoch {epoch}/{config['epochs']}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if epoch % config["sample_every"] == 0 or epoch == config["epochs"]:
            save_samples(model, val_loader, device, samples_dir, epoch, config["num_samples"], config["max_new_tokens"])

        save_trainable_checkpoint(output_dir / "last.pt", model, {"epoch": epoch, "config": config})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
            save_trainable_checkpoint(output_dir / "best.pt", model, {"epoch": epoch, "config": config, "val_loss": best_val_loss})
        else:
            epochs_since_best += 1
            if patience and epochs_since_best >= patience:
                print(f"Early stopping: no val loss improvement for {patience} epochs.")
                break

    logger.close()
    print(f"Done in {(time.time() - start) / 60:.1f} min. Best val loss: {best_val_loss:.4f}. Checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
