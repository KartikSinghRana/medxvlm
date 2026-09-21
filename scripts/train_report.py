#!/usr/bin/env python3
"""Real training run: Stage 3 report generation (LoRA on the projected-image decoder).

Config-driven (configs/report_gen.yaml). Loads its frozen vision encoder
from a trained Stage 1 checkpoint; only the projector MLP and LoRA
adapters are optimized. Saves a handful of generated report samples
alongside their reference report every `sample_every` epochs, and
checkpoints only the trainable parameters (projector + LoRA), not the
~1.6B frozen decoder weights.

v2c additions (configs/report_gen_v2c.yaml): `findings_conditioning` puts the
Stage 2 ensemble's flagged findings into each image's instruction
(`medxvlm.data.findings_prompt`), and `select_by: generation` picks the
checkpoint by clinical-efficacy F1 of greedy generations on a fixed val
subsample instead of teacher-forced val loss -- v2b showed val NLL prefers
the collapsed model (docs/v2_rebuild.md §3.6.4).

Usage:
    python scripts/train_report.py --stage1-checkpoint checkpoints/align/local_on/best.pt
        [--config configs/report_gen.yaml] [--epochs N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import ReportGenDataset, labels_to_training_target  # noqa: E402
from medxvlm.data.findings_prompt import FindingsLookup  # noqa: E402
from medxvlm.data.labeler import aggregate_report  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402
from medxvlm.engine.loop_utils import EpochLogger, load_config, load_vision_encoder, set_seed  # noqa: E402
from medxvlm.models.report_generator import ReportGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "report_gen.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override grad_accum_steps (e.g. --batch-size 2 --grad-accum-steps 8 keeps the effective batch at 16 with less VRAM).")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Smoke-test only: stop each train epoch after N batches.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Continue from a saved trainable checkpoint (last.pt/best.pt): loads projector+LoRA, starts at its epoch+1, "
                             "keeps its selection score as the incumbent best. Optimizer moments are not saved, so AdamW restarts cold.")
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
    generated = model.generate(pixel_values, list(instructions[:num_samples]), max_new_tokens=max_new_tokens)

    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"epoch_{epoch:03d}.txt"
    with open(out_path, "w") as f:
        for i, (gen, ref, instr) in enumerate(zip(generated, reports[:num_samples], instructions[:num_samples])):
            f.write(f"=== sample {i} ===\nINSTRUCTION: {instr}\nREFERENCE: {ref}\nGENERATED: {gen}\n\n")
    print(f"  wrote {num_samples} generation samples to {out_path}")


def _labels_to_binary(labels: dict[str, float]) -> np.ndarray:
    v = np.array([labels[n] for n in CHEXPERT_LABELS], dtype=float)
    v = np.where(np.isnan(v), 0.0, v)
    return np.where(v == -1.0, 1.0, v).astype(int)


def make_generation_val_subset(val_dataset: ReportGenDataset, n_images: int, seed: int) -> Subset:
    """Fixed, class-balanced val subsample for generation-based model selection (half abnormal, half normal)."""
    rows = val_dataset.rows.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    abnormal = np.where(rows["No Finding"] != 1.0)[0]
    normal = np.where(rows["No Finding"] == 1.0)[0]
    k = n_images // 2
    idx = np.concatenate([rng.choice(abnormal, min(k, len(abnormal)), replace=False), rng.choice(normal, min(n_images - k, len(normal)), replace=False)])
    return Subset(val_dataset, sorted(idx.tolist()))


@torch.no_grad()
def run_generation_eval(model: ReportGenerator, subset: Subset, device, batch_size: int, max_new_tokens: int) -> dict[str, float]:
    """Greedy-decode the subset and score it the way eval_v2_score_reports.py does: labeler CE macro F1 + distinct count.

    # why greedy and short: this runs every epoch on ~48 images; the ranking
    # between epochs is what matters, not the absolute number (which is
    # reported with beam search on the 150-image test subsample afterwards).
    """
    model.eval()
    rows = subset.dataset.rows.reset_index(drop=True).iloc[subset.indices]
    truth = np.stack([_labels_to_binary({l: r[l] for l in CHEXPERT_LABELS}) for _, r in rows.iterrows()])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    generated: list[str] = []
    for pixel_values, instructions, _reports in loader:
        generated.extend(t.strip() for t in model.generate(pixel_values.to(device), list(instructions), max_new_tokens=max_new_tokens))
    pred = np.stack([_labels_to_binary(aggregate_report(t)) for t in generated])
    supported = [j for j in range(len(CHEXPERT_LABELS)) if truth[:, j].sum() > 0]
    macro_f1 = float(np.mean([f1_score(truth[:, j], pred[:, j], zero_division=0) for j in supported]))
    n_distinct = len(set(generated))
    return {"gen/ce_macro_f1": macro_f1, "gen/n_distinct": float(n_distinct), "gen/n_images": float(len(generated)),
            "gen/modal_share": float(max(generated.count(g) for g in set(generated)) / len(generated))}


def main() -> int:
    args = parse_args()
    config = load_config(args.config, {"epochs": args.epochs, "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps})
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
    lookup_train = lookup_eval = None
    if config.get("findings_conditioning", False):
        lookup_train = FindingsLookup(REPO_ROOT / config["findings_predictions"], REPO_ROOT / config["findings_thresholds"],
                                      dropout=config.get("findings_dropout", 0.0), seed=config["seed"])
        lookup_eval = FindingsLookup(REPO_ROOT / config["findings_predictions"], REPO_ROOT / config["findings_thresholds"], dropout=0.0)
    train_dataset = ReportGenDataset(
        manifest, "train", projections=projections,
        instruction_fn=(lambda f: lookup_train.instruction(f, training=True)) if lookup_train else None,
    )
    sampler = make_balanced_sampler(train_dataset, config.get("abnormal_fraction", 0.5)) if config.get("oversample_abnormal", False) else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"], shuffle=sampler is None, sampler=sampler, num_workers=config["num_workers"],
        drop_last=True, collate_fn=collate,
    )
    val_dataset = ReportGenDataset(manifest, "val", projections=projections, instruction_fn=lookup_eval.instruction if lookup_eval else None)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=collate)
    select_by = config.get("select_by", "val_loss")
    assert select_by in ("val_loss", "generation"), select_by
    gen_subset = make_generation_val_subset(val_dataset, config.get("gen_val_images", 48), config["seed"]) if select_by == "generation" else None

    start_epoch, resumed = 1, None
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(resumed["trainable_state_dict"], strict=False)
        assert all(("lora_" not in m and "projector" not in m) for m in missing), missing
        assert not unexpected, unexpected
        start_epoch = int(resumed["epoch"]) + 1
        print(f"Resumed trainable weights from {args.resume} (epoch {resumed['epoch']}); continuing at epoch {start_epoch}")

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
        f"trainable params: {n_trainable:,} | frozen params: {n_frozen:,} | "
        f"findings_conditioning: {lookup_train is not None} (dropout {config.get('findings_dropout', 0.0)}) | select_by: {select_by}"
        + (f" on {len(gen_subset)} val images" if gen_subset is not None else "")
    )
    if lookup_train is not None:
        print(f"  example train instruction: {train_dataset[0][1]!r}")

    output_dir = REPO_ROOT / config["output_dir"]
    logger = EpochLogger(REPO_ROOT / config["log_dir"])
    samples_dir = REPO_ROOT / config["samples_dir"]
    best_val_loss = float("inf")
    best_score = -float("inf")  # higher is better: -val_loss, or generation CE F1 (tie-break: more distinct outputs)
    best_epoch = 0
    patience = config.get("early_stopping_patience", 0)
    epochs_since_best = 0
    if resumed is not None:
        # The resumed checkpoint is the incumbent: later epochs must beat it to replace best.pt.
        r_metrics = resumed.get("metrics", {})
        best_val_loss = float(resumed.get("val_loss", float("inf")))
        best_score = (r_metrics["gen/ce_macro_f1"] + 1e-4 * r_metrics["gen/n_distinct"]) if (select_by == "generation" and "gen/ce_macro_f1" in r_metrics) else -best_val_loss
        best_epoch = int(resumed["epoch"])
        print(f"  incumbent best: epoch {best_epoch}, score {best_score:.4f} ({select_by})")
    start = time.time()

    for epoch in range(start_epoch, config["epochs"] + 1):
        train_loss = run_train_epoch(model, train_loader, device, optimizer, config["grad_accum_steps"], args.limit_train_batches)
        val_loss = run_val_epoch(model, val_loader, device)
        metrics = {"train/loss": train_loss, "val/loss": val_loss}
        if gen_subset is not None:
            t_gen = time.time()
            metrics.update(run_generation_eval(model, gen_subset, device, batch_size=8, max_new_tokens=config.get("gen_max_new_tokens", 96)))
            metrics["gen/seconds"] = time.time() - t_gen
        logger.log(epoch, metrics)
        extra = (f" gen_ce_f1={metrics['gen/ce_macro_f1']:.4f} distinct={int(metrics['gen/n_distinct'])}/{int(metrics['gen/n_images'])} "
                 f"modal={metrics['gen/modal_share']:.2f} ({metrics['gen/seconds']:.0f}s)") if gen_subset is not None else ""
        print(f"[epoch {epoch}/{config['epochs']}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}{extra} | {(time.time() - start) / 60:.1f} min", flush=True)

        if epoch % config["sample_every"] == 0 or epoch == config["epochs"]:
            save_samples(model, val_loader, device, samples_dir, epoch, config["num_samples"], config["max_new_tokens"])

        best_val_loss = min(best_val_loss, val_loss)
        score = (metrics["gen/ce_macro_f1"] + 1e-4 * metrics["gen/n_distinct"]) if select_by == "generation" else -val_loss
        save_trainable_checkpoint(output_dir / "last.pt", model, {"epoch": epoch, "config": config, "val_loss": val_loss, "metrics": metrics})
        if score > best_score:
            best_score, best_epoch, epochs_since_best = score, epoch, 0
            save_trainable_checkpoint(output_dir / "best.pt", model, {"epoch": epoch, "config": config, "val_loss": val_loss, "metrics": metrics, "select_by": select_by})
        else:
            epochs_since_best += 1
            if patience and epochs_since_best >= patience:
                print(f"Early stopping: no {select_by} improvement for {patience} epochs.")
                break

    logger.close()
    print(f"Done in {(time.time() - start) / 60:.1f} min. Best epoch {best_epoch} by {select_by} (score {best_score:.4f}); best val loss {best_val_loss:.4f}. Checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
