#!/usr/bin/env python3
"""Phase 5: generate reports with MedXVLM's Stage 3 model on a fixed subsample
of its own held-out test split, for later scoring against CheXOne on the
identical images (see docs/cross_dataset_evaluation.md).

Writes:
- docs/phase5_artifacts/report_gen_subsample.csv (filename, reference) --
  the shared image list CheXOne's run also uses (external_repro/run_chexone_on_medxvlm_testsplit.py).
- docs/phase5_artifacts/medxvlm_stage3_generations.csv (filename, reference, generated).

Usage:
    python scripts/eval_phase5_report_generation.py [--n 150] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medxvlm.data.dataset import DEFAULT_REPORT_INSTRUCTION, ReportGenDataset  # noqa: E402
from medxvlm.engine.loop_utils import load_checkpoint, load_stage1_encoders  # noqa: E402
from medxvlm.models.report_generator import ReportGenerator  # noqa: E402

STAGE1_CHECKPOINT = REPO_ROOT / "checkpoints" / "align" / "local_on" / "best.pt"
STAGE3_CHECKPOINT = REPO_ROOT / "checkpoints" / "report_gen" / "best.pt"
MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts"


def build_report_generator(device):
    vision_encoder, _text_encoder = load_stage1_encoders(STAGE1_CHECKPOINT, device)
    ckpt = load_checkpoint(STAGE3_CHECKPOINT)
    config = ckpt["config"]
    model = ReportGenerator(
        vision_encoder,
        decoder_name=config["decoder_name"],
        pooled_grid_size=config["pooled_grid_size"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    # Only LoRA + projector were saved; everything else in `missing` must be
    # frozen base-decoder/vision-encoder params already loaded correctly.
    assert all(("lora_" not in m and "projector" not in m) for m in missing), missing
    assert not unexpected, unexpected
    model.eval()
    print(f"[info] loaded report_gen checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ReportGenDataset(MANIFEST, "test")
    n = min(args.n, len(dataset))
    sample_positions = dataset.rows.sample(n=n, random_state=args.seed).index.tolist()
    print(f"[info] {n} of {len(dataset)} test images sampled (seed={args.seed})")

    dataset.rows.loc[sample_positions, ["filename", "report_text"]].rename(columns={"report_text": "reference"}).to_csv(
        OUT_DIR / "report_gen_subsample.csv", index=False
    )

    model = build_report_generator(device)

    generations = []
    t0 = time.time()
    for i, pos in enumerate(sample_positions):
        filename = dataset.rows.loc[pos, "filename"]
        image, _instruction, reference = dataset[pos]
        pixel_values = image.unsqueeze(0).to(device)
        generated = model.generate(pixel_values, DEFAULT_REPORT_INSTRUCTION, max_new_tokens=args.max_new_tokens)[0]
        generations.append({"filename": filename, "reference": reference, "generated": generated})
        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"[progress] {i+1}/{n}, {elapsed:.0f}s elapsed", flush=True)

    out_df = pd.DataFrame(generations)
    out_df.to_csv(OUT_DIR / "medxvlm_stage3_generations.csv", index=False)
    print(f"[done] wrote {OUT_DIR / 'medxvlm_stage3_generations.csv'} ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
