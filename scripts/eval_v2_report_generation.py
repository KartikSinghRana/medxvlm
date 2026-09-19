#!/usr/bin/env python3
"""Generate reports with the v2 Stage 3 model on the fixed Phase 5 test subsample.

Uses the identical 150 test images (docs/phase5_artifacts/report_gen_subsample.csv)
that the Phase 3 Stage 3 model and CheXOne already generated for, so all three
can be scored side by side by scripts/eval_v2_score_reports.py. The subsample is
80 lateral / 70 frontal; the v2 model trained frontal-only, so the scorer also
reports the 70-frontal subset separately.

Decoding is greedy with the same max_new_tokens as the Phase 3 run -- the
protocol is held fixed so the only variable is the model.

Usage:
    python scripts/eval_v2_report_generation.py [--checkpoint checkpoints/report_gen_v2/best.pt]
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

from medxvlm.data.dataset import DEFAULT_REPORT_INSTRUCTION, build_transforms  # noqa: E402
from medxvlm.engine.loop_utils import load_checkpoint, load_vision_encoder  # noqa: E402
from medxvlm.models.report_generator import ReportGenerator  # noqa: E402
from PIL import Image  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images" / "images_normalized"
SUBSAMPLE = REPO_ROOT / "docs" / "phase5_artifacts" / "report_gen_subsample.csv"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"


def build_report_generator(checkpoint: Path, device):
    ckpt = load_checkpoint(checkpoint)
    config = ckpt["config"]
    vision_checkpoint = REPO_ROOT / config.get("vision_checkpoint", config.get("stage1_checkpoint"))
    vision_encoder = load_vision_encoder(vision_checkpoint, device)
    model = ReportGenerator(
        vision_encoder,
        decoder_name=config["decoder_name"],
        pooled_grid_size=config["pooled_grid_size"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    assert all(("lora_" not in m and "projector" not in m) for m in missing), missing
    assert not unexpected, unexpected
    model.eval()
    print(f"[info] {checkpoint}: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, vision={vision_checkpoint.name} ({vision_encoder.backbone_name}), visual tokens={model.num_visual_tokens}")
    return model, config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "report_gen_v2" / "best.pt")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "medxvlm_stage3_v2_generations.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(SUBSAMPLE)
    manifest = pd.read_csv(MANIFEST)[["filename", "projection", "report_text"]]
    rows = sub[["filename"]].merge(manifest, on="filename", how="left", validate="one_to_one")
    assert rows.report_text.notna().all()
    print(f"[info] {len(rows)} images: {rows.projection.value_counts().to_dict()}")

    model, _config = build_report_generator(args.checkpoint, device)
    transform = build_transforms("test")

    generated: list[str] = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows.iloc[start : start + args.batch_size]
        pixel_values = torch.stack([transform(Image.open(IMAGES_DIR / f).convert("RGB")) for f in batch.filename]).to(device)
        generated.extend(model.generate(pixel_values, DEFAULT_REPORT_INSTRUCTION, max_new_tokens=args.max_new_tokens))
        print(f"[progress] {min(start + args.batch_size, len(rows))}/{len(rows)}, {time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame({
        "filename": rows.filename,
        "projection": rows.projection,
        "reference": rows.report_text,  # cleaned text (current manifest)
        "generated": generated,
    })
    out.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out} | distinct generations: {out.generated.nunique()} / {len(out)} | {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
