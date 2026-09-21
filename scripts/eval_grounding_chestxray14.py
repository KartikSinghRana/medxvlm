#!/usr/bin/env python3
"""Quantitative grounding check: do the saliency maps land on the radiologist's box?

NIH ChestX-ray14 ships 984 hand-drawn bounding boxes (880 test images, 8
findings) -- the only pixel-level ground truth this project has access to,
since Indiana/Open-i has none. Every saliency method in
`medxvlm.explain.saliency` is scored on every box with:

  - pointing game   -- is the map's argmax pixel inside the box? (Zhang et al. 2018)
  - mass ratio      -- share of (min-max normalised) map mass inside the box,
                       divided by the box's share of the image; 1.0 = no better
                       than a uniform map
  - area-matched IoU -- IoU between the box and the map's top-k pixels, k = box area

against two baselines: the expected hit rate of a uniformly random point (the
box's area fraction) and always pointing at the image centre.

Label mapping (NIH box label -> MedXVLM head label): five exact (Atelectasis,
Cardiomegaly, Effusion->Pleural Effusion, Pneumonia, Pneumothorax), Mass and
Nodule -> Lung Lesion (the CheXpert definition of that class), Infiltrate ->
Lung Opacity (approximate).

Usage:
    PYTHONPATH=src python scripts/eval_grounding_chestxray14.py [--limit 50] [--examples 2]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_external_chestxray14 import find_nih_root  # noqa: E402
from medxvlm.data.dataset import IMAGE_SIZE, build_transforms  # noqa: E402
from medxvlm.engine.loop_utils import load_stage1_alignment, load_vision_encoder  # noqa: E402
from medxvlm.explain import saliency as S  # noqa: E402
from medxvlm.explain.attention_maps import overlay_attention  # noqa: E402
from medxvlm.inference import DEFAULT_CLASSIFIER_ENSEMBLE, MedXVLMPipeline  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"
BBOX_LABEL_MAP = {  # NIH box label -> MedXVLM head label
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax",
    "Mass": "Lung Lesion", "Nodule": "Lung Lesion", "Infiltrate": "Lung Opacity",
}
APPROXIMATE = {"Infiltrate"}
STAGE1_CHECKPOINTS = {  # text-grounded methods
    "text_v3b_findings": REPO_ROOT / "checkpoints/align_v3/findings/best.pt",
    "text_v2_report": REPO_ROOT / "checkpoints/align_v2/local_on/best.pt",
    "text_phase3_local_on": REPO_ROOT / "checkpoints/align/local_on/best.pt",
    "text_phase3_local_off": REPO_ROOT / "checkpoints/align/local_off/best.pt",
}
PROMPT = "There is {}."  # the pos-only protocol behind the 0.734 zero-shot number


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def score_box(up: np.ndarray, box: tuple[float, float, float, float]) -> dict[str, float]:
    """`up`: [S, S] raw map at image resolution; `box`: (x0, y0, x1, y1) in the same pixels."""
    size = up.shape[0]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, min(x0, size - 1)), max(0, min(y0, size - 1))
    x1, y1 = max(x0 + 1, min(x1, size)), max(y0 + 1, min(y1, size))
    area_frac = ((x1 - x0) * (y1 - y0)) / (size * size)
    ay, ax = np.unravel_index(int(np.argmax(up)), up.shape)
    hit = float(x0 <= ax < x1 and y0 <= ay < y1)
    norm = S.minmax(up)
    total = float(norm.sum())
    inside = float(norm[y0:y1, x0:x1].sum())
    mass_ratio = (inside / total) / area_frac if total > 0 else 1.0
    k = max(1, int(round(area_frac * size * size)))
    thresh = np.partition(up.ravel(), -k)[-k]
    top = up >= thresh
    box_mask = np.zeros_like(top)
    box_mask[y0:y1, x0:x1] = True
    iou = float((top & box_mask).sum() / max(1, (top | box_mask).sum()))
    return {"hit": hit, "mass_ratio": float(mass_ratio), "iou_area_matched": iou, "area_frac": float(area_frac)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-root", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: first N boxes only.")
    ap.add_argument("--examples", type=int, default=2, help="Overlay examples per class to save.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_nih_root(args.nih_root)
    boxes = pd.read_csv(next(root.rglob("BBox_List_2017.csv")))
    boxes = boxes.rename(columns={"Bbox [x": "x", "h]": "h"})[["Image Index", "Finding Label", "x", "y", "w", "h"]]
    boxes = boxes[boxes["Finding Label"].isin(BBOX_LABEL_MAP)].reset_index(drop=True)
    if args.limit:
        boxes = boxes.head(args.limit)
    index = {p.name: p for p in root.rglob("*.png")}
    boxes = boxes[boxes["Image Index"].isin(index)].reset_index(drop=True)
    print(f"[data] {len(boxes)} boxes on {boxes['Image Index'].nunique()} images; classes: {boxes['Finding Label'].value_counts().to_dict()}", flush=True)

    # ---- models ---------------------------------------------------------
    pipe = MedXVLMPipeline(device=device)  # 4-seed ensemble
    members = pipe.classifiers
    rollout_tower = load_vision_encoder(DEFAULT_CLASSIFIER_ENSEMBLE[0], device).eval()  # separate copy -> eager attention
    grounders: dict[str, S.TextGrounder] = {}
    prompt_embeds: dict[str, torch.Tensor] = {}
    for name, ck in STAGE1_CHECKPOINTS.items():
        if not ck.exists():
            print(f"[skip] {name}: {ck} missing"); continue
        g = S.TextGrounder(load_stage1_alignment(ck, device), device)
        grounders[name] = g
        prompt_embeds[name] = g.embed_prompts([PROMPT.format(l.lower()) for l in CHEXPERT_LABELS])  # [14, 512]
    transform = build_transforms("test")
    print(f"[models] ensemble n={len(members)}, text grounders: {list(grounders)}", flush=True)

    def maps_for(pixel_values: torch.Tensor, label_idx: list[int]) -> dict[str, torch.Tensor]:
        """Every method's map for the requested labels: name -> [len(label_idx), N] (rollout: same map per label)."""
        out: dict[str, torch.Tensor] = {}
        attn = [S.attention_pooling_map(v, h, pixel_values)[0, label_idx] for v, h, _ in members]
        out["stage2_attn_seed0"] = attn[0]
        out["stage2_attn_ensemble"] = torch.stack(attn).mean(0)
        cams = [S.gradcam_map(v, h, pixel_values, label_idx)[0] for v, h, _ in members]
        out["stage2_gradcam_seed0"] = cams[0]
        out["stage2_gradcam_ensemble"] = torch.stack(cams).mean(0)
        roll = S.attention_rollout(rollout_tower, pixel_values)[0]
        out["rollout_stage2_tower"] = roll.unsqueeze(0).expand(len(label_idx), -1)
        for name, g in grounders.items():
            out[name] = g.map(pixel_values, prompt_embeds[name][label_idx])[0]
        return out

    # ---- loop ---------------------------------------------------------------
    rows: list[dict] = []
    examples_saved: dict[str, int] = defaultdict(int)
    ex_dir = args.out_dir / "grounding_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    example_methods = ["stage2_attn_ensemble", "stage2_gradcam_ensemble", "rollout_stage2_tower", "text_v3b_findings"]
    t0 = time.time()
    for n_img, (img_name, group) in enumerate(boxes.groupby("Image Index", sort=False)):
        image = Image.open(index[img_name]).convert("RGB")
        w_img, h_img = image.size
        sx, sy = IMAGE_SIZE / w_img, IMAGE_SIZE / h_img
        pixel_values = transform(image).unsqueeze(0).to(device)
        labels = sorted({BBOX_LABEL_MAP[l] for l in group["Finding Label"]})
        label_idx = [CHEXPERT_LABELS.index(l) for l in labels]
        maps = maps_for(pixel_values, label_idx)
        for _, b in group.iterrows():
            head_label = BBOX_LABEL_MAP[b["Finding Label"]]
            j = labels.index(head_label)
            box = (b["x"] * sx, b["y"] * sy, (b["x"] + b["w"]) * sx, (b["y"] + b["h"]) * sy)
            base = {"image": img_name, "nih_label": b["Finding Label"], "head_label": head_label,
                    "approximate": b["Finding Label"] in APPROXIMATE}
            ups = {name: S.upsample_map(m[j], IMAGE_SIZE) for name, m in maps.items()}
            for name, up in ups.items():
                rows.append({**base, "method": name, **score_box(up, box)})
            # baselines
            x0, y0, x1, y1 = box
            area_frac = ((x1 - x0) * (y1 - y0)) / (IMAGE_SIZE * IMAGE_SIZE)
            c = IMAGE_SIZE / 2
            rows.append({**base, "method": "baseline_center", "hit": float(x0 <= c < x1 and y0 <= c < y1),
                         "mass_ratio": float("nan"), "iou_area_matched": float("nan"), "area_frac": area_frac})
            rows.append({**base, "method": "baseline_random_expected", "hit": area_frac,
                         "mass_ratio": 1.0, "iou_area_matched": float("nan"), "area_frac": area_frac})
            if examples_saved[b["Finding Label"]] < args.examples:
                panels = []
                for name in example_methods:
                    if name not in maps:
                        continue
                    ov = Image.fromarray(overlay_attention(pixel_values[0].cpu(), maps[name][j].cpu()))
                    d = ImageDraw.Draw(ov)
                    d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
                    panels.append((name, ov))
                canvas = Image.new("RGB", (IMAGE_SIZE * len(panels), IMAGE_SIZE + 14), (0, 0, 0))
                for k, (name, ov) in enumerate(panels):
                    canvas.paste(ov, (k * IMAGE_SIZE, 14))
                    ImageDraw.Draw(canvas).text((k * IMAGE_SIZE + 3, 1), f"{name} | {b['Finding Label']}", fill=(255, 255, 255))
                canvas.save(ex_dir / f"{b['Finding Label'].lower()}_{examples_saved[b['Finding Label']]}_{img_name}.jpg", quality=85)
                examples_saved[b["Finding Label"]] += 1
        if (n_img + 1) % 50 == 0:
            print(f"[{n_img + 1}/{boxes['Image Index'].nunique()} images] {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "grounding_chestxray14_raw.csv", index=False)

    # ---- aggregate ------------------------------------------------------------
    summary: dict = {"n_boxes": int(len(boxes)), "n_images": int(boxes["Image Index"].nunique()),
                     "label_map": BBOX_LABEL_MAP, "approximate_labels": sorted(APPROXIMATE), "prompt": PROMPT,
                     "metrics": {"hit": "pointing game: argmax pixel of the 224x224-upsampled map inside the box",
                                 "mass_ratio": "(map mass inside box / total) / (box area / image area); 1.0 = uniform",
                                 "iou_area_matched": "IoU of box vs top-k pixels of the map, k = box area"},
                     "per_method": {}}
    for method, dm in df.groupby("method"):
        per_class = {}
        for cls, dc in dm.groupby("nih_label"):
            k, n = int(round(dc["hit"].sum())), len(dc)
            lo, hi = wilson(k, n) if method != "baseline_random_expected" else (float("nan"), float("nan"))
            per_class[cls] = {"n": n, "hit_rate": float(dc["hit"].mean()), "hit_ci95": [lo, hi],
                              "mass_ratio_mean": float(dc["mass_ratio"].mean()), "iou_mean": float(dc["iou_area_matched"].mean())}
        k, n = int(round(dm["hit"].sum())), len(dm)
        lo, hi = wilson(k, n) if method != "baseline_random_expected" else (float("nan"), float("nan"))
        summary["per_method"][method] = {
            "overall": {"n": n, "hit_rate": float(dm["hit"].mean()), "hit_ci95": [lo, hi],
                        "mass_ratio_mean": float(dm["mass_ratio"].mean()), "iou_mean": float(dm["iou_area_matched"].mean())},
            "macro_hit_rate_8_classes": float(np.mean([v["hit_rate"] for v in per_class.values()])),
            "per_class": per_class,
        }
    (args.out_dir / "grounding_chestxray14_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'method':28s} {'hit%':>6s} {'95% CI':>14s} {'macro8':>7s} {'mass':>6s} {'IoU':>6s}")
    order = sorted(summary["per_method"], key=lambda m: -summary["per_method"][m]["overall"]["hit_rate"])
    for m in order:
        o = summary["per_method"][m]["overall"]
        ci = "" if math.isnan(o["hit_ci95"][0]) else f"[{o['hit_ci95'][0]*100:.1f}, {o['hit_ci95'][1]*100:.1f}]"
        print(f"{m:28s} {o['hit_rate']*100:6.1f} {ci:>14s} {summary['per_method'][m]['macro_hit_rate_8_classes']*100:7.1f} "
              f"{o['mass_ratio_mean']:6.2f} {o['iou_mean']:6.3f}")
    print(f"[done] wrote {args.out_dir / 'grounding_chestxray14_summary.json'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
