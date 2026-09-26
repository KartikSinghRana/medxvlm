#!/usr/bin/env python3
"""RQ1 -- does the image encoder's pretraining matter? Evaluate the encoder comparison.

Every checkpoint below was trained with the identical v2 Stage 2 recipe
(configs/classifier_v2.yaml: image-only per-label attention head, ASL, split LR,
early stopping); only the ViT-B tower's starting weights differ:

  X-ray DINOv2  domain-specific, self-supervised on chest X-rays   (classifier_v2/{local_on,seed1,seed2,seed3})
  BiomedCLIP    domain-specific, contrastive on PubMed figure-captions (classifier_v2/rq1_biomedclip_s*)
  OpenAI CLIP   general-domain, contrastive on web image-text      (classifier_v2/rq1_clip_openai_s*)
  ImageNet ViT  general-domain, supervised ImageNet-21k -> 1k       (classifier_v2/rq1_imagenet_vit_s*)

For each checkpoint: macro AUROC on the 377 held-out frontal Indiana test images
(8 findings shared with RadZero / 5 with >=10 positives / all 13), macro AUROC on
the NIH ChestX-ray14 test split (8 mapped findings, a hospital never seen), and
the pointing-game hit rate of its attention maps on the 984 NIH radiologist boxes.

Writes docs/phase5_artifacts/v2/rq1_backbones_summary.json and
rq1_backbones_test_predictions.csv (seed-0 per-image probabilities, consumed by
eval_bootstrap_classification.py for paired confidence intervals).

Usage:
    PYTHONPATH=src python scripts/eval_rq1_backbones.py [--skip-nih] [--skip-grounding]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_external_chestxray14 import LABEL_MAP, NIH_LABELS, NihDataset, find_nih_root  # noqa: E402
from eval_grounding_chestxray14 import BBOX_LABEL_MAP, score_box  # noqa: E402
from eval_v2_classification import per_class_auroc, summarize  # noqa: E402
from medxvlm.data.dataset import IMAGE_SIZE, ClassificationDataset, build_transforms  # noqa: E402
from medxvlm.explain import saliency as S  # noqa: E402
from medxvlm.inference import MedXVLMPipeline  # noqa: E402
from medxvlm.models.labels import CHEXPERT_LABELS  # noqa: E402

CK = REPO_ROOT / "checkpoints" / "classifier_v2"
OUT = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"
MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
GROUPS = {  # display name -> (pretraining description, [run dirs, seed 0 first])
    "RQ1 X-ray DINOv2": ("domain-specific, self-supervised (chest X-rays)", ["local_on", "seed1", "seed2", "seed3"]),
    "RQ1 BiomedCLIP": ("domain-specific, contrastive (PubMed figures)", ["rq1_biomedclip_s0", "rq1_biomedclip_s1"]),
    "RQ1 OpenAI CLIP": ("general-domain, contrastive (web image-text)", ["rq1_clip_openai_s0", "rq1_clip_openai_s1"]),
    "RQ1 ImageNet ViT": ("general-domain, supervised (ImageNet)", ["rq1_imagenet_vit_s0", "rq1_imagenet_vit_s1"]),
}


def load(run: str, device):
    pipe = MedXVLMPipeline(device=device, classifier_checkpoints=[CK / run / "best.pt"], report_checkpoint=None)
    vision, head, _ = pipe.classifiers[0]
    ckpt_cfg = torch.load(CK / run / "best.pt", map_location="cpu", weights_only=False)
    return vision, head, {"best_epoch": ckpt_cfg.get("epoch"), "val_macro_auroc": ckpt_cfg.get("val_macro_auroc")}


@torch.no_grad()
def probs_for(vision, head, loader, device) -> np.ndarray:
    out = []
    for batch in loader:
        pv = batch[0].to(device, non_blocking=True)
        out.append(torch.sigmoid(head(vision(pv).patch_tokens).logits).cpu())
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nih", action="store_true")
    ap.add_argument("--skip-grounding", action="store_true")
    ap.add_argument("--groups", nargs="*", default=None, help="Only these groups (smoke tests).")
    ap.add_argument("--out-suffix", default="", help="Suffix for the output files (smoke tests).")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # --- data ------------------------------------------------------------
    test_ds = ClassificationDataset(MANIFEST, "test", projections=("Frontal",))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4,
                             collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])))
    targets = torch.cat([t for _, t in test_loader]).numpy().astype(int)
    support = {c: int(targets[:, i].sum()) for i, c in enumerate(CHEXPERT_LABELS)}
    filenames = test_ds.rows.filename.tolist()

    if not args.skip_nih or not args.skip_grounding:
        root = find_nih_root(None)
        index = {p.name: p for p in root.rglob("*.png")}
    if not args.skip_nih:
        entry = pd.read_csv(next(root.rglob("Data_Entry_2017*.csv")))
        test_list = set(next(root.rglob("test_list.txt")).read_text().split())
        nih = entry[entry["Image Index"].isin(test_list) & entry["Image Index"].isin(index)].copy()
        for lab in NIH_LABELS:
            nih[lab] = nih["Finding Labels"].str.split("|").apply(lambda xs: int(lab in xs))
        nih_loader = DataLoader(NihDataset(nih, index), batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
    if not args.skip_grounding:
        boxes = pd.read_csv(next(root.rglob("BBox_List_2017.csv"))).rename(columns={"Bbox [x": "x", "h]": "h"})
        boxes = boxes[boxes["Finding Label"].isin(BBOX_LABEL_MAP) & boxes["Image Index"].isin(index)]
        transform = build_transforms("test")
        box_images = {}
        for name in boxes["Image Index"].unique():
            img = Image.open(index[name]).convert("RGB")
            box_images[name] = (transform(img), img.size)

    # --- evaluate ----------------------------------------------------------
    results: dict = {"groups": {}, "recipe": "configs/classifier_v2.yaml (identical for all); only the tower's initial weights differ",
                     "n_test_images": len(filenames)}
    seed0_preds = {"filename": filenames}
    for group, (desc, runs) in GROUPS.items():
        if args.groups and group not in args.groups:
            continue
        present = [r for r in runs if (CK / r / "best.pt").exists()]
        if not present:
            print(f"[skip] {group}: no checkpoints"); continue
        per_seed = {}
        for run in present:
            vision, head, meta = load(run, device)
            p = probs_for(vision, head, test_loader, device)
            s = summarize(per_class_auroc(p, targets, list(CHEXPERT_LABELS)), support)
            row = {**meta, "backbone": vision.backbone_name,
                   "test_auroc_8": s["macro_auroc_8_shared_classes"],
                   "test_auroc_5_stable": s["macro_auroc_shared_classes_with_ge10_positives"],
                   "test_auroc_13": s["macro_auroc_all_valid_classes"], "per_class_auroc": s["per_class_auroc"]}
            if run == present[0]:
                for i, c in enumerate(CHEXPERT_LABELS):
                    seed0_preds[f"{group}__{c}"] = p[:, i]
            if not args.skip_nih:
                arr = np.zeros((len(nih), len(CHEXPERT_LABELS)), dtype=np.float32)
                with torch.no_grad():
                    for pv, idx in nih_loader:
                        arr[idx.numpy()] = torch.sigmoid(head(vision(pv.to(device, non_blocking=True)).patch_tokens).logits).cpu().numpy()
                row["nih_auroc_8"] = float(np.mean([roc_auc_score(nih[LABEL_MAP[c]].to_numpy(), arr[:, CHEXPERT_LABELS.index(c)]) for c in LABEL_MAP]))
            if not args.skip_grounding:
                hits, n = 0, 0
                for name, group_boxes in boxes.groupby("Image Index"):
                    pv, (w, h) = box_images[name]
                    attn = S.attention_pooling_map(vision, head, pv.unsqueeze(0).to(device))[0].cpu()
                    for _, b in group_boxes.iterrows():
                        j = CHEXPERT_LABELS.index(BBOX_LABEL_MAP[b["Finding Label"]])
                        sx, sy = IMAGE_SIZE / w, IMAGE_SIZE / h
                        box = (b["x"] * sx, b["y"] * sy, (b["x"] + b["w"]) * sx, (b["y"] + b["h"]) * sy)
                        hits += score_box(S.upsample_map(attn[j], IMAGE_SIZE), box)["hit"]
                        n += 1
                row["pointing_game_hit_rate"] = hits / n
                row["pointing_game_n_boxes"] = n
            per_seed[run] = row
            print(f"[{group}] {run}: test-8 {row['test_auroc_8']:.3f} | test-5 {row['test_auroc_5_stable']:.3f} | "
                  f"NIH {row.get('nih_auroc_8', float('nan')):.3f} | pointing {row.get('pointing_game_hit_rate', float('nan')):.3f} "
                  f"| best epoch {row['best_epoch']} | {time.time() - t0:.0f}s", flush=True)
            del vision, head
            torch.cuda.empty_cache()

        def agg(key: str) -> dict:
            vals = np.array([r[key] for r in per_seed.values() if key in r])
            return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else None, "n_seeds": int(len(vals))} if len(vals) else {}

        results["groups"][group] = {
            "pretraining": desc, "backbone": next(iter(per_seed.values()))["backbone"], "per_seed": per_seed,
            "test_auroc_8": agg("test_auroc_8"), "test_auroc_5_stable": agg("test_auroc_5_stable"),
            "test_auroc_13": agg("test_auroc_13"), "nih_auroc_8": agg("nih_auroc_8"),
            "pointing_game_hit_rate": agg("pointing_game_hit_rate"),
        }

    pd.DataFrame(seed0_preds).to_csv(OUT / f"rq1_backbones_test_predictions{args.out_suffix}.csv", index=False)
    (OUT / f"rq1_backbones_summary{args.out_suffix}.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'encoder':20s} {'seeds':>5s} {'test AUROC-8':>14s} {'test AUROC-5':>14s} {'NIH AUROC-8':>14s} {'pointing':>10s}")
    for g, r in results["groups"].items():
        f = lambda k: (f"{r[k]['mean']:.3f}" + (f"±{r[k]['std']:.3f}" if r[k].get("std") else "")) if r[k] else "—"  # noqa: E731
        print(f"{g:20s} {r['test_auroc_8']['n_seeds']:5d} {f('test_auroc_8'):>14s} {f('test_auroc_5_stable'):>14s} "
              f"{f('nih_auroc_8'):>14s} {f('pointing_game_hit_rate'):>10s}")
    print(f"[done] {time.time() - t0:.0f}s -> {OUT / f'rq1_backbones_summary{args.out_suffix}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
