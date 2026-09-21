#!/usr/bin/env python3
"""Sentence-level visual attribution of the Stage 3 generator (RQ3/RQ4 explainability).

Two questions, both answered with `medxvlm.explain.sentence_attribution`:

1. **How much does each generated sentence look at the image?** On all 377
   frontal Indiana held-out test images: per-sentence visual attention share (raw and
   with the attention sink removed), split by what the labeler says the
   sentence is (positive finding / negation / other).
2. **When a sentence names a finding, does its attention land on the
   radiologist's box?** On a stratified sample of NIH ChestX-ray14 box images:
   for every generated sentence the labeler marks POSITIVE for the box's
   label, pointing-game hit of that sentence's 8x8 map vs the box, against the
   centre baseline. Mentions are rare (the generator is mostly "normal"), so
   the counts are reported alongside the rates.

Beam-3 + repetition penalty 1.2 decoding (the project default), batched.

Usage:
    PYTHONPATH=src python scripts/eval_sentence_attribution.py [--nih-per-class 40] [--examples 6]
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
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_external_chestxray14 import find_nih_root  # noqa: E402
from eval_grounding_chestxray14 import APPROXIMATE, BBOX_LABEL_MAP, score_box, wilson  # noqa: E402
from medxvlm.data.dataset import DEFAULT_REPORT_INSTRUCTION, IMAGE_SIZE, _load_split, _resolve_image_path  # noqa: E402
from medxvlm.data.labeler import NEGATIVE, POSITIVE, UNCERTAIN, find_mentions  # noqa: E402
from medxvlm.explain import saliency as S  # noqa: E402
from medxvlm.explain.attention_maps import overlay_attention  # noqa: E402
from medxvlm.explain.sentence_attribution import attribute_sentences  # noqa: E402
from medxvlm.inference import DEFAULT_DECODING, MedXVLMPipeline  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
OUT_DIR = REPO_ROOT / "docs" / "phase5_artifacts" / "v2"


def sentence_kind(sentence: str) -> tuple[str, list[str]]:
    """'positive' (any POSITIVE/UNCERTAIN finding mention), 'negation' (only NEGATIVE mentions) or 'other'."""
    mentions = [m for m in find_mentions(sentence) if m.label != "No Finding"]
    pos = sorted({m.label for m in mentions if m.value in (POSITIVE, UNCERTAIN)})
    if pos:
        return "positive", pos
    if any(m.value == NEGATIVE for m in mentions):
        return "negation", []
    return "other", []


def generate_batched(pipe: MedXVLMPipeline, images: list[Image.Image], batch_size: int) -> list[str]:
    texts: list[str] = []
    gen = pipe.report_generator
    base = gen.decoder.get_base_model()
    base.set_attn_implementation("sdpa")  # fast path for decoding; attribution switches to eager itself
    for i in range(0, len(images), batch_size):
        pv = torch.stack([pipe.preprocess(im) for im in images[i:i + batch_size]]).to(pipe.device)
        texts.extend(t.strip() for t in gen.generate(pv, DEFAULT_REPORT_INSTRUCTION, max_new_tokens=128, **DEFAULT_DECODING))
    return texts


def save_example(path: Path, pixel_values: torch.Tensor, atts, box=None, title: str = "") -> None:
    panels = []
    for a in atts[:5]:
        ov = Image.fromarray(overlay_attention(pixel_values, a.visual_map))
        if box is not None:
            ImageDraw.Draw(ov).rectangle(list(box), outline=(255, 255, 255), width=2)
        panels.append((f"vis={a.visual_share_excl_sink:.2f} | {a.sentence[:34]}", ov))
    if not panels:
        return
    canvas = Image.new("RGB", (IMAGE_SIZE * len(panels), IMAGE_SIZE + 26), (0, 0, 0))
    ImageDraw.Draw(canvas).text((3, 1), title[:120], fill=(255, 255, 255))
    for k, (label, ov) in enumerate(panels):
        canvas.paste(ov, (k * IMAGE_SIZE, 26))
        ImageDraw.Draw(canvas).text((k * IMAGE_SIZE + 3, 13), label, fill=(255, 255, 0))
    canvas.save(path, quality=85)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-root", type=Path, default=None)
    ap.add_argument("--nih-per-class", type=int, default=40, help="Box images sampled per NIH finding (seed 0).")
    ap.add_argument("--limit-indiana", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    pipe = MedXVLMPipeline(device=device, load_report_generator=True)
    ex_dir = args.out_dir / "sentence_attribution_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"decoding": DEFAULT_DECODING, "checkpoint": pipe.report_meta}

    # ---- 1. Indiana frontal test: visual share by sentence kind ----------------------
    df = _load_split(MANIFEST, "test", projections=("Frontal",))
    if args.limit_indiana:
        df = df.head(args.limit_indiana)
    images = [Image.open(_resolve_image_path(MANIFEST, f)).convert("RGB") for f in df["filename"]]
    texts = generate_batched(pipe, images, args.batch_size)
    print(f"[indiana] generated {len(texts)} reports in {time.time() - t0:.0f}s", flush=True)
    rows = []
    n_examples = 0
    for (_, r), img, text in zip(df.iterrows(), images, texts):
        pv = pipe.preprocess(img).unsqueeze(0).to(device)
        atts = attribute_sentences(pipe.report_generator, pv, DEFAULT_REPORT_INSTRUCTION, text)
        for k, a in enumerate(atts):
            kind, labels = sentence_kind(a.sentence)
            ent = float(-(a.visual_map * (a.visual_map + 1e-12).log()).sum() / np.log(len(a.visual_map)))
            rows.append({"dataset": "indiana_frontal_test", "image": r["filename"], "sentence_idx": k, "sentence": a.sentence,
                         "kind": kind, "labels": "|".join(labels), "token_count": a.token_count,
                         "visual_share": a.visual_share, "visual_share_excl_sink": a.visual_share_excl_sink,
                         "prompt_text_share": a.text_share_prompt, "map_entropy_norm": ent})
        if n_examples < args.examples and any(sentence_kind(a.sentence)[0] == "positive" for a in atts):
            save_example(ex_dir / f"indiana_{r['filename']}.jpg", pv[0].cpu(), atts, title=f"{r['filename']} | {text[:100]}")
            n_examples += 1
    ind = pd.DataFrame(rows)
    by_kind = {}
    for kind, dk in ind.groupby("kind"):
        by_kind[kind] = {"n_sentences": int(len(dk)), "visual_share_mean": float(dk["visual_share"].mean()),
                         "visual_share_excl_sink_mean": float(dk["visual_share_excl_sink"].mean()),
                         "prompt_text_share_mean": float(dk["prompt_text_share"].mean()),
                         "map_entropy_norm_mean": float(dk["map_entropy_norm"].mean())}
    summary["indiana_frontal_test"] = {
        "n_images": int(len(df)), "n_sentences": int(len(ind)),
        "uniform_visual_share_reference": "64 visual tokens out of ~120 attended positions would give ~0.5 if attention were uniform",
        "by_sentence_kind": by_kind,
        "visual_share_excl_sink_mean_all": float(ind["visual_share_excl_sink"].mean()),
        "visual_share_excl_sink_p90": float(ind["visual_share_excl_sink"].quantile(0.9)),
    }
    print(f"[indiana] {len(ind)} sentences; visual share (excl. sink) by kind: "
          + ", ".join(f"{k}={v['visual_share_excl_sink_mean']:.3f} (n={v['n_sentences']})" for k, v in by_kind.items()), flush=True)

    # ---- 2. NIH boxes: do finding sentences point at the box? -----------------------
    root = find_nih_root(args.nih_root)
    boxes = pd.read_csv(next(root.rglob("BBox_List_2017.csv")))
    boxes = boxes.rename(columns={"Bbox [x": "x", "h]": "h"})[["Image Index", "Finding Label", "x", "y", "w", "h"]]
    boxes = boxes[boxes["Finding Label"].isin(BBOX_LABEL_MAP)]
    sample = boxes.groupby("Finding Label", group_keys=False).apply(
        lambda g: g.sample(n=min(args.nih_per_class, len(g)), random_state=0)).reset_index(drop=True)
    index = {p.name: p for p in root.rglob("*.png")}
    sample = sample[sample["Image Index"].isin(index)].reset_index(drop=True)
    img_names = list(dict.fromkeys(sample["Image Index"]))
    nih_images = [Image.open(index[n]).convert("RGB") for n in img_names]
    t1 = time.time()
    nih_texts = dict(zip(img_names, generate_batched(pipe, nih_images, args.batch_size)))
    print(f"[nih] generated {len(img_names)} reports in {time.time() - t1:.0f}s", flush=True)
    nih_rows, n_ex = [], 0
    for name, img in zip(img_names, nih_images):
        pv = pipe.preprocess(img).unsqueeze(0).to(device)
        atts = attribute_sentences(pipe.report_generator, pv, DEFAULT_REPORT_INSTRUCTION, nih_texts[name])
        w_img, h_img = img.size
        sx, sy = IMAGE_SIZE / w_img, IMAGE_SIZE / h_img
        for _, b in sample[sample["Image Index"] == name].iterrows():
            head_label = BBOX_LABEL_MAP[b["Finding Label"]]
            box = (b["x"] * sx, b["y"] * sy, (b["x"] + b["w"]) * sx, (b["y"] + b["h"]) * sy)
            hits = [a for a in atts if head_label in sentence_kind(a.sentence)[1]]
            base = {"image": name, "nih_label": b["Finding Label"], "head_label": head_label,
                    "approximate": b["Finding Label"] in APPROXIMATE, "report": nih_texts[name]}
            if not hits:
                nih_rows.append({**base, "mentioned": False})
                continue
            for a in hits:
                sc = score_box(S.upsample_map(a.visual_map, IMAGE_SIZE), box)
                x0, y0, x1, y1 = box
                c = IMAGE_SIZE / 2
                nih_rows.append({**base, "mentioned": True, "sentence": a.sentence,
                                 "visual_share_excl_sink": a.visual_share_excl_sink, **sc,
                                 "center_hit": float(x0 <= c < x1 and y0 <= c < y1)})
            if n_ex < args.examples:
                save_example(ex_dir / f"nih_{b['Finding Label'].lower()}_{name}.jpg", pv[0].cpu(), hits, box=box,
                             title=f"{name} box={b['Finding Label']} | {nih_texts[name][:90]}")
                n_ex += 1
    nih = pd.DataFrame(nih_rows)
    per_class = {}
    for cls, dc in nih.groupby("nih_label"):
        m = dc[dc["mentioned"]]
        k, n = (int(m["hit"].sum()), len(m)) if len(m) else (0, 0)
        per_class[cls] = {"n_boxes": int(dc["image"].nunique()), "n_mentioning_sentences": n,
                          "hit_rate": (k / n) if n else None, "hit_ci95": list(wilson(k, n)) if n else None,
                          "center_hit_rate": float(m["center_hit"].mean()) if n else None,
                          "mass_ratio_mean": float(m["mass_ratio"].mean()) if n else None}
    m_all = nih[nih["mentioned"]]
    k, n = int(m_all["hit"].sum()) if len(m_all) else 0, len(m_all)
    summary["nih_boxes"] = {
        "n_images": int(len(img_names)), "n_boxes": int(len(sample)), "per_class_sampled": args.nih_per_class,
        "n_mentioning_sentences_total": n,
        "hit_rate_overall": (k / n) if n else None, "hit_ci95_overall": list(wilson(k, n)) if n else None,
        "center_hit_rate_overall": float(m_all["center_hit"].mean()) if n else None,
        "mass_ratio_mean_overall": float(m_all["mass_ratio"].mean()) if n else None,
        "per_class": per_class,
    }
    print(f"[nih] {n} box-matching finding sentences; hit {k}/{n}" + (f" = {k/n:.3f} (center {m_all['center_hit'].mean():.3f})" if n else ""), flush=True)
    for cls, v in per_class.items():
        print(f"   {cls:14s} boxes={v['n_boxes']:3d} mentions={v['n_mentioning_sentences']:3d} hit={v['hit_rate']} center={v['center_hit_rate']}")

    pd.concat([ind.assign(split="indiana"), nih.assign(split="nih")], ignore_index=True).to_csv(
        args.out_dir / "sentence_attribution_raw.csv", index=False)
    (args.out_dir / "sentence_attribution_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] {time.time() - t0:.0f}s -> {args.out_dir / 'sentence_attribution_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
