"""Single-image inference over the trained MedXVLM checkpoints.

One object, `MedXVLMPipeline`, wraps everything the demo app (and any other
consumer) needs: the v2 image-only classifier with its per-label attention
maps, and -- optionally, because it costs ~7 GB of VRAM -- the v2 LoRA report
generator. Preprocessing is `build_transforms("test")`, i.e. exactly what the
models were validated with (CLAHE -> resize -> ImageNet normalisation).

# why a module and not code inside app/app.py: the Streamlit script should be
# a thin view; keeping load/classify/generate here makes the pipeline usable
# from a notebook or a test without importing Streamlit, and lets the app be
# exercised headlessly through streamlit.testing.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import json

from medxvlm.data.dataset import DEFAULT_REPORT_INSTRUCTION, build_transforms
from medxvlm.data.findings_prompt import LISTABLE_LABELS, findings_instruction
from medxvlm.engine.loop_utils import load_checkpoint, load_stage1_alignment, load_vision_encoder
from medxvlm.explain import saliency
from medxvlm.explain.attention_maps import overlay_attention
from medxvlm.explain.sentence_attribution import SentenceAttribution, attribute_sentences
from medxvlm.models.classifier_head import ClassifierHead
from medxvlm.models.labels import CHEXPERT_LABELS
from medxvlm.models.report_generator import ReportGenerator
from medxvlm.models.vision_encoder import VisionEncoder, infer_backbone_from_state_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIER_CHECKPOINT = REPO_ROOT / "checkpoints" / "classifier_v2" / "local_on" / "best.pt"
# The 4-seed ensemble (docs/v2_rebuild.md §3.6.1): 0.868 macro AUROC vs 0.837 +- 0.015 for any
# single seed. Each member is ~0.35 GB of VRAM, so four cost less than the report decoder.
DEFAULT_CLASSIFIER_ENSEMBLE = tuple(
    REPO_ROOT / "checkpoints" / "classifier_v2" / run / "best.pt" for run in ("local_on", "seed1", "seed2", "seed3")
)
# Report generators the app can serve. "v2" writes from the image alone; "v2c" is findings-conditioned:
# its prompt carries the classifier ensemble's flagged findings (docs/v2_rebuild.md §3.7), built here from the
# *live* ensemble probabilities and the val-selected thresholds -- the same rule that built its training prompts.
REPORT_CHECKPOINTS = {
    "v2": REPO_ROOT / "checkpoints" / "report_gen_v2" / "best.pt",
    "v2c": REPO_ROOT / "checkpoints" / "report_gen_v2c" / "best.pt",
}
DEFAULT_REPORT_CHECKPOINT = REPORT_CHECKPOINTS["v2"]
FINDINGS_THRESHOLDS_PATH = REPO_ROOT / "docs" / "phase5_artifacts" / "v2" / "stage2_ensemble_thresholds.json"
# Beam search + mild repetition penalty: CE F1 0.152 -> 0.176 and device hallucination
# 10% -> 3% on the same checkpoint (docs/v2_rebuild.md §3.6.2). Greedy was the wrong default.
DEFAULT_DECODING = {"num_beams": 3, "repetition_penalty": 1.2}
# Stage 1 checkpoint behind the text-grounded heatmaps (the 0.734 zero-shot head, docs/v2_rebuild.md §3.6.3).
DEFAULT_TEXT_GROUNDING_CHECKPOINT = REPO_ROOT / "checkpoints" / "align_v3" / "findings" / "best.pt"
# Heatmap methods offered by `heatmaps()`; pointing-game accuracy on the 984 NIH ChestX-ray14 boxes is in
# docs/phase5_artifacts/v2/grounding_chestxray14_summary.json (attention 41%, Grad-CAM 19%, text 11%; random 7.5%).
HEATMAP_METHODS = ("attention", "gradcam", "text")


@dataclass
class ClassificationResult:
    """Per-label probabilities plus the attention map each probability was pooled from."""

    probabilities: dict[str, float]
    attention_weights: torch.Tensor  # [num_labels, num_patches], CPU (mean over ensemble members)
    pixel_values: torch.Tensor  # [3, H, W], the normalised tensor the model saw, CPU
    checkpoint: str  # comma-separated when an ensemble was used
    n_models: int = 1

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)

    def heatmap(self, label: str, style: str = "classic") -> np.ndarray:
        """RGB uint8 overlay of `label`'s attention over the preprocessed image."""
        return overlay_attention(self.pixel_values, self.attention_weights[CHEXPERT_LABELS.index(label)], style=style)


@dataclass
class ReportResult:
    text: str
    checkpoint: str
    max_new_tokens: int
    decoding: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)  # honesty flags for the UI, e.g. device-mention warning
    instruction: str = DEFAULT_REPORT_INSTRUCTION  # the exact prompt used (carries the flagged findings for v2c)
    flagged_findings: list[str] = field(default_factory=list)  # classifier findings put into the prompt (v2c only)


@dataclass
class SentenceExplanation:
    sentence: str
    visual_share: float  # attention mass on the 64 visual tokens, sink-corrected (0-1)
    overlay: np.ndarray  # RGB uint8 heatmap of that sentence's visual attention


# Terms the v2 generator is known to hallucinate (docs/v2_rebuild.md §3.4: 10% of
# frontal generations mention a device/surgery absent from the reference).
_DEVICE_TERMS = (
    "catheter", "picc", "tube", " line", "clip", "wire", "pacemaker", "lobectomy", "sternotomy", "port",
)


def _flag_report(text: str) -> list[str]:
    lowered = f" {text.lower()}"
    flags: list[str] = []
    if any(term in lowered for term in _DEVICE_TERMS):
        flags.append(
            "Mentions a support device or surgical change. The v2 generator hallucinated such mentions in "
            "10% of held-out frontal cases with greedy decoding (3% with the default beam search) -- verify against the image before trusting."
        )
    return flags


class MedXVLMPipeline:
    """Load once, then `classify()` / `generate_report()` any PIL image."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        classifier_checkpoints: Sequence[str | Path] = DEFAULT_CLASSIFIER_ENSEMBLE,
        report_checkpoint: str | Path | None = DEFAULT_REPORT_CHECKPOINT,
        load_report_generator: bool = False,
    ) -> None:
        """`classifier_checkpoints`: one path = single model; several = probability-averaged ensemble.
        Members that are missing on disk are skipped with a note in `classifier_meta["skipped"]`."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = build_transforms("test")
        paths = [Path(p) for p in classifier_checkpoints]
        present = [p for p in paths if p.exists()]
        if not present:
            raise FileNotFoundError(f"No classifier checkpoint found among {paths}")
        self.classifier_checkpoints = present
        self.classifier_checkpoint = present[0]  # kept for callers that expect a single path
        self.report_checkpoint = Path(report_checkpoint) if report_checkpoint else None
        self.classifiers = [self._load_classifier(p) for p in present]
        self.vision_encoder, self.head, self.classifier_meta = self.classifiers[0]
        self.classifier_meta = dict(self.classifier_meta, n_models=len(present), skipped=[str(p) for p in paths if p not in present])
        self.report_generator: ReportGenerator | None = None
        self.report_meta: dict = {}
        # Hosting safeguards: one decoder call at a time (the GPU is shared by every visitor), and an optional
        # idle timeout that frees the ~7 GB report model (and the Stage 1 grounder) when nobody has used them
        # for a while -- so a publicly hosted demo does not sit on VRAM the owner needs for other work.
        self._gpu_lock = threading.RLock()
        self._idle_timer: threading.Timer | None = None
        self.idle_unload_seconds: float | None = None
        self.text_grounder: saliency.TextGrounder | None = None
        self.text_grounding_checkpoint = DEFAULT_TEXT_GROUNDING_CHECKPOINT
        if load_report_generator:
            self.load_report_generator()

    # ---- loading -----------------------------------------------------------
    def _load_classifier(self, path: Path) -> tuple[VisionEncoder, ClassifierHead, dict]:
        ckpt = load_checkpoint(path)
        state = ckpt["model_state_dict"]
        vision = VisionEncoder(pretrained=False, backbone=infer_backbone_from_state_dict(state)).to(self.device)
        vision.load_state_dict({k[len("vision_encoder."):]: v for k, v in state.items() if k.startswith("vision_encoder.")})
        head_state = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        text_dim = head_state["text_proj.weight"].shape[1] if "text_proj.weight" in head_state else None
        head = ClassifierHead(embed_dim=vision.embed_dim, text_dim=text_dim).to(self.device)
        head.load_state_dict(head_state)
        vision.eval()
        head.eval()
        meta = {
            "epoch": ckpt.get("epoch"),
            "val_macro_auroc": ckpt.get("val_macro_auroc"),
            "backbone": vision.backbone_name,
            "use_text_fusion": ckpt.get("config", {}).get("use_text_fusion", text_dim is not None),
        }
        return vision, head, meta

    def load_report_generator(self, checkpoint: str | Path | None = None) -> None:
        """Load Stage 3 lazily -- the 1.5B decoder is the expensive part (~7 GB VRAM in fp32).

        Passing a different `checkpoint` swaps the generator (one decoder is kept in memory at a time)."""
        with self._gpu_lock:
            self._load_report_generator(checkpoint)
        self._touch()

    def _load_report_generator(self, checkpoint: str | Path | None) -> None:
        if checkpoint is not None and Path(checkpoint) != self.report_checkpoint:
            self.unload_report_generator()
            self.report_checkpoint = Path(checkpoint)
        if self.report_generator is not None or self.report_checkpoint is None:
            return
        ckpt = load_checkpoint(self.report_checkpoint)
        config = ckpt["config"]
        vision_path = REPO_ROOT / config.get("vision_checkpoint", config.get("stage1_checkpoint"))
        vision = load_vision_encoder(vision_path, self.device)
        model = ReportGenerator(
            vision,
            decoder_name=config["decoder_name"],
            pooled_grid_size=config["pooled_grid_size"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
        ).to(self.device)
        missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        assert all(("lora_" not in m and "projector" not in m) for m in missing), missing
        assert not unexpected, unexpected
        model.eval()
        self.report_generator = model
        self.report_meta = {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "decoder": config["decoder_name"],
            "visual_tokens": model.num_visual_tokens,
            "vision_checkpoint": str(vision_path.relative_to(REPO_ROOT)),
            "findings_conditioning": bool(config.get("findings_conditioning", False)),
            "select_by": ckpt.get("select_by", "val_loss"),
        }
        if self.report_meta["findings_conditioning"] and not hasattr(self, "_findings_thresholds"):
            self._findings_thresholds = json.load(open(FINDINGS_THRESHOLDS_PATH))["thresholds"]

    def unload_report_generator(self) -> None:
        with self._gpu_lock:
            self.report_generator = None
            self.report_meta = {}
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _touch(self) -> None:
        """Restart the idle countdown after any use of the lazily loaded models."""
        if not self.idle_unload_seconds:
            return
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self.idle_unload_seconds, self._unload_idle_models)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _unload_idle_models(self) -> None:
        with self._gpu_lock:
            self.text_grounder = None
            self.unload_report_generator()

    def load_text_grounder(self) -> saliency.TextGrounder:
        """Lazily load the Stage 1 alignment used for prompt -> patch similarity maps (~1 GB VRAM)."""
        if self.text_grounder is None:
            if not self.text_grounding_checkpoint.exists():
                raise FileNotFoundError(self.text_grounding_checkpoint)
            self.text_grounder = saliency.TextGrounder(load_stage1_alignment(self.text_grounding_checkpoint, self.device), self.device)
        return self.text_grounder

    def _set_decoder_attention(self, implementation: str) -> None:
        # sdpa for decoding speed, eager when attention weights must be read back (sentence attribution)
        base = self.report_generator.decoder.get_base_model()
        if base.config._attn_implementation != implementation:
            base.set_attn_implementation(implementation)

    # ---- inference ---------------------------------------------------------
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))

    @torch.no_grad()
    def classify(self, image: Image.Image) -> ClassificationResult:
        pixel_values = self.preprocess(image)
        batch = pixel_values.unsqueeze(0).to(self.device)
        probs_sum, attn_sum = None, None
        for vision, head, _meta in self.classifiers:
            out = head(vision(batch).patch_tokens, text_embeds=None)
            p, a = torch.sigmoid(out.logits)[0].cpu(), out.attention_weights[0].cpu()
            probs_sum = p if probs_sum is None else probs_sum + p
            attn_sum = a if attn_sum is None else attn_sum + a
        n = len(self.classifiers)
        probs, attn = probs_sum / n, attn_sum / n
        return ClassificationResult(
            probabilities={label: float(probs[i]) for i, label in enumerate(CHEXPERT_LABELS)},
            attention_weights=attn,
            pixel_values=pixel_values,
            checkpoint=", ".join(str(p.relative_to(REPO_ROOT)) for p in self.classifier_checkpoints),
            n_models=n,
        )

    def heatmaps(self, image: Image.Image, labels: Sequence[str], method: str = "attention", style: str = "classic") -> dict[str, np.ndarray]:
        """Label -> RGB overlay for one of `HEATMAP_METHODS`.

        "attention": Stage 2's per-label attention weights, averaged over the ensemble (what `classify()` returns).
        "gradcam":   Grad-CAM on the final patch tokens, averaged over the ensemble.
        "text":      cosine similarity of each patch to the prompt "There is <label>." in the Stage 1 space.
        """
        if method not in HEATMAP_METHODS:
            raise ValueError(f"method must be one of {HEATMAP_METHODS}, got {method!r}")
        pixel_values = self.preprocess(image)
        batch = pixel_values.unsqueeze(0).to(self.device)
        idx = [CHEXPERT_LABELS.index(l) for l in labels]
        if method == "attention":
            maps = torch.stack([saliency.attention_pooling_map(v, h, batch)[0, idx] for v, h, _ in self.classifiers]).mean(0)
        elif method == "gradcam":
            maps = torch.stack([saliency.gradcam_map(v, h, batch, idx)[0] for v, h, _ in self.classifiers]).mean(0)
        else:
            grounder = self.load_text_grounder()
            maps = grounder.map(batch, grounder.embed_prompts([f"There is {l.lower()}." for l in labels]))[0]
        return {label: overlay_attention(pixel_values, maps[i].cpu(), style=style) for i, label in enumerate(labels)}

    def flagged_findings(self, probabilities: dict[str, float]) -> list[str]:
        """Classifier findings above their val-selected thresholds, most confident first (the v2c prompt rule)."""
        thr = self._findings_thresholds
        names = [l for l in LISTABLE_LABELS if probabilities[l] >= thr[l]]
        return sorted(names, key=lambda l: -probabilities[l])

    def report_instruction(self, image: Image.Image) -> tuple[str, list[str]]:
        """The prompt the loaded generator expects: plain for v2, findings-conditioned (from live ensemble probs) for v2c."""
        if self.report_meta.get("findings_conditioning"):
            flagged = self.flagged_findings(self.classify(image).probabilities)
            return findings_instruction(flagged), flagged
        return DEFAULT_REPORT_INSTRUCTION, []

    @torch.no_grad()
    def attribute_report(self, image: Image.Image, text: str, instruction: str | None = None, style: str = "classic") -> list[SentenceExplanation]:
        with self._gpu_lock:
            result = self._attribute_report(image, text, instruction, style)
        self._touch()
        return result

    def _attribute_report(self, image: Image.Image, text: str, instruction: str | None, style: str) -> list[SentenceExplanation]:
        """Per-sentence visual attention of the generator for an already generated `text` (one extra decoder pass)."""
        if self.report_generator is None:
            self.load_report_generator()
        pixel_values = self.preprocess(image)
        if instruction is None:
            instruction = self.report_instruction(image)[0]
        self._set_decoder_attention("eager")
        atts: list[SentenceAttribution] = attribute_sentences(
            self.report_generator, pixel_values.unsqueeze(0).to(self.device), instruction, text
        )
        return [SentenceExplanation(a.sentence, a.visual_share_excl_sink, overlay_attention(pixel_values, a.visual_map, style=style)) for a in atts]

    @torch.no_grad()
    def generate_report(self, image: Image.Image, max_new_tokens: int = 128, decoding: dict | None = None) -> ReportResult:
        with self._gpu_lock:
            result = self._generate_report(image, max_new_tokens, decoding)
        self._touch()
        return result

    def _generate_report(self, image: Image.Image, max_new_tokens: int, decoding: dict | None) -> ReportResult:
        if self.report_generator is None:
            self.load_report_generator()
        assert self.report_generator is not None, "no report checkpoint configured"
        self._set_decoder_attention("sdpa")
        decoding = DEFAULT_DECODING if decoding is None else decoding
        instruction, flagged = self.report_instruction(image)
        pixel_values = self.preprocess(image).unsqueeze(0).to(self.device)
        text = self.report_generator.generate(pixel_values, instruction, max_new_tokens=max_new_tokens, **decoding)[0].strip()
        return ReportResult(
            text=text,
            checkpoint=str(self.report_checkpoint.relative_to(REPO_ROOT)),
            max_new_tokens=max_new_tokens,
            decoding=dict(decoding),
            flags=_flag_report(text),
            instruction=instruction,
            flagged_findings=flagged,
        )
