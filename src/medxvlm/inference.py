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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from medxvlm.data.dataset import DEFAULT_REPORT_INSTRUCTION, build_transforms
from medxvlm.engine.loop_utils import load_checkpoint, load_vision_encoder
from medxvlm.explain.attention_maps import overlay_attention
from medxvlm.models.classifier_head import ClassifierHead
from medxvlm.models.labels import CHEXPERT_LABELS
from medxvlm.models.report_generator import ReportGenerator
from medxvlm.models.vision_encoder import VisionEncoder, infer_backbone_from_state_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIER_CHECKPOINT = REPO_ROOT / "checkpoints" / "classifier_v2" / "local_on" / "best.pt"
DEFAULT_REPORT_CHECKPOINT = REPO_ROOT / "checkpoints" / "report_gen_v2" / "best.pt"


@dataclass
class ClassificationResult:
    """Per-label probabilities plus the attention map each probability was pooled from."""

    probabilities: dict[str, float]
    attention_weights: torch.Tensor  # [num_labels, num_patches], CPU
    pixel_values: torch.Tensor  # [3, H, W], the normalised tensor the model saw, CPU
    checkpoint: str

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)

    def heatmap(self, label: str) -> np.ndarray:
        """RGB uint8 overlay of `label`'s attention over the preprocessed image."""
        return overlay_attention(self.pixel_values, self.attention_weights[CHEXPERT_LABELS.index(label)])


@dataclass
class ReportResult:
    text: str
    checkpoint: str
    max_new_tokens: int
    flags: list[str] = field(default_factory=list)  # honesty flags for the UI, e.g. device-mention warning


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
            "10% of held-out frontal cases (27% on lateral views) -- verify against the image before trusting."
        )
    return flags


class MedXVLMPipeline:
    """Load once, then `classify()` / `generate_report()` any PIL image."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        classifier_checkpoint: str | Path = DEFAULT_CLASSIFIER_CHECKPOINT,
        report_checkpoint: str | Path | None = DEFAULT_REPORT_CHECKPOINT,
        load_report_generator: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = build_transforms("test")
        self.classifier_checkpoint = Path(classifier_checkpoint)
        self.report_checkpoint = Path(report_checkpoint) if report_checkpoint else None
        self.vision_encoder, self.head, self.classifier_meta = self._load_classifier(self.classifier_checkpoint)
        self.report_generator: ReportGenerator | None = None
        self.report_meta: dict = {}
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

    def load_report_generator(self) -> None:
        """Load Stage 3 lazily -- the 1.5B decoder is the expensive part (~7 GB VRAM in fp32)."""
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
        }

    def unload_report_generator(self) -> None:
        self.report_generator = None
        self.report_meta = {}
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- inference ---------------------------------------------------------
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))

    @torch.no_grad()
    def classify(self, image: Image.Image) -> ClassificationResult:
        pixel_values = self.preprocess(image)
        out = self.head(self.vision_encoder(pixel_values.unsqueeze(0).to(self.device)).patch_tokens, text_embeds=None)
        probs = torch.sigmoid(out.logits)[0].cpu()
        return ClassificationResult(
            probabilities={label: float(probs[i]) for i, label in enumerate(CHEXPERT_LABELS)},
            attention_weights=out.attention_weights[0].cpu(),
            pixel_values=pixel_values,
            checkpoint=str(self.classifier_checkpoint.relative_to(REPO_ROOT)),
        )

    @torch.no_grad()
    def generate_report(self, image: Image.Image, max_new_tokens: int = 128) -> ReportResult:
        if self.report_generator is None:
            self.load_report_generator()
        assert self.report_generator is not None, "no report checkpoint configured"
        pixel_values = self.preprocess(image).unsqueeze(0).to(self.device)
        text = self.report_generator.generate(pixel_values, DEFAULT_REPORT_INSTRUCTION, max_new_tokens=max_new_tokens)[0].strip()
        return ReportResult(
            text=text,
            checkpoint=str(self.report_checkpoint.relative_to(REPO_ROOT)),
            max_new_tokens=max_new_tokens,
            flags=_flag_report(text),
        )
