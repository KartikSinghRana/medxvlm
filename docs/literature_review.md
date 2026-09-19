# Literature Review

This review covers the literature MedXVLM's three-stage architecture
builds on, grouped by theme: supervised chest X-ray classification,
contrastive vision-language pretraining, zero-shot classification, report
generation, explainability, and parameter-efficient fine-tuning. It closes
by naming the specific gap MedXVLM's architecture targets and the papers
closest to it in scope.

## Theme 1 — Chest X-ray classification (supervised)

**ChestX-ray8 / ChestX-ray14 (Wang, Peng, Lu, Lu, Bagheri & Summers, 2017,
CVPR).** Introduced the NIH dataset of 108,948 frontal-view chest
radiographs from 32,717 patients, with disease labels mined from the
associated radiology reports via NLP rather than manual annotation, plus a
weakly-supervised classification/localization benchmark. The paper itself
introduces 8 labels; the publicly released dataset was later extended to
14 and is commonly called "ChestX-ray14" — both names refer to the same
underlying dataset and paper. It matters here as the dataset/labeling
paradigm nearly every later paper in this review trains or evaluates
against, and as a contrast case for MedXVLM's choice of a paired
image-report corpus instead of NLP-mined single labels.

**CheXNet (Rajpurkar, Irvin, Zhu et al., 2017, arXiv:1711.05225 —
technical report, no peer-reviewed venue).** Trained a 121-layer DenseNet
on ChestX-ray14 to detect pneumonia (and later all 14 pathologies),
reporting an F1 score exceeding the average of four practicing
radiologists on a reader study. It matters here as the canonical example
of purely supervised, single-task CNN classification on chest X-rays, and
as motivation for why later work (Themes 2–3) moved toward
representations that transfer across labels and tasks rather than one
model per finding.

**CheXpert (Irvin, Rajpurkar, Ko et al., 2019, AAAI).** Released 224,316
chest radiographs from 65,240 patients labeled by a rule-based text
labeler applied to the associated reports across 14 observations, with
explicit positive/negative/uncertain label states rather than forced
binary labels. It matters here directly: MedXVLM's Stage 2 classification
head targets this same 14-class CheXpert label set, and CheXpert is the
standard downstream benchmark most Theme 2/3 papers below fine-tune or
zero-shot-evaluate against.

**CheXbert (Smit, Jain, Rajpurkar, Pareek, Ng & Lungren, 2020, EMNLP).**
Replaced CheXpert's rule-based labeler with a BERT model trained on the
rule-based labeler's output plus a small expert-corrected, backtranslation
augmented dataset, improving label accuracy over the original labeler. It
matters here as the standard automatic tool for scoring generated
radiology reports' clinical accuracy (via CheXbert-labeled F1 between a
generated and reference report) — directly relevant to evaluating
MedXVLM's Stage 3 output.

## Theme 2 — Contrastive vision-language pretraining

**CLIP (Radford, Kim, Hallacy et al., 2021, ICML).** Trained paired image
and text encoders on 400M web (image, text) pairs with a symmetric
InfoNCE contrastive loss, aligning both modalities into a shared
embedding space and enabling zero-shot classification via natural-language
prompts. It matters here as the generic ancestor of the global InfoNCE
half of MedXVLM's Stage 1 objective, and of every medical
vision-language-pretraining paper below.

**ConVIRT (Zhang, Jiang, Miura, Manning & Langlotz; arXiv:2010.00747,
2020, formally published MLHC 2022).** Applied bidirectional
image-report contrastive learning (a ResNet50 image encoder and a
ClinicalBERT-initialised text encoder) to paired chest X-rays and
reports, predating CLIP's public release, and showed the resulting image
encoder needed roughly 10% of the labeled data to match ImageNet-pretrained
supervised baselines on downstream classification. It matters here as the
direct precedent for the *global-only* half of MedXVLM's Stage 1 loss —
GLoRIA (next) is what MedXVLM adds on top of this.

**GLoRIA (Huang, Shen, Lungren & Yeung, 2021, ICCV).** Added an
attention-weighted **local** contrastive loss — aligning image sub-regions
with individual words in the paired report — on top of a global
image-report contrastive loss, without requiring a pretrained object
detector, improving label-efficient classification, retrieval, and
segmentation. This is the direct architectural precedent for MedXVLM's
Stage 1 design ("global InfoNCE + local patch-word loss") and the single
most load-bearing citation in this review for that design choice.

**BioViL (Boecking, Usuyama, Bannur et al., 2022, ECCV).** Improved chest
X-ray vision-language pretraining mainly through better **text**
modeling — a domain-adapted language model (CXR-BERT) tuned to radiology
report semantics such as negation and uncertainty — paired with a
global-only contrastive objective, and released a phrase-grounding
annotated dataset. It matters here as a counterpoint to GLoRIA: strong
text-encoder quality alone, without an explicit local loss, is also a
viable route to good zero-shot and grounding performance, which frames a
design trade-off MedXVLM's local-loss choice is implicitly weighing
against.

**BiomedCLIP (Zhang, Xu, Usuyama et al.; arXiv:2303.00915, 2023, later
NEJM AI 2025).** A CLIP-style PubMedBERT + ViT dual encoder contrastively
pretrained on PMC-15M — 15 million figure-caption pairs spanning diverse
biomedical imaging modalities, not radiology-specific — that reached new
state-of-the-art results on biomedical retrieval, classification, and VQA,
in places outperforming radiology-specialist models despite its broader,
less domain-specific training data. It matters here directly: MedXVLM
initialises both the Stage 1 image and text encoders from BiomedCLIP
rather than general-domain CLIP or random initialisation, betting that
broad biomedical pretraining is a better starting point than
general-domain web images for a small, single-GPU fine-tuning budget.

## Theme 3 — Zero-shot medical image classification

**CheXzero (Tiu, Talius, Patel, Langlotz, Ng & Rajpurkar, 2022, Nature
Biomedical Engineering).** Applied CLIP-style contrastive pretraining
directly to 377,110 MIMIC-CXR image-report pairs with no manual pathology
labels at all, then performed zero-shot classification by comparing image
embeddings against text-prompt embeddings (e.g. "pneumonia" vs. "no
pneumonia"), matching board-certified radiologists on several pathologies
without any task-specific label training. It matters here as the closest
precedent for evaluating MedXVLM's Stage 1 embeddings' zero-shot
classification quality *before* Stage 2's supervised classification head
is layered on top.

**RadZero (Park, Yoon, Kim & Choi; arXiv:2504.07416, 2025, NeurIPS 2025
poster).** A very recent paper proposing similarity-based cross-attention
between text embeddings and local image features (VL-CABS) for
interpretable, fine-grained vision-language alignment on chest X-rays,
using an LLM to extract concise semantic sentences from reports and
training with multi-positive contrastive learning; it reports
state-of-the-art zero-shot classification, phrase grounding, and
segmentation, with pixel-level similarity maps for explainability. Unlike
the more established papers above, this is emerging 2025 work rather than
a foundational reference, but it matters here precisely because it
targets the same combination MedXVLM does — zero-shot alignment *and*
built-in explainability — from a single contrastive representation.

## Theme 4 — Report generation

**R2Gen (Chen, Song, Chang & Wan, 2020, EMNLP).** A Transformer report
generator augmented with a relational-memory module and memory-driven
conditional layer normalization, designed to handle the repetitive,
templated structure of radiology reports; evaluated on IU X-Ray (this
project's own dataset, under its NLM/Open-i name) and MIMIC-CXR. It
matters here both as a widely used report-generation baseline architecture
and as prior evidence that Indiana/Open-i specifically is a workable
corpus for report generation, not just classification.

**LLaVA (Liu, Li, Wu & Lee, 2023, NeurIPS, oral).** Used text-only GPT-4
to synthesize multimodal instruction-following data from image captions,
then connected a frozen pretrained CLIP ViT-L/14 encoder to a Vicuna LLM
through a single projection layer, training end-to-end on roughly 80K
images to reach GPT-4-like multimodal chat behaviour. It matters here as
the direct architectural template for MedXVLM's Stage 3: a frozen vision
tower, a lightweight projection into an LLM's token embedding space, and
an instruction-tuned language model consuming the projected visual tokens
as a soft prompt.

**LLaVA-Med (Li, Wong, Zhang et al., 2023, NeurIPS Datasets and
Benchmarks track, spotlight).** Adapted LLaVA's visual-instruction-tuning
recipe to biomedicine by curating PubMed Central figure-caption pairs and
using text-only GPT-4 to self-instruct open-ended biomedical Q&A data from
those captions, training the full pipeline in under 15 hours on 8 A100s
with a two-stage concept-alignment-then-instruction curriculum. It matters
here as direct evidence that the LLaVA recipe transfers to biomedical
imagery broadly (not radiology-specific) on a comparatively modest compute
budget, which is the same bet MedXVLM's Stage 3 makes at an even smaller
scale (a single GPU, a 0.5B-parameter LLM).

**CheXagent (Chen, Varma, Xu et al., 2024, arXiv:2401.12208 — technical
report, no peer-reviewed venue found).** A chest-X-ray-specific foundation
model combining a clinical LLM, a vision encoder, and a bridging network,
trained in four stages across 28 public datasets spanning 8 CXR tasks
(classification, VQA, report generation and summarization, grounding),
introducing the CheXinstruct instruction-tuning dataset and CheXbench
evaluation suite, and outperforming LLaVA-Med and other generalist medical
VLMs per radiologist evaluation. It matters here as the closest existing
system to MedXVLM's overall scope — classification and generation from
one shared backbone — at roughly two orders of magnitude more parameters,
data, and compute than a solo single-GPU project can target.

## Theme 5 — Explainability

**Grad-CAM (Selvaraju, Cogswell, Das, Vedantam, Parikh & Batra, 2017,
ICCV; extended version in IJCV 2020).** Uses the gradient of a target
class score with respect to a convolutional layer's feature maps to
produce a coarse, class-discriminative localization heatmap, without any
architecture change or retraining. It matters here as the standard
post-hoc visual explainability tool MedXVLM applies to the vision
encoder/classifier to show which image regions drove a given prediction.

**Attention rollout (Abnar & Zuidema, 2020, ACL).** Shows that raw
last-layer self-attention weights are an unreliable explanation in
Transformers, since information mixes across layers, and proposes
"attention rollout" — recursively combining attention weights across all
layers, treated as a DAG of information flow — to produce a more faithful
token-to-token or token-to-patch attribution map. It matters here as the
explainability method suited to MedXVLM's ViT and PubMedBERT encoders,
where Grad-CAM's convolutional assumptions do not apply as directly.

## Theme 6 — Parameter-efficient fine-tuning

**LoRA (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen; arXiv:2106.09685,
2021, published ICLR 2022).** Freezes a pretrained model's weight
matrices and injects small trainable low-rank decomposition matrices into
each targeted layer, cutting trainable parameters by orders of magnitude
and optimizer memory substantially, while matching or beating full
fine-tuning quality across several model families, with no added
inference latency once the low-rank update is merged back in. It matters
here directly: MedXVLM fine-tunes Qwen2.5-0.5B-Instruct for report
generation with LoRA specifically so Stage 3 fits on a single 16 GB GPU
without full-parameter fine-tuning.

## The gap this project targets

Most of the literature above optimizes one stage in isolation:
GLoRIA/BioViL/BiomedCLIP for alignment and zero-shot transfer, R2Gen/
LLaVA-Med for generation. Where a single system does span multiple
capabilities, as CheXagent does, it does so with multi-source pretraining
and a compute budget (28 datasets, a full clinical LLM backbone) that is
out of reach for a solo, single-GPU project on a fixed timeline. MedXVLM
is not proposing a new contrastive loss, a new projection mechanism, or a
new fine-tuning method — GLoRIA's local loss, LLaVA's projection, and
LoRA are each already established. The gap it targets is *integration at
small scale*: building alignment, classification, and generation around
one shared, frozen vision backbone trained once, on a corpus and hardware
budget a single student can actually complete, with explainability
instrumented across all three stages rather than retrofitted onto only
the classifier.

The most directly comparable recent papers are **GLoRIA** (the direct
source of Stage 1's global+local loss design), **CheXagent** (the closest
system in overall scope — classification and generation from one
backbone — at a scale MedXVLM deliberately does not attempt to match),
and **RadZero** (2025), whose combination of zero-shot alignment with
built-in, similarity-based explainability is the closest existing target
for what MedXVLM's Stage 1 representation is meant to support once
Grad-CAM/attention-rollout/text-grounded heatmaps are layered on top.
