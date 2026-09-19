# Environment Validation

**Date:** 2026-08-30
**Command:** `python scripts/verify_env.py` (Conda environment `ml`)

## Machine

| Component | Observed |
|---|---|
| OS | Ubuntu 26.04 LTS on WSL2 (kernel `6.18.33.2-microsoft-standard-WSL2`) |
| CPU | AMD Ryzen 7 5700X3D (16 logical cores visible to WSL2) |
| RAM visible to WSL2 | 15 GiB total (see note below) |
| GPU | NVIDIA GeForce RTX 5080, 16 GB VRAM, compute capability `sm_120` (Blackwell) |
| CUDA driver toolkit (`nvcc`) | 12.6 |

**WSL2 RAM note:** the host machine has 32 GB of physical RAM, but WSL2's
default memory ballooning currently caps the Linux VM at ~15 GiB. This is a
`.wslconfig` setting (`memory=`), not a hardware limit — it only matters if a
later training run needs more host RAM than WSL2 is currently allowed to use
(e.g. large `num_workers` dataloader prefetching), and can be raised in
`%UserProfile%\.wslconfig` on the Windows side if that becomes a bottleneck.

## GPU wheel note

The RTX 5080 is a Blackwell-generation GPU (`sm_120`). The system CUDA
toolkit (`nvcc`, 12.6) is *older* than what the GPU needs for PyTorch's
prebuilt kernels — that number is about the toolkit optionally used to
compile custom CUDA extensions, not about whether PyTorch's own bundled
CUDA runtime supports the GPU. Because PyTorch ships its own CUDA runtime
inside the wheel, what actually matters is installing a wheel built against
CUDA 12.8+ (`sm_120` kernels were only added starting with the cu128
wheel index). This environment has `torch`/`torchvision` installed from the
cu130 index, confirmed working below.

## Result

Exit code: **0** (pass)

```
=== Python & PyTorch ===
Python version        : 3.11.15
PyTorch version        : 2.11.0+cu130
PyTorch CUDA build     : 13.0
CUDA available         : True
GPU                    : NVIDIA GeForce RTX 5080
Compute capability     : sm_120
GPU memory (total)     : 15.9 GB

=== Package imports (requirements.txt) ===
[PASS] torch                    -> torch          2.11.0+cu130
[PASS] torchvision              -> torchvision    0.26.0+cu130
[PASS] timm                     -> timm           1.0.28
[PASS] transformers             -> transformers   5.8.0
[PASS] peft                     -> peft           0.20.0
[PASS] open_clip_torch          -> open_clip      3.3.0
[PASS] opencv-python-headless   -> cv2            5.0.0
[PASS] scikit-learn             -> sklearn        1.8.0
[PASS] sacrebleu                -> sacrebleu      2.6.0
[PASS] rouge-score              -> rouge_score    unknown version
[PASS] nltk                     -> nltk           3.10.2
[PASS] pandas                   -> pandas         2.3.3
[PASS] numpy                    -> numpy          2.4.3
[PASS] pillow                   -> PIL            12.1.1
[PASS] pyyaml                   -> yaml           6.0.3
[PASS] kagglehub                -> kagglehub      1.0.2
[PASS] tensorboard              -> tensorboard    2.21.0
[PASS] streamlit                -> streamlit      1.61.1

=== Summary ===
CUDA available         : True
All packages import    : True
Environment OK.
```

`rouge-score` reports "unknown version" because that package does not
expose `__version__`; the pinned version in `requirements.txt`
(`rouge-score==0.1.2`) is what's installed and importing correctly — this
is a reporting quirk of the library, not a failure.

## Conclusion

The environment satisfies the Phase 1 exit criterion: `scripts/verify_env.py`
passes with exit code 0, CUDA is available, and every pinned dependency in
`requirements.txt` imports successfully against the RTX 5080.
