#!/usr/bin/env python3
"""Validate the local Python / PyTorch / CUDA environment for MedXVLM.

Checks the Python and PyTorch versions, confirms a CUDA-capable GPU is
visible, and attempts to import every package pinned in requirements.txt,
printing a pass/fail line for each. Exits non-zero if CUDA is unavailable
or any package fails to import, so it can be used as a simple setup gate.

Usage:
    python scripts/verify_env.py
"""

from __future__ import annotations

import importlib
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

LABEL_WIDTH = 22

# Distribution name (as pinned in requirements.txt) -> module name used to
# import it, for the cases where the two differ.
DIST_TO_IMPORT_NAME: dict[str, str] = {
    "open_clip_torch": "open_clip",
    "opencv-python-headless": "cv2",
    "scikit-learn": "sklearn",
    "rouge-score": "rouge_score",
    "pillow": "PIL",
    "pyyaml": "yaml",
}


@dataclass(frozen=True)
class PackageCheck:
    """Result of attempting to import a single required package."""

    distribution: str
    import_name: str
    ok: bool
    detail: str


def parse_requirements(requirements_path: Path) -> list[str]:
    """Extract bare distribution names from a pinned requirements file.

    Blank lines, comments, and pip flags (e.g. ``--index-url``) are
    skipped; version pins such as ``torch==2.11.0`` are reduced to
    ``torch``.
    """
    names: list[str] = []
    for raw_line in requirements_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = re.split(r"[=<>~!\s]", line, maxsplit=1)[0]
        if name:
            names.append(name)
    return names


def check_package(distribution: str) -> PackageCheck:
    """Attempt to import one required package and report its version."""
    import_name = DIST_TO_IMPORT_NAME.get(distribution, distribution.replace("-", "_"))
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # broad on purpose: any import-time failure is a FAIL line
        return PackageCheck(distribution, import_name, ok=False, detail=str(exc))
    version = str(getattr(module, "__version__", "unknown version"))
    return PackageCheck(distribution, import_name, ok=True, detail=version)


def _print_row(label: str, value: object) -> None:
    print(f"{label:<{LABEL_WIDTH}}: {value}")


def report_python_and_cuda() -> bool:
    """Print interpreter, PyTorch, and GPU/CUDA details.

    Returns:
        True if a CUDA-capable GPU is visible to PyTorch, False otherwise.
    """
    print("=== Python & PyTorch ===")
    _print_row("Python version", platform.python_version())

    try:
        import torch
    except ImportError as exc:
        _print_row("PyTorch version", f"NOT INSTALLED ({exc})")
        _print_row("CUDA available", "False (torch not importable)")
        return False

    _print_row("PyTorch version", torch.__version__)
    _print_row("PyTorch CUDA build", torch.version.cuda)
    cuda_available = torch.cuda.is_available()
    _print_row("CUDA available", cuda_available)

    if cuda_available:
        device_index = 0
        name = torch.cuda.get_device_name(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        total_mem_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        _print_row("GPU", name)
        _print_row("Compute capability", f"sm_{major}{minor}")
        _print_row("GPU memory (total)", f"{total_mem_gb:.1f} GB")
    else:
        _print_row("GPU", "none visible to PyTorch")

    return cuda_available


def report_package_imports(requirements_path: Path) -> bool:
    """Print a pass/fail import line for every package in requirements.txt.

    Returns:
        True if every listed package imported successfully.
    """
    print()
    print("=== Package imports (requirements.txt) ===")
    results = [check_package(dist) for dist in parse_requirements(requirements_path)]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.distribution:<24} -> {result.import_name:<14} {result.detail}")
    return all(result.ok for result in results)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    requirements_path = repo_root / "requirements.txt"

    cuda_ok = report_python_and_cuda()
    packages_ok = report_package_imports(requirements_path)

    print()
    print("=== Summary ===")
    _print_row("CUDA available", cuda_ok)
    _print_row("All packages import", packages_ok)

    if cuda_ok and packages_ok:
        print("Environment OK.")
        return 0

    print("Environment check FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
