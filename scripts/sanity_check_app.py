#!/usr/bin/env python3
"""Headless functional check of the Streamlit demo (app/app.py).

Drives the real app script through `streamlit.testing.v1.AppTest` -- no
browser, no server -- and asserts that each user path renders without an
exception and produces the elements a user would see:

  1. cold start: title + disclaimer render, no model errors
  2. switch to the upload source -> uploader shown, clean stop, no override
  3. pick a held-out test image -> probabilities chart, ground-truth table,
     attention heatmaps appear
  4. enable report generation -> generator loads, "Generate findings" runs,
     report text appears (skipped with --no-generator to keep it light)

Follows the repo's sanity_check_*.py convention: prints PASS/FAIL per check,
exit code 1 on any failure.

Usage:
    PYTHONPATH=src python scripts/sanity_check_app.py [--no-generator]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = REPO_ROOT / "app" / "app.py"


def _fail_on_exception(at: AppTest, step: str) -> None:
    if at.exception:
        raise AssertionError(f"{step}: app raised {at.exception[0].value!r}\n{at.exception[0].stack_trace}")


def check_cold_start(at: AppTest) -> None:
    _fail_on_exception(at, "cold start")
    assert any("MedXVLM" in t.value for t in at.title), "title missing"
    assert any("not a medical device" in e.value for e in at.error), "disclaimer banner missing"
    assert at.toggle(key=None) is not None or at.toggle, "report toggle missing"
    print("[PASS] cold start renders title, disclaimer and sidebar")


def check_upload_path_renders(at: AppTest) -> None:
    """Switch to the upload source: the uploader must appear and the app must stop cleanly, not crash."""
    at.radio[0].set_value("Upload an X-ray").run(timeout=60)
    _fail_on_exception(at, "upload source selected")
    assert any("Upload an image" in i.value for i in at.info), "upload prompt missing"
    assert not at.selectbox, "test-image selectbox should not render on the upload path"
    at.radio[0].set_value("Held-out test image").run(timeout=120)
    _fail_on_exception(at, "switch back to test images")
    print("[PASS] upload path: uploader renders, no test-image override, clean stop")


def check_test_image_path(at: AppTest) -> None:
    boxes = at.selectbox
    assert boxes, "test-image selectbox missing (is data/processed/manifest.csv present?)"
    boxes[0].select_index(0).run(timeout=120)
    _fail_on_exception(at, "test image selected")
    subheaders = [s.value for s in at.subheader]
    assert any("probabilities" in s for s in subheaders), f"probabilities section missing: {subheaders}"
    assert any("looked" in s for s in subheaders), "explainability section missing"
    assert at.dataframe, "ground-truth comparison table missing"
    captions = " ".join(c.value for c in at.caption)
    assert "Grey bars" in captions, "thin-support caption missing"
    print(f"[PASS] test-image path: chart, ground-truth table and {len(at.multiselect[0].value)} default heatmap labels rendered")


def check_report_generation(at: AppTest) -> None:
    at.toggle[0].set_value(True).run(timeout=180)
    _fail_on_exception(at, "enable report generation")
    buttons = [b for b in at.button if "Generate" in b.label]
    assert buttons, "Generate button missing after enabling generator"
    buttons[0].click().run(timeout=300)
    _fail_on_exception(at, "generate findings")
    body = " ".join(m.value for m in at.markdown) + " ".join(getattr(w, "value", "") for w in at.get("text"))
    assert "Model" in body or any("Reference" in m.value for m in at.markdown), "generated-report section missing"
    print("[PASS] report generation: decoder loaded, findings generated and rendered")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-generator", action="store_true", help="Skip the ~7 GB decoder check.")
    args = ap.parse_args()

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    failures = 0
    for check in (check_cold_start, check_upload_path_renders, check_test_image_path) + (() if args.no_generator else (check_report_generation,)):
        try:
            check(at)
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {check.__name__}: {exc}")
    print("OK: all app checks passed." if not failures else f"FAILED: {failures} check(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
