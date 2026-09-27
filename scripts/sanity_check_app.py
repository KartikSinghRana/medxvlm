#!/usr/bin/env python3
"""Headless functional check of the Streamlit demo (app/app.py + app/app_pages/).

Drives the real app through `streamlit.testing.v1.AppTest` -- no browser, no
server -- and asserts that each user path renders without an exception and
produces the elements a user would see:

  1. cold start: page title, disclaimer line, input source control
  2. upload source -> uploader shown, clean empty state, no test-image override
  3. held-out test image -> findings badges + table, heatmaps
  4. heatmap methods: attention, Grad-CAM and text-grounded overlays all render
  5. report (v2): generator loads on "Generate report", text + radiologist
     reference render, "Show sentence attention" renders attribution panels
  6. report (v2c): switching model flips decoding to greedy, the report shows
     the findings given to the writer
  7. model card page renders its tabs, metrics and charts

Steps 5-6 load the ~7 GB decoder; skip them with --no-generator.
Prints PASS/FAIL per check, exit code 1 on any failure.

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


def _markdown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def _captions(at: AppTest) -> str:
    return " ".join(c.value for c in at.caption)


def check_cold_start(at: AppTest) -> None:
    _fail_on_exception(at, "cold start")
    assert any("Chest X-ray analysis" in t.value for t in at.title), "page title missing"
    assert "not a medical device" in _captions(at), "disclaimer line missing"
    assert at.segmented_control(key="source") is not None, "input source control missing"
    print("[PASS] cold start: title, disclaimer line and input source render")


def check_upload_path(at: AppTest) -> None:
    at.segmented_control(key="source").set_value("upload").run(timeout=60)
    _fail_on_exception(at, "upload source selected")
    assert "Upload a frontal chest X-ray" in _captions(at), "upload empty state missing"
    assert not at.selectbox, "test-image selectbox should not render on the upload path"
    at.segmented_control(key="source").set_value("test").run(timeout=120)
    _fail_on_exception(at, "switch back to test images")
    print("[PASS] upload path: uploader + empty state, no test-image override")


def check_test_image_path(at: AppTest) -> None:
    assert at.selectbox, "test-image selectbox missing (is data/processed/manifest.csv present?)"
    at.selectbox(key="test_image").select_index(0).run(timeout=120)
    _fail_on_exception(at, "test image selected")
    subheaders = " ".join(s.value for s in at.subheader)
    for section in ("Findings", "Where the model looked", "Findings report"):
        assert section in subheaders, f"section {section!r} missing: {subheaders}"
    assert at.dataframe, "findings table missing"
    assert "Model flags" in _captions(at) and "Report label (automatic)" in _captions(at), "findings summary missing"
    table = at.dataframe[0].value
    assert set(table["Model"]) <= {"positive", "negative"}, f"model calls should be positive/negative: {set(table['Model'])}"
    assert "Percentile" in table.columns and "Report label" in table.columns, f"columns: {list(table.columns)}"
    n_bal = int((table["Model"] == "positive").sum())
    at.segmented_control(key="operating_point").set_value("sensitive").run(timeout=120)
    _fail_on_exception(at, "sensitive operating point")
    n_sens = int((at.dataframe[0].value["Model"] == "positive").sum())
    at.segmented_control(key="operating_point").set_value("balanced").run(timeout=120)
    labels = at.multiselect[0].value
    print(f"[PASS] test-image path: summary + table; positives balanced {n_bal} -> sensitive {n_sens}; {len(labels)} default heatmap findings")


def check_heatmap_methods(at: AppTest) -> None:
    for method in ("gradcam", "text", "attention"):
        at.segmented_control(key="heatmap_method").set_value(method).run(timeout=180)
        _fail_on_exception(at, f"heatmap method {method}")
        assert len(at.get("image")) >= 3, f"{method}: heatmaps missing (x-ray + >=2 overlays expected)"
        assert not [w for w in at.warning if "checkpoint for this method is missing" in w.value], f"{method}: checkpoint missing"
    print("[PASS] heatmap methods: attention, Grad-CAM and text-grounded overlays all render")


def check_report_v2(at: AppTest) -> None:
    at.button(key="generate").click().run(timeout=300)
    _fail_on_exception(at, "generate report (v2)")
    captions = _captions(at)
    assert "MedXVLM v2, beam search" in captions, "generated report header missing"
    assert any(c.value == "Radiologist" for c in at.caption), "radiologist reference missing"
    at.toggle(key="explain_sentences").set_value(True).run(timeout=300)
    _fail_on_exception(at, "show sentence attention")
    assert "Each map shows where one sentence looked" in _captions(at), "attribution explanation missing"
    print("[PASS] report v2: decoder loaded on demand, report + reference + sentence attribution rendered")


def check_report_v2c(at: AppTest) -> None:
    at.segmented_control(key="report_model").set_value("v2c").run(timeout=120)
    _fail_on_exception(at, "switch to v2c")
    assert at.segmented_control(key="decoding").value == "greedy", "decoding did not switch to greedy for v2c"
    at.button(key="generate").click().run(timeout=300)
    _fail_on_exception(at, "generate report (v2c)")
    assert "Findings given to the writer" in _captions(at), "v2c prompt findings missing"
    print("[PASS] report v2c: greedy decoding auto-selected, report + prompt findings rendered")


def check_model_card(at: AppTest) -> None:
    at.switch_page("app_pages/model_card.py").run(timeout=120)
    _fail_on_exception(at, "model card page")
    assert any("Model card" in t.value for t in at.title), "model card title missing"
    assert len(at.tabs) == 4, f"expected 4 tabs, got {len(at.tabs)}"
    assert len(at.metric) >= 10, f"expected >=10 metrics, got {len(at.metric)}"
    assert len(at.get("vega_lite_chart")) >= 4, "charts missing"
    print(f"[PASS] model card: {len(at.tabs)} tabs, {len(at.metric)} metrics, {len(at.get('vega_lite_chart'))} charts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-generator", action="store_true", help="Skip the ~7 GB decoder checks.")
    args = ap.parse_args()

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    checks = [check_cold_start, check_upload_path, check_test_image_path, check_heatmap_methods]
    if not args.no_generator:
        checks += [check_report_v2, check_report_v2c]
    checks.append(check_model_card)
    failures = 0
    for check in checks:
        try:
            check(at)
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {check.__name__}: {exc}")
    print("OK: all app checks passed." if not failures else f"FAILED: {failures} check(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
