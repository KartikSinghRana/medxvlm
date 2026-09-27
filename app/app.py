"""MedXVLM demo — entry point.

The proposal's Week-14 deliverable: upload (or pick) a chest X-ray, get the
4-model classifier ensemble's findings, see where it looked, and draft a
findings report. Pages live in `app_pages/`; shared cached resources in
`common.py`; all model code in `medxvlm.inference.MedXVLMPipeline`.

Run from the repo root:
    conda activate ml && PYTHONPATH=src streamlit run app/app.py
Theme: app/.streamlit/config.toml (light + dark).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
# Streamlit re-executes the page scripts on every interaction but keeps imported helper modules cached, so an
# edited app/common.py only took effect after a server restart (an ImportError for any newly added helper).
# Dropping it makes every run import the current file; its st.cache_* functions keep their caches, which
# Streamlit keys by the function's source, so the loaded models are not reloaded.
sys.modules.pop("common", None)

st.set_page_config(page_title="MedXVLM", page_icon=":material/pulmonology:", layout="wide")
st.logo(str(APP_DIR / "static" / "logo.svg"), icon_image=":material/pulmonology:", size="large")

page = st.navigation(
    [
        st.Page("app_pages/analyze.py", title="Analyze an X-ray", icon=":material/radiology:", default=True),
        st.Page("app_pages/model_card.py", title="Model card", icon=":material/monitoring:"),
    ]
)
page.run()
