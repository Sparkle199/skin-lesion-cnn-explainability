"""Streamlit demo, Page 2 -- full model comparison.

Embeds the existing results dashboard (scripts/generate_dashboard.py) rather than
re-implementing the same charts natively in Streamlit -- one source of truth for the
training curves, confusion matrices, and summary table, generated straight from
results/*.json.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model comparison")

dashboard_path = Path("results/training_dashboard.html")
if not dashboard_path.exists():
    st.error(
        "results/training_dashboard.html not found. Run "
        "`python scripts/generate_dashboard.py` first (requires results/epoch_metrics.json "
        "and results/confusion_summary.json -- see docs/pipeline/07-evaluation.md)."
    )
    st.stop()

components.html(dashboard_path.read_text(encoding="utf-8"), height=3200, scrolling=True)
