"""Streamlit demo, Page 3 -- accuracy vs. interpretability trade-off.

Renders src.trade_off's own output (results/trade_off_summary.json) rather than
re-deriving or re-wording its conclusion, so this page and the dissertation text can't
drift out of sync with each other (see docs/pipeline/10-streamlit-app.md).
"""

import pandas as pd
import streamlit as st

from src.webapp.common import load_trade_off

st.set_page_config(page_title="Trade-off View", layout="wide")
st.title("Accuracy vs. interpretability trade-off")

trade_off = load_trade_off()
if trade_off is None:
    st.error(
        "results/trade_off_summary.json not found. Run `python -m src.trade_off` first "
        "(see docs/pipeline/09-trade-off-analysis.md)."
    )
    st.stop()

st.markdown(trade_off["summary"].replace("\n", "  \n"))

df = pd.DataFrame(trade_off["comparison_table"]).set_index("architecture")
df.index = df.index.map({"resnet50": "ResNet-50", "efficientnetb4": "EfficientNetB4", "vgg16": "VGG16"})

st.subheader("Seven-class accuracy vs. Grad-CAM faithfulness (IoU)")
st.caption(
    "Grad-CAM only, at the largest sample size available per architecture (n=150 "
    "where present, else n=30) -- see the note below the table for why SHAP isn't "
    "included here and where the full Grad-CAM-vs-SHAP comparison is reported."
)
st.bar_chart(df[["seven_class_accuracy", "faithfulness_iou"]])

st.subheader("Binary task: accuracy vs. skin-tone consistency")
st.caption("Skin-tone accuracy spread = max-minus-min accuracy across DDI's Fitzpatrick groups; smaller means more consistent across skin tones.")
st.bar_chart(df[["binary_accuracy", "binary_skin_tone_accuracy_spread"]])

st.subheader("Full comparison table")
st.dataframe(
    df.style.format(
        {
            "seven_class_accuracy": "{:.3f}",
            "seven_class_kappa": "{:.3f}",
            "isic2018_test_accuracy": "{:.3f}",
            "faithfulness_iou": "{:.3f}",
            "faithfulness_dice": "{:.3f}",
            "faithfulness_n": "{:.0f}",
            "binary_accuracy": "{:.3f}",
            "binary_skin_tone_accuracy_spread": "{:.3f}",
        },
        na_rep="—",
    ),
    use_container_width=True,
)
