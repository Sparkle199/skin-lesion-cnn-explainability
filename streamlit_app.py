"""Streamlit demo, Page 1 -- single-image prediction + live Grad-CAM.

Run from the repository root: streamlit run streamlit_app.py
Reads only models/*.keras (Stage 6 output) -- no training happens here.
"""

import numpy as np
import streamlit as st
from PIL import Image

from src.webapp.common import (
    ARCHITECTURES,
    available_runs,
    load_model,
    predict_and_explain,
    sample_images,
)

st.set_page_config(page_title="Skin Lesion Classifier Demo", layout="wide")
st.title("Skin lesion CNN demo")
st.caption(
    "Experimental artefact, not a diagnostic tool. Predictions are from models trained "
    "for a dissertation project comparing CNN architectures on HAM10000 (+ DDI for the "
    "binary task) -- see the Model Comparison and Trade-off pages for full metrics."
)

runs = available_runs()
if not runs:
    st.error(
        "No trained models found under models/. Run src.train for at least one "
        "(architecture, task) pair first (see docs/pipeline/06-training.md)."
    )
    st.stop()

available_architectures = sorted({a for a, _ in runs})
task_by_arch = {a: sorted({t for arch, t in runs if arch == a}) for a in available_architectures}

col_a, col_b = st.columns(2)
with col_a:
    architecture = st.selectbox(
        "Architecture",
        available_architectures,
        format_func=lambda a: {"resnet50": "ResNet-50", "efficientnetb4": "EfficientNetB4", "vgg16": "VGG16"}.get(a, a),
    )
with col_b:
    task = st.selectbox(
        "Task",
        task_by_arch[architecture],
        format_func=lambda t: "Seven-class" if t == "seven_class" else "Binary (malignant/benign)",
    )

model, base = load_model(architecture, task)
if model is None:
    st.error(f"models/{architecture}_{task}.keras not found.")
    st.stop()

st.divider()

source = st.radio("Image source", ["Sample from dataset", "Upload"], horizontal=True)

image_array = None
image_label = None

if source == "Upload":
    uploaded = st.file_uploader("Upload a dermoscopic image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_array = np.array(Image.open(uploaded).convert("RGB"))
        image_label = uploaded.name
else:
    samples = sample_images(task)
    if not samples:
        st.warning(
            "No local dataset found to sample from (this app isn't running where the "
            "extracted HAM10000 data lives). Use Upload instead."
        )
    else:
        choice = st.selectbox("Pick a sample image", samples, format_func=lambda s: s["label"])
        image_array = np.array(Image.open(choice["path"]).convert("RGB"))
        image_label = choice["label"]

if image_array is not None:
    with st.spinner("Running prediction + Grad-CAM..."):
        classes, probabilities, overlay = predict_and_explain(model, base, architecture, task, image_array)

    img_col, cam_col = st.columns(2)
    with img_col:
        st.image(image_array, caption=f"Input ({image_label})", use_container_width=True)
    with cam_col:
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    st.subheader("Class probabilities")
    order = np.argsort(probabilities)[::-1]
    st.bar_chart({classes[i]: float(probabilities[i]) for i in order})

    top_idx = int(np.argmax(probabilities))
    st.metric("Top prediction", classes[top_idx], f"{probabilities[top_idx]:.1%} confidence")
else:
    st.info("Choose a sample image or upload one to see a prediction.")
