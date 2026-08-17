"""Shared data/model loading for the Streamlit demo (Stage 10, docs/pipeline/10-streamlit-app.md).

Reads only pre-computed artifacts produced by earlier pipeline stages (models/*.keras
from Stage 6, results/*.json from Stage 7 and 9) -- this module does no training and no
SHAP (too slow for interactive use, per the Stage 10 doc). Grad-CAM is cheap enough
(single backward pass) to compute live for an uploaded image.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from src import config
from src.train import TASKS
from src.xai.gradcam import make_gradcam_heatmap, overlay_heatmap

ARCHITECTURES = list(config.IMAGE_SIZE)
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
DATA_ROOT = config.DATA_ROOT


@st.cache_resource(show_spinner="Loading model...")
def load_model(architecture: str, task: str):
    """Load a trained model and its backbone submodel (for Grad-CAM), cached per
    (architecture, task) so re-selecting the same pair doesn't reload from disk."""
    model_path = MODELS_DIR / f"{architecture}_{task}.keras"
    if not model_path.exists():
        return None, None
    model = tf.keras.models.load_model(model_path)
    base = model.get_layer(config.BACKBONE_LAYER_NAME[architecture])
    return model, base


@st.cache_data(show_spinner=False)
def load_results(architecture: str, task: str) -> dict | None:
    path = RESULTS_DIR / f"{architecture}_{task}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_trade_off() -> dict | None:
    path = RESULTS_DIR / "trade_off_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def available_runs() -> list[tuple[str, str]]:
    """(architecture, task) pairs that actually have a saved model on disk."""
    return [
        (a, t)
        for a in ARCHITECTURES
        for t in TASKS
        if (MODELS_DIR / f"{a}_{t}.keras").exists()
    ]


@st.cache_data(show_spinner=False)
def sample_images(task: str, n: int = 6) -> list[dict]:
    """A handful of real images from the extracted dataset for the "try a sample"
    picker, so the demo works without requiring an upload. Returns
    [{"label": ..., "path": ...}, ...]. Empty if the data isn't present (e.g. running
    this app somewhere other than the training pod)."""
    from src.data import ham10000

    if not config.HAM10000_IMAGE_DIR.exists():
        return []

    df = ham10000.load_metadata()
    picked = df.sample(n=min(n, len(df)), random_state=config.RANDOM_SEED)
    return [
        {"label": f"{row.image_id} ({row.dx})", "path": row.image_path}
        for row in picked.itertuples()
    ]


def predict_and_explain(model, base, architecture: str, task: str, image: np.ndarray):
    """image: raw pixel-scale (0-255) HxWx3 array, any size -- resized internally to
    the architecture's expected input. Returns (class_names, probabilities, overlay).
    """
    classes = TASKS[task]["classes"]
    image_size = config.IMAGE_SIZE[architecture]

    resized = tf.image.resize(image, image_size)
    batch = resized[tf.newaxis, ...]

    proba = model.predict(batch, verbose=0)[0]
    if len(classes) == 2:
        probabilities = np.array([1 - proba[0], proba[0]])
    else:
        probabilities = proba

    heatmap = make_gradcam_heatmap(model, base, architecture, batch)
    overlay = overlay_heatmap(resized.numpy().astype("uint8"), heatmap)

    return classes, probabilities, overlay
