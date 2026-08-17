"""Wiring test for src.xai.shap_explain against a stub predict function -- there is no
trained model to test against yet, so this checks that build_explainer/explain_images
run end-to-end and return correctly-shaped output, not that real attributions would be
meaningful (that needs an actual trained model, on Kaggle -- see agent journal)."""

import numpy as np

from src.xai.shap_explain import build_explainer, explain_images

IMAGE_SHAPE = (32, 32, 3)  # small on purpose: this is a wiring check, not a real model
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class StubModel:
    """Deterministic fake softmax scores from mean pixel value, just to exercise the
    masker/explainer plumbing without a real trained network."""

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        means = batch.mean(axis=(1, 2, 3))
        logits = np.stack([means * (i + 1) for i in range(len(CLASSES))], axis=1)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


def test_build_explainer_returns_partition_explainer():
    explainer = build_explainer(StubModel(), IMAGE_SHAPE, CLASSES)

    assert type(explainer).__name__ == "PartitionExplainer"


def test_explain_images_returns_expected_shape():
    explainer = build_explainer(StubModel(), IMAGE_SHAPE, CLASSES)
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(2, *IMAGE_SHAPE)).astype("float32")

    result = explain_images(explainer, images, max_evals=100, batch_size=10)

    values = np.array(result.values)
    assert values.shape[0] == 2
    assert values.shape[-1] == len(CLASSES)
