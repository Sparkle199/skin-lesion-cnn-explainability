"""SHAP wrapper, applied per spec to each trained model after training.

Uses shap.Explainer with an Image masker (Partition algorithm) rather than
DeepExplainer/GradientExplainer: it treats the model as a black-box predict function,
so it works uniformly across all three architectures and both task types without
depending on internal layer access or a specific Keras backend version.
"""

from typing import Protocol

import numpy as np
import shap


class PredictsBatches(Protocol):
    """Anything with a Keras-style .predict(batch, verbose=...) -- deliberately not
    typed as tf.keras.Model, since this module treats the model as a black box and has
    no TensorFlow dependency of its own."""

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray: ...


def build_explainer(
    model: PredictsBatches, image_shape: tuple[int, int, int], classes: list[str]
) -> shap.Explainer:
    """image_shape: (H, W, 3) for the target architecture (see src.config.IMAGE_SIZE).

    No separate preprocessing wrapper is needed here: src.models.build.build_model bakes
    each architecture's preprocessing into the model itself, so `model.predict` already
    accepts raw pixel-scale images directly, which is what the Image masker perturbs.
    """
    masker = shap.maskers.Image("blur(128,128)", image_shape)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, verbose=0)

    return shap.Explainer(predict_fn, masker, output_names=classes)


def explain_images(
    explainer: shap.Explainer, images: np.ndarray, max_evals: int = 500, batch_size: int = 50
):
    """images: (n, H, W, 3) raw pixel-scale array. Returns a shap.Explanation with one
    attribution map per image per class, for visual/faithfulness comparison against
    Grad-CAM (src.xai.gradcam) on the same images."""
    return explainer(images, max_evals=max_evals, batch_size=batch_size)
