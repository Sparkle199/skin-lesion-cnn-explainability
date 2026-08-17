"""Grad-CAM wrapper, applied per spec to each trained model after training.

Works against the (model, base) pair returned by src.models.build.build_model: the
gradient of the predicted class score is taken with respect to the backbone's last
conv/activation layer (src.config.LAST_CONV_LAYER), then the classification head
(`model`'s final Dense layer) is re-applied to the backbone's pooled output to compute
that class score -- this avoids needing the backbone's internal tensors to be exposed
through the outer wrapped model's graph.
"""

import numpy as np
import tensorflow as tf

from src import config
from src.models.build import preprocess_for


def make_gradcam_heatmap(
    model: tf.keras.Model,
    base: tf.keras.Model,
    architecture: str,
    raw_images: tf.Tensor,
    pred_index: int | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image.

    raw_images: a (1, H, W, 3) batch in raw pixel scale (0-255) -- preprocessing for
    the given architecture is applied internally, matching how the model itself
    preprocesses input in src.models.build.build_model.
    pred_index: which class to explain. Defaults to the model's own top prediction for
    the seven-class task; ignored for the binary task, which has a single sigmoid unit.
    """
    last_conv_layer_name = config.LAST_CONV_LAYER[architecture]
    grad_model = tf.keras.Model(base.input, [base.get_layer(last_conv_layer_name).output, base.output])
    dense = model.get_layer("predictions")

    preprocessed = preprocess_for(architecture, raw_images)

    with tf.GradientTape() as tape:
        conv_output, pooled_features = grad_model(preprocessed)
        predictions = dense(pooled_features)
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resize `heatmap` to `image`'s size and blend it over the image (jet colormap)
    for visual inspection, per the spec's faithfulness assessment."""
    import matplotlib.cm as cm

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], image.shape[:2]).numpy().squeeze()
    colored = cm.jet(heatmap_resized)[..., :3]
    image_float = image.astype("float32") / 255.0 if image.max() > 1.0 else image.astype("float32")
    blended = colored * alpha + image_float * (1 - alpha)
    return (np.clip(blended, 0, 1) * 255).astype("uint8")
