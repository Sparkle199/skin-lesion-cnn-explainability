"""Quantitative XAI faithfulness check for HAM10000-derived predictions, per spec.

Compares a thresholded Grad-CAM/SHAP attention map against the ground-truth lesion
segmentation mask bundled with HAM10000 (src.data.ham10000.load_segmentation_mask),
using IoU and Dice overlap. This check only applies to HAM10000-derived predictions --
DDI carries no equivalent mask, so its faithfulness assessment stays visual-only (see
spec/specification.md and the proposal's Evaluation section).
"""

import numpy as np


def compute_overlap(attention_map: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict:
    """IoU and Dice between a thresholded attention map and the ground-truth mask.

    attention_map: 2D array in [0, 1] (e.g. src.xai.gradcam.make_gradcam_heatmap's
    output, or a single class's SHAP attribution map rescaled to [0, 1]). Resized to
    the mask's resolution if the two don't already match -- Grad-CAM heatmaps are
    produced at the backbone's coarse conv-layer resolution, while SHAP's Image-masker
    attributions are already at the input image's resolution. TensorFlow is only
    imported here, lazily, so the rest of this module's pure-numpy math has no
    TensorFlow dependency of its own.
    """
    if attention_map.shape != mask.shape:
        import tensorflow as tf

        attention_map = (
            tf.image.resize(attention_map[..., np.newaxis], mask.shape[:2]).numpy().squeeze()
        )

    predicted = attention_map >= threshold
    truth = mask.astype(bool)

    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    predicted_area = predicted.sum()
    truth_area = truth.sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / (predicted_area + truth_area)) if (predicted_area + truth_area) > 0 else 0.0

    return {"iou": iou, "dice": dice}


def mean_overlap(attention_maps: list[np.ndarray], masks: list[np.ndarray], threshold: float = 0.5) -> dict:
    """Mean IoU/Dice across a sample of (attention_map, mask) pairs, for reporting a
    single faithfulness score per model/architecture rather than per image."""
    if len(attention_maps) != len(masks):
        raise ValueError("attention_maps and masks must be the same length")

    scores = [compute_overlap(a, m, threshold) for a, m in zip(attention_maps, masks)]
    return {
        "mean_iou": float(np.mean([s["iou"] for s in scores])),
        "mean_dice": float(np.mean([s["dice"] for s in scores])),
        "n": len(scores),
    }
