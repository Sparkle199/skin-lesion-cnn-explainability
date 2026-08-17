"""Augmentation and class-weighting shared by both tasks (per spec: rotation, flip,
zoom, brightness augmentation + class weighting to address class imbalance)."""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def build_augmentation_layer() -> tf.keras.Sequential:
    """Augmentation pipeline applied during training only (not at eval/inference)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def compute_class_weights(labels: list[str]) -> dict[str, float]:
    """Per-class weights (inverse frequency) for use as `class_weight` in model.fit."""
    classes = sorted(set(labels))
    weights = compute_class_weight(class_weight="balanced", classes=np.array(classes), y=labels)
    return dict(zip(classes, weights))
