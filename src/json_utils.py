"""JSON serialization helper for numpy types, reused wherever evaluation results
(confusion matrices, metric dicts from src.evaluate) get written to disk. No TensorFlow
dependency, unlike most of this project, so it can be tested without it."""

import numpy as np


def numpy_json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Cannot JSON-serialise {type(obj)}")
