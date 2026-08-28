"""Exports raw per-example predictions (y_true, y_proba) needed for Tier 1 experiments
(decision-threshold calibration, bootstrap confidence intervals).

src/evaluate.py's saved results only contain aggregated metrics (confusion matrix,
accuracy, kappa at the default 0.5 threshold) -- these cannot be re-thresholded or
resampled after the fact, only raw per-example probabilities can. This script re-runs
inference (no training) for every model needed by scripts/tier1_analysis.py, in one
process, so TensorFlow/XLA warmup is paid once rather than once per model:

- Seven-class validation predictions for all three architectures (for the
  architecture-ranking bootstrap CI).
- DDI held-out validation predictions (the same ddi_val split used by
  src.evaluate_ddi's finetuned/joint modes) for all three binary approaches (joint,
  zero-shot, fine-tuned) and all three architectures -- evaluating the zero-shot model
  on this same held-out subset, rather than the full 656-image DDI set
  src.evaluate_ddi uses for it, so the three-way comparison is on identical examples.

Requires TensorFlow and the extracted datasets; run on the training pod.

Usage: python -m src.export_predictions
"""

import json

import tensorflow as tf

from src import config
from src.data import ddi
from src.data.pipeline import make_dataset
from src.evaluate_run import predict_dataset
from src.json_utils import numpy_json_default
from src.train import prepare_data

ARCHITECTURES = list(config.IMAGE_SIZE)
BINARY_CLASSES = ["benign", "malignant"]


def _predict_raw(model_path: str, df, label_col: str, classes: list[str], image_size):
    model = tf.keras.models.load_model(model_path)
    ds = make_dataset(df, label_col, classes, image_size, batch_size=32, training=False)
    y_true, _, y_proba = predict_dataset(model, ds, num_classes=len(classes))
    return y_true, y_proba


def main():
    out = {}

    _, seven_val = prepare_data("seven_class")
    ddi_df = ddi.load_metadata()
    _, ddi_val = ddi.split_ddi(ddi_df)

    for arch in ARCHITECTURES:
        image_size = config.IMAGE_SIZE[arch]

        y_true, y_proba = _predict_raw(
            f"models/{arch}_seven_class.keras", seven_val, "dx", config.SEVEN_CLASSES, image_size
        )
        out[f"{arch}_seven_class"] = {
            "y_true": y_true.tolist(),
            "y_proba": y_proba.tolist(),
            "classes": config.SEVEN_CLASSES,
        }
        print(f"{arch}: seven_class done ({len(y_true)} examples)")

        for label, model_file in [
            ("joint", f"{arch}_binary.keras"),
            ("zero_shot", f"{arch}_binary_ham_only.keras"),
            ("finetuned", f"{arch}_binary_ddi_finetuned.keras"),
        ]:
            y_true, y_proba = _predict_raw(
                f"models/{model_file}", ddi_val, "binary_label", BINARY_CLASSES, image_size
            )
            out[f"{arch}_ddi_{label}"] = {
                "y_true": y_true.tolist(),
                "y_proba": y_proba.tolist(),
                "classes": BINARY_CLASSES,
            }
            print(f"{arch}: ddi_{label} done ({len(y_true)} examples)")

    with open("results/raw_predictions.json", "w") as f:
        json.dump(out, f, default=numpy_json_default)
    print("Wrote results/raw_predictions.json")


if __name__ == "__main__":
    main()
