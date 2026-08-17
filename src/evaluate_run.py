"""Evaluation entrypoint: loads a trained model, predicts on its task's validation set
(plus, for the seven-class task, the ISIC2018 held-out test set), computes the full
metric suite via src.evaluate, and -- for a sample of HAM10000-derived predictions --
the quantitative Grad-CAM faithfulness score against ground-truth lesion masks.

Usage: python -m src.evaluate_run --architecture resnet50 --task seven_class \
           --model-path models/resnet50_seven_class.keras

Requires TensorFlow and a model already trained via src.train; intended to run on
Kaggle GPU infrastructure per the proposal, not locally (see agent journal).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src import config
from src.data import ham10000
from src.data.pipeline import make_dataset
from src.evaluate import evaluate_predictions, stratified_binary_metrics
from src.json_utils import numpy_json_default
from src.train import TASKS, prepare_data
from src.xai.faithfulness import mean_overlap
from src.xai.gradcam import make_gradcam_heatmap


def predict_dataset(model: tf.keras.Model, dataset: tf.data.Dataset, num_classes: int):
    """Run inference over an unshuffled dataset and return (y_true, y_pred, y_proba) as
    integer class indices / probabilities, matching src.evaluate.evaluate_predictions.
    """
    proba_batches, label_batches = [], []
    for images, labels in dataset:
        proba_batches.append(model.predict(images, verbose=0))
        label_batches.append(labels.numpy())

    y_proba = np.concatenate(proba_batches)
    labels_arr = np.concatenate(label_batches)

    if num_classes == 2:
        y_true = labels_arr.astype(int)
        y_proba = y_proba.reshape(-1)
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_true = np.argmax(labels_arr, axis=1)
        y_pred = np.argmax(y_proba, axis=1)

    return y_true, y_pred, y_proba


def run_faithfulness_check(
    model: tf.keras.Model,
    base: tf.keras.Model,
    architecture: str,
    ham_rows,
    n_samples: int = 30,
    seed: int = config.RANDOM_SEED,
):
    """Sample up to n_samples HAM10000-derived rows and score Grad-CAM heatmaps against
    their ground-truth lesion segmentation masks. Returns None if ham_rows is empty
    (e.g. no HAM10000 rows in a DDI-only sample -- DDI has no equivalent ground truth,
    per spec).
    """
    if len(ham_rows) == 0:
        return None

    sample = ham_rows.sample(n=min(n_samples, len(ham_rows)), random_state=seed)
    image_size = config.IMAGE_SIZE[architecture]

    heatmaps, masks = [], []
    for _, row in sample.iterrows():
        raw = tf.io.read_file(row["image_path"])
        image = tf.io.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, image_size)[tf.newaxis, ...]

        heatmaps.append(make_gradcam_heatmap(model, base, architecture, image))
        masks.append(ham10000.load_segmentation_mask(row["image_id"]))

    return mean_overlap(heatmaps, masks)


def evaluate(architecture: str, task_name: str, model_path: str, faithfulness_samples: int = 30) -> dict:
    task = TASKS[task_name]
    classes, label_col = task["classes"], task["label_col"]
    image_size = config.IMAGE_SIZE[architecture]
    num_classes = len(classes)

    model = tf.keras.models.load_model(model_path)

    _, val_df = prepare_data(task_name)
    val_ds = make_dataset(val_df, label_col, classes, image_size, training=False)
    y_true, y_pred, y_proba = predict_dataset(model, val_ds, num_classes)

    results = {"architecture": architecture, "task": task_name}
    results["validation"] = evaluate_predictions(y_true, y_pred, y_proba, classes)

    if task_name == "seven_class":
        isic_df = ham10000.load_isic2018_test()
        isic_ds = make_dataset(isic_df, "dx", classes, image_size, training=False)
        isic_true, isic_pred, isic_proba = predict_dataset(model, isic_ds, num_classes)
        results["isic2018_test"] = evaluate_predictions(isic_true, isic_pred, isic_proba, classes)
        ham_rows = val_df
    else:
        val_df = val_df.reset_index(drop=True)
        val_df["y_true"], val_df["y_pred"] = y_true, y_pred
        results["skin_tone_stratified"] = stratified_binary_metrics(val_df, "y_true", "y_pred")
        ham_rows = val_df[val_df["source"] == "ham10000"]

    base = model.get_layer(config.BACKBONE_LAYER_NAME[architecture])
    faithfulness = run_faithfulness_check(model, base, architecture, ham_rows, faithfulness_samples)
    if faithfulness is not None:
        results["faithfulness"] = faithfulness

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=list(config.IMAGE_SIZE))
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--faithfulness-samples", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = evaluate(args.architecture, args.task, args.model_path, args.faithfulness_samples)

    output_path = Path(args.output or f"results/{args.architecture}_{args.task}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=numpy_json_default)
    print(f"Wrote evaluation results to {output_path}")


if __name__ == "__main__":
    main()
