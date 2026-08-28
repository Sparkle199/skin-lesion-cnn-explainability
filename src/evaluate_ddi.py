"""Evaluation for the DDI methodology's three comparison checkpoints (journal,
2026-08-17/2026-08-19 entries):

- --mode zero_shot: how well does a HAM10000-only binary model (never trained on any
  DDI image) generalise to DDI's imaging modality and skin-tone distribution? Uses
  DDI's *entire* 656 images -- none of it was used for training this model, so none of
  it needs holding back, unlike the two modes below.
- --mode finetuned: how does the DDI-fine-tuned model perform on DDI's own held-out
  validation split (untouched by src.finetune_ddi's training), and has fine-tuning on
  DDI cost it HAM10000 binary performance (a retention/forgetting check)?
- --mode joint: the *original* joint/mixed model (src.train, task=binary,
  DDI-oversampled into training from the first batch) scored on the exact same DDI
  held-out validation split as --mode finetuned uses (same ddi.split_ddi call, same
  seed -- build_binary_corpus's val split is constructed from that identical ddi_val),
  so all three approaches are finally comparable on the same held-out DDI images
  instead of the joint model's only-ever-reported number being a HAM10000+DDI-mixed
  overall figure that isn't directly comparable to the other two modes' DDI-only
  numbers.

Neither mode computes Grad-CAM faithfulness: DDI carries no ground-truth segmentation
mask (per spec/specification.md), so its faithfulness assessment stays visual-only,
generated separately (Stage 8, not this script).

Usage:
  python -m src.evaluate_ddi --architecture resnet50 --mode zero_shot \
      --model-path models/resnet50_binary_ham_only.keras
  python -m src.evaluate_ddi --architecture resnet50 --mode finetuned \
      --model-path models/resnet50_binary_ddi_finetuned.keras
  python -m src.evaluate_ddi --architecture resnet50 --mode joint \
      --model-path models/resnet50_binary.keras
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src import config
from src.data import ddi
from src.data.pipeline import make_dataset
from src.evaluate import evaluate_predictions, stratified_binary_metrics
from src.evaluate_run import predict_dataset
from src.json_utils import numpy_json_default
from src.train import prepare_data

BINARY_CLASSES = ["benign", "malignant"]


def _evaluate_on(model, df, image_size, batch_size=32):
    ds = make_dataset(df, "binary_label", BINARY_CLASSES, image_size, batch_size, training=False)
    y_true, y_pred, y_proba = predict_dataset(model, ds, num_classes=2)
    metrics = evaluate_predictions(y_true, y_pred, y_proba, BINARY_CLASSES)

    df = df.reset_index(drop=True)
    df["y_true"], df["y_pred"] = y_true, y_pred
    if "skin_tone_group" in df.columns:
        strat = stratified_binary_metrics(df, "y_true", "y_pred")
        if strat:
            metrics["skin_tone_stratified"] = strat
    return metrics


def evaluate_zero_shot(architecture: str, model_path: str) -> dict:
    """The HAM10000-only model, never exposed to DDI, scored on DDI's full 656 images
    (a pure held-out generalisation/bias stress test) and on its own HAM10000 binary
    validation split, for a direct side-by-side of in-distribution vs. out-of-
    distribution performance in one result file."""
    image_size = config.IMAGE_SIZE[architecture]
    model = tf.keras.models.load_model(model_path)

    ddi_df = ddi.load_metadata()
    _, ham_val = prepare_data("binary_ham_only")

    return {
        "architecture": architecture,
        "mode": "zero_shot",
        "ham10000_validation": _evaluate_on(model, ham_val, image_size),
        "ddi_full_zero_shot": _evaluate_on(model, ddi_df, image_size),
    }


def evaluate_finetuned(architecture: str, model_path: str) -> dict:
    """The DDI-fine-tuned model, scored on DDI's held-out validation split (the same
    split src.finetune_ddi excluded from its own training data) and on HAM10000's
    binary validation split, to check whether fine-tuning on DDI cost it HAM10000
    performance."""
    image_size = config.IMAGE_SIZE[architecture]
    model = tf.keras.models.load_model(model_path)

    ddi_df = ddi.load_metadata()
    _, ddi_val = ddi.split_ddi(ddi_df)
    _, ham_val = prepare_data("binary_ham_only")

    return {
        "architecture": architecture,
        "mode": "finetuned",
        "ham10000_validation_retention": _evaluate_on(model, ham_val, image_size),
        "ddi_held_out_validation": _evaluate_on(model, ddi_val, image_size),
    }


def evaluate_joint_on_ddi_val(architecture: str, model_path: str) -> dict:
    """The original joint/mixed model, scored on exactly the same DDI held-out
    validation split --mode finetuned uses -- the joint model *did* train on DDI, but
    only on ddi_train (via build_binary_corpus/make_oversampled_binary_dataset);
    ddi_val was never in its training data either, so this remains a genuine held-out
    evaluation, not a train-set re-check. Filters prepare_data("binary")'s combined
    validation split down to its DDI rows (source == "ddi") rather than calling
    ddi.split_ddi directly, so this is guaranteed to be exactly the same rows the joint
    model's own validation loop saw as its DDI-sourced validation examples during
    training, not a coincidentally-matching independent split."""
    image_size = config.IMAGE_SIZE[architecture]
    model = tf.keras.models.load_model(model_path)

    _, val_binary = prepare_data("binary")
    ddi_val = val_binary[val_binary["source"] == "ddi"].reset_index(drop=True)

    return {
        "architecture": architecture,
        "mode": "joint",
        "ddi_held_out_validation": _evaluate_on(model, ddi_val, image_size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=list(config.IMAGE_SIZE))
    parser.add_argument("--mode", required=True, choices=["zero_shot", "finetuned", "joint"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "zero_shot":
        results = evaluate_zero_shot(args.architecture, args.model_path)
        default_name = f"{args.architecture}_binary_ham_only_zero_shot_ddi.json"
    elif args.mode == "finetuned":
        results = evaluate_finetuned(args.architecture, args.model_path)
        default_name = f"{args.architecture}_binary_ddi_finetuned.json"
    else:
        results = evaluate_joint_on_ddi_val(args.architecture, args.model_path)
        default_name = f"{args.architecture}_binary_joint_ddi_val.json"

    output_path = Path(args.output or f"results/{default_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=numpy_json_default)
    print(f"Wrote evaluation results to {output_path}")


if __name__ == "__main__":
    main()
