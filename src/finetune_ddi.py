"""Stage 2 of the sequential DDI methodology: fine-tune an already-trained HAM10000-only
binary model (src.train, task=binary_ham_only) further on DDI alone.

Per the revised design (journal, 2026-08-17 entry): rather than mixing DDI into binary
training from the first batch (the original "binary" task, still kept as a comparison
arm), get a solid HAM10000-only binary baseline first, measure its zero-shot
generalisation to DDI (src.evaluate_ddi's zero-shot mode), and only then adapt it to
DDI specifically -- as a genuine fine-tuning step on top of an already-converged model,
at a low learning rate, rather than training from scratch on a mixed corpus.

Usage: python -m src.finetune_ddi --architecture resnet50 \
           --base-model-path models/resnet50_binary_ham_only.keras

Tier 2 additions (journal, 2026-08-28 entry) -- two optional ablations, off by default
so the baseline (no flags) reproduces the original committed
models/{arch}_binary_ddi_finetuned.keras exactly:
- --class-weight: apply class weighting to the DDI fine-tuning loss (DDI's train split
  is ~74%/26% benign/malignant; the baseline run applied none, unlike src.train's
  HAM10000 training, which always does).
- --unfreeze-batchnorm: let BatchNormalization layers within the already-unfrozen top
  `--unfreeze-layers` backbone layers train during fine-tuning, instead of staying
  frozen (src.models.build.unfreeze_top_layers freezes BN even in the unfrozen region;
  this means every prior fine-tuning run left BN running statistics calibrated to
  HAM10000, never adapting to DDI's different imaging distribution -- named as the
  leading unconfirmed hypothesis for the calibration-shift finding in the 2026-08-18
  entries, tested here for the first time).

Requires TensorFlow and DDI's extracted images (see src/config.py); intended to run on
the same pod as training, not locally.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src import config
from src.data import ddi
from src.data.augmentation import compute_class_weights
from src.data.pipeline import make_dataset
from src.json_utils import numpy_json_default

BINARY_CLASSES = ["benign", "malignant"]


def _unfreeze_batchnorm_in_top_layers(base: tf.keras.Model, num_layers: int) -> None:
    """Let BatchNormalization layers within the top `num_layers` backbone layers train,
    reversing the freeze src.models.build.unfreeze_top_layers applied when the base
    model was originally built. Deliberately scoped to only the same top region already
    unfrozen for conv weights -- the deeper, still-frozen layers' BN statistics have no
    reason to drift independently of their (frozen) conv weights."""
    for layer in base.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def finetune(
    architecture: str,
    base_model_path: str,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-6,
    use_class_weight: bool = False,
    unfreeze_batchnorm: bool = False,
    unfreeze_layers: int = 30,
):
    """Continue training an already-trained HAM10000-only binary model on DDI's own
    train split only (ddi.split_ddi's train half) -- DDI's validation half stays fully
    held out, so src.evaluate_ddi's fine-tuned-model evaluation on it is a genuine
    generalisation check, not a re-check of training-set performance.

    learning_rate defaults far lower than either of src.train's two phases (1e-3 frozen
    warmup, 1e-5 fine-tune) -- this model has already converged on a related but
    distinct distribution (HAM10000), and DDI is small (656 images total, ~557 in this
    train split); a higher rate risks catastrophic forgetting of what HAM10000 taught it
    rather than a gentle adaptation.
    """
    image_size = config.IMAGE_SIZE[architecture]

    ddi_df = ddi.load_metadata()
    ddi_train, ddi_val = ddi.split_ddi(ddi_df)

    train_ds = make_dataset(ddi_train, "binary_label", BINARY_CLASSES, image_size, batch_size, training=True)
    val_ds = make_dataset(ddi_val, "binary_label", BINARY_CLASSES, image_size, batch_size, training=False)

    model = tf.keras.models.load_model(base_model_path)

    if unfreeze_batchnorm:
        base = model.get_layer(config.BACKBONE_LAYER_NAME[architecture])
        _unfreeze_batchnorm_in_top_layers(base, unfreeze_layers)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

    class_weight = None
    if use_class_weight:
        weight_by_name = compute_class_weights(ddi_train["binary_label"].tolist())
        class_weight = {BINARY_CLASSES.index(name): w for name, w in weight_by_name.items()}

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight)

    return model, {"loss": history.history["loss"], "val_loss": history.history["val_loss"],
                    "accuracy": history.history["accuracy"], "val_accuracy": history.history["val_accuracy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=list(config.IMAGE_SIZE))
    parser.add_argument("--base-model-path", required=True, help="Path to a models/{arch}_binary_ham_only.keras")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--class-weight", action="store_true", help="Apply class weighting to the fine-tuning loss (off by default, matching the original baseline run).")
    parser.add_argument("--unfreeze-batchnorm", action="store_true", help="Let BatchNorm layers in the top --unfreeze-layers train during fine-tuning (off by default).")
    parser.add_argument("--unfreeze-layers", type=int, default=30, help="How many top backbone layers to consider 'unfrozen' for --unfreeze-batchnorm's scope.")
    parser.add_argument("--output-suffix", default=None, help="Appended to the output model/history filenames, e.g. '_cw_bn'. Defaults to auto-derived from the flags used.")
    args = parser.parse_args()

    model, history = finetune(
        args.architecture, args.base_model_path, args.epochs, args.batch_size, args.learning_rate,
        args.class_weight, args.unfreeze_batchnorm, args.unfreeze_layers,
    )

    suffix = args.output_suffix
    if suffix is None:
        suffix = ("_cw" if args.class_weight else "") + ("_bn" if args.unfreeze_batchnorm else "")

    out_path = f"models/{args.architecture}_binary_ddi_finetuned{suffix}.keras"
    model.save(out_path)
    print(f"Saved DDI-fine-tuned model to {out_path}")

    history_path = Path(f"results/history_{args.architecture}_binary_ddi_finetuned{suffix}.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=numpy_json_default)
    print(f"Saved fine-tuning history to {history_path}")


if __name__ == "__main__":
    main()
