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

Requires TensorFlow and DDI's extracted images (see src/config.py); intended to run on
the same pod as training, not locally.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src import config
from src.data import ddi
from src.data.pipeline import make_dataset
from src.json_utils import numpy_json_default


def finetune(
    architecture: str,
    base_model_path: str,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-6,
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

    train_ds = make_dataset(ddi_train, "binary_label", ["benign", "malignant"], image_size, batch_size, training=True)
    val_ds = make_dataset(ddi_val, "binary_label", ["benign", "malignant"], image_size, batch_size, training=False)

    model = tf.keras.models.load_model(base_model_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return model, {"loss": history.history["loss"], "val_loss": history.history["val_loss"],
                    "accuracy": history.history["accuracy"], "val_accuracy": history.history["val_accuracy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=list(config.IMAGE_SIZE))
    parser.add_argument("--base-model-path", required=True, help="Path to a models/{arch}_binary_ham_only.keras")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()

    model, history = finetune(
        args.architecture, args.base_model_path, args.epochs, args.batch_size, args.learning_rate
    )

    out_path = f"models/{args.architecture}_binary_ddi_finetuned.keras"
    model.save(out_path)
    print(f"Saved DDI-fine-tuned model to {out_path}")

    history_path = Path(f"results/history_{args.architecture}_binary_ddi_finetuned.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=numpy_json_default)
    print(f"Saved fine-tuning history to {history_path}")


if __name__ == "__main__":
    main()
