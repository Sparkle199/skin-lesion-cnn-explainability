"""Training entrypoint for one (architecture, task) pair.

Usage: python -m src.train --architecture resnet50 --task seven_class
       python -m src.train --architecture efficientnetb4 --task binary

Requires TensorFlow and the extracted/mounted datasets (see src/config.py for expected
paths); intended to run on Kaggle GPU infrastructure per the proposal, not locally.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src import config
from src.data import binary_corpus, ddi, ham10000
from src.data.augmentation import compute_class_weights
from src.data.oversampling import compute_steps_per_epoch
from src.data.pipeline import make_dataset, make_oversampled_binary_dataset
from src.json_utils import numpy_json_default
from src.models.build import build_model, unfreeze_top_layers

TASKS = {
    "seven_class": {"classes": config.SEVEN_CLASSES, "label_col": "dx"},
    "binary": {"classes": ["benign", "malignant"], "label_col": "binary_label"},
    # HAM10000-only binary pretraining stage, per the revised DDI methodology (see
    # spec/specification.md and journal, 2026-08-17): train a solid HAM10000 binary
    # baseline first, then either (a) evaluate it zero-shot on DDI as a pure
    # generalisation/bias stress test (src/evaluate_ddi.py), or (b) fine-tune it
    # further on DDI (src/finetune_ddi.py) -- rather than mixing DDI into training
    # from the first batch, as the original "binary" task above still does. Both
    # "binary" and "binary_ham_only" are kept: the joint-mixed approach isn't
    # discarded, it becomes one comparison arm alongside the sequential one.
    "binary_ham_only": {"classes": ["benign", "malignant"], "label_col": "binary_label"},
}

# Columns kept when relabelling HAM10000 alone to binary, matching
# src.data.binary_corpus's own column set so a "binary_ham_only" DataFrame is
# structurally identical to the HAM10000 half of a joint "binary" corpus (same
# columns evaluate_run.py and src.evaluate expect: image_path, binary_label, source,
# skin_tone_group, image_id).
_HAM_ONLY_BINARY_COLUMNS = ["image_path", "binary_label", "source", "skin_tone_group", "image_id"]


def prepare_data(task_name: str):
    """Return (train_df, val_df) for the requested task.

    All tasks start from the same lesion-level HAM10000 split, so a lesion never
    appears in one task's training set and another task's validation set.
    """
    ham_df = ham10000.load_metadata()
    train_df, val_df = ham10000.lesion_level_split(ham_df)

    if task_name == "seven_class":
        return train_df, val_df

    if task_name == "binary":
        ddi_df = ddi.load_metadata()
        ddi_train, ddi_val = ddi.split_ddi(ddi_df)
        train_binary = binary_corpus.build_binary_corpus(train_df, ddi_train)
        val_binary = binary_corpus.build_binary_corpus(val_df, ddi_val)
        return train_binary, val_binary

    if task_name == "binary_ham_only":
        train_binary = binary_corpus.relabel_ham10000_binary(train_df)[_HAM_ONLY_BINARY_COLUMNS]
        val_binary = binary_corpus.relabel_ham10000_binary(val_df)[_HAM_ONLY_BINARY_COLUMNS]
        return train_binary, val_binary

    raise ValueError(f"Unknown task '{task_name}', expected one of {list(TASKS)}")


def train(
    architecture: str,
    task_name: str,
    epochs_frozen: int = 5,
    epochs_finetune: int = 10,
    unfreeze_layers: int = 30,
    batch_size: int = 32,
    ddi_fraction: float = 0.3,
):
    """Two-phase transfer learning: train the frozen-backbone head, then fine-tune the
    top `unfreeze_layers` backbone layers at a lower learning rate.

    For the binary task, training batches are drawn from HAM10000 and DDI at a fixed
    `ddi_fraction` via make_oversampled_binary_dataset, rather than plain shuffling of
    the combined corpus -- DDI is only ~6% of that corpus by volume, so an unweighted
    shuffle would barely feature it per batch, diluting the reason it was added (see
    agent journal, 2026-08-10 entry). Validation always uses the real, un-oversampled
    distribution (make_dataset), since oversampling is a training-only correction.
    """
    task = TASKS[task_name]
    classes, label_col = task["classes"], task["label_col"]
    image_size = config.IMAGE_SIZE[architecture]

    train_df, val_df = prepare_data(task_name)
    val_ds = make_dataset(val_df, label_col, classes, image_size, batch_size, training=False)

    if task_name == "binary":
        train_ds = make_oversampled_binary_dataset(train_df, image_size, batch_size, ddi_fraction)
        n_ham10000 = (train_df["source"] == "ham10000").sum()
        steps_per_epoch = compute_steps_per_epoch(n_ham10000, batch_size, ddi_fraction)
    else:
        train_ds = make_dataset(train_df, label_col, classes, image_size, batch_size, training=True)
        steps_per_epoch = None

    class_weight_by_name = compute_class_weights(train_df[label_col].tolist())
    class_weight = {classes.index(name): w for name, w in class_weight_by_name.items()}

    model, base = build_model(architecture, num_classes=len(classes), task_name=task_name)
    loss = "binary_crossentropy" if len(classes) == 2 else "categorical_crossentropy"

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_frozen,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight,
    )

    unfreeze_top_layers(base, unfreeze_layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss, metrics=["accuracy"])
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight,
    )

    # Both phases' per-epoch metrics, concatenated with a "phase" marker per epoch --
    # history_frozen's own History object was previously discarded entirely (its fit()
    # return value was never captured), so only the fine-tune phase's 10 epochs were
    # ever visible to callers, even though both phases print to stdout. Fixed here so
    # main() can save the full 15-epoch (5 frozen + 10 fine-tune) curve to disk, rather
    # than that data only existing as unstructured training-log text.
    history = {"phase": ["frozen"] * epochs_frozen + ["finetune"] * epochs_finetune}
    for key in history_finetune.history:
        history[key] = history_frozen.history[key] + history_finetune.history[key]

    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=list(config.IMAGE_SIZE))
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--epochs-frozen", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--ddi-fraction",
        type=float,
        default=0.3,
        help="Target share of each training batch drawn from DDI for the binary task (ignored for seven_class).",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Appended to the output model/history filenames, e.g. '_seed2' -- for "
        "re-running the same (architecture, task) multiple times without overwriting "
        "the original (Tier 3 multi-seed variance check; no global TF seed is set "
        "anywhere in this pipeline, so each invocation gets different weight "
        "initialisation/dropout even though the data split and shuffle order stay "
        "identical, since those are seeded separately -- see journal, 2026-08-28 entry).",
    )
    args = parser.parse_args()

    model, history = train(
        args.architecture,
        args.task,
        args.epochs_frozen,
        args.epochs_finetune,
        args.unfreeze_layers,
        args.batch_size,
        args.ddi_fraction,
    )
    out_path = f"models/{args.architecture}_{args.task}{args.output_suffix}.keras"
    model.save(out_path)
    print(f"Saved trained model to {out_path}")

    history_path = Path(f"results/history_{args.architecture}_{args.task}{args.output_suffix}.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=numpy_json_default)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
