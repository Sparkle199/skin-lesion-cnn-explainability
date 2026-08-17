"""tf.data pipeline shared by both tasks: reads (image_path, label) pairs from a
DataFrame, decodes/resizes images, one-hot/binary-encodes labels, and applies
augmentation on the training split only.

Also provides make_oversampled_binary_dataset, which corrects for DDI being a small
fraction (~6%) of the binary task's combined HAM10000+DDI corpus by volume -- without
it, per-label class weighting alone leaves DDI's contribution to gradient updates
diluted in proportion to its size, undermining why it was added (see agent journal,
2026-08-10 entry). The oversampling ratio itself is validated in src.data.oversampling,
kept free of a TensorFlow dependency so it can be tested without it.
"""

import pandas as pd
import tensorflow as tf

from src.data.augmentation import build_augmentation_layer
from src.data.oversampling import validate_ddi_fraction

_AUGMENT = build_augmentation_layer()


def _load_image(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=3)
    return tf.image.resize(image, image_size)


def _encode_labels(df: pd.DataFrame, label_col: str, classes: list[str]):
    class_to_index = {name: i for i, name in enumerate(classes)}
    label_indices = df[label_col].map(class_to_index).to_numpy()
    if len(classes) == 2:
        return label_indices.astype("float32")
    return tf.keras.utils.to_categorical(label_indices, num_classes=len(classes))


def _build_example_dataset(
    df: pd.DataFrame,
    label_col: str,
    classes: list[str],
    image_size: tuple[int, int],
    training: bool,
) -> tf.data.Dataset:
    """Unbatched (image, label) dataset: shuffle + load + resize + augment (if
    training). Shared by make_dataset and make_oversampled_binary_dataset so both paths
    decode/augment images identically.
    """
    labels = _encode_labels(df, label_col, classes)
    paths = df["image_path"].to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: (_load_image(path, image_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        dataset = dataset.map(
            lambda image, label: (_AUGMENT(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset


def make_dataset(
    df: pd.DataFrame,
    label_col: str,
    classes: list[str],
    image_size: tuple[int, int],
    batch_size: int = 32,
    training: bool = False,
) -> tf.data.Dataset:
    """Build a batched, shuffled (if training) tf.data.Dataset from a metadata DataFrame.

    Multi-class labels (len(classes) > 2) are one-hot encoded for softmax + categorical
    crossentropy; binary labels are encoded as a single 0/1 float for sigmoid + binary
    crossentropy, matching the two head types in src.models.build.build_model.

    Used as-is (no oversampling) for the primary seven-class task, and for evaluation on
    any split -- oversampling is a training-distribution correction only; validation and
    test sets must reflect the real combined distribution, not an inflated one.
    """
    dataset = _build_example_dataset(df, label_col, classes, image_size, training)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_oversampled_binary_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int = 32,
    ddi_fraction: float = 0.3,
) -> tf.data.Dataset:
    """Training-only dataset for the binary task that samples each batch from HAM10000
    and DDI at a fixed target ratio, rather than relying on random shuffling of the
    combined corpus (which would draw DDI at close to its ~6% share by chance).

    df must be the combined corpus from src.data.binary_corpus.build_binary_corpus
    (i.e. carries a "source" column of "ham10000"/"ddi"). The returned dataset repeats
    indefinitely, since sample_from_datasets needs both component datasets to never run
    dry -- callers must bound training with steps_per_epoch
    (src.data.oversampling.compute_steps_per_epoch), not dataset exhaustion.
    """
    validate_ddi_fraction(ddi_fraction)

    ham_df = df[df["source"] == "ham10000"]
    ddi_df = df[df["source"] == "ddi"]
    if len(ham_df) == 0 or len(ddi_df) == 0:
        raise ValueError(
            f"make_oversampled_binary_dataset requires both sources present; got "
            f"{len(ham_df)} ham10000 rows and {len(ddi_df)} ddi rows."
        )

    ham_ds = _build_example_dataset(ham_df, "binary_label", ["benign", "malignant"], image_size, training=True)
    ddi_ds = _build_example_dataset(ddi_df, "binary_label", ["benign", "malignant"], image_size, training=True)

    combined = tf.data.Dataset.sample_from_datasets(
        [ham_ds.repeat(), ddi_ds.repeat()],
        weights=[1 - ddi_fraction, ddi_fraction],
        seed=42,
        stop_on_empty_dataset=False,
    )
    return combined.batch(batch_size).prefetch(tf.data.AUTOTUNE)
