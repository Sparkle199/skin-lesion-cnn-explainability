"""End-to-end smoke test against real HAM10000 data and a real (tiny) training run.

Fills the gap flagged repeatedly in journal/agent-journal.md: every TensorFlow-dependent
module (src.models.build, src.data.pipeline, src.train, src.evaluate_run, src.xai.gradcam)
had only ever been syntax-checked, never runtime-tested, since no local environment had
both TensorFlow and the extracted datasets. This test requires both, and is skipped
automatically wherever either is unavailable -- it is not meant to run on every dev
machine, only somewhere with real TensorFlow and the real extracted data (e.g. the
Runpod training pod).

Deliberately small (a handful of images, 1 epoch per phase) -- this checks that the real
pipeline wiring works end-to-end (data loads, model builds, trains, saves, reloads,
predicts, explains), not that the resulting model is any good. Model quality is what the
real six training runs (src/train.py, run properly) are for.
"""

import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow", reason="requires TensorFlow, not installed here")

from src import config  # noqa: E402
from src.data.oversampling import compute_steps_per_epoch  # noqa: E402
from src.data.pipeline import make_dataset, make_oversampled_binary_dataset  # noqa: E402
from src.evaluate import evaluate_predictions, stratified_binary_metrics  # noqa: E402
from src.evaluate_run import predict_dataset  # noqa: E402
from src.models.build import build_model, unfreeze_top_layers  # noqa: E402
from src.train import prepare_data  # noqa: E402
from src.xai.faithfulness import compute_overlap  # noqa: E402
from src.xai.gradcam import make_gradcam_heatmap  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.HAM10000_IMAGE_DIR.exists(),
    reason=f"real HAM10000 data not found at {config.HAM10000_IMAGE_DIR}",
)


@pytest.mark.parametrize("architecture", ["resnet50", "efficientnetb4", "vgg16"])
def test_train_evaluate_explain_roundtrip_on_real_data_subset(tmp_path, architecture):
    image_size = config.IMAGE_SIZE[architecture]
    batch_size = 8

    train_df, val_df = prepare_data("seven_class")
    train_subset = train_df.head(32).reset_index(drop=True)
    val_subset = val_df.head(16).reset_index(drop=True)

    train_ds = make_dataset(
        train_subset, "dx", config.SEVEN_CLASSES, image_size, batch_size, training=True
    )
    val_ds = make_dataset(
        val_subset, "dx", config.SEVEN_CLASSES, image_size, batch_size, training=False
    )

    model, base = build_model(architecture, num_classes=len(config.SEVEN_CLASSES), task_name="seven_class")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)

    unfreeze_top_layers(base, 5)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)

    # Round-trip through save/load, exactly as src.evaluate_run does for real runs --
    # this specifically checks the previously-unverified assumption in config.py that
    # BACKBONE_LAYER_NAME resolves the backbone submodel on a *reloaded* model.
    model_path = tmp_path / "smoke_model.keras"
    model.save(model_path)
    reloaded = tf.keras.models.load_model(model_path)
    reloaded_base = reloaded.get_layer(config.BACKBONE_LAYER_NAME[architecture])
    assert reloaded_base is not None

    y_true, y_pred, y_proba = predict_dataset(reloaded, val_ds, num_classes=len(config.SEVEN_CLASSES))
    assert y_true.shape == y_pred.shape == (len(val_subset),)
    assert y_proba.shape == (len(val_subset), len(config.SEVEN_CLASSES))

    metrics = evaluate_predictions(y_true, y_pred, y_proba, config.SEVEN_CLASSES)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["precision_per_class"]) == set(config.SEVEN_CLASSES)

    # Grad-CAM + faithfulness on one real HAM10000 image against its real ground-truth mask.
    from src.data import ham10000

    sample_row = val_subset.iloc[0]
    raw = tf.io.read_file(sample_row["image_path"])
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, image_size)[tf.newaxis, ...]

    heatmap = make_gradcam_heatmap(reloaded, reloaded_base, architecture, image)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6

    mask = ham10000.load_segmentation_mask(sample_row["image_id"])
    overlap = compute_overlap(heatmap, mask)
    assert 0.0 <= overlap["iou"] <= 1.0
    assert 0.0 <= overlap["dice"] <= 1.0


@pytest.mark.skipif(
    not config.DDI_METADATA_CSV.exists(),
    reason=f"real DDI data not found at {config.DDI_METADATA_CSV}",
)
def test_binary_task_roundtrip_on_real_data_subset(tmp_path):
    """Covers the binary task's own machinery, untouched by the seven-class test above:
    the combined HAM10000+DDI corpus, DDI-oversampled batching (make_oversampled_binary_
    dataset), steps_per_epoch, and skin-tone-stratified evaluation on real DDI rows.
    Uses resnet50 only -- per-architecture correctness is already confirmed by the
    parametrized seven-class test above; what's new here is the binary-specific data path,
    not re-proving architecture handling.
    """
    architecture = "resnet50"
    image_size = config.IMAGE_SIZE[architecture]
    batch_size = 8
    ddi_fraction = 0.3

    train_binary, val_binary = prepare_data("binary")

    # .head() alone would grab HAM10000-only rows, since build_binary_corpus concatenates
    # HAM10000 rows before DDI rows -- pull a slice of each source explicitly so both are
    # present, which make_oversampled_binary_dataset requires.
    def _subset(df, n_ham: int, n_ddi: int):
        ham_rows = df[df["source"] == "ham10000"].head(n_ham)
        ddi_rows = df[df["source"] == "ddi"].head(n_ddi)
        return pd.concat([ham_rows, ddi_rows], ignore_index=True)

    train_subset = _subset(train_binary, n_ham=24, n_ddi=8)
    val_subset = _subset(val_binary, n_ham=12, n_ddi=4)

    train_ds = make_oversampled_binary_dataset(train_subset, image_size, batch_size, ddi_fraction)
    val_ds = make_dataset(val_subset, "binary_label", ["benign", "malignant"], image_size, batch_size, training=False)

    n_ham10000 = (train_subset["source"] == "ham10000").sum()
    steps_per_epoch = compute_steps_per_epoch(n_ham10000, batch_size, ddi_fraction)

    model, base = build_model(architecture, num_classes=2, task_name="binary")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=1, steps_per_epoch=steps_per_epoch, verbose=0)

    unfreeze_top_layers(base, 5)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=1, steps_per_epoch=steps_per_epoch, verbose=0)

    model_path = tmp_path / "smoke_model_binary.keras"
    model.save(model_path)
    reloaded = tf.keras.models.load_model(model_path)
    reloaded_base = reloaded.get_layer(config.BACKBONE_LAYER_NAME[architecture])

    y_true, y_pred, y_proba = predict_dataset(reloaded, val_ds, num_classes=2)
    assert y_true.shape == y_pred.shape == y_proba.shape == (len(val_subset),)

    metrics = evaluate_predictions(y_true, y_pred, y_proba, ["benign", "malignant"])
    assert 0.0 <= metrics["accuracy"] <= 1.0

    val_subset = val_subset.reset_index(drop=True)
    val_subset["y_true"], val_subset["y_pred"] = y_true, y_pred
    strat = stratified_binary_metrics(val_subset, "y_true", "y_pred")
    # Only DDI rows carry a non-null skin_tone_group -- HAM10000 rows are excluded by
    # design (src/evaluate.py), so the stratified result must cover DDI rows only.
    assert set(strat) <= {"FST_I_II", "FST_III_IV", "FST_V_VI"}
    assert sum(group["n"] for group in strat.values()) == (val_subset["source"] == "ddi").sum()

    # Grad-CAM + faithfulness on a HAM10000-derived row from the binary corpus, exactly
    # as run_faithfulness_check filters to ham_rows in src/evaluate_run.py.
    from src.data import ham10000

    ham_rows = val_subset[val_subset["source"] == "ham10000"]
    sample_row = ham_rows.iloc[0]
    raw = tf.io.read_file(sample_row["image_path"])
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, image_size)[tf.newaxis, ...]

    heatmap = make_gradcam_heatmap(reloaded, reloaded_base, architecture, image)
    mask = ham10000.load_segmentation_mask(sample_row["image_id"])
    overlap = compute_overlap(heatmap, mask)
    assert 0.0 <= overlap["iou"] <= 1.0
    assert 0.0 <= overlap["dice"] <= 1.0
