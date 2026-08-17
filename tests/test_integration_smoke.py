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

import pytest

tf = pytest.importorskip("tensorflow", reason="requires TensorFlow, not installed here")

from src import config  # noqa: E402
from src.data.pipeline import make_dataset  # noqa: E402
from src.evaluate import evaluate_predictions  # noqa: E402
from src.evaluate_run import predict_dataset  # noqa: E402
from src.models.build import build_model, unfreeze_top_layers  # noqa: E402
from src.train import prepare_data  # noqa: E402
from src.xai.faithfulness import compute_overlap  # noqa: E402
from src.xai.gradcam import make_gradcam_heatmap  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.HAM10000_IMAGE_DIR.exists(),
    reason=f"real HAM10000 data not found at {config.HAM10000_IMAGE_DIR}",
)


def test_train_evaluate_explain_roundtrip_on_real_data_subset(tmp_path):
    architecture = "resnet50"
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
