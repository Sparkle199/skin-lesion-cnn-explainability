import numpy as np
import pandas as pd

from src.evaluate import evaluate_predictions, stratified_binary_metrics


def test_evaluate_predictions_seven_class_shapes_and_ranges():
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    rng = np.random.default_rng(1)
    n = 500

    y_true = rng.integers(0, 7, size=n)
    y_pred = y_true.copy()
    y_pred[rng.choice(n, size=50, replace=False)] = rng.integers(0, 7, size=50)
    y_proba = np.eye(7)[y_pred] * 0.9 + 0.1 / 7

    metrics = evaluate_predictions(y_true, y_pred, y_proba, classes)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["precision_per_class"]) == set(classes)
    assert metrics["confusion_matrix"].shape == (7, 7)
    assert -1.0 <= metrics["cohens_kappa"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_predictions_binary_shapes_and_ranges():
    rng = np.random.default_rng(1)
    n = 500

    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    y_pred[rng.choice(n, size=40, replace=False)] = rng.integers(0, 2, size=40)
    y_proba = np.clip(y_pred * 0.8 + rng.normal(0, 0.05, size=n), 0, 1)

    metrics = evaluate_predictions(y_true, y_pred, y_proba, ["benign", "malignant"])

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_predictions_perfect_predictions_score_maximally():
    classes = ["benign", "malignant"]
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.05, 0.95, 0.8, 0.2])
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = evaluate_predictions(y_true, y_pred, y_proba, classes)

    assert metrics["accuracy"] == 1.0
    assert metrics["cohens_kappa"] == 1.0


def test_stratified_binary_metrics_excludes_unlabelled_rows():
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 0, 1],
            "y_pred": [0, 1, 1, 1, 0, 0],
            "skin_tone_group": ["FST_I_II", "FST_I_II", "FST_V_VI", None, None, "FST_V_VI"],
        }
    )

    result = stratified_binary_metrics(df, "y_true", "y_pred")

    assert set(result.keys()) == {"FST_I_II", "FST_V_VI"}
    assert sum(v["n"] for v in result.values()) == df["skin_tone_group"].notna().sum()
