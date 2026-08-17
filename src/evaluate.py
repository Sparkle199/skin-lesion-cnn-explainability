"""Metric computation for both tasks, per spec: accuracy, precision, recall, per-class
F1, ROC-AUC, Cohen's Kappa, confusion matrices, and (binary task only) skin-tone-group-
stratified performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: list[str]
) -> dict:
    """Core multi-metric evaluation, shared by the seven-class and binary tasks.

    y_true, y_pred: integer class indices. y_proba: predicted probabilities
    (n_samples, n_classes) for the seven-class task, or (n_samples,) for binary.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )

    if len(classes) == 2:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", labels=range(len(classes)))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_per_class": dict(zip(classes, precision)),
        "recall_per_class": dict(zip(classes, recall)),
        "f1_per_class": dict(zip(classes, f1)),
        "roc_auc": roc_auc,
        "cohens_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(classes))),
    }


def stratified_binary_metrics(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str = "skin_tone_group"
) -> dict[str, dict]:
    """Break binary-task accuracy/precision/recall/F1 down by skin-tone group.

    Only rows with a non-null group_col are included (HAM10000 rows carry no skin-tone
    label and are excluded here, per spec -- this reports DDI's contribution only).
    """
    results = {}
    labelled = df[df[group_col].notna()]
    for group, group_df in labelled.groupby(group_col):
        precision, recall, f1, _ = precision_recall_fscore_support(
            group_df[y_true_col], group_df[y_pred_col], labels=[0, 1], zero_division=0
        )
        results[group] = {
            "n": len(group_df),
            "accuracy": accuracy_score(group_df[y_true_col], group_df[y_pred_col]),
            "precision": dict(zip(["benign", "malignant"], precision)),
            "recall": dict(zip(["benign", "malignant"], recall)),
            "f1": dict(zip(["benign", "malignant"], f1)),
        }
    return results
