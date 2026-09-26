"""
Evaluation metrics for HAR classification.

Reports the metrics used in both the DeepConvLSTM paper (weighted F1) and the
DAGHAR benchmark (accuracy, macro F1), plus a confusion matrix.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    """Return a dict of scalar + array metrics."""
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "f1_per_class": f1_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def summarize_folds(fold_metrics: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-fold scalar metrics into mean +/- std.
    `fold_metrics` maps target_name -> metrics dict.
    """
    keys = ["accuracy", "f1_weighted", "f1_macro"]
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics.values() if k in m]
        if vals:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary
