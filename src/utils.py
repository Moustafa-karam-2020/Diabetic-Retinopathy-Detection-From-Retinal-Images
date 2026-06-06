from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


def get_cost_sensitive_weights(labels: Sequence[int]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for use with
    ``torch.nn.CrossEntropyLoss(weight=...)``.

    Each class weight is calculated as:

        w_c = N_total / (N_classes * N_c)

    where N_c is the number of samples belonging to class c.  This ensures
    that rare classes receive a proportionally higher penalty, forcing the
    model to pay more attention to under-represented grades.

    Args:
        labels: 1-D array-like of integer class labels (values 0-4).

    Returns:
        weights: float32 tensor of shape (num_classes,) on CPU.
                 Pass directly to ``CrossEntropyLoss(weight=weights.to(device))``.

    Example:
        >>> weights = get_cost_sensitive_weights(train_df["label"].tolist())
        >>> criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels_arr)
    n_classes = len(classes)
    n_total = len(labels_arr)

    weights = np.zeros(n_classes, dtype=np.float32)
    for c in classes:
        n_c = np.sum(labels_arr == c)
        weights[c] = n_total / (n_classes * n_c)

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
) -> dict[str, float]:
    """
    Compute a suite of classification metrics for a 5-class DR grading task.

    Metrics
    -------
    accuracy   : Fraction of correctly classified samples.
    precision  : Macro-averaged precision across all 5 classes.
    recall     : Macro-averaged recall across all 5 classes.
    f1         : Macro-averaged F1-score across all 5 classes.
    qwk        : Quadratic Weighted Kappa — the primary DR competition metric.
                 Penalises disagreements quadratically; chance agreement = 0,
                 perfect agreement = 1.

    All multi-class metrics use ``average='macro'`` and
    ``zero_division=0`` so that missing predictions for a class do not raise
    exceptions during early training epochs.

    Args:
        true_labels : Ground-truth integer labels (0-4).
        pred_labels : Model-predicted integer labels (0-4).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1', 'qwk'.

    Example:
        >>> metrics = evaluate(all_targets, all_preds)
        >>> print(f"QWK: {metrics['qwk']:.4f}  |  F1: {metrics['f1']:.4f}")
    """
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(pred_labels, dtype=np.int64)

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall":    float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1":        float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "qwk":       float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ),
    }
