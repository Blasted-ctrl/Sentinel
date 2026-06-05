"""Classification metrics. Recall is the headline metric for wildfire risk —
missing a real fire is far costlier than a false alarm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class ClassificationMetrics:
    """Headline binary-classification metrics for the positive (fire) class."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    false_alarm_rate: float  # FP / (FP + TN) — fraction of safe tiles flagged
    threshold: float
    support_positive: int
    support_negative: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute metrics from ground-truth labels and positive-class probabilities."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    # AUC is undefined when only one class is present in y_true.
    if len(np.unique(y_true_arr)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    tn, fp, _fn, _tp = _confusion_counts(y_true_arr, y_pred)
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true_arr, y_pred)),
        precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred, zero_division=0)),
        auc=auc,
        false_alarm_rate=float(false_alarm),
        threshold=float(threshold),
        support_positive=int((y_true_arr == 1).sum()),
        support_negative=int((y_true_arr == 0).sum()),
    )


def _confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` robust to single-class predictions."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)
