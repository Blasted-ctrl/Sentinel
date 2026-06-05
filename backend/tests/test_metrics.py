"""Tests for classification metrics."""

from __future__ import annotations

import math

from sentinel.training.metrics import compute_classification_metrics


def test_perfect_predictions() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = compute_classification_metrics(y_true, y_prob)
    assert m.recall == 1.0
    assert m.precision == 1.0
    assert m.f1 == 1.0
    assert m.auc == 1.0
    assert m.false_alarm_rate == 0.0


def test_false_alarm_rate() -> None:
    # Two negatives, one flagged -> false alarm rate 0.5; both positives caught.
    y_true = [0, 0, 1, 1]
    y_prob = [0.9, 0.1, 0.9, 0.9]
    m = compute_classification_metrics(y_true, y_prob)
    assert m.recall == 1.0
    assert m.false_alarm_rate == 0.5
    assert m.support_positive == 2
    assert m.support_negative == 2


def test_single_class_auc_is_nan() -> None:
    m = compute_classification_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert math.isnan(m.auc)
    assert m.recall == 1.0
