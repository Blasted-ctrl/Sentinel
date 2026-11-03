import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pytest

from sentinel.data.forecast_split import split_forecast_windows
from sentinel.training.train_reliable import (
    alarm_threshold,
    probability_report,
    summarize_windows,
    train_reliable_climate,
)


def test_threshold_uses_recall_constraint_and_summaries_use_past_only() -> None:
    y = np.array([0, 1, 0, 1])
    prob = np.array([0.1, 0.4, 0.7, 0.8])
    assert alarm_threshold(y, prob, 1.0) == 0.4
    assert alarm_threshold(y, prob, 0.5) == 0.8
    x = np.arange(12, dtype=float).reshape(2, 3, 2)
    summary = summarize_windows(x)
    assert summary.shape == (2, 10)
    np.testing.assert_array_equal(summary[:, :2], x[:, -1])
    with pytest.raises(ValueError, match="finite"):
        summarize_windows(np.full((2, 3, 2), np.nan))
    probabilities = np.arange(11) / 10
    report = probability_report(np.arange(11) % 2, probabilities, 0.5)
    assert sum(b["count"] for b in report["reliability_bins"]) == 11


def test_training_roundtrip_and_test_labels_cannot_change_fitted_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    days = [date(2010, 1, 1) + timedelta(days=i) for i in range(400)]
    dates = days * 12
    groups = np.repeat([f"r{i}" for i in range(12)], len(days))
    x = rng.normal(size=(len(dates), 3, 3))
    y = (x[:, -1, 0] + rng.normal(size=len(x)) * 0.7 > 0).astype(int)
    cfg = {"train_end": days[99], "calibration_end": days[199], "tune_end": days[299]}
    cache = tmp_path / "climate.npz"
    arrays = {
        "x": x,
        "y": y,
        "groups": groups,
        "blocks": groups,
        "dates": np.array([day.isoformat() for day in dates]),
        "feature_names": np.array(["temperature_max", "season_sin", "season_cos"]),
        "metadata": json.dumps({"lookback": 3, "horizon": 2, "citation": "synthetic test"}),
    }
    np.savez(cache, **arrays)
    output = tmp_path / "first"
    report = train_reliable_climate(cache, output, **cfg)
    bundle = joblib.load(output / "climate.joblib")
    assert report["protocol"]["sample_interval_overlap"] is False
    assert 0 <= report["test"]["brier"] <= 1
    assert (
        sum(b["count"] for b in report["test"]["reliability_bins"])
        == report["splits"]["test"]["samples"]
    )
    assert json.loads((output / "metrics.json").read_text())["data_sha256"] == report["data_sha256"]
    split = split_forecast_windows(groups.tolist(), dates, lookback=3, horizon=2, **cfg)
    arrays["y"][split["test"]] = 1 - arrays["y"][split["test"]]
    np.savez(cache, **arrays)
    second = train_reliable_climate(cache, tmp_path / "second", **cfg)
    other = joblib.load(tmp_path / "second" / "climate.joblib")
    assert bundle["threshold"] == other["threshold"]
    np.testing.assert_array_equal(bundle["calibrator"].coef_, other["calibrator"].coef_)
    np.testing.assert_array_equal(
        bundle["model"].predict_proba(summarize_windows(x)),
        other["model"].predict_proba(summarize_windows(x)),
    )
    assert report["test"]["brier"] != second["test"]["brier"]
    arrays["y"][:] = 0
    np.savez(cache, **arrays)
    with pytest.raises(ValueError, match="both classes"):
        train_reliable_climate(cache, tmp_path / "invalid", **cfg)
    assert not (tmp_path / "invalid").exists()
