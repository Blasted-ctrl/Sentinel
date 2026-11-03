"""CPU climate benchmark with independent fit, calibration, alarm, and test data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)

from sentinel.data.forecast_split import split_forecast_windows
from sentinel.training.metrics import compute_classification_metrics


def summarize_windows(x: np.ndarray) -> np.ndarray:
    """Keep recent conditions, mean, extremes, and change; no future features."""
    if x.ndim != 3 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite (samples, lookback, features) array")
    return np.concatenate([x[:, -1], x.mean(1), x.min(1), x.max(1), x[:, -1] - x[:, 0]], 1)


def alarm_threshold(y: np.ndarray, prob: np.ndarray, min_recall: float) -> float:
    """Highest threshold meeting the requested recall on alarm-tuning data only."""
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1]")
    _, recall, thresholds = precision_recall_curve(y, prob)
    return float(thresholds[recall[:-1] >= min_recall].max())


def probability_report(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    report = compute_classification_metrics(y, prob, threshold=threshold).as_dict()
    report.update(
        average_precision=float(average_precision_score(y, prob)),
        brier=float(brier_score_loss(y, prob)),
        log_loss=float(log_loss(y, prob, labels=[0, 1])),
        prevalence=float(y.mean()),
    )
    bin_ids = np.minimum((prob * 10).astype(int), 9)
    report["reliability_bins"] = [
        {
            "lower": bin_id / 10,
            "upper": (bin_id + 1) / 10,
            "count": int(mask.sum()),
            "mean_probability": float(prob[mask].mean()),
            "observed_rate": float(y[mask].mean()),
        }
        for bin_id in range(10)
        if (mask := bin_ids == bin_id).any()
    ]
    return report


def train_reliable_climate(
    data_path: Path,
    output_dir: Path,
    *,
    train_end: date,
    calibration_end: date,
    tune_end: date,
    min_recall: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Train an experimental candidate; never replace the serving model implicitly.

    The NPZ comes from prepare-climate. Calibration and threshold fitting retain
    the observed prevalence. No class balancing or random internal validation.
    """
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1]")
    with np.load(data_path, allow_pickle=False) as data:
        x, y = data["x"], data["y"]
        groups, blocks = data["groups"].astype(str), data["blocks"].astype(str)
        dates = [date.fromisoformat(day) for day in data["dates"].astype(str)]
        feature_names = data["feature_names"].astype(str).tolist()
        metadata = json.loads(str(data["metadata"]))
    if (
        y.ndim != 1
        or len(y) != len(x)
        or len(groups) != len(y)
        or len(blocks) != len(y)
        or len(dates) != len(y)
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("cache arrays must align and labels must be binary")
    if x.ndim != 3 or x.shape[1] != metadata["lookback"] or x.shape[2] != len(feature_names):
        raise ValueError("cache feature shape does not match metadata")
    if len(set(zip(groups, dates, strict=True))) != len(y):
        raise ValueError("cache must contain unique region/date samples")
    if any(len(set(blocks[groups == group])) != 1 for group in set(groups)):
        raise ValueError("each region must belong to exactly one spatial block")
    splits = split_forecast_windows(
        blocks.tolist(),
        dates,
        train_end=train_end,
        calibration_end=calibration_end,
        tune_end=tune_end,
        lookback=metadata["lookback"],
        horizon=metadata["horizon"],
        seed=seed,
    )
    for name, idx in splits.items():
        if len(np.unique(y[idx])) != 2:
            raise ValueError(f"{name} needs both classes; expand data, do not search test seeds")

    features = summarize_windows(x)
    train, cal, tune, test = (splits[k] for k in ("train", "calibration", "tune", "test"))
    counts = Counter(groups[train])
    weights = np.array([1 / counts[group] for group in groups[train]])
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    ).fit(features[train], y[train], sample_weight=weights)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
        model.decision_function(features[cal]).reshape(-1, 1), y[cal]
    )

    def predict(idx: np.ndarray) -> np.ndarray:
        return np.asarray(
            calibrator.predict_proba(model.decision_function(features[idx]).reshape(-1, 1))[:, 1]
        )

    threshold = alarm_threshold(y[tune], predict(tune), min_recall)
    season_cols = [feature_names.index(name) for name in ("season_sin", "season_cos")]
    season = x[:, -1, season_cols]
    baseline = LogisticRegression(max_iter=1000, random_state=seed).fit(
        season[train], y[train], sample_weight=weights
    )
    baseline_threshold = alarm_threshold(
        y[tune], baseline.predict_proba(season[tune])[:, 1], min_recall
    )
    prob = predict(test)
    raw_prob = model.predict_proba(features[test])[:, 1]
    report = {
        "model": "experimental-region-balanced-hist-gradient-boosting",
        "generated_at": datetime.now(UTC).isoformat(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "source": metadata,
        "protocol": {
            "train_end": train_end.isoformat(),
            "calibration_end": calibration_end.isoformat(),
            "tune_end": tune_end.isoformat(),
            "min_tuning_recall": min_recall,
            "region_overlap": False,
            "sample_interval_overlap": False,
            "spatial_assignment": "seeded label-blind block permutation; 55/15/15/15 approx",
            "dropped_samples": len(y) - sum(len(idx) for idx in splits.values()),
        },
        "splits": {
            name: {
                "samples": len(idx),
                "regions": len(set(groups[idx])),
                "blocks": sorted(set(blocks[idx])),
                "prevalence": float(y[idx].mean()),
                "first_window_end": min(dates[i] for i in idx).isoformat(),
                "last_window_end": max(dates[i] for i in idx).isoformat(),
            }
            for name, idx in splits.items()
        },
        "test": probability_report(y[test], prob, threshold),
        "uncalibrated_test": probability_report(y[test], raw_prob, 0.5),
        "seasonal_baseline_test": probability_report(
            y[test], baseline.predict_proba(season[test])[:, 1], baseline_threshold
        ),
        "test_by_region": {
            str(group): probability_report(y[test][mask], prob[mask], threshold)
            for group in sorted(set(groups[test]))
            if len(np.unique(y[test][mask := groups[test] == group])) == 2
        },
        "single_class_test_regions": [
            str(group)
            for group in sorted(set(groups[test]))
            if len(np.unique(y[test][groups[test] == group])) < 2
        ],
        "status": "experimental; retrospective evaluation does not certify deployment reliability",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "threshold": threshold,
            "feature_names": feature_names,
            "lookback": metadata["lookback"],
            "horizon": metadata["horizon"],
        },
        output_dir / "climate.joblib",
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        y=y[test],
        probability=prob,
        groups=groups[test],
        dates=np.asarray([dates[i].isoformat() for i in test]),
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
