"""LSTM training + evaluation on climate time-series.

Builds region-grouped sliding windows of daily weather (Open-Meteo) labelled by
FPA-FOD fire occurrences, applies the same leakage-safe geospatial split as the
CNN, trains a class-weighted Keras LSTM, and persists the model plus a
``metrics.json`` source of truth. It also caches the artifacts the ensemble
needs (per-sample LSTM probabilities, labels, region keys/centroids).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.data.climate import (
    GridRegion,
    assemble_climate_dataset,
    fetch_weather_for_regions,
    fire_dates_by_region,
    grid_regions,
    load_fod_fires,
    standardize,
)
from sentinel.data.geosplit import GeoSplit, assert_no_region_leakage, split_by_groups
from sentinel.geo import BBox
from sentinel.ingest.weather import WeatherDaily
from sentinel.logging import get_logger
from sentinel.models.lstm import LstmConfig, build_lstm, seed_tensorflow
from sentinel.training.metrics import compute_classification_metrics
from sentinel.training.seed import seed_everything

logger = get_logger(__name__)

# Default area of interest: Northern California (fire-prone, dense FOD coverage).
DEFAULT_BBOX = BBox(-124.0, 36.0, -117.0, 42.0)


@dataclass(slots=True)
class LstmTrainConfig:
    """Hyperparameters and run settings for LSTM training."""

    fod_sqlite: Path
    output_dir: Path
    bbox: BBox = DEFAULT_BBOX
    cell_size: float = 1.0
    start: date = date(2010, 1, 1)
    end: date = date(2015, 12, 31)
    lookback: int = 14
    horizon: int = 7
    epochs: int = 8
    batch_size: int = 256
    units: int = 64
    lr: float = 1e-3
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    threshold: float = 0.5
    model_filename: str = "lstm_climate.keras"
    metrics_filename: str = "metrics.json"

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fod_sqlite"] = str(self.fod_sqlite)
        data["output_dir"] = str(self.output_dir)
        data["bbox"] = list(self.bbox)
        data["start"] = self.start.isoformat()
        data["end"] = self.end.isoformat()
        return data


def _class_weight(labels: np.ndarray) -> dict[int, float]:
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    return {
        0: n / (2.0 * max(n_neg, 1)),
        1: n / (2.0 * max(n_pos, 1)),
    }


def _positive_rate(y: np.ndarray, idx: list[int]) -> float:
    return round(float(y[idx].mean()), 4) if idx else 0.0


def train_lstm(
    cfg: LstmTrainConfig,
    *,
    regions: list[GridRegion] | None = None,
    weather_by_region: dict[str, list[WeatherDaily]] | None = None,
    fires_by_region: dict[str, set[date]] | None = None,
) -> dict[str, Any]:
    """Train the climate LSTM and return the metrics report (also saved).

    ``regions`` defaults to a grid over ``cfg.bbox``; pass an explicit list
    (e.g. globally-selected fire-active cells) to train a worldwide model.
    """
    seed_everything(cfg.seed)
    seed_tensorflow(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if regions is None:
        regions = grid_regions(cfg.bbox, cfg.cell_size)
    if weather_by_region is None:
        weather_by_region = fetch_weather_for_regions(regions, cfg.start, cfg.end)
    if fires_by_region is None:
        fires = load_fod_fires(cfg.fod_sqlite, cfg.bbox, cfg.start.year, cfg.end.year)
        fires_by_region = fire_dates_by_region(regions, fires)

    dataset = assemble_climate_dataset(
        regions,
        weather_by_region,
        fires_by_region,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )
    split = split_by_groups(
        dataset.groups,
        dataset.y.tolist(),
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )
    assert_no_region_leakage(split)
    logger.info(
        "lstm.dataset",
        samples=len(dataset.y),
        positives=int(dataset.y.sum()),
        regions=split.region_counts(),
        sizes=split.sizes(),
    )

    x_train, mean, std = standardize(dataset.x[split.train])
    x_val, _, _ = standardize(dataset.x[split.val], mean, std)
    x_test, _, _ = standardize(dataset.x[split.test], mean, std)
    y_train, y_val, y_test = (
        dataset.y[split.train],
        dataset.y[split.val],
        dataset.y[split.test],
    )

    model = build_lstm(
        LstmConfig(
            lookback=cfg.lookback,
            n_features=dataset.x.shape[-1],
            units=cfg.units,
            learning_rate=cfg.lr,
        )
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val) if len(y_val) else None,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        class_weight=_class_weight(y_train),
        verbose=2,
    )

    test_prob = model.predict(x_test, verbose=0).ravel()
    test_metrics = compute_classification_metrics(y_test, test_prob, threshold=cfg.threshold)
    val_metrics = None
    if len(y_val):
        val_prob = model.predict(x_val, verbose=0).ravel()
        val_metrics = compute_classification_metrics(y_val, val_prob, threshold=cfg.threshold)

    report = {
        "model": "keras-lstm",
        "task": "climate-sequence-ignition-risk",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": cfg.seed,
        "dataset": {
            "n_total": len(dataset.y),
            "positives": int(dataset.y.sum()),
            "negatives": int(len(dataset.y) - dataset.y.sum()),
            "features": dataset.feature_names,
            "lookback": cfg.lookback,
            "horizon_days": cfg.horizon,
            "regions": split.region_counts(),
        },
        "split": {
            "sizes": split.sizes(),
            "positive_rate": {
                "train": _positive_rate(dataset.y, split.train),
                "val": _positive_rate(dataset.y, split.val),
                "test": _positive_rate(dataset.y, split.test),
            },
            "geospatial": True,
            "cell_size_deg": cfg.cell_size,
            "region_leakage": False,
        },
        "config": cfg.public_dict(),
        "val": val_metrics.as_dict() if val_metrics else None,
        "test": test_metrics.as_dict(),
        "history": {
            "loss": [round(float(v), 5) for v in history.history.get("loss", [])],
            "val_auc": [round(float(v), 5) for v in history.history.get("val_auc", [])],
        },
    }

    _save_lstm_artifacts(cfg, model, mean, std, dataset, split, regions, report)
    logger.info("lstm.complete", test=test_metrics.as_dict())
    return report


def _save_lstm_artifacts(
    cfg: LstmTrainConfig,
    model: Any,
    mean: np.ndarray,
    std: np.ndarray,
    dataset: Any,
    split: GeoSplit,
    regions: list[GridRegion],
    report: dict[str, Any],
) -> None:
    model.save(cfg.output_dir / cfg.model_filename)
    np.savez(cfg.output_dir / "scaler.npz", mean=mean, std=std)

    # All-sample LSTM probabilities + region metadata for the ensemble step.
    x_all, _, _ = standardize(dataset.x, mean, std)
    prob_all = model.predict(x_all, verbose=0).ravel()
    region_bboxes = np.array([list(r.bbox) for r in regions], dtype=float)
    np.savez(
        cfg.output_dir / "ensemble_inputs.npz",
        y=dataset.y,
        groups=np.array(dataset.groups),
        lstm_prob=prob_all,
        train_idx=np.array(split.train),
        val_idx=np.array(split.val),
        test_idx=np.array(split.test),
        region_keys=np.array([r.key for r in regions]),
        region_bboxes=region_bboxes,
    )
    (cfg.output_dir / cfg.metrics_filename).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("lstm.artifacts.saved", output=str(cfg.output_dir))
