"""Smoke test for the full LSTM training pipeline (tiny, synthetic, no network).

Skips automatically if TensorFlow is unavailable.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from sentinel.data.climate import grid_regions
from sentinel.geo import BBox
from sentinel.ingest.weather import WeatherDaily
from sentinel.training.train_lstm import LstmTrainConfig, train_lstm

BBOX = BBox(-122.0, 37.0, -119.0, 40.0)  # 3x3 = 9 regions at cell_size 1.0


def _weather(start: date, days: int, region_index: int) -> list[WeatherDaily]:
    return [
        WeatherDaily(
            date=start + timedelta(days=i),
            temperature_max=25.0 + (i + region_index) % 10,
            temperature_min=12.0,
            precipitation=float((i + region_index) % 4),
            wind_speed_max=8.0 + region_index % 3,
            evapotranspiration=4.0 + (i % 3),
        )
        for i in range(days)
    ]


def test_train_lstm_smoke(tmp_path: Path) -> None:
    regions = grid_regions(BBOX, 1.0)
    start = date(2012, 1, 1)
    weather = {r.key: _weather(start, 80, i) for i, r in enumerate(regions)}
    # Give several regions positive fire days so both classes appear across splits.
    fires = {r.key: set() for r in regions}
    for i, r in enumerate(regions):
        if i % 2 == 0:
            fires[r.key] = {start + timedelta(days=40), start + timedelta(days=55)}

    cfg = LstmTrainConfig(
        fod_sqlite=tmp_path / "unused.sqlite",
        output_dir=tmp_path / "lstm",
        bbox=BBOX,
        cell_size=1.0,
        start=start,
        end=start + timedelta(days=79),
        lookback=7,
        horizon=3,
        epochs=1,
        batch_size=32,
        units=8,
        seed=42,
    )
    report = train_lstm(cfg, weather_by_region=weather, fires_by_region=fires)

    assert (cfg.output_dir / "metrics.json").exists()
    assert (cfg.output_dir / "ensemble_inputs.npz").exists()
    saved = json.loads((cfg.output_dir / "metrics.json").read_text())
    assert saved["split"]["region_leakage"] is False
    for key in ("recall", "precision", "false_alarm_rate"):
        assert 0.0 <= report["test"][key] <= 1.0
