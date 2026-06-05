"""Tests for the climate time-series dataset builder."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from sentinel.data.climate import (
    N_FEATURES,
    assemble_climate_dataset,
    build_region_windows,
    fire_dates_by_region,
    grid_regions,
    load_fod_fires,
    standardize,
)
from sentinel.geo import BBox
from sentinel.ingest.weather import WeatherDaily


def _weather_series(start: date, days: int) -> list[WeatherDaily]:
    return [
        WeatherDaily(
            date=start + timedelta(days=i),
            temperature_max=30.0 + i % 5,
            temperature_min=15.0,
            precipitation=float(i % 3),
            wind_speed_max=10.0,
            evapotranspiration=5.0,
        )
        for i in range(days)
    ]


def test_grid_regions_tiles_bbox() -> None:
    regions = grid_regions(BBox(-122.0, 37.0, -120.0, 39.0), 1.0)
    assert len(regions) == 4  # 2 x 2
    assert len({r.key for r in regions}) == 4


def test_load_fod_filters_by_bbox_and_year(tmp_path: Path) -> None:
    db = tmp_path / "fod.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE Fires (FIRE_YEAR INT, DISCOVERY_DOY INT, LONGITUDE REAL, LATITUDE REAL)"
    )
    conn.executemany(
        "INSERT INTO Fires VALUES (?,?,?,?)",
        [
            (2012, 200, -121.0, 38.0),  # inside bbox + year
            (2012, 100, -150.0, 10.0),  # outside bbox
            (2000, 200, -121.0, 38.0),  # outside year range
        ],
    )
    conn.commit()
    conn.close()

    fires = load_fod_fires(db, BBox(-122.0, 37.0, -120.0, 39.0), 2010, 2015)
    assert len(fires) == 1
    assert fires[0][0] == date(2012, 7, 18)  # doy 200 of 2012


def test_fire_dates_bucketed_into_regions() -> None:
    regions = grid_regions(BBox(-122.0, 37.0, -120.0, 39.0), 1.0)
    fires = [(date(2012, 7, 1), -121.5, 37.5), (date(2012, 7, 2), -120.5, 38.5)]
    by_region = fire_dates_by_region(regions, fires)
    populated = {k: v for k, v in by_region.items() if v}
    assert len(populated) == 2  # two different cells


def test_window_labels_use_horizon() -> None:
    weather = _weather_series(date(2012, 1, 1), 30)
    fire_days = {date(2012, 1, 21)}  # day index 20
    x, y, groups, dates = build_region_windows(
        "r", weather, fire_days, lookback=5, horizon=3
    )
    assert x.shape[1:] == (5, N_FEATURES)
    # Window ending 2012-01-18 (idx 17): horizon covers 19,20,21 -> positive.
    idx = dates.index(date(2012, 1, 18))
    assert y[idx] == 1
    # Window ending 2012-01-10: horizon 11,12,13 -> negative.
    assert y[dates.index(date(2012, 1, 10))] == 0
    assert set(groups) == {"r"}


def test_assemble_and_standardize() -> None:
    regions = grid_regions(BBox(-122.0, 37.0, -120.0, 39.0), 1.0)
    weather = {r.key: _weather_series(date(2012, 1, 1), 40) for r in regions}
    fires = {r.key: set() for r in regions}
    fires[regions[0].key] = {date(2012, 1, 30)}

    ds = assemble_climate_dataset(regions, weather, fires, lookback=7, horizon=3)
    assert ds.x.shape[0] == len(ds.y) == len(ds.groups)
    assert ds.x.shape[1:] == (7, N_FEATURES)

    x_std, _mean, _std = standardize(ds.x)
    assert x_std.shape == ds.x.shape
    assert np.allclose(x_std.reshape(-1, N_FEATURES).mean(axis=0), 0.0, atol=1e-5)
