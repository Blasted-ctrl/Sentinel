"""Cache a fixed, outcome-independent FPA-FOD/weather grid for repeatable runs."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from sentinel.data.climate import (
    assemble_climate_dataset,
    fetch_weather_for_regions,
    fire_dates_by_region,
    grid_regions,
    load_fod_fires,
)
from sentinel.geo import BBox
from sentinel.ingest.weather import WeatherClient


def prepare_climate_cache(
    fod_sqlite: Path,
    output: Path,
    bbox: BBox,
    start: date,
    end: date,
    *,
    citation: str,
    cell_size: float = 1.0,
    block_size: float = 4.0,
    lookback: int = 14,
    horizon: int = 7,
) -> Path:
    if not fod_sqlite.is_file():
        raise ValueError(f"FPA-FOD SQLite file not found: {fod_sqlite}")
    if start >= end or not 0 < cell_size <= block_size or lookback < 1 or horizon < 1:
        raise ValueError("need start < end, 0 < cell_size <= block_size, positive window lengths")
    with sqlite3.connect(f"{fod_sqlite.resolve().as_uri()}?mode=ro", uri=True) as conn:
        first_year, last_year = conn.execute(
            "SELECT MIN(FIRE_YEAR), MAX(FIRE_YEAR) FROM Fires"
        ).fetchone()
    if first_year is None or not first_year <= start.year <= end.year <= last_year:
        raise ValueError(f"requested years exceed FPA-FOD data extent: {first_year}..{last_year}")
    regions = grid_regions(bbox, cell_size)
    blocks = {
        r.key: f"{math.floor(r.centroid[0] / block_size)}:{math.floor(r.centroid[1] / block_size)}"
        for r in regions
    }
    if len(set(blocks.values())) < 8:
        raise ValueError("need at least 8 spatial blocks; expand the area or reduce block size")
    fires = load_fod_fires(fod_sqlite, bbox, start.year, end.year)
    client = WeatherClient(model="era5")
    try:
        weather = fetch_weather_for_regions(regions, start, end, client=client)
    finally:
        client.close()
    dataset = assemble_climate_dataset(
        regions, weather, fire_dates_by_region(regions, fires), lookback=lookback, horizon=horizon
    )
    with fod_sqlite.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "citation": citation,
        "fire_source_sha256": source_hash,
        "bbox": list(bbox),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cell_size_deg": cell_size,
        "block_size_deg": block_size,
        "lookback": lookback,
        "horizon": horizon,
        "target": "at least one reported FPA-FOD discovery in (t, t+horizon]",
        "weather_source": "Open-Meteo historical archive; retrospective reanalysis",
        "weather_model": "era5",
        "sampling": "all grid cells and available consecutive daily windows; no outcome filtering",
        "missing_weather": "gapped windows dropped; nulls forward-filled, initially zero",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(
            stream,
            x=dataset.x,
            y=dataset.y,
            groups=np.asarray(dataset.groups),
            blocks=np.asarray([blocks[g] for g in dataset.groups]),
            dates=np.asarray([day.isoformat() for day in dataset.dates]),
            feature_names=np.asarray(dataset.feature_names),
            metadata=json.dumps(metadata),
        )
    return output
