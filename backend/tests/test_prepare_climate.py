import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from sentinel.data import prepare_climate
from sentinel.geo import BBox
from sentinel.ingest.weather import WeatherDaily


def test_prepare_cache_keeps_nonfire_regions_and_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fod.sqlite"
    with sqlite3.connect(source) as conn:
        conn.execute(
            "CREATE TABLE Fires (FIRE_YEAR INT, DISCOVERY_DOY INT, LONGITUDE REAL, LATITUDE REAL)"
        )
        conn.execute("INSERT INTO Fires VALUES (2010, 20, -122, 34)")
    weather = [
        WeatherDaily(date(2010, 1, 1) + timedelta(days=i), 30, 15, 0, 10, 5) for i in range(40)
    ]
    fetch = Mock(
        side_effect=lambda regions, start, end, **kwargs: {r.key: weather for r in regions}
    )
    monkeypatch.setattr(prepare_climate, "fetch_weather_for_regions", fetch)
    output = tmp_path / "cache.npz"
    prepare_climate.prepare_climate_cache(
        source,
        output,
        BBox(-124, 32, -104, 48),
        date(2010, 1, 1),
        date(2010, 2, 9),
        citation="synthetic fixture",
        cell_size=4,
        block_size=4,
    )
    with np.load(output, allow_pickle=False) as cache:
        assert len(set(cache["groups"])) == 20
        assert (cache["y"] == 0).any() and (cache["y"] == 1).any()
        assert cache["x"].shape == (400, 14, 7)
        metadata = json.loads(str(cache["metadata"]))
        assert metadata["citation"] == "synthetic fixture"
        assert metadata["weather_model"] == "era5"
        assert metadata["fire_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="data extent"):
        prepare_climate.prepare_climate_cache(
            source,
            output,
            BBox(-124, 32, -104, 48),
            date(2009, 1, 1),
            date(2010, 2, 9),
            citation="synthetic fixture",
        )
    fetch.assert_called_once()
