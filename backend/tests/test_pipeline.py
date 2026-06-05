"""Tests for the ingestion pipeline orchestration (DB writes patched out).

These verify the pipeline wires clients -> rows -> storage correctly without a
live database. The real DB round-trip is covered by ``test_db_integration.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from moto import mock_aws

from sentinel.geo import BBox
from sentinel.ingest import pipeline as pipeline_mod
from sentinel.ingest.firms import FireDetection
from sentinel.ingest.imagery import StacItem
from sentinel.ingest.pipeline import IngestPipeline
from sentinel.ingest.storage import ObjectStorage
from sentinel.ingest.weather import WeatherDaily


@pytest.fixture
def storage() -> Iterator[ObjectStorage]:
    with mock_aws():
        yield ObjectStorage("sentinel-test", endpoint_url=None, region="us-east-1")


@pytest.fixture
def patched_db(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, list[dict[str, Any]]]]:
    """Replace region creation + upsert with in-memory capture."""
    captured: list[tuple[Any, list[dict[str, Any]]]] = []

    def fake_region(session: Any, name: str, bbox: BBox) -> Any:
        return SimpleNamespace(id=7, name=name)

    def fake_upsert(
        session: Any, model: Any, rows: list[dict[str, Any]], index_elements: list[str]
    ) -> int:
        captured.append((model, list(rows)))
        return len(rows)

    monkeypatch.setattr(pipeline_mod, "get_or_create_region", fake_region)
    monkeypatch.setattr(pipeline_mod, "_upsert", fake_upsert)
    return captured


def _stac_item() -> StacItem:
    return StacItem(
        item_id="S2A_TEST_20240801",
        collection="sentinel-2-l2a",
        captured_at=datetime(2024, 8, 1, 18, 55, tzinfo=UTC),
        cloud_cover=3.2,
        geometry_wkt="POLYGON((-120.5 38.5,-120 38.5,-120 39,-120.5 39,-120.5 38.5))",
        assets={"thumbnail": "https://tiles.example/thumb.jpg"},
    )


def test_pipeline_runs_all_sources(
    sample_bbox: BBox,
    storage: ObjectStorage,
    patched_db: list[tuple[Any, list[dict[str, Any]]]],
) -> None:
    firms = MagicMock()
    firms.fetch.return_value = [
        FireDetection(
            latitude=38.7,
            longitude=-120.2,
            acq_date=date(2024, 8, 1),
            acq_time="0712",
            brightness=330.5,
            frp=12.4,
            confidence="n",
            satellite="N",
            instrument="VIIRS",
            daynight="D",
            source="VIIRS_SNPP_NRT",
        )
    ]
    weather = MagicMock()
    weather.fetch_daily.return_value = [
        WeatherDaily(date(2024, 8, 1), 35.1, 18.2, 0.0, 14.5, 6.1)
    ]
    stac = MagicMock()
    stac.search.return_value = [_stac_item()]
    stac.download_asset.return_value = (b"\xff\xd8\xff jpeg", "image/jpeg")

    pipe = IngestPipeline(
        MagicMock(), firms=firms, weather=weather, stac=stac, storage=storage
    )
    summary = pipe.run(
        "sierra", sample_bbox, date(2024, 8, 1), date(2024, 8, 1), tile_limit=5
    )

    assert summary.weather_readings == 1
    assert summary.fire_events == 1
    assert summary.satellite_tiles == 1
    assert summary.region_id == 7

    # The thumbnail was uploaded to object storage.
    stored_key = next(
        rows[0]["s3_key"]
        for model, rows in patched_db
        if model.__name__ == "SatelliteTile"
    )
    assert storage.exists(stored_key)

    # Weather rows carry the region id and parsed features.
    weather_rows = next(
        rows for model, rows in patched_db if model.__name__ == "WeatherReading"
    )
    assert weather_rows[0]["region_id"] == 7
    assert weather_rows[0]["temperature_max"] == pytest.approx(35.1)


def test_pipeline_respects_source_selection(
    sample_bbox: BBox,
    patched_db: list[tuple[Any, list[dict[str, Any]]]],
) -> None:
    weather = MagicMock()
    weather.fetch_daily.return_value = [
        WeatherDaily(date(2024, 8, 1), 30.0, 15.0, 0.0, 10.0, 5.0)
    ]
    pipe = IngestPipeline(MagicMock(), weather=weather)
    summary = pipe.run(
        "sierra", sample_bbox, date(2024, 8, 1), date(2024, 8, 1), sources=["weather"]
    )

    assert summary.weather_readings == 1
    assert summary.fire_events == 0
    assert summary.satellite_tiles == 0
    weather.fetch_daily.assert_called_once()


def test_pipeline_missing_client_raises(sample_bbox: BBox) -> None:
    pipe = IngestPipeline(MagicMock())
    region = SimpleNamespace(id=1, name="x")
    with pytest.raises(ValueError):
        pipe.ingest_weather(region, sample_bbox, date(2024, 8, 1), date(2024, 8, 1))
