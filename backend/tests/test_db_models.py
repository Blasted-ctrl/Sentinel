"""Tests for the ORM schema definition (no database required)."""

from __future__ import annotations

from sentinel.db import models
from sentinel.db.base import Base

EXPECTED_TABLES = {
    "regions",
    "weather_readings",
    "fire_events",
    "satellite_tiles",
}


def test_all_tables_registered() -> None:
    assert set(Base.metadata.tables) >= EXPECTED_TABLES


def test_weather_has_fire_relevant_features() -> None:
    cols = set(models.WeatherReading.__table__.columns.keys())
    assert {
        "temperature_max",
        "temperature_min",
        "precipitation",
        "wind_speed_max",
        "evapotranspiration",
    } <= cols


def test_fire_event_unique_detection_constraint() -> None:
    constraint_names = {c.name for c in models.FireEvent.__table__.constraints}
    assert "uq_fire_unique_detection" in constraint_names


def test_geometry_columns_use_wgs84() -> None:
    assert models.Region.__table__.c.geom.type.srid == 4326
    assert models.WeatherReading.__table__.c.geom.type.srid == 4326
    assert models.FireEvent.__table__.c.geom.type.srid == 4326
    assert models.SatelliteTile.__table__.c.geom.type.srid == 4326


def test_geometry_types() -> None:
    assert models.Region.__table__.c.geom.type.geometry_type == "POLYGON"
    assert models.FireEvent.__table__.c.geom.type.geometry_type == "POINT"
    assert models.SatelliteTile.__table__.c.geom.type.geometry_type == "POLYGON"
