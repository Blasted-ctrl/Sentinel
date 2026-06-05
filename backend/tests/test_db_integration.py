"""PostGIS integration test.

Runs only when ``DATABASE_URL`` points at a live PostGIS instance (provided by
docker-compose locally and by a service container in CI). Otherwise skipped.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from geoalchemy2 import WKTElement
from sqlalchemy import Engine, func, select

from sentinel.db.base import get_engine, get_sessionmaker
from sentinel.db.models import FireEvent, Region, SatelliteTile, WeatherReading
from sentinel.db.schema import init_db
from sentinel.geo import BBox
from sentinel.ingest.pipeline import get_or_create_region

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("DATABASE_URL")
POLYGON = "POLYGON((-120.5 38.5,-120 38.5,-120 39,-120.5 39,-120.5 38.5))"


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set; skipping PostGIS integration test")
    eng = get_engine(DATABASE_URL)
    init_db(eng, drop=True)
    yield eng
    eng.dispose()


def test_geo_roundtrip_and_spatial_query(engine: Engine) -> None:
    session_factory = get_sessionmaker(engine)
    with session_factory() as session:
        region = Region(name="itest", geom=WKTElement(POLYGON, srid=4326))
        session.add(region)
        session.flush()

        session.add(
            WeatherReading(
                region_id=region.id,
                date=date(2024, 8, 1),
                geom=WKTElement("POINT(-120.25 38.75)", srid=4326),
                temperature_max=35.0,
            )
        )
        session.add(
            FireEvent(
                region_id=region.id,
                latitude=38.7,
                longitude=-120.2,
                geom=WKTElement("POINT(-120.2 38.7)", srid=4326),
                acq_date=date(2024, 8, 1),
                acq_time="0712",
                frp=12.4,
                source="firms",
            )
        )
        session.add(
            SatelliteTile(
                region_id=region.id,
                item_id="S2_ITEST",
                asset="thumbnail",
                captured_at=datetime(2024, 8, 1, tzinfo=UTC),
                geom=WKTElement(POLYGON, srid=4326),
                s3_key="sentinel-2-l2a/itest/S2_ITEST/thumbnail.jpg",
            )
        )
        session.commit()

        # The fire point falls inside the region polygon -> spatial join works.
        inside = session.scalar(
            select(func.count())
            .select_from(FireEvent)
            .where(func.ST_Within(FireEvent.geom, region.geom))
        )
        assert inside == 1


def test_get_or_create_region_is_idempotent(engine: Engine) -> None:
    bbox = BBox(-121.0, 37.0, -120.0, 38.0)
    session_factory = get_sessionmaker(engine)
    with session_factory() as session:
        first = get_or_create_region(session, "idempotent", bbox)
        session.commit()
        second = get_or_create_region(session, "idempotent", bbox)
        assert first.id == second.id
