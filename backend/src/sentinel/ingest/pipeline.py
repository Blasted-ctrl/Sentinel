"""Ingestion pipeline: fetch FIRMS / weather / imagery and persist to PostGIS.

The pipeline is deliberately dependency-injected — callers pass in the clients,
object storage, and a SQLAlchemy session — so it can be unit-tested with mocks
and integration-tested against a live PostGIS instance.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

from geoalchemy2 import WKTElement
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from sentinel.config import Settings, get_settings
from sentinel.db.models import FireEvent, Region, SatelliteTile, WeatherReading
from sentinel.geo import BBox
from sentinel.ingest.firms import FireDetection, FirmsClient
from sentinel.ingest.imagery import StacClient, StacItem
from sentinel.ingest.storage import ObjectStorage
from sentinel.ingest.weather import WeatherClient, WeatherDaily
from sentinel.logging import get_logger

logger = get_logger(__name__)

SRID = 4326
DEFAULT_SOURCES = ("fire", "weather", "imagery")


@dataclass(slots=True)
class IngestSummary:
    """Counts of newly inserted rows from an ingestion run."""

    region: str
    region_id: int
    weather_readings: int = 0
    fire_events: int = 0
    satellite_tiles: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "region_id": self.region_id,
            "weather_readings": self.weather_readings,
            "fire_events": self.fire_events,
            "satellite_tiles": self.satellite_tiles,
        }


def _point_wkt(lon: float, lat: float) -> WKTElement:
    return WKTElement(f"POINT({lon} {lat})", srid=SRID)


def _ext_for(content_type: str) -> str:
    return mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".bin"


def get_or_create_region(session: Session, name: str, bbox: BBox) -> Region:
    """Return the named region, creating it from ``bbox`` if absent."""
    existing = session.execute(
        select(Region).where(Region.name == name)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    region = Region(name=name, geom=WKTElement(bbox.to_wkt(), srid=SRID))
    session.add(region)
    session.flush()  # assign region.id
    return region


def _upsert(
    session: Session,
    model: type[Any],
    rows: Sequence[dict[str, Any]],
    index_elements: list[str],
) -> int:
    """Insert ``rows`` skipping conflicts; return the number actually inserted."""
    if not rows:
        return 0
    stmt = pg_insert(model).values(list(rows)).on_conflict_do_nothing(
        index_elements=index_elements
    )
    result = cast("CursorResult[Any]", session.execute(stmt))
    return result.rowcount or 0


class IngestPipeline:
    """Coordinates data acquisition and persistence for a region/date range."""

    def __init__(
        self,
        session: Session,
        *,
        firms: FirmsClient | None = None,
        weather: WeatherClient | None = None,
        stac: StacClient | None = None,
        storage: ObjectStorage | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.firms = firms
        self.weather = weather
        self.stac = stac
        self.storage = storage
        self.settings = settings or get_settings()

    # -- individual stages -------------------------------------------------

    def ingest_weather(self, region: Region, bbox: BBox, start: date, end: date) -> int:
        if self.weather is None:
            raise ValueError("weather client not configured")
        lon, lat = bbox.centroid
        readings: Iterable[WeatherDaily] = self.weather.fetch_daily(lat, lon, start, end)
        rows = [
            {
                "region_id": region.id,
                "date": r.date,
                "geom": _point_wkt(lon, lat),
                "temperature_max": r.temperature_max,
                "temperature_min": r.temperature_min,
                "precipitation": r.precipitation,
                "wind_speed_max": r.wind_speed_max,
                "evapotranspiration": r.evapotranspiration,
                "source": "open-meteo",
            }
            for r in readings
        ]
        inserted = _upsert(
            self.session, WeatherReading, rows, ["region_id", "date"]
        )
        logger.info("weather.ingested", region=region.name, inserted=inserted)
        return inserted

    def ingest_fire(self, region: Region, bbox: BBox, start: date, end: date) -> int:
        if self.firms is None:
            raise ValueError("firms client not configured")
        detections: Iterable[FireDetection] = self.firms.fetch(bbox, start, end)
        rows = [
            {
                "region_id": region.id,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "geom": _point_wkt(d.longitude, d.latitude),
                "acq_date": d.acq_date,
                "acq_time": d.acq_time,
                "brightness": d.brightness,
                "frp": d.frp,
                "confidence": d.confidence,
                "satellite": d.satellite,
                "instrument": d.instrument,
                "daynight": d.daynight,
                "source": d.source,
            }
            for d in detections
        ]
        inserted = _upsert(
            self.session,
            FireEvent,
            rows,
            ["latitude", "longitude", "acq_date", "acq_time", "source"],
        )
        logger.info("fire.ingested", region=region.name, inserted=inserted)
        return inserted

    def ingest_imagery(
        self,
        region: Region,
        bbox: BBox,
        start: date,
        end: date,
        *,
        asset: str = "thumbnail",
        max_cloud_cover: int | None = None,
        limit: int = 10,
    ) -> int:
        if self.stac is None or self.storage is None:
            raise ValueError("stac client and storage must both be configured")
        cloud = (
            max_cloud_cover
            if max_cloud_cover is not None
            else self.settings.stac_max_cloud_cover
        )
        items: list[StacItem] = self.stac.search(
            bbox, start, end, max_cloud_cover=cloud, limit=limit
        )
        if not items:
            logger.info("imagery.none", region=region.name)
            return 0

        self.storage.ensure_bucket()
        rows: list[dict[str, Any]] = []
        for item in items:
            try:
                data, content_type = self.stac.download_asset(item, asset)
            except KeyError as exc:
                logger.warning("imagery.asset_missing", item=item.item_id, error=str(exc))
                continue
            key = (
                f"{item.collection}/{region.name}/{item.item_id}/"
                f"{asset}{_ext_for(content_type)}"
            )
            self.storage.put_bytes(key, data, content_type=content_type)
            rows.append(
                {
                    "region_id": region.id,
                    "item_id": item.item_id,
                    "collection": item.collection,
                    "asset": asset,
                    "captured_at": item.captured_at,
                    "cloud_cover": item.cloud_cover,
                    "geom": WKTElement(item.geometry_wkt, srid=SRID),
                    "s3_key": key,
                    "content_type": content_type,
                    "size_bytes": len(data),
                    "source": "stac",
                }
            )
        inserted = _upsert(self.session, SatelliteTile, rows, ["item_id", "asset"])
        logger.info("imagery.ingested", region=region.name, inserted=inserted)
        return inserted

    # -- orchestration -----------------------------------------------------

    def run(
        self,
        name: str,
        bbox: BBox,
        start: date,
        end: date,
        *,
        sources: Sequence[str] = DEFAULT_SOURCES,
        asset: str = "thumbnail",
        max_cloud_cover: int | None = None,
        tile_limit: int = 10,
    ) -> IngestSummary:
        """Run the requested ingestion stages and return a summary of inserts."""
        region = get_or_create_region(self.session, name, bbox)
        summary = IngestSummary(region=name, region_id=region.id)

        if "weather" in sources:
            summary.weather_readings = self.ingest_weather(region, bbox, start, end)
        if "fire" in sources:
            summary.fire_events = self.ingest_fire(region, bbox, start, end)
        if "imagery" in sources:
            summary.satellite_tiles = self.ingest_imagery(
                region,
                bbox,
                start,
                end,
                asset=asset,
                max_cloud_cover=max_cloud_cover,
                limit=tile_limit,
            )

        self.session.commit()
        logger.info("ingest.complete", **summary.as_dict())
        return summary


def build_pipeline(
    session: Session,
    *,
    settings: Settings | None = None,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> IngestPipeline:
    """Construct a pipeline with real clients for the requested ``sources``."""
    settings = settings or get_settings()
    firms = (
        FirmsClient(
            settings.firms_map_key,
            settings.firms_source,
            base_url=settings.firms_base_url,
        )
        if "fire" in sources
        else None
    )
    weather = (
        WeatherClient(archive_url=settings.open_meteo_archive_url)
        if "weather" in sources
        else None
    )
    stac = (
        StacClient(api_url=settings.stac_api_url, collection=settings.stac_collection)
        if "imagery" in sources
        else None
    )
    storage = ObjectStorage.from_settings(settings) if "imagery" in sources else None
    return IngestPipeline(
        session,
        firms=firms,
        weather=weather,
        stac=stac,
        storage=storage,
        settings=settings,
    )
