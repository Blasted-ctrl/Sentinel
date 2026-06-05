"""PostGIS ORM models.

Tables
------
``regions``           Areas of interest (polygons) we forecast for.
``weather_readings``  Daily weather features per region (time + geo indexed).
``fire_events``       Active-fire detections from NASA FIRMS (geo indexed).
``satellite_tiles``   Sentinel-2 scene metadata + the S3 key of the stored tile.

The spatial columns use SRID 4326 (WGS84). GeoAlchemy2 creates a GiST spatial
index automatically for every ``Geometry`` column with ``spatial_index=True``.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel.db.base import Base


class Region(Base):
    """An area of interest we forecast wildfire risk for."""

    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    geom: Mapped[Any] = mapped_column(
        Geometry("POLYGON", srid=4326, spatial_index=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    weather_readings: Mapped[list[WeatherReading]] = relationship(
        back_populates="region", cascade="all, delete-orphan"
    )
    fire_events: Mapped[list[FireEvent]] = relationship(
        back_populates="region", cascade="all, delete-orphan"
    )
    satellite_tiles: Mapped[list[SatelliteTile]] = relationship(
        back_populates="region", cascade="all, delete-orphan"
    )
    risk_scores: Mapped[list[RiskScore]] = relationship(
        back_populates="region", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Region id={self.id} name={self.name!r}>"


class WeatherReading(Base):
    """A daily weather feature vector sampled at a region's centroid."""

    __tablename__ = "weather_readings"
    __table_args__ = (
        UniqueConstraint("region_id", "date", name="uq_weather_region_date"),
        Index("ix_weather_region_date", "region_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(
        ForeignKey("regions.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    geom: Mapped[Any] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=True), nullable=False
    )

    temperature_max: Mapped[float | None] = mapped_column(Float)
    temperature_min: Mapped[float | None] = mapped_column(Float)
    precipitation: Mapped[float | None] = mapped_column(Float)
    wind_speed_max: Mapped[float | None] = mapped_column(Float)
    evapotranspiration: Mapped[float | None] = mapped_column(Float)

    source: Mapped[str] = mapped_column(String(64), default="open-meteo")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    region: Mapped[Region] = relationship(back_populates="weather_readings")


class FireEvent(Base):
    """An active-fire detection from NASA FIRMS."""

    __tablename__ = "fire_events"
    __table_args__ = (
        UniqueConstraint(
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "source",
            name="uq_fire_unique_detection",
        ),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_fire_lat"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="ck_fire_lon"),
        Index("ix_fire_acq_date", "acq_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int | None] = mapped_column(
        ForeignKey("regions.id", ondelete="CASCADE")
    )
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    geom: Mapped[Any] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=True), nullable=False
    )
    acq_date: Mapped[date_type] = mapped_column(Date, nullable=False)
    acq_time: Mapped[str] = mapped_column(String(8), nullable=False)

    brightness: Mapped[float | None] = mapped_column(Float)
    frp: Mapped[float | None] = mapped_column(Float)  # fire radiative power (MW)
    confidence: Mapped[str | None] = mapped_column(String(16))
    satellite: Mapped[str | None] = mapped_column(String(32))
    instrument: Mapped[str | None] = mapped_column(String(32))
    daynight: Mapped[str | None] = mapped_column(String(1))

    source: Mapped[str] = mapped_column(String(64), default="firms")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    region: Mapped[Region | None] = relationship(back_populates="fire_events")


class SatelliteTile(Base):
    """Metadata for a Sentinel-2 scene whose imagery is stored in S3."""

    __tablename__ = "satellite_tiles"
    __table_args__ = (
        UniqueConstraint("item_id", "asset", name="uq_tile_item_asset"),
        Index("ix_tile_captured_at", "captured_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int | None] = mapped_column(
        ForeignKey("regions.id", ondelete="CASCADE")
    )
    item_id: Mapped[str] = mapped_column(String(256), nullable=False)
    collection: Mapped[str] = mapped_column(String(64), default="sentinel-2-l2a")
    asset: Mapped[str] = mapped_column(String(64), default="thumbnail")
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cloud_cover: Mapped[float | None] = mapped_column(Float)
    geom: Mapped[Any] = mapped_column(
        Geometry("POLYGON", srid=4326, spatial_index=True), nullable=False
    )

    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int | None] = mapped_column(Integer)

    source: Mapped[str] = mapped_column(String(64), default="stac")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    region: Mapped[Region | None] = relationship(back_populates="satellite_tiles")


class RiskScore(Base):
    """A fused wildfire-risk score for a region on a given day."""

    __tablename__ = "risk_scores"
    __table_args__ = (
        UniqueConstraint("region_id", "date", name="uq_risk_region_date"),
        Index("ix_risk_region_date", "region_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(
        ForeignKey("regions.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date_type] = mapped_column(Date, nullable=False)

    ensemble_score: Mapped[float] = mapped_column(Float, nullable=False)
    cnn_score: Mapped[float | None] = mapped_column(Float)
    lstm_score: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    region: Mapped[Region] = relationship(back_populates="risk_scores")
