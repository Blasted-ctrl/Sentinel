"""Database package: SQLAlchemy engine, ORM models, and schema bootstrap."""

from sentinel.db.base import Base, get_engine, get_sessionmaker, session_scope
from sentinel.db.models import (
    FireEvent,
    Region,
    RiskScore,
    SatelliteTile,
    WeatherReading,
)

__all__ = [
    "Base",
    "FireEvent",
    "Region",
    "RiskScore",
    "SatelliteTile",
    "WeatherReading",
    "get_engine",
    "get_sessionmaker",
    "session_scope",
]
