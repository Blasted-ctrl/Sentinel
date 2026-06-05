"""Data acquisition: FIRMS fire labels, Open-Meteo weather, STAC imagery."""

from sentinel.ingest.firms import FireDetection, FirmsClient
from sentinel.ingest.imagery import StacClient, StacItem
from sentinel.ingest.storage import ObjectStorage
from sentinel.ingest.weather import WeatherClient, WeatherDaily

__all__ = [
    "FireDetection",
    "FirmsClient",
    "ObjectStorage",
    "StacClient",
    "StacItem",
    "WeatherClient",
    "WeatherDaily",
]
