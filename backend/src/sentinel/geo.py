"""Geospatial helpers for working with bounding boxes.

A bounding box is represented as ``(min_lon, min_lat, max_lon, max_lat)`` in
WGS84 (EPSG:4326), which is the convention used by FIRMS, Open-Meteo, and STAC.
"""

from __future__ import annotations

from typing import NamedTuple

from shapely.geometry import box


class BBox(NamedTuple):
    """An axis-aligned bounding box in WGS84 degrees."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the ``(lon, lat)`` centre of the box."""
        return (
            (self.min_lon + self.max_lon) / 2.0,
            (self.min_lat + self.max_lat) / 2.0,
        )

    def to_wkt(self) -> str:
        """Return the box as a WGS84 POLYGON in WKT."""
        return str(box(self.min_lon, self.min_lat, self.max_lon, self.max_lat).wkt)

    def as_list(self) -> list[float]:
        """Return ``[min_lon, min_lat, max_lon, max_lat]`` (STAC/GeoJSON order)."""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


def parse_bbox(value: str) -> BBox:
    """Parse a ``"min_lon,min_lat,max_lon,max_lat"`` string into a :class:`BBox`.

    Raises:
        ValueError: if the string is malformed or the coordinates are out of
            range / non-ascending.
    """
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"bbox must have 4 comma-separated numbers, got {len(parts)}: {value!r}"
        )
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:  # pragma: no cover - message clarity
        raise ValueError(f"bbox contains non-numeric values: {value!r}") from exc

    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError(f"longitude out of range [-180, 180]: {value!r}")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError(f"latitude out of range [-90, 90]: {value!r}")
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"bbox min must be strictly less than max: {value!r}")

    return BBox(min_lon, min_lat, max_lon, max_lat)
