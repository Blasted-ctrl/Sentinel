"""Curated, globally-distributed fire-prone regions for the demo overview map.

These give a meaningful worldwide snapshot (real fire regions on every inhabited
continent) rather than an ocean-cluttered grid. Per-point prediction anywhere is
available via the API's ``/risk/point`` endpoint.
"""

from __future__ import annotations

from sentinel.geo import BBox

# (display name, centre longitude, centre latitude)
_WORLD_FIRE_REGIONS: list[tuple[str, float, float]] = [
    ("California, USA", -120.5, 38.5),
    ("Pacific NW, USA", -121.5, 44.5),
    ("Colorado, USA", -106.0, 39.3),
    ("British Columbia, Canada", -121.5, 51.0),
    ("Mediterranean, Greece", 22.5, 38.7),
    ("Portugal", -8.0, 40.0),
    ("Andalusia, Spain", -4.6, 37.4),
    ("Kabylie, Algeria", 4.5, 36.6),
    ("Victoria, Australia", 145.5, -37.5),
    ("New South Wales, Australia", 150.0, -33.5),
    ("Western Australia", 116.0, -31.5),
    ("Cerrado, Brazil", -48.0, -12.0),
    ("Amazonia, Brazil", -52.0, -5.0),
    ("Central Chile", -71.0, -35.0),
    ("Western Cape, South Africa", 19.2, -33.9),
    ("Sumatra, Indonesia", 102.0, -0.8),
    ("Yakutia, Siberia", 129.0, 62.0),
    ("Uttarakhand, India", 79.0, 30.2),
]


def world_fire_regions(box_size: float = 1.2) -> list[tuple[str, BBox]]:
    """Return ``(name, bbox)`` for each curated region (a ``box_size`` deg cell)."""
    half = box_size / 2.0
    out: list[tuple[str, BBox]] = []
    for name, lon, lat in _WORLD_FIRE_REGIONS:
        out.append(
            (name, BBox(lon - half, lat - half, lon + half, lat + half))
        )
    return out
