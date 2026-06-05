"""Dataset acquisition, loading, and leakage-safe geospatial splitting."""

from sentinel.data.geosplit import GeoSplit, assign_regions, geospatial_split, region_key
from sentinel.data.wildfire import (
    TileSample,
    WildfireTileDataset,
    build_transforms,
    discover_samples,
)

__all__ = [
    "GeoSplit",
    "TileSample",
    "WildfireTileDataset",
    "assign_regions",
    "build_transforms",
    "discover_samples",
    "geospatial_split",
    "region_key",
]
