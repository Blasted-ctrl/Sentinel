"""Dataset acquisition, loading, and leakage-safe geospatial splitting."""

from sentinel.data.geosplit import (
    GeoSplit,
    assert_no_region_leakage,
    assign_regions,
    geospatial_split,
    region_key,
    split_by_groups,
)
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
    "assert_no_region_leakage",
    "assign_regions",
    "build_transforms",
    "discover_samples",
    "geospatial_split",
    "region_key",
    "split_by_groups",
]
