"""Tests for global fire aggregation + globally-spread region selection."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from sentinel.data.global_fire import (
    build_global_fire_regions,
    count_fires_per_cell,
    fire_dates_for_regions,
    select_regions,
)


def _write_csv(path: Path) -> None:
    rows: list[tuple[float, float, str]] = []
    for d in ["2020-06-01", "2020-06-02", "2020-06-03", "2020-06-10"]:
        rows.append((40.1, 5.1, d))  # Mediterranean cell
    for d in ["2020-07-01", "2020-07-02", "2020-07-03"]:
        rows.append((38.2, -120.2, d))  # California cell
    for d in ["2020-12-01", "2020-12-02", "2020-12-03", "2020-12-04", "2020-12-05"]:
        rows.append((-35.0, 145.5, d))  # SE Australia cell
    pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date"]).to_csv(
        path, index=False
    )


def test_count_and_select(tmp_path: Path) -> None:
    csv = tmp_path / "fire.csv"
    _write_csv(csv)

    counts, min_date, max_date = count_fires_per_cell(csv, 2.0, chunksize=3)
    assert sum(counts.values()) == 12
    assert min_date == date(2020, 6, 1)
    assert max_date == date(2020, 12, 5)

    regions = select_regions(counts, 2.0, max_regions=10, min_fires=3)
    assert len(regions) == 3

    dates = fire_dates_for_regions(csv, regions, 2.0, chunksize=3)
    assert sum(len(v) for v in dates.values()) == 12


def test_min_fires_filters_sparse_cells(tmp_path: Path) -> None:
    csv = tmp_path / "fire.csv"
    _write_csv(csv)
    counts, _, _ = count_fires_per_cell(csv, 2.0)
    regions = select_regions(counts, 2.0, max_regions=10, min_fires=4)
    # Only the Mediterranean (4) and Australia (5) cells qualify; California (3) drops.
    assert len(regions) == 2


def test_build_end_to_end_is_globally_spread(tmp_path: Path) -> None:
    csv = tmp_path / "fire.csv"
    _write_csv(csv)
    regions, fires, _span = build_global_fire_regions(
        csv, cell_size=2.0, max_regions=10, min_fires=3
    )
    assert len(regions) == 3
    assert all(fires[r.key] for r in regions)
    lons = [r.bbox.centroid[0] for r in regions]
    assert max(lons) - min(lons) > 100  # spans the globe, not one region
