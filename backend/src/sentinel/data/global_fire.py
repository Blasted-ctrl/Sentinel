"""Global fire-occurrence dataset → leakage-safe, globally-spread training regions.

Source: NASA MODIS active-fire detections (global, one year), a CSV with
``latitude, longitude, acq_date`` columns and tens of millions of rows.

We cannot fetch weather for the whole globe, so we pick the most fire-active
grid cells *with geographic spread* (round-robin across coarse macro-cells) so
the LSTM sees California, the Mediterranean, Australia, the Amazon, Siberia,
African savanna, etc. — not just the single most fire-dense region.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.data.climate import GridRegion
from sentinel.geo import BBox
from sentinel.logging import get_logger

logger = get_logger(__name__)

_USE_COLS = {"latitude", "longitude", "acq_date"}


def _read_chunks(csv_path: str | Path, chunksize: int) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(
        csv_path,
        usecols=lambda c: c.lower() in _USE_COLS,
        chunksize=chunksize,
    ):
        chunk.columns = [c.lower() for c in chunk.columns]
        yield chunk


def _cell_keys(lon: np.ndarray, lat: np.ndarray, cell_size: float) -> np.ndarray:
    cx = np.floor(lon / cell_size).astype(int)
    cy = np.floor(lat / cell_size).astype(int)
    return np.char.add(np.char.add(cx.astype(str), ":"), cy.astype(str))


def _bbox_for_key(key: str, cell_size: float) -> BBox:
    cx_s, cy_s = key.split(":")
    cx, cy = int(cx_s), int(cy_s)
    return BBox(cx * cell_size, cy * cell_size, (cx + 1) * cell_size, (cy + 1) * cell_size)


def count_fires_per_cell(
    csv_path: str | Path, cell_size: float, *, chunksize: int = 2_000_000
) -> tuple[dict[str, int], date, date]:
    """First pass: fire count per cell + the global ``(min_date, max_date)``."""
    counts: dict[str, int] = defaultdict(int)
    min_date: date | None = None
    max_date: date | None = None
    for chunk in _read_chunks(csv_path, chunksize):
        keys = _cell_keys(
            chunk["longitude"].to_numpy(), chunk["latitude"].to_numpy(), cell_size
        )
        for key, n in pd.Series(keys).value_counts().items():
            counts[str(key)] += int(n)
        days = pd.to_datetime(chunk["acq_date"]).dt.date
        cmin, cmax = days.min(), days.max()
        min_date = cmin if min_date is None else min(min_date, cmin)
        max_date = cmax if max_date is None else max(max_date, cmax)
    assert min_date is not None and max_date is not None
    logger.info("global_fire.counted", cells=len(counts), span=f"{min_date}..{max_date}")
    return dict(counts), min_date, max_date


def select_regions(
    counts: dict[str, int],
    cell_size: float,
    *,
    max_regions: int,
    min_fires: int,
    macro_size: float = 15.0,
) -> list[GridRegion]:
    """Pick up to ``max_regions`` fire-active cells, spread across the globe."""
    macro: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for key, count in counts.items():
        if count < min_fires:
            continue
        bbox = _bbox_for_key(key, cell_size)
        lon, lat = bbox.centroid
        macro_key = (math.floor(lon / macro_size), math.floor(lat / macro_size))
        macro[macro_key].append((key, count))

    pools = [sorted(cells, key=lambda t: -t[1]) for cells in macro.values()]
    pools.sort(key=lambda p: -p[0][1])  # macro-cells with hotter peaks first

    selected: list[str] = []
    while len(selected) < max_regions and any(pools):
        for pool in pools:
            if len(selected) >= max_regions:
                break
            if pool:
                selected.append(pool.pop(0)[0])
        pools = [p for p in pools if p]

    logger.info(
        "global_fire.selected", regions=len(selected), macro_cells=len(macro)
    )
    return [GridRegion(key=k, bbox=_bbox_for_key(k, cell_size)) for k in selected]


def fire_dates_for_regions(
    csv_path: str | Path,
    regions: list[GridRegion],
    cell_size: float,
    *,
    chunksize: int = 2_000_000,
) -> dict[str, set[date]]:
    """Second pass: collect fire dates for the selected regions only."""
    wanted = {r.key for r in regions}
    out: dict[str, set[date]] = {r.key: set() for r in regions}
    for chunk in _read_chunks(csv_path, chunksize):
        keys = _cell_keys(
            chunk["longitude"].to_numpy(), chunk["latitude"].to_numpy(), cell_size
        )
        chunk = chunk.assign(_key=keys)
        chunk = chunk[chunk["_key"].isin(wanted)]
        if chunk.empty:
            continue
        days = pd.to_datetime(chunk["acq_date"]).dt.date
        for key, day in zip(chunk["_key"].to_numpy(), days.to_numpy(), strict=True):
            out[str(key)].add(day)
    return out


def build_global_fire_regions(
    csv_path: str | Path,
    *,
    cell_size: float = 2.0,
    max_regions: int = 250,
    min_fires: int = 30,
    macro_size: float = 15.0,
) -> tuple[list[GridRegion], dict[str, set[date]], tuple[date, date]]:
    """End-to-end: select globally-spread fire regions + their fire dates."""
    counts, min_date, max_date = count_fires_per_cell(csv_path, cell_size)
    regions = select_regions(
        counts,
        cell_size,
        max_regions=max_regions,
        min_fires=min_fires,
        macro_size=macro_size,
    )
    fires_by_region = fire_dates_for_regions(csv_path, regions, cell_size)
    return regions, fires_by_region, (min_date, max_date)
