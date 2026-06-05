"""Climate time-series dataset for the LSTM ignition-risk model.

Builds labelled sliding windows of daily weather per region:

* **Regions** are grid cells over an area of interest.
* **Weather** comes from the Open-Meteo archive (multi-year ERA5 reanalysis).
* **Labels** come from the FPA-FOD US wildfire-occurrence database: a window
  ending on day *t* is positive if a fire ignites in the region within the next
  ``horizon`` days.

Everything is grouped by region so the same leakage-safe geospatial split used
for the CNN applies here too.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from sentinel.geo import BBox
from sentinel.ingest.weather import WeatherClient, WeatherDaily
from sentinel.logging import get_logger

logger = get_logger(__name__)

# Per-day weather features fed to the LSTM (plus two seasonality channels).
WEATHER_FEATURES = [
    "temperature_max",
    "temperature_min",
    "precipitation",
    "wind_speed_max",
    "evapotranspiration",
]
FEATURE_NAMES = [*WEATHER_FEATURES, "season_sin", "season_cos"]
N_FEATURES = len(FEATURE_NAMES)


@dataclass(slots=True)
class GridRegion:
    """A grid-cell region with a key, bbox, and centroid."""

    key: str
    bbox: BBox

    @property
    def centroid(self) -> tuple[float, float]:
        return self.bbox.centroid


@dataclass(slots=True)
class ClimateDataset:
    """Windowed climate sequences with labels and region groups."""

    x: np.ndarray  # (n, lookback, n_features)
    y: np.ndarray  # (n,)
    groups: list[str]  # region key per sample
    dates: list[date]  # window end-date per sample
    feature_names: list[str]


def grid_regions(bbox: BBox, cell_size: float) -> list[GridRegion]:
    """Tile ``bbox`` into ``cell_size``-degree grid-cell regions."""
    regions: list[GridRegion] = []
    lon = bbox.min_lon
    while lon < bbox.max_lon:
        lat = bbox.min_lat
        while lat < bbox.max_lat:
            cell = BBox(
                lon,
                lat,
                min(lon + cell_size, bbox.max_lon),
                min(lat + cell_size, bbox.max_lat),
            )
            cx = math.floor(lon / cell_size)
            cy = math.floor(lat / cell_size)
            regions.append(GridRegion(key=f"{cx}:{cy}", bbox=cell))
            lat += cell_size
        lon += cell_size
    return regions


def _fod_date(fire_year: int, discovery_doy: int) -> date:
    return date(fire_year, 1, 1) + timedelta(days=int(discovery_doy) - 1)


def load_fod_fires(
    sqlite_path: str | Path,
    bbox: BBox,
    start_year: int,
    end_year: int,
) -> list[tuple[date, float, float]]:
    """Load ``(date, lon, lat)`` fire occurrences from the FPA-FOD SQLite db."""
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.execute(
            """
            SELECT FIRE_YEAR, DISCOVERY_DOY, LONGITUDE, LATITUDE
            FROM Fires
            WHERE FIRE_YEAR BETWEEN ? AND ?
              AND LONGITUDE BETWEEN ? AND ?
              AND LATITUDE BETWEEN ? AND ?
              AND DISCOVERY_DOY IS NOT NULL
            """,
            (start_year, end_year, bbox.min_lon, bbox.max_lon, bbox.min_lat, bbox.max_lat),
        )
        fires = [
            (_fod_date(int(year), int(doy)), float(lon), float(lat))
            for year, doy, lon, lat in cur.fetchall()
        ]
    finally:
        conn.close()
    logger.info("fod.loaded", count=len(fires), years=f"{start_year}-{end_year}")
    return fires


def fire_dates_by_region(
    regions: Sequence[GridRegion], fires: Sequence[tuple[date, float, float]]
) -> dict[str, set[date]]:
    """Bucket fire occurrences into the region cell that contains them."""
    by_region: dict[str, set[date]] = {r.key: set() for r in regions}
    for region in regions:
        b = region.bbox
        for fdate, lon, lat in fires:
            if b.min_lon <= lon < b.max_lon and b.min_lat <= lat < b.max_lat:
                by_region[region.key].add(fdate)
    return by_region


def _daily_feature_rows(weather: Sequence[WeatherDaily]) -> tuple[np.ndarray, list[date]]:
    """Build a forward-filled feature matrix (n_days, n_features) and dates."""
    rows: list[list[float]] = []
    dates: list[date] = []
    last = dict.fromkeys(WEATHER_FEATURES, 0.0)
    for day in sorted(weather, key=lambda w: w.date):
        values = {
            "temperature_max": day.temperature_max,
            "temperature_min": day.temperature_min,
            "precipitation": day.precipitation,
            "wind_speed_max": day.wind_speed_max,
            "evapotranspiration": day.evapotranspiration,
        }
        feats = []
        for name in WEATHER_FEATURES:
            v = values[name]
            if v is None:
                v = last[name]  # forward fill
            last[name] = v
            feats.append(float(v))
        angle = 2.0 * math.pi * (day.date.timetuple().tm_yday / 365.0)
        feats.append(math.sin(angle))
        feats.append(math.cos(angle))
        rows.append(feats)
        dates.append(day.date)
    return np.asarray(rows, dtype=np.float32), dates


def build_region_windows(
    region_key: str,
    weather: Sequence[WeatherDaily],
    fire_days: set[date],
    *,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[date]]:
    """Build sliding windows for one region.

    A window ending on day ``t`` is labelled 1 if any fire occurs in
    ``(t, t + horizon]``.
    """
    features, dates = _daily_feature_rows(weather)
    n = len(dates)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[str] = []
    end_dates: list[date] = []
    for t in range(lookback - 1, n - horizon):
        window = features[t - lookback + 1 : t + 1]
        if window.shape[0] != lookback:
            continue
        horizon_days = {dates[t] + timedelta(days=h) for h in range(1, horizon + 1)}
        label = 1 if horizon_days & fire_days else 0
        xs.append(window)
        ys.append(label)
        groups.append(region_key)
        end_dates.append(dates[t])
    if not xs:
        empty = np.empty((0, lookback, N_FEATURES), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64), [], []
    return np.stack(xs), np.asarray(ys, dtype=np.int64), groups, end_dates


def fetch_weather_for_regions(
    regions: Sequence[GridRegion],
    start: date,
    end: date,
    *,
    client: WeatherClient | None = None,
    delay: float = 1.5,
) -> dict[str, list[WeatherDaily]]:
    """Fetch each region's daily weather series (one Open-Meteo call per region).

    ``delay`` seconds are slept between calls to stay under Open-Meteo's free
    rate limit.
    """
    own_client = client is None
    weather_client = client or WeatherClient()
    out: dict[str, list[WeatherDaily]] = {}
    try:
        for i, region in enumerate(regions):
            if i > 0 and delay > 0:
                time.sleep(delay)
            lon, lat = region.centroid
            out[region.key] = weather_client.fetch_daily(lat, lon, start, end)
            logger.info("weather.fetched", region=region.key, days=len(out[region.key]))
    finally:
        if own_client:
            weather_client.close()
    return out


def assemble_climate_dataset(
    regions: Sequence[GridRegion],
    weather_by_region: dict[str, list[WeatherDaily]],
    fires_by_region: dict[str, set[date]],
    *,
    lookback: int = 14,
    horizon: int = 7,
) -> ClimateDataset:
    """Assemble all regions' windows into a single :class:`ClimateDataset`."""
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    groups: list[str] = []
    dates: list[date] = []
    for region in regions:
        weather = weather_by_region.get(region.key, [])
        if len(weather) < lookback + horizon:
            continue
        xr, yr, gr, dr = build_region_windows(
            region.key,
            weather,
            fires_by_region.get(region.key, set()),
            lookback=lookback,
            horizon=horizon,
        )
        if xr.shape[0] == 0:
            continue
        x_parts.append(xr)
        y_parts.append(yr)
        groups.extend(gr)
        dates.extend(dr)

    if not x_parts:
        raise ValueError("no climate windows could be built; check inputs")
    return ClimateDataset(
        x=np.concatenate(x_parts),
        y=np.concatenate(y_parts),
        groups=groups,
        dates=dates,
        feature_names=FEATURE_NAMES,
    )


def standardize(
    x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize (n, L, F) sequences per feature; fit stats if not provided."""
    if mean is None or std is None:
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (x - mean) / std_safe, mean, std_safe
