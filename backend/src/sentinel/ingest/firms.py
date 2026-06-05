"""NASA FIRMS active-fire detection client.

FIRMS exposes a CSV "area" API:

    {base}/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{day_range}/{date}

``day_range`` is 1-10 and the response covers ``day_range`` days starting at
``date``. We split arbitrary date ranges into <=10-day windows and concatenate.

Get a free MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.geo import BBox

_MAX_WINDOW_DAYS = 10


@dataclass(slots=True)
class FireDetection:
    """A single active-fire detection."""

    latitude: float
    longitude: float
    acq_date: date
    acq_time: str
    brightness: float | None
    frp: float | None
    confidence: str | None
    satellite: str | None
    instrument: str | None
    daynight: str | None
    source: str


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _date_windows(start: date, end: date) -> list[tuple[date, int]]:
    """Split ``[start, end]`` inclusive into ``(window_start, day_range)`` chunks."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    windows: list[tuple[date, int]] = []
    cursor = start
    while cursor <= end:
        remaining = (end - cursor).days + 1
        span = min(remaining, _MAX_WINDOW_DAYS)
        windows.append((cursor, span))
        cursor += timedelta(days=span)
    return windows


class FirmsClient:
    """Fetches active-fire detections from NASA FIRMS."""

    def __init__(
        self,
        map_key: str,
        source: str = "VIIRS_SNPP_NRT",
        *,
        base_url: str = "https://firms.modaps.eosdis.nasa.gov/api",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not map_key:
            raise ValueError("FIRMS map_key is required (set FIRMS_MAP_KEY)")
        self.map_key = map_key
        self.source = source
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _get(self, url: str) -> str:
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.text

    def _parse_csv(self, text: str) -> list[FireDetection]:
        reader = csv.DictReader(io.StringIO(text))
        detections: list[FireDetection] = []
        for row in reader:
            lat = _to_float(row.get("latitude"))
            lon = _to_float(row.get("longitude"))
            acq_date_raw = row.get("acq_date")
            if lat is None or lon is None or not acq_date_raw:
                continue
            # FIRMS reports brightness as bright_ti4 (VIIRS) or brightness (MODIS).
            brightness = _to_float(row.get("bright_ti4")) or _to_float(
                row.get("brightness")
            )
            detections.append(
                FireDetection(
                    latitude=lat,
                    longitude=lon,
                    acq_date=date.fromisoformat(acq_date_raw),
                    acq_time=(row.get("acq_time") or "").strip() or "0000",
                    brightness=brightness,
                    frp=_to_float(row.get("frp")),
                    confidence=(row.get("confidence") or "").strip() or None,
                    satellite=(row.get("satellite") or "").strip() or None,
                    instrument=(row.get("instrument") or "").strip() or None,
                    daynight=(row.get("daynight") or "").strip() or None,
                    source=self.source,
                )
            )
        return detections

    def fetch(self, bbox: BBox, start: date, end: date) -> list[FireDetection]:
        """Fetch all detections in ``bbox`` over ``[start, end]`` inclusive."""
        area = f"{bbox.min_lon},{bbox.min_lat},{bbox.max_lon},{bbox.max_lat}"
        results: list[FireDetection] = []
        for window_start, span in _date_windows(start, end):
            url = (
                f"{self.base_url}/area/csv/{self.map_key}/{self.source}/"
                f"{area}/{span}/{window_start.isoformat()}"
            )
            results.extend(self._parse_csv(self._get(url)))
        return results

    def close(self) -> None:
        self._client.close()
