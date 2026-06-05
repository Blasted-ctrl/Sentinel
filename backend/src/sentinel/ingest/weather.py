"""Open-Meteo daily weather client (historical archive, no API key required).

We sample one daily feature vector per region at its centroid. The features are
chosen for fire relevance: temperature, precipitation, wind, and reference
evapotranspiration (a dryness proxy).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
]


@dataclass(slots=True)
class WeatherDaily:
    """A single day's weather features at a point."""

    date: date
    temperature_max: float | None
    temperature_min: float | None
    precipitation: float | None
    wind_speed_max: float | None
    evapotranspiration: float | None


class WeatherClient:
    """Fetches daily weather features from the Open-Meteo archive API."""

    def __init__(
        self,
        *,
        archive_url: str = "https://archive-api.open-meteo.com/v1/archive",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.archive_url = archive_url
        self._client = client or httpx.Client(timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _get(self, params: dict[str, str]) -> dict[str, object]:
        resp = self._client.get(self.archive_url, params=params)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return data

    def fetch_daily(
        self, lat: float, lon: float, start: date, end: date
    ) -> list[WeatherDaily]:
        """Return one :class:`WeatherDaily` per day in ``[start, end]`` inclusive."""
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(_DAILY_VARS),
            "timezone": "UTC",
        }
        payload = self._get(params)
        daily = payload.get("daily")
        if not isinstance(daily, dict):
            return []

        times = daily.get("time") or []
        if not isinstance(times, list):
            return []

        def col(name: str) -> list[object]:
            value = daily.get(name)
            return value if isinstance(value, list) else []

        temp_max = col("temperature_2m_max")
        temp_min = col("temperature_2m_min")
        precip = col("precipitation_sum")
        wind = col("wind_speed_10m_max")
        et0 = col("et0_fao_evapotranspiration")

        readings: list[WeatherDaily] = []
        for i, day in enumerate(times):
            readings.append(
                WeatherDaily(
                    date=date.fromisoformat(str(day)),
                    temperature_max=_at(temp_max, i),
                    temperature_min=_at(temp_min, i),
                    precipitation=_at(precip, i),
                    wind_speed_max=_at(wind, i),
                    evapotranspiration=_at(et0, i),
                )
            )
        return readings

    def close(self) -> None:
        self._client.close()


def _at(values: list[object], index: int) -> float | None:
    if index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
