"""Tests for the Open-Meteo weather client (HTTP mocked with respx)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from sentinel.ingest.weather import WeatherClient

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_PAYLOAD = {
    "daily": {
        "time": ["2024-08-01", "2024-08-02"],
        "temperature_2m_max": [35.1, 36.4],
        "temperature_2m_min": [18.2, 19.0],
        "precipitation_sum": [0.0, 1.2],
        "wind_speed_10m_max": [14.5, 22.1],
        "et0_fao_evapotranspiration": [6.1, 6.8],
    }
}


@respx.mock
def test_weather_fetch_daily_parses_rows() -> None:
    respx.get(url__startswith=_ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=_PAYLOAD)
    )
    client = WeatherClient(archive_url=_ARCHIVE_URL, client=httpx.Client())
    rows = client.fetch_daily(38.75, -120.25, date(2024, 8, 1), date(2024, 8, 2))

    assert len(rows) == 2
    assert rows[0].date == date(2024, 8, 1)
    assert rows[0].temperature_max == pytest.approx(35.1)
    assert rows[0].precipitation == pytest.approx(0.0)
    assert rows[1].wind_speed_max == pytest.approx(22.1)
    assert rows[1].evapotranspiration == pytest.approx(6.8)


@respx.mock
def test_weather_handles_missing_values() -> None:
    payload = {
        "daily": {
            "time": ["2024-08-01"],
            "temperature_2m_max": [None],
            "precipitation_sum": [],
        }
    }
    respx.get(url__startswith=_ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=payload)
    )
    client = WeatherClient(archive_url=_ARCHIVE_URL, client=httpx.Client())
    rows = client.fetch_daily(0.0, 0.0, date(2024, 8, 1), date(2024, 8, 1))
    assert len(rows) == 1
    assert rows[0].temperature_max is None
    assert rows[0].precipitation is None


def test_weather_rejects_reversed_range() -> None:
    client = WeatherClient(client=httpx.Client())
    with pytest.raises(ValueError):
        client.fetch_daily(0.0, 0.0, date(2024, 8, 2), date(2024, 8, 1))
