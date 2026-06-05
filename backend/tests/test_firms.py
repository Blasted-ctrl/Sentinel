"""Tests for the NASA FIRMS client (HTTP mocked with respx)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from sentinel.geo import BBox
from sentinel.ingest.firms import FirmsClient, _date_windows

_VIIRS_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "38.7,-120.2,330.5,0.5,0.5,2024-08-01,0712,N,VIIRS,n,2.0NRT,295.1,12.4,D\n"
    "38.8,-120.1,310.0,0.5,0.5,2024-08-01,0712,N,VIIRS,h,2.0NRT,290.0,5.0,N\n"
)


def test_date_windows_splits_into_10_day_chunks() -> None:
    windows = _date_windows(date(2024, 8, 1), date(2024, 8, 25))
    assert windows == [
        (date(2024, 8, 1), 10),
        (date(2024, 8, 11), 10),
        (date(2024, 8, 21), 5),
    ]


def test_date_windows_rejects_reversed_range() -> None:
    with pytest.raises(ValueError):
        _date_windows(date(2024, 8, 2), date(2024, 8, 1))


def test_firms_requires_map_key() -> None:
    with pytest.raises(ValueError):
        FirmsClient("")


@respx.mock
def test_firms_fetch_parses_detections(sample_bbox: BBox) -> None:
    route = respx.get(url__regex=r".*/area/csv/.*").mock(
        return_value=httpx.Response(200, text=_VIIRS_CSV)
    )
    client = FirmsClient("FAKEKEY", "VIIRS_SNPP_NRT", client=httpx.Client())
    detections = client.fetch(sample_bbox, date(2024, 8, 1), date(2024, 8, 1))

    assert route.called
    assert len(detections) == 2
    first = detections[0]
    assert first.latitude == pytest.approx(38.7)
    assert first.longitude == pytest.approx(-120.2)
    assert first.acq_date == date(2024, 8, 1)
    assert first.acq_time == "0712"
    assert first.brightness == pytest.approx(330.5)
    assert first.frp == pytest.approx(12.4)
    assert first.confidence == "n"
    assert first.source == "VIIRS_SNPP_NRT"


@respx.mock
def test_firms_skips_malformed_rows(sample_bbox: BBox) -> None:
    bad_csv = (
        "latitude,longitude,acq_date,acq_time,frp\n"
        ",,2024-08-01,0712,1.0\n"  # missing coords -> skipped
        "39.0,-120.3,2024-08-01,0800,2.5\n"
    )
    respx.get(url__regex=r".*/area/csv/.*").mock(
        return_value=httpx.Response(200, text=bad_csv)
    )
    client = FirmsClient("FAKEKEY", client=httpx.Client())
    detections = client.fetch(sample_bbox, date(2024, 8, 1), date(2024, 8, 1))
    assert len(detections) == 1
    assert detections[0].latitude == pytest.approx(39.0)
