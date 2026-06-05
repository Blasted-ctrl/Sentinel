"""Tests for the STAC imagery client (HTTP mocked with respx)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from sentinel.geo import BBox
from sentinel.ingest.imagery import StacClient

_API = "https://earth-search.aws.element84.com/v1"

_SEARCH_RESPONSE = {
    "type": "FeatureCollection",
    "features": [
        {
            "id": "S2A_T10SFH_20240801_0_L2A",
            "collection": "sentinel-2-l2a",
            "properties": {
                "datetime": "2024-08-01T18:55:01Z",
                "eo:cloud_cover": 3.2,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-120.5, 38.5],
                        [-120.0, 38.5],
                        [-120.0, 39.0],
                        [-120.5, 39.0],
                        [-120.5, 38.5],
                    ]
                ],
            },
            "assets": {
                "thumbnail": {"href": "https://tiles.example/thumb.jpg"},
                "visual": {"href": "https://tiles.example/visual.tif"},
            },
        }
    ],
}


@respx.mock
def test_stac_search_parses_items(sample_bbox: BBox) -> None:
    respx.post(f"{_API}/search").mock(
        return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
    )
    client = StacClient(api_url=_API, client=httpx.Client())
    items = client.search(sample_bbox, date(2024, 8, 1), date(2024, 8, 2))

    assert len(items) == 1
    item = items[0]
    assert item.item_id == "S2A_T10SFH_20240801_0_L2A"
    assert item.cloud_cover == pytest.approx(3.2)
    assert item.captured_at.year == 2024
    assert item.geometry_wkt.startswith("POLYGON((")
    assert "thumbnail" in item.assets


@respx.mock
def test_stac_download_asset(sample_bbox: BBox) -> None:
    respx.post(f"{_API}/search").mock(
        return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
    )
    respx.get("https://tiles.example/thumb.jpg").mock(
        return_value=httpx.Response(
            200, content=b"\xff\xd8\xff\xe0jpegdata", headers={"content-type": "image/jpeg"}
        )
    )
    client = StacClient(api_url=_API, client=httpx.Client())
    item = client.search(sample_bbox, date(2024, 8, 1), date(2024, 8, 2))[0]
    data, content_type = client.download_asset(item, "thumbnail")

    assert data.startswith(b"\xff\xd8\xff")
    assert content_type == "image/jpeg"


@respx.mock
def test_stac_download_missing_asset_raises(sample_bbox: BBox) -> None:
    respx.post(f"{_API}/search").mock(
        return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
    )
    client = StacClient(api_url=_API, client=httpx.Client())
    item = client.search(sample_bbox, date(2024, 8, 1), date(2024, 8, 2))[0]
    with pytest.raises(KeyError):
        client.download_asset(item, "nonexistent")
