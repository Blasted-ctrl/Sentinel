"""STAC imagery client for Sentinel-2 L2A scenes (Element84 Earth Search).

We query a STAC API for scenes intersecting a bbox over a date range, filtered
by cloud cover, then download a chosen asset (default: the scene thumbnail) so
it can be persisted to object storage. No API key is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.geo import BBox


@dataclass(slots=True)
class StacItem:
    """A minimal view of a STAC item we care about."""

    item_id: str
    collection: str
    captured_at: datetime
    cloud_cover: float | None
    geometry_wkt: str
    assets: dict[str, str]  # asset key -> href


def _geometry_to_wkt(geometry: dict[str, Any]) -> str:
    """Convert a GeoJSON Polygon geometry to a WGS84 POLYGON WKT string."""
    coords = geometry.get("coordinates") or []
    if geometry.get("type") != "Polygon" or not coords:
        # Fallback: empty polygon is invalid; callers should guard, but keep safe.
        return "POLYGON EMPTY"
    ring = coords[0]
    pairs = ", ".join(f"{lon} {lat}" for lon, lat in ring)
    return f"POLYGON(({pairs}))"


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class StacClient:
    """Searches a STAC API and downloads scene assets."""

    def __init__(
        self,
        *,
        api_url: str = "https://earth-search.aws.element84.com/v1",
        collection: str = "sentinel-2-l2a",
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.collection = collection
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def search(
        self,
        bbox: BBox,
        start: date,
        end: date,
        *,
        max_cloud_cover: int = 30,
        limit: int = 10,
    ) -> list[StacItem]:
        """Search for scenes intersecting ``bbox`` within ``[start, end]``."""
        body = {
            "collections": [self.collection],
            "bbox": bbox.as_list(),
            "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
            "limit": limit,
        }
        resp = self._client.post(f"{self.api_url}/search", json=body)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        items: list[StacItem] = []
        for feat in features:
            props = feat.get("properties", {})
            assets = {
                key: asset.get("href", "")
                for key, asset in (feat.get("assets") or {}).items()
                if asset.get("href")
            }
            items.append(
                StacItem(
                    item_id=feat["id"],
                    collection=feat.get("collection", self.collection),
                    captured_at=_parse_datetime(props["datetime"]),
                    cloud_cover=props.get("eo:cloud_cover"),
                    geometry_wkt=_geometry_to_wkt(feat.get("geometry", {})),
                    assets=assets,
                )
            )
        return items

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def download_asset(
        self, item: StacItem, asset: str = "thumbnail"
    ) -> tuple[bytes, str]:
        """Download an asset's bytes, returning ``(data, content_type)``."""
        href = item.assets.get(asset)
        if not href:
            available = ", ".join(sorted(item.assets)) or "<none>"
            raise KeyError(
                f"asset {asset!r} not found on item {item.item_id!r}; "
                f"available: {available}"
            )
        resp = self._client.get(href)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, content_type

    def close(self) -> None:
        self._client.close()
