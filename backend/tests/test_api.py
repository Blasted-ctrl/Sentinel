"""Tests for the FastAPI risk endpoints (repository + DB mocked)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from sentinel.api.app import create_app
from sentinel.api.deps import get_db
from sentinel.serving import repository


def _region(rid: int = 1, name: str = "sierra") -> SimpleNamespace:
    return SimpleNamespace(id=rid, name=name)


def _score(day: date = date(2024, 8, 1)) -> SimpleNamespace:
    return SimpleNamespace(
        date=day,
        ensemble_score=0.72,
        cnn_score=0.6,
        lstm_score=0.81,
        model_version="cnn-lstm-ensemble-v1",
        created_at=datetime(2024, 8, 1, 6, 0, tzinfo=UTC),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_db] = lambda: MagicMock()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_regions(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repository, "list_regions", lambda db: [_region(1, "sierra")])
    resp = client.get("/regions")
    assert resp.status_code == 200
    assert resp.json() == [{"id": 1, "name": "sierra"}]


def test_risk_latest_for_region(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repository, "get_region", lambda db, ref: _region())
    monkeypatch.setattr(
        repository, "latest_risk_score", lambda db, rid, **kw: _score()
    )
    resp = client.get("/risk", params={"region": "sierra"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["region_name"] == "sierra"
    assert body["ensemble_score"] == 0.72
    assert body["lstm_score"] == 0.81


def test_risk_region_not_found(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repository, "get_region", lambda db, ref: None)
    resp = client.get("/risk", params={"region": "nowhere"})
    assert resp.status_code == 404


def test_risk_no_score_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repository, "get_region", lambda db, ref: _region())
    monkeypatch.setattr(repository, "latest_risk_score", lambda db, rid, **kw: None)
    resp = client.get("/risk", params={"region": "sierra"})
    assert resp.status_code == 404


def test_risk_latest_all(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repository, "latest_scores_all", lambda db: [(_region(), _score())]
    )
    resp = client.get("/risk/latest")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["ensemble_score"] == 0.72


def test_risk_geojson(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                "properties": {"region_id": 1, "region_name": "sierra", "ensemble_score": 0.72},
            }
        ],
    }
    monkeypatch.setattr(repository, "regions_geojson", lambda db: fc)
    resp = client.get("/risk/geojson")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert body["features"][0]["properties"]["ensemble_score"] == 0.72
