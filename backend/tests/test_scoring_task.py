"""Tests for the Celery re-scoring task (scorer + DB mocked)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.geo import BBox
from sentinel.serving.scorer import RiskComponents
from sentinel.tasks import scoring


class _FakeScorer:
    model_version = "test-v1"

    def score(self, key: str, bbox: BBox, day: date) -> RiskComponents:
        return RiskComponents(ensemble=0.7, cnn=0.6, lstm=0.8)


def test_rescore_regions_scores_each(monkeypatch: pytest.MonkeyPatch) -> None:
    regions = [
        SimpleNamespace(id=1, name="a", geom=None),
        SimpleNamespace(id=2, name="b", geom=None),
    ]
    monkeypatch.setattr(scoring.repository, "list_regions", lambda s: regions)

    captured: list[tuple[int, dict[str, Any]]] = []
    monkeypatch.setattr(
        scoring.repository,
        "upsert_risk_score",
        lambda s, rid, day, **kw: captured.append((rid, kw)),
    )

    summary = scoring.rescore_regions(
        MagicMock(),
        _FakeScorer(),
        date(2024, 8, 1),
        bbox_fn=lambda r: BBox(-1.0, -1.0, 1.0, 1.0),
    )

    assert summary == {"date": "2024-08-01", "scored": 2}
    assert {rid for rid, _ in captured} == {1, 2}
    assert captured[0][1]["ensemble_score"] == 0.7
    assert captured[0][1]["model_version"] == "test-v1"


def test_celery_task_and_beat_registered() -> None:
    from sentinel.tasks.celery_app import celery_app

    assert "daily-rescore-regions" in celery_app.conf.beat_schedule
    assert "sentinel.tasks.scoring.rescore_all_regions" in celery_app.tasks
