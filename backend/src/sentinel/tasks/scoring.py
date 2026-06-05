"""Scheduled re-scoring: compute and store risk scores for all tracked regions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from sentinel.config import get_settings
from sentinel.db.base import session_scope
from sentinel.db.models import Region
from sentinel.geo import BBox
from sentinel.logging import get_logger
from sentinel.serving import repository
from sentinel.serving.scorer import RegionRiskScorer
from sentinel.tasks.celery_app import celery_app

logger = get_logger(__name__)

_scorer: RegionRiskScorer | None = None


def region_to_bbox(region: Region) -> BBox:
    """Derive a region's bounding box from its PostGIS polygon."""
    min_lon, min_lat, max_lon, max_lat = to_shape(region.geom).bounds
    return BBox(min_lon, min_lat, max_lon, max_lat)


def get_scorer() -> RegionRiskScorer:
    """Lazily build (and cache) the heavy scorer from configured artifacts."""
    global _scorer
    if _scorer is None:
        s = get_settings()
        _scorer = RegionRiskScorer.from_artifacts(
            s.lstm_dir, s.cnn_checkpoint, s.ensemble_path
        )
    return _scorer


def rescore_regions(
    session: Session,
    scorer: RegionRiskScorer,
    day: date,
    *,
    bbox_fn: Callable[[Region], BBox] = region_to_bbox,
) -> dict[str, Any]:
    """Score every region for ``day`` and upsert the results. Returns a summary."""
    scored = 0
    for region in repository.list_regions(session):
        components = scorer.score(region.name, bbox_fn(region), day)
        repository.upsert_risk_score(
            session,
            region.id,
            day,
            ensemble_score=components.ensemble,
            cnn_score=components.cnn,
            lstm_score=components.lstm,
            model_version=scorer.model_version,
        )
        scored += 1
    logger.info("rescore.done", date=day.isoformat(), scored=scored)
    return {"date": day.isoformat(), "scored": scored}


@celery_app.task(name="sentinel.tasks.scoring.rescore_all_regions")
def rescore_all_regions(day_iso: str | None = None) -> dict[str, Any]:
    """Celery entry point: re-score all regions for ``day_iso`` (default today)."""
    day = date.fromisoformat(day_iso) if day_iso else date.today()
    with session_scope() as session:
        return rescore_regions(session, get_scorer(), day)
