"""Database access helpers for regions and risk scores."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from sentinel.db.models import Region, RiskScore


def get_region(session: Session, ref: str) -> Region | None:
    """Look up a region by numeric id or by name."""
    if ref.isdigit():
        return session.get(Region, int(ref))
    return session.execute(
        select(Region).where(Region.name == ref)
    ).scalar_one_or_none()


def list_regions(session: Session) -> list[Region]:
    return list(session.execute(select(Region).order_by(Region.name)).scalars())


def get_risk_score(session: Session, region_id: int, day: date) -> RiskScore | None:
    return session.execute(
        select(RiskScore).where(
            RiskScore.region_id == region_id, RiskScore.date == day
        )
    ).scalar_one_or_none()


def latest_risk_score(
    session: Session, region_id: int, *, on_or_before: date | None = None
) -> RiskScore | None:
    stmt = select(RiskScore).where(RiskScore.region_id == region_id)
    if on_or_before is not None:
        stmt = stmt.where(RiskScore.date <= on_or_before)
    return session.execute(
        stmt.order_by(desc(RiskScore.date)).limit(1)
    ).scalar_one_or_none()


def latest_scores_all(session: Session) -> list[tuple[Region, RiskScore]]:
    """Return each region paired with its most recent risk score (if any)."""
    out: list[tuple[Region, RiskScore]] = []
    for region in list_regions(session):
        score = latest_risk_score(session, region.id)
        if score is not None:
            out.append((region, score))
    return out


def regions_geojson(session: Session) -> dict[str, Any]:
    """Return regions as a GeoJSON FeatureCollection with their latest risk score."""
    rows = session.execute(
        select(Region.id, Region.name, func.ST_AsGeoJSON(Region.geom))
    ).all()
    features: list[dict[str, Any]] = []
    for region_id, name, geom_json in rows:
        score = latest_risk_score(session, region_id)
        features.append(
            {
                "type": "Feature",
                "geometry": json.loads(geom_json),
                "properties": {
                    "region_id": region_id,
                    "region_name": name,
                    "date": score.date.isoformat() if score else None,
                    "ensemble_score": score.ensemble_score if score else None,
                    "cnn_score": score.cnn_score if score else None,
                    "lstm_score": score.lstm_score if score else None,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def upsert_risk_score(
    session: Session,
    region_id: int,
    day: date,
    *,
    ensemble_score: float,
    cnn_score: float | None = None,
    lstm_score: float | None = None,
    model_version: str | None = None,
) -> None:
    """Insert or update the risk score for ``(region_id, day)``."""
    stmt = pg_insert(RiskScore).values(
        region_id=region_id,
        date=day,
        ensemble_score=ensemble_score,
        cnn_score=cnn_score,
        lstm_score=lstm_score,
        model_version=model_version,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["region_id", "date"],
        set_={
            "ensemble_score": stmt.excluded.ensemble_score,
            "cnn_score": stmt.excluded.cnn_score,
            "lstm_score": stmt.excluded.lstm_score,
            "model_version": stmt.excluded.model_version,
        },
    )
    session.execute(stmt)
