"""FastAPI application factory and routes.

The web process stays lightweight: it serves risk scores that the Celery worker
computed and stored in PostGIS. Heavy model loading lives in the worker, not here.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from sentinel import __version__
from sentinel.api.deps import get_db
from sentinel.api.schemas import HealthOut, RegionOut, RiskScoreOut
from sentinel.config import get_settings
from sentinel.db.models import Region, RiskScore
from sentinel.serving import repository


def _to_score_out(region: Region, score: RiskScore) -> RiskScoreOut:
    return RiskScoreOut(
        region_id=region.id,
        region_name=region.name,
        date=score.date,
        ensemble_score=score.ensemble_score,
        cnn_score=score.cnn_score,
        lstm_score=score.lstm_score,
        model_version=score.model_version,
        computed_at=score.created_at,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Sentinel API",
        version=__version__,
        summary="Wildfire ignition-risk scores by region.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health() -> HealthOut:
        return HealthOut(status="ok", version=__version__)

    @app.get("/regions", response_model=list[RegionOut], tags=["regions"])
    def regions(db: Annotated[Session, Depends(get_db)]) -> list[RegionOut]:
        return [RegionOut(id=r.id, name=r.name) for r in repository.list_regions(db)]

    @app.get("/risk", response_model=RiskScoreOut, tags=["risk"])
    def risk(
        db: Annotated[Session, Depends(get_db)],
        region: Annotated[str, Query(description="region id or name")],
        day: Annotated[
            date | None, Query(alias="date", description="YYYY-MM-DD; default latest")
        ] = None,
    ) -> RiskScoreOut:
        region_obj = repository.get_region(db, region)
        if region_obj is None:
            raise HTTPException(status_code=404, detail=f"region {region!r} not found")

        if day is not None:
            score = repository.get_risk_score(db, region_obj.id, day)
            if score is None:
                score = repository.latest_risk_score(
                    db, region_obj.id, on_or_before=day
                )
        else:
            score = repository.latest_risk_score(db, region_obj.id)

        if score is None:
            raise HTTPException(
                status_code=404,
                detail=f"no risk score available for region {region_obj.name!r}",
            )
        return _to_score_out(region_obj, score)

    @app.get("/risk/latest", response_model=list[RiskScoreOut], tags=["risk"])
    def risk_latest(
        db: Annotated[Session, Depends(get_db)],
    ) -> list[RiskScoreOut]:
        return [
            _to_score_out(region, score)
            for region, score in repository.latest_scores_all(db)
        ]

    @app.get("/risk/geojson", tags=["risk"])
    def risk_geojson(db: Annotated[Session, Depends(get_db)]) -> dict[str, object]:
        """Regions as a GeoJSON FeatureCollection with latest scores (for the map)."""
        return repository.regions_geojson(db)

    @app.get("/risk/point", tags=["risk"])
    def risk_point(
        lat: Annotated[float, Query(ge=-90, le=90)],
        lon: Annotated[float, Query(ge=-180, le=180)],
        day: Annotated[date | None, Query(alias="date")] = None,
        size: Annotated[float, Query(gt=0, le=5, description="cell size in degrees")] = 0.5,
    ) -> dict[str, object]:
        """Score an arbitrary point on demand (anywhere on Earth).

        Loads the models on first use, then fuses live LSTM climate risk + CNN
        imagery prior for a small cell around ``(lat, lon)``.
        """
        from sentinel.geo import BBox  # local import: keeps base import light
        from sentinel.tasks.scoring import get_scorer

        half = size / 2.0
        bbox = BBox(
            max(-180.0, lon - half),
            max(-90.0, lat - half),
            min(180.0, lon + half),
            min(90.0, lat + half),
        )
        when = day or date.today()
        components = get_scorer().score(f"point:{lat:.3f},{lon:.3f}", bbox, when)
        return {
            "lat": lat,
            "lon": lon,
            "date": when.isoformat(),
            "ensemble_score": components.ensemble,
            "cnn_score": components.cnn,
            "lstm_score": components.lstm,
        }

    return app


app = create_app()
