"""Serving layer: risk repository + scorer used by the API and Celery tasks."""

from sentinel.serving.repository import (
    get_region,
    get_risk_score,
    latest_risk_score,
    latest_scores_all,
    list_regions,
    upsert_risk_score,
)

__all__ = [
    "get_region",
    "get_risk_score",
    "latest_risk_score",
    "latest_scores_all",
    "list_regions",
    "upsert_risk_score",
]
