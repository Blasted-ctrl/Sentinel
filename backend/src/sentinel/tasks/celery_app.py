"""Celery application with a daily re-scoring beat schedule."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from sentinel.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sentinel",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["sentinel.tasks.scoring"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# Re-score every tracked region daily at 06:00 UTC.
celery_app.conf.beat_schedule = {
    "daily-rescore-regions": {
        "task": "sentinel.tasks.scoring.rescore_all_regions",
        "schedule": crontab(hour=6, minute=0),
    },
}
