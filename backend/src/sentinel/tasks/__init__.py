"""Celery application and scheduled scoring tasks."""

from sentinel.tasks.celery_app import celery_app

__all__ = ["celery_app"]
