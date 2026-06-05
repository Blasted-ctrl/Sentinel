"""Application configuration loaded from environment / `.env`.

All secrets and connection strings live here. Nothing in the codebase should
read ``os.environ`` directly — go through :func:`get_settings` so configuration
is validated and overridable in tests.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database ---
    database_url: str = "postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Object storage (S3 / MinIO) ---
    s3_endpoint_url: str | None = None
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "sentinel-tiles"
    s3_region: str = "us-east-1"

    # --- NASA FIRMS ---
    firms_map_key: str = ""
    firms_source: str = "VIIRS_SNPP_NRT"
    firms_base_url: str = "https://firms.modaps.eosdis.nasa.gov/api"

    # --- Open-Meteo ---
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"

    # --- STAC imagery (Sentinel-2) ---
    stac_api_url: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-l2a"
    stac_max_cloud_cover: int = Field(default=30, ge=0, le=100)

    # --- Model artifacts (used by the Celery scorer) ---
    cnn_checkpoint: str = "metrics/cnn/cnn_resnet18.pt"
    lstm_dir: str = "metrics/lstm"
    ensemble_path: str = "metrics/ensemble/ensemble.joblib"

    # --- API ---
    cors_origins: str = "http://localhost:3000"

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
