"""Region risk scorer: fuse live LSTM climate risk + CNN imagery prior.

Loads the trained artifacts (Keras LSTM + scaler, PyTorch CNN, ensemble
meta-model) and computes a fused 0-1 ignition-risk score for a region/date from
freshly fetched weather and imagery. Heavy to construct, so the API and Celery
task build it once and reuse it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sentinel.data.climate import _daily_feature_rows
from sentinel.geo import BBox
from sentinel.ingest.imagery import StacClient
from sentinel.ingest.weather import WeatherClient
from sentinel.logging import get_logger
from sentinel.models.ensemble import EnsembleScorer
from sentinel.models.inference import CnnPredictor

logger = get_logger(__name__)

NEUTRAL_PROB = 0.5


@dataclass(slots=True)
class RiskComponents:
    """The fused score plus its component probabilities."""

    ensemble: float
    cnn: float
    lstm: float


class RegionRiskScorer:
    """Computes fused wildfire-risk scores for a region on a date."""

    def __init__(
        self,
        *,
        lstm_model: Any,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        cnn_predictor: CnnPredictor,
        ensemble: EnsembleScorer,
        lookback: int = 14,
        weather_client: WeatherClient | None = None,
        stac_client: StacClient | None = None,
        imagery_start: date = date(2023, 6, 1),
        imagery_end: date = date(2023, 9, 30),
        stac_max_cloud: int = 40,
        model_version: str = "cnn-lstm-ensemble-v1",
    ) -> None:
        self.lstm_model = lstm_model
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.cnn_predictor = cnn_predictor
        self.ensemble = ensemble
        self.lookback = lookback
        self.weather_client = weather_client or WeatherClient()
        self.stac_client = stac_client or StacClient()
        self.imagery_start = imagery_start
        self.imagery_end = imagery_end
        self.stac_max_cloud = stac_max_cloud
        self.model_version = model_version
        self._cnn_cache: dict[str, float] = {}

    @classmethod
    def from_artifacts(
        cls,
        lstm_dir: str | Path,
        cnn_checkpoint: str | Path,
        ensemble_path: str | Path,
        **kwargs: Any,
    ) -> RegionRiskScorer:
        import tensorflow as tf

        lstm_dir = Path(lstm_dir)
        model = tf.keras.models.load_model(lstm_dir / "lstm_climate.keras")
        scaler = np.load(lstm_dir / "scaler.npz")
        return cls(
            lstm_model=model,
            scaler_mean=scaler["mean"],
            scaler_std=scaler["std"],
            cnn_predictor=CnnPredictor(cnn_checkpoint),
            ensemble=EnsembleScorer.load(ensemble_path),
            **kwargs,
        )

    def _lstm_prob(self, lat: float, lon: float, day: date) -> float:
        # Fetch a generous window then take the last `lookback` days up to `day`.
        start = day - timedelta(days=self.lookback * 3)
        weather = self.weather_client.fetch_daily(lat, lon, start, day)
        features, _dates = _daily_feature_rows(weather)
        if features.shape[0] < self.lookback:
            raise ValueError(
                f"need {self.lookback} days of weather, got {features.shape[0]}"
            )
        window = features[-self.lookback :]
        std_safe = np.where(self.scaler_std == 0, 1.0, self.scaler_std)
        norm = (window - self.scaler_mean) / std_safe
        prob = self.lstm_model.predict(norm[np.newaxis, ...], verbose=0).ravel()[0]
        return float(prob)

    def _cnn_prior(self, region_key: str, bbox: BBox) -> float:
        if region_key in self._cnn_cache:
            return self._cnn_cache[region_key]
        prob = NEUTRAL_PROB
        try:
            items = self.stac_client.search(
                bbox,
                self.imagery_start,
                self.imagery_end,
                max_cloud_cover=self.stac_max_cloud,
                limit=1,
            )
            if items:
                data, _ = self.stac_client.download_asset(items[0], "thumbnail")
                with Image.open(io.BytesIO(data)) as img:
                    prob = self.cnn_predictor.predict_prob(img)
        except Exception as exc:
            logger.warning("scorer.cnn_failed", region=region_key, error=str(exc))
        self._cnn_cache[region_key] = prob
        return prob

    def score(
        self, region_key: str, bbox: BBox, day: date
    ) -> RiskComponents:
        """Compute the fused risk score for a region on a date."""
        lon, lat = bbox.centroid
        lstm_prob = self._lstm_prob(lat, lon, day)
        cnn_prob = self._cnn_prior(region_key, bbox)
        ensemble_prob = self.ensemble.score(cnn_prob, lstm_prob)
        logger.info(
            "scorer.scored",
            region=region_key,
            date=day.isoformat(),
            ensemble=round(ensemble_prob, 4),
        )
        return RiskComponents(ensemble=ensemble_prob, cnn=cnn_prob, lstm=lstm_prob)
