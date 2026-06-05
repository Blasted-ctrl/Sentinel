"""Tests for the RegionRiskScorer fusion logic (models + clients mocked)."""

from __future__ import annotations

import io
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from sentinel.geo import BBox
from sentinel.ingest.weather import WeatherDaily
from sentinel.models.ensemble import EnsembleScorer
from sentinel.serving.scorer import RegionRiskScorer


def _weather(n: int) -> list[WeatherDaily]:
    return [
        WeatherDaily(date(2024, 1, 1) + timedelta(days=i), 30.0, 15.0, 0.0, 10.0, 5.0)
        for i in range(n)
    ]


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 50, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _scorer(*, lstm_p: float, weather_days: int, stac: MagicMock) -> RegionRiskScorer:
    lstm = MagicMock()
    lstm.predict.return_value = np.array([[lstm_p]])
    cnn = MagicMock()
    cnn.predict_prob.return_value = 0.6
    weather = MagicMock()
    weather.fetch_daily.return_value = _weather(weather_days)
    return RegionRiskScorer(
        lstm_model=lstm,
        scaler_mean=np.zeros(7),
        scaler_std=np.ones(7),
        cnn_predictor=cnn,
        ensemble=EnsembleScorer(meta=None, cnn_weight=0.5),
        lookback=14,
        weather_client=weather,
        stac_client=stac,
    )


def test_score_fuses_cnn_and_lstm() -> None:
    stac = MagicMock()
    stac.search.return_value = [MagicMock()]
    stac.download_asset.return_value = (_png_bytes(), "image/png")
    scorer = _scorer(lstm_p=0.8, weather_days=40, stac=stac)

    comp = scorer.score("r1", BBox(-1.0, -1.0, 1.0, 1.0), date(2024, 2, 1))
    assert comp.lstm == pytest.approx(0.8)
    assert comp.cnn == pytest.approx(0.6)
    assert comp.ensemble == pytest.approx(0.5 * 0.6 + 0.5 * 0.8)  # weighted average

    # The per-region CNN prior is cached — no second STAC search.
    scorer.score("r1", BBox(-1.0, -1.0, 1.0, 1.0), date(2024, 2, 2))
    stac.search.assert_called_once()


def test_score_neutral_prior_when_no_imagery() -> None:
    stac = MagicMock()
    stac.search.return_value = []  # no scene available
    scorer = _scorer(lstm_p=0.3, weather_days=40, stac=stac)

    comp = scorer.score("r", BBox(-1.0, -1.0, 1.0, 1.0), date(2024, 2, 1))
    assert comp.cnn == pytest.approx(0.5)  # neutral
    scorer.cnn_predictor.predict_prob.assert_not_called()  # type: ignore[attr-defined]


def test_score_requires_enough_weather() -> None:
    scorer = _scorer(lstm_p=0.5, weather_days=5, stac=MagicMock())
    with pytest.raises(ValueError):
        scorer.score("r", BBox(-1.0, -1.0, 1.0, 1.0), date(2024, 2, 1))
