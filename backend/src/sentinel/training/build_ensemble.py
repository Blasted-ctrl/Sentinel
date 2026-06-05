"""Build + evaluate the CNN+LSTM ensemble end to end.

Reads the LSTM run's cached per-sample climate probabilities, derives a static
per-region CNN imagery prior (one Sentinel-2 tile per region), fits a logistic
meta-model on the (geospatially separate) validation split, and reports honest
test metrics — including the false-alarm rate — versus single-model baselines.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sentinel.geo import BBox
from sentinel.ingest.imagery import StacClient
from sentinel.logging import get_logger
from sentinel.models.ensemble import EnsembleScorer, fit_meta_model
from sentinel.models.inference import CnnPredictor
from sentinel.training.metrics import ClassificationMetrics, compute_classification_metrics

logger = get_logger(__name__)

NEUTRAL_PROB = 0.5


@dataclass(slots=True)
class EnsembleConfig:
    """Settings for assembling and evaluating the ensemble."""

    lstm_dir: Path  # contains ensemble_inputs.npz + metrics.json
    cnn_checkpoint: Path
    output_dir: Path
    imagery_start: date = date(2023, 6, 1)
    imagery_end: date = date(2023, 9, 30)
    stac_max_cloud: int = 40
    threshold: float = 0.5
    seed: int = 42
    metrics_filename: str = "metrics.json"
    scorer_filename: str = "ensemble.joblib"

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("lstm_dir", "cnn_checkpoint", "output_dir"):
            data[key] = str(getattr(self, key))
        data["imagery_start"] = self.imagery_start.isoformat()
        data["imagery_end"] = self.imagery_end.isoformat()
        return data


def cnn_region_priors(
    region_keys: list[str],
    region_bboxes: np.ndarray,
    predictor: CnnPredictor,
    *,
    stac: StacClient | None = None,
    start: date,
    end: date,
    max_cloud: int,
) -> dict[str, float]:
    """Fetch one Sentinel-2 tile per region and return its CNN wildfire prob."""
    client = stac or StacClient()
    own = stac is None
    priors: dict[str, float] = {}
    try:
        for key, box in zip(region_keys, region_bboxes, strict=True):
            bbox = BBox(*[float(v) for v in box])
            try:
                items = client.search(
                    bbox, start, end, max_cloud_cover=max_cloud, limit=1
                )
                if not items:
                    priors[key] = NEUTRAL_PROB
                    continue
                data, _ = client.download_asset(items[0], "thumbnail")
                with Image.open(io.BytesIO(data)) as img:
                    priors[key] = predictor.predict_prob(img)
            except Exception as exc:
                logger.warning("ensemble.tile_failed", region=key, error=str(exc))
                priors[key] = NEUTRAL_PROB
    finally:
        if own:
            client.close()
    return priors


def build_ensemble(
    cfg: EnsembleConfig,
    *,
    cnn_region_probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit + evaluate the ensemble; return the metrics report (also saved)."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(cfg.lstm_dir / "ensemble_inputs.npz", allow_pickle=True)
    y = data["y"].astype(int)
    groups = data["groups"].astype(str)
    lstm_prob = data["lstm_prob"].astype(float)
    val_idx = data["val_idx"].astype(int)
    test_idx = data["test_idx"].astype(int)
    region_keys = data["region_keys"].astype(str).tolist()
    region_bboxes = data["region_bboxes"]

    if cnn_region_probs is None:
        predictor = CnnPredictor(cfg.cnn_checkpoint)
        cnn_region_probs = cnn_region_priors(
            region_keys,
            region_bboxes,
            predictor,
            start=cfg.imagery_start,
            end=cfg.imagery_end,
            max_cloud=cfg.stac_max_cloud,
        )

    cnn_prob = np.array(
        [cnn_region_probs.get(g, NEUTRAL_PROB) for g in groups], dtype=float
    )

    # Fit the meta-model on validation (geospatially disjoint from test).
    meta = fit_meta_model(
        cnn_prob[val_idx], lstm_prob[val_idx], y[val_idx], seed=cfg.seed
    )
    scorer = EnsembleScorer(meta=meta)

    ensemble_test = scorer.score_batch(cnn_prob[test_idx], lstm_prob[test_idx])
    wavg_test = 0.5 * cnn_prob[test_idx] + 0.5 * lstm_prob[test_idx]

    def m(prob: np.ndarray) -> ClassificationMetrics:
        return compute_classification_metrics(y[test_idx], prob, threshold=cfg.threshold)

    report: dict[str, Any] = {
        "model": "cnn-lstm-ensemble",
        "task": "regional-ignition-risk",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": cfg.seed,
        "components": {
            "cnn": "resnet18-transfer (imagery prior)",
            "lstm": "keras-lstm (climate)",
            "combiner": "logistic meta-model",
        },
        "config": cfg.public_dict(),
        "test": {
            "ensemble": m(ensemble_test).as_dict(),
            "lstm_only": m(lstm_prob[test_idx]).as_dict(),
            "cnn_only": m(cnn_prob[test_idx]).as_dict(),
            "weighted_avg": m(wavg_test).as_dict(),
        },
        "meta_model": {
            "coef": [float(c) for c in np.ravel(meta.coef_)],
            "intercept": float(meta.intercept_[0]),
            "feature_order": ["cnn_prob", "lstm_prob"],
        },
    }

    scorer.save(cfg.output_dir / cfg.scorer_filename)
    (cfg.output_dir / cfg.metrics_filename).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info(
        "ensemble.complete",
        ensemble=report["test"]["ensemble"],
        lstm_only_recall=report["test"]["lstm_only"]["recall"],
    )
    return report
