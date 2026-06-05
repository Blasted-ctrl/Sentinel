"""CNN + LSTM ensemble: fuse imagery risk and climate risk into one 0-1 score.

The CNN provides a (static) landscape-flammability prior from satellite imagery;
the LSTM provides the (dynamic) climate-driven ignition risk. A small logistic
meta-model learns how to weight them. A plain weighted average is kept as a
transparent baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(slots=True)
class EnsembleScorer:
    """Combine CNN and LSTM probabilities into a single fire-risk score."""

    meta: Any = None  # fitted sklearn estimator (predict_proba) or None
    cnn_weight: float = 0.5

    def score_batch(self, cnn_prob: ArrayLike, lstm_prob: ArrayLike) -> np.ndarray:
        cnn = np.asarray(cnn_prob, dtype=float)
        lstm = np.asarray(lstm_prob, dtype=float)
        if self.meta is not None:
            features = np.column_stack([cnn, lstm])
            return np.asarray(self.meta.predict_proba(features)[:, 1], dtype=float)
        return self.cnn_weight * cnn + (1.0 - self.cnn_weight) * lstm

    def score(self, cnn_prob: float, lstm_prob: float) -> float:
        return float(self.score_batch([cnn_prob], [lstm_prob])[0])

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"meta": self.meta, "cnn_weight": self.cnn_weight}, path)

    @classmethod
    def load(cls, path: str | Path) -> EnsembleScorer:
        import joblib

        data = joblib.load(path)
        return cls(meta=data["meta"], cnn_weight=data["cnn_weight"])


def fit_meta_model(
    cnn_prob: ArrayLike, lstm_prob: ArrayLike, y: ArrayLike, *, seed: int = 42
) -> Any:
    """Fit a class-balanced logistic meta-model on the two component scores."""
    from sklearn.linear_model import LogisticRegression

    features = np.column_stack(
        [np.asarray(cnn_prob, dtype=float), np.asarray(lstm_prob, dtype=float)]
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(features, np.asarray(y, dtype=int))
    return clf
