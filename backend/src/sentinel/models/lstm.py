"""TensorFlow/Keras LSTM for climate-sequence ignition risk.

TensorFlow is imported lazily so the rest of the package (and the ingestion CLI)
does not pay its heavy import cost.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LstmConfig:
    """Configuration for the climate LSTM."""

    lookback: int = 14
    n_features: int = 7
    units: int = 64
    dropout: float = 0.3
    learning_rate: float = 1e-3


def seed_tensorflow(seed: int = 42) -> None:
    """Seed TensorFlow and enable deterministic ops where possible."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    with contextlib.suppress(Exception):  # older TF without the API
        tf.config.experimental.enable_op_determinism()


def build_lstm(config: LstmConfig | None = None) -> Any:
    """Build and compile the climate LSTM (binary ignition-risk classifier)."""
    cfg = config or LstmConfig()
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(cfg.lookback, cfg.n_features)),
            layers.LSTM(cfg.units),
            layers.Dropout(cfg.dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model
