"""Label-blind spatial blocks and purged chronological forecast evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np


def split_forecast_windows(
    blocks: Sequence[str],
    dates: Sequence[date],
    *,
    train_end: date,
    calibration_end: date,
    tune_end: date,
    lookback: int,
    horizon: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Use different spatial blocks AND later time intervals for every stage.

    A sample consumes [t-lookback+1, t+horizon]. Keep that entire interval
    inside its stage. Block assignment never reads labels or fire counts.
    """
    if len(blocks) != len(dates) or not dates:
        raise ValueError("blocks and dates must have the same nonzero length")
    if not train_end < calibration_end < tune_end:
        raise ValueError("cutoffs must satisfy train_end < calibration_end < tune_end")
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    unique = sorted(set(blocks))
    if len(unique) < 8:
        raise ValueError("need at least 8 spatial blocks; expand the area or reduce block size")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_holdout = max(1, int(len(unique) * 0.15))
    block_sets = dict(
        zip(
            ("test", "tune", "calibration", "train"),
            (
                set(part.tolist())
                for part in np.split(shuffled, [n_holdout, 2 * n_holdout, 3 * n_holdout])
            ),
            strict=True,
        )
    )
    bounds = {
        "train": (None, train_end),
        "calibration": (train_end, calibration_end),
        "tune": (calibration_end, tune_end),
        "test": (tune_end, None),
    }
    splits = {}
    for name, (start, end) in bounds.items():
        indices = [
            i
            for i, (block, day) in enumerate(zip(blocks, dates, strict=True))
            if block in block_sets[name]
            and (start is None or day - timedelta(days=lookback - 1) > start)
            and (end is None or day + timedelta(days=horizon) <= end)
        ]
        if not indices:
            raise ValueError(f"{name} split is empty after spatial/time purging; expand inputs")
        splits[name] = np.asarray(indices, dtype=int)
    return splits
