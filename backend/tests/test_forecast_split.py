from datetime import date, timedelta

import numpy as np
import pytest

from sentinel.data.forecast_split import split_forecast_windows


def test_forecast_split_separates_blocks_and_entire_sample_intervals() -> None:
    days = [date(2010, 1, 1) + timedelta(days=i) for i in range(200)]
    blocks = [str(block) for block in range(20) for _ in days]
    dates = days * 20
    cfg = {
        "train_end": days[49],
        "calibration_end": days[99],
        "tune_end": days[149],
        "lookback": 14,
        "horizon": 7,
    }
    splits = split_forecast_windows(blocks, dates, **cfg)
    repeated = split_forecast_windows(blocks, dates, **cfg)
    names = list(splits)
    for pos, name in enumerate(names):
        idx = splits[name]
        assert np.array_equal(idx, repeated[name])
        for later in names[pos + 1 :]:
            assert {blocks[i] for i in idx}.isdisjoint(blocks[i] for i in splits[later])
            assert max(dates[i] + timedelta(days=7) for i in idx) < min(
                dates[i] - timedelta(days=13) for i in splits[later]
            )


def test_forecast_split_fails_on_insufficient_blocks_or_purged_data() -> None:
    cfg = {
        "train_end": date(2010, 1, 1),
        "calibration_end": date(2011, 1, 1),
        "tune_end": date(2012, 1, 1),
        "lookback": 14,
        "horizon": 7,
    }
    with pytest.raises(ValueError, match="at least 8"):
        split_forecast_windows(["a"], [date(2010, 1, 1)], **cfg)
    with pytest.raises(ValueError, match="empty"):
        split_forecast_windows([str(i) for i in range(20)], [date(2000, 1, 1)] * 20, **cfg)
