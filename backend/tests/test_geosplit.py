"""Tests for leakage-safe geospatial splitting."""

from __future__ import annotations

import pytest

from sentinel.data.geosplit import (
    assert_no_region_leakage,
    geospatial_split,
    region_key,
)


def _grid_dataset(
    n_regions: int = 20, per_region: int = 6
) -> tuple[list[tuple[float, float]], list[int]]:
    """Build coords/labels across many well-separated 0.5deg cells."""
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    for r in range(n_regions):
        base_lon = -120.0 + r * 1.0
        base_lat = 30.0 + r * 0.3
        for k in range(per_region):
            coords.append((base_lon + 0.01 * k, base_lat + 0.01 * k))
            labels.append(k % 2)  # both classes within each region
    return coords, labels


def test_region_key_snaps_to_grid() -> None:
    assert region_key(-120.4, 38.2, 0.5) == region_key(-120.1, 38.4, 0.5)
    assert region_key(-120.4, 38.2, 0.5) != region_key(-119.4, 38.2, 0.5)


def test_split_has_no_region_leakage() -> None:
    coords, labels = _grid_dataset()
    split = geospatial_split(coords, labels, cell_size=0.5, seed=42)
    # Must not raise.
    assert_no_region_leakage(split)

    train_r = {split.groups[i] for i in split.train}
    val_r = {split.groups[i] for i in split.val}
    test_r = {split.groups[i] for i in split.test}
    assert train_r and val_r and test_r
    assert train_r.isdisjoint(test_r)
    assert val_r.isdisjoint(test_r)


def test_split_covers_all_indices_once() -> None:
    coords, labels = _grid_dataset()
    split = geospatial_split(coords, labels, seed=7)
    combined = sorted(split.train + split.val + split.test)
    assert combined == list(range(len(labels)))


def test_split_is_deterministic() -> None:
    coords, labels = _grid_dataset()
    a = geospatial_split(coords, labels, seed=123)
    b = geospatial_split(coords, labels, seed=123)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_split_rejects_too_few_regions() -> None:
    coords = [(-120.0, 38.0), (-120.01, 38.01)]  # same cell
    labels = [0, 1]
    with pytest.raises(ValueError):
        geospatial_split(coords, labels, cell_size=0.5)
