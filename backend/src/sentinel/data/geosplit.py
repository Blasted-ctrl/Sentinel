"""Stratified geospatial train/val/test splitting.

Wildfire tiles are geotagged. If two tiles from the same place land in different
splits, the model can memorise that location and the evaluation is optimistic —
"region leakage". We prevent it by snapping each tile's ``(lon, lat)`` to a grid
cell (the *region*) and keeping every cell wholly within one split, while
stratifying on the fire/no-fire label via scikit-learn's ``StratifiedGroupKFold``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(slots=True)
class GeoSplit:
    """Index lists for each split plus the per-sample region key."""

    train: list[int]
    val: list[int]
    test: list[int]
    groups: list[str]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def region_counts(self) -> dict[str, int]:
        def uniq(idx: list[int]) -> int:
            return len({self.groups[i] for i in idx})

        return {
            "train": uniq(self.train),
            "val": uniq(self.val),
            "test": uniq(self.test),
            "total": len(set(self.groups)),
        }


def region_key(lon: float, lat: float, cell_size: float) -> str:
    """Snap a coordinate to a grid cell id, e.g. ``"-148:91"`` for 0.5° cells."""
    cx = math.floor(lon / cell_size)
    cy = math.floor(lat / cell_size)
    return f"{cx}:{cy}"


def assign_regions(coords: Sequence[tuple[float, float]], cell_size: float) -> list[str]:
    """Map each ``(lon, lat)`` to its grid-cell region key."""
    return [region_key(lon, lat, cell_size) for lon, lat in coords]


def _first_fold(
    n_splits: int, indices: np.ndarray, labels: np.ndarray, groups: Sequence[str], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rest_index, holdout_index)`` from the first stratified-group fold."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_local, hold_local = next(iter(splitter.split(indices, labels, groups)))
    return indices[rest_local], indices[hold_local]


def split_by_groups(
    groups: Sequence[str],
    labels: Sequence[int],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> GeoSplit:
    """Split indices given pre-computed region group keys (no region crosses a split).

    Raises:
        ValueError: if there are too few distinct regions to split without leakage.
    """
    groups = list(groups)
    n = len(groups)
    if n != len(labels):
        raise ValueError("groups and labels must be the same length")
    n_groups = len(set(groups))
    if n_groups < 3:
        raise ValueError(
            f"need at least 3 distinct regions for a leak-free split, got {n_groups}; "
            f"reduce cell_size or supply more spatially diverse data"
        )

    y = np.asarray(labels)
    all_idx = np.arange(n)

    n_test_folds = min(max(2, round(1.0 / test_frac)), n_groups)
    rest_idx, test_idx = _first_fold(n_test_folds, all_idx, y, groups, seed)

    rest_groups = [groups[i] for i in rest_idx]
    n_rest_groups = len(set(rest_groups))
    n_val_folds = min(max(2, round((1.0 - test_frac) / val_frac)), n_rest_groups)
    if n_val_folds < 2:
        train_idx, val_idx = rest_idx, np.array([], dtype=int)
    else:
        train_idx, val_idx = _first_fold(
            n_val_folds, rest_idx, y[rest_idx], rest_groups, seed
        )

    return GeoSplit(
        train=sorted(train_idx.tolist()),
        val=sorted(val_idx.tolist()),
        test=sorted(test_idx.tolist()),
        groups=groups,
    )


def geospatial_split(
    coords: Sequence[tuple[float, float]],
    labels: Sequence[int],
    *,
    cell_size: float = 0.5,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> GeoSplit:
    """Split sample indices into train/val/test with no region crossing a split.

    Region keys are derived by snapping ``coords`` to a ``cell_size`` grid.
    """
    if len(labels) != len(coords):
        raise ValueError("coords and labels must be the same length")
    groups = assign_regions(coords, cell_size)
    return split_by_groups(
        groups, labels, val_frac=val_frac, test_frac=test_frac, seed=seed
    )


def assert_no_region_leakage(split: GeoSplit) -> None:
    """Raise ``AssertionError`` if any region appears in more than one split."""
    train_r = {split.groups[i] for i in split.train}
    val_r = {split.groups[i] for i in split.val}
    test_r = {split.groups[i] for i in split.test}
    assert train_r.isdisjoint(val_r), "region leakage between train and val"
    assert train_r.isdisjoint(test_r), "region leakage between train and test"
    assert val_r.isdisjoint(test_r), "region leakage between val and test"
