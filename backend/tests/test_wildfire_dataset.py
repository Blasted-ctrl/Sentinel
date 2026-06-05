"""Tests for wildfire dataset discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.data.wildfire import (
    LABEL_NOWILDFIRE,
    LABEL_WILDFIRE,
    WildfireTileDataset,
    balanced_subset,
    build_transforms,
    discover_samples,
    parse_coords,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("-57.119,51.472.jpg", (-57.119, 51.472)),
        ("12.5,-3.25.png", (12.5, -3.25)),
        ("not_coords.jpg", None),
        ("1,2,3.jpg", None),
        ("200,10.jpg", None),  # lon out of range
    ],
)
def test_parse_coords(name: str, expected: tuple[float, float] | None) -> None:
    assert parse_coords(name) == expected


def test_discover_samples_labels_and_coords(tiny_wildfire_dataset: Path) -> None:
    samples = discover_samples(tiny_wildfire_dataset)
    assert len(samples) == 48  # 12 regions x (2 wildfire + 2 nowildfire)
    labels = {s.label for s in samples}
    assert labels == {LABEL_WILDFIRE, LABEL_NOWILDFIRE}
    # Coordinates parsed from filenames are within the generated range.
    assert all(-121.0 <= s.lon <= -108.0 for s in samples)


def test_balanced_subset_is_class_balanced(tiny_wildfire_dataset: Path) -> None:
    samples = discover_samples(tiny_wildfire_dataset)
    subset = balanced_subset(samples, limit=10, seed=1)
    assert len(subset) == 10
    pos = sum(1 for s in subset if s.label == LABEL_WILDFIRE)
    assert pos == 5  # balanced


def test_dataset_getitem_shapes(tiny_wildfire_dataset: Path) -> None:
    samples = discover_samples(tiny_wildfire_dataset)
    transform = build_transforms(image_size=32, train=False)
    ds = WildfireTileDataset(samples, list(range(len(samples))), transform)
    tensor, label = ds[0]
    assert tensor.shape == (3, 32, 32)
    assert label in (0, 1)
    assert len(ds) == len(samples)


def test_discover_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_samples(tmp_path / "does-not-exist")
