"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from sentinel.geo import BBox


@pytest.fixture(autouse=True)
def _fake_aws_credentials() -> Iterator[None]:
    """Ensure boto3/moto never reach real AWS during tests."""
    saved = {k: os.environ.get(k) for k in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    )}
    os.environ.update(
        AWS_ACCESS_KEY_ID="testing",
        AWS_SECRET_ACCESS_KEY="testing",
        AWS_SECURITY_TOKEN="testing",
        AWS_SESSION_TOKEN="testing",
        AWS_DEFAULT_REGION="us-east-1",
    )
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def sample_bbox() -> BBox:
    """A small bbox over the Sierra Nevada (fire-prone) for tests."""
    return BBox(-120.5, 38.5, -120.0, 39.0)


def _make_tile(path: Path, channel: int, seed: int) -> None:
    """Write a tiny 32x32 RGB image biased toward one channel (weak signal)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 60, size=(32, 32, 3), dtype="uint8")
    arr[..., channel] = np.clip(arr[..., channel].astype(int) + 180, 0, 255).astype(
        "uint8"
    )
    Image.fromarray(arr).save(path)


@pytest.fixture
def tiny_wildfire_dataset(tmp_path: Path) -> Path:
    """Build a small, leak-test-friendly wildfire image dataset.

    12 well-separated regions (one 0.5deg grid cell each), each with 2 wildfire
    and 2 nowildfire coordinate-named tiles. Fire tiles are red-biased, no-fire
    tiles green-biased, giving a faint learnable signal for smoke training.
    """
    root = tmp_path / "wildfire"
    seed = 0
    for r in range(12):
        base_lon = -120.0 + r * 1.0
        base_lat = 38.0 + r * 0.2
        for cls, channel in (("wildfire", 0), ("nowildfire", 1)):
            folder = root / "train" / cls
            folder.mkdir(parents=True, exist_ok=True)
            for k in range(2):
                lon = round(base_lon + 0.01 * k, 5)
                lat = round(base_lat + 0.01 * k, 5)
                _make_tile(folder / f"{lon},{lat}.png", channel, seed)
                seed += 1
    return root
