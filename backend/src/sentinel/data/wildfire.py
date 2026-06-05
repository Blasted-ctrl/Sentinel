"""Wildfire satellite-tile dataset (real/no-fire image classification).

Expects the public "Wildfire Prediction Dataset" layout: an arbitrary tree of
``wildfire/`` and ``nowildfire/`` folders whose image files are named
``"<lon>,<lat>.jpg"``. The coordinates in the filename drive the leakage-safe
geospatial split (see :mod:`sentinel.data.geosplit`).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from sentinel.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# ImageNet normalisation stats (ResNet was pretrained with these).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

LABEL_WILDFIRE = 1
LABEL_NOWILDFIRE = 0


@dataclass(slots=True)
class TileSample:
    """One labelled, geotagged satellite tile."""

    path: Path
    lon: float
    lat: float
    label: int  # 1 = wildfire, 0 = nowildfire


def parse_coords(filename: str) -> tuple[float, float] | None:
    """Parse ``"<lon>,<lat>"`` from a filename stem; ``None`` if not parseable."""
    stem = Path(filename).stem
    parts = stem.split(",")
    if len(parts) != 2:
        return None
    try:
        lon, lat = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return lon, lat


def _label_for(folder_name: str) -> int | None:
    name = folder_name.lower()
    if name == "nowildfire":
        return LABEL_NOWILDFIRE
    if name == "wildfire":
        return LABEL_WILDFIRE
    return None


def discover_samples(root: str | Path) -> list[TileSample]:
    """Recursively find labelled, coordinate-named tiles under ``root``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    samples: list[TileSample] = []
    skipped = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = _label_for(path.parent.name)
        if label is None:
            continue
        coords = parse_coords(path.name)
        if coords is None:
            skipped += 1
            continue
        samples.append(TileSample(path=path, lon=coords[0], lat=coords[1], label=label))

    logger.info(
        "dataset.discovered",
        root=str(root),
        samples=len(samples),
        skipped_no_coords=skipped,
    )
    if not samples:
        raise ValueError(
            f"no coordinate-named wildfire/nowildfire tiles found under {root}"
        )
    return samples


def balanced_subset(
    samples: list[TileSample], limit: int | None, *, seed: int = 42
) -> list[TileSample]:
    """Return up to ``limit`` samples, balanced across the two classes."""
    if limit is None or limit >= len(samples):
        return samples
    rng = random.Random(seed)
    pos = [s for s in samples if s.label == LABEL_WILDFIRE]
    neg = [s for s in samples if s.label == LABEL_NOWILDFIRE]
    rng.shuffle(pos)
    rng.shuffle(neg)
    per_class = limit // 2
    chosen = pos[:per_class] + neg[: limit - per_class]
    rng.shuffle(chosen)
    return chosen


def build_transforms(image_size: int, *, train: bool) -> transforms.Compose:
    """Build train (augmented) or eval image transforms."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class WildfireTileDataset(Dataset[tuple[torch.Tensor, int]]):
    """Torch dataset over a fixed list of tiles and a subset of indices."""

    def __init__(
        self,
        samples: Sequence[TileSample],
        indices: Sequence[int],
        transform: transforms.Compose,
    ) -> None:
        self.samples = list(samples)
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[self.indices[i]]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
            tensor: torch.Tensor = self.transform(image)
        return tensor, sample.label
