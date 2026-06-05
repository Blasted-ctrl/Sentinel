"""Acquire the wildfire image dataset from Kaggle, a local zip, or a folder."""

from __future__ import annotations

import zipfile
from pathlib import Path

from sentinel.data.wildfire import discover_samples
from sentinel.logging import get_logger

logger = get_logger(__name__)

KAGGLE_WILDFIRE_DATASET = "abdelghaniaaba/wildfire-prediction-dataset"


def download_from_kaggle(dataset: str = KAGGLE_WILDFIRE_DATASET) -> Path:
    """Download a Kaggle dataset via kagglehub and return its local root path.

    Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or a
    ``~/.kaggle/access_token`` / ``KAGGLE_API_TOKEN``).
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - dependency present in this project
        raise RuntimeError(
            "kagglehub is required for --source kaggle (pip install kagglehub)"
        ) from exc

    logger.info("kaggle.download.start", dataset=dataset)
    path = Path(kagglehub.dataset_download(dataset))
    logger.info("kaggle.download.done", path=str(path))
    return path


def extract_zip(zip_path: str | Path, dest: str | Path) -> Path:
    """Extract a dataset zip into ``dest`` and return ``dest``."""
    zip_path = Path(zip_path)
    dest = Path(dest)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    logger.info("zip.extracted", zip=str(zip_path), dest=str(dest))
    return dest


def resolve_wildfire_dataset(
    source: str,
    *,
    path: str | Path | None = None,
    dataset: str = KAGGLE_WILDFIRE_DATASET,
    dest: str | Path = "data/wildfire",
) -> Path:
    """Resolve a usable dataset root from the chosen ``source``.

    Args:
        source: ``"kaggle"``, ``"zip"``, or ``"local"``.
        path: zip file (``zip``) or existing folder (``local``).
        dataset: Kaggle dataset slug (``kaggle``).
        dest: extraction target for ``zip``.
    """
    if source == "kaggle":
        root = download_from_kaggle(dataset)
    elif source == "zip":
        if path is None:
            raise ValueError("--path is required for --source zip")
        root = extract_zip(path, dest)
    elif source == "local":
        if path is None:
            raise ValueError("--path is required for --source local")
        root = Path(path)
    else:
        raise ValueError(f"unknown source {source!r}; use kaggle|zip|local")

    # Validate it is actually loadable (raises if no labelled tiles are found).
    n = len(discover_samples(root))
    logger.info("dataset.resolved", root=str(root), samples=n)
    return root
