"""CNN training + evaluation pipeline.

End to end: discover tiles -> leak-free geospatial split -> class-weighted
fine-tuning of a pretrained ResNet -> honest evaluation on a held-out test set
-> persist the model and a ``metrics.json`` that is the single source of truth
for the numbers we report.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sentinel.data.geosplit import GeoSplit, assert_no_region_leakage, geospatial_split
from sentinel.data.wildfire import (
    TileSample,
    WildfireTileDataset,
    balanced_subset,
    build_transforms,
    discover_samples,
)
from sentinel.logging import get_logger
from sentinel.models.cnn import CnnConfig, build_resnet_cnn, trainable_parameters
from sentinel.training.metrics import ClassificationMetrics, compute_classification_metrics
from sentinel.training.seed import seed_everything

logger = get_logger(__name__)

CLASS_NAMES = ["nowildfire", "wildfire"]


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters and run settings for CNN training."""

    data_root: Path
    output_dir: Path
    image_size: int = 96
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    limit: int | None = None
    cell_size: float = 0.5
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    pretrained: bool = True
    unfreeze_layer4: bool = True
    num_workers: int = 0
    device: str = "cpu"
    threshold: float = 0.5
    select_metric: str = "recall"  # val metric used to pick the best epoch
    model_filename: str = "cnn_resnet18.pt"
    metrics_filename: str = "metrics.json"

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_root"] = str(self.data_root)
        data["output_dir"] = str(self.output_dir)
        return data


def _class_weights(labels: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency class weights to counter fire/no-fire imbalance."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _positive_rate(samples: Sequence[TileSample], indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    pos = sum(samples[i].label for i in indices)
    return pos / len(indices)


def _make_loaders(
    samples: list[TileSample], split: GeoSplit, cfg: TrainConfig
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_tf = build_transforms(cfg.image_size, train=True)
    eval_tf = build_transforms(cfg.image_size, train=False)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        WildfireTileDataset(samples, split.train, train_tf),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        WildfireTileDataset(samples, split.val, eval_tf),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        WildfireTileDataset(samples, split.test, eval_tf),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def predict_proba(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_prob_positive)`` over a loader."""
    model.eval()
    probs: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        trues.append(targets.numpy())
    if not probs:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(trues), np.concatenate(probs)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        n += inputs.size(0)
    return total_loss / max(n, 1)


def train_cnn(cfg: TrainConfig) -> dict[str, Any]:
    """Run the full training pipeline and return the metrics report (also saved)."""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    samples = balanced_subset(discover_samples(cfg.data_root), cfg.limit, seed=cfg.seed)
    coords = [(s.lon, s.lat) for s in samples]
    labels = [s.label for s in samples]

    split = geospatial_split(
        coords,
        labels,
        cell_size=cfg.cell_size,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )
    assert_no_region_leakage(split)
    logger.info("split.ready", sizes=split.sizes(), regions=split.region_counts())

    train_loader, val_loader, test_loader = _make_loaders(samples, split, cfg)

    model = build_resnet_cnn(
        CnnConfig(
            num_classes=2,
            pretrained=cfg.pretrained,
            freeze_backbone=True,
            unfreeze_layer4=cfg.unfreeze_layer4,
        )
    ).to(device)

    train_labels = [samples[i].label for i in split.train]
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_labels).to(device))
    optimizer = torch.optim.Adam(
        trainable_parameters(model), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_val: ClassificationMetrics | None = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)

        if len(split.val) > 0:
            y_true, y_prob = predict_proba(model, val_loader, device)
            val_metrics = compute_classification_metrics(
                y_true, y_prob, threshold=cfg.threshold
            )
            score = getattr(val_metrics, cfg.select_metric)
        else:  # tiny datasets: fall back to selecting on train loss
            val_metrics = None
            score = -train_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val": val_metrics.as_dict() if val_metrics else None,
            }
        )
        logger.info(
            "epoch.done",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_recall=round(val_metrics.recall, 4) if val_metrics else None,
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = val_metrics

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_prob = predict_proba(model, test_loader, device)
    test_metrics = compute_classification_metrics(y_true, y_prob, threshold=cfg.threshold)

    report = _build_report(cfg, samples, split, history, best_val, test_metrics)
    _save_artifacts(cfg, model, report)
    logger.info("train.complete", test=test_metrics.as_dict())
    return report


def _build_report(
    cfg: TrainConfig,
    samples: list[TileSample],
    split: GeoSplit,
    history: list[dict[str, Any]],
    best_val: ClassificationMetrics | None,
    test_metrics: ClassificationMetrics,
) -> dict[str, Any]:
    labels = [s.label for s in samples]
    return {
        "model": "resnet18-transfer",
        "task": "wildfire-tile-classification",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": cfg.seed,
        "dataset": {
            "root": str(cfg.data_root),
            "n_total": len(samples),
            "positives": int(sum(labels)),
            "negatives": int(len(labels) - sum(labels)),
            "regions": split.region_counts(),
        },
        "split": {
            "sizes": split.sizes(),
            "positive_rate": {
                "train": round(_positive_rate(samples, split.train), 4),
                "val": round(_positive_rate(samples, split.val), 4),
                "test": round(_positive_rate(samples, split.test), 4),
            },
            "geospatial": True,
            "cell_size_deg": cfg.cell_size,
            "region_leakage": False,
        },
        "config": cfg.public_dict(),
        "val_best": best_val.as_dict() if best_val else None,
        "test": test_metrics.as_dict(),
        "history": history,
    }


def _save_artifacts(cfg: TrainConfig, model: nn.Module, report: dict[str, Any]) -> None:
    model_path = cfg.output_dir / cfg.model_filename
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": cfg.image_size,
            "classes": CLASS_NAMES,
            "arch": "resnet18",
            "unfreeze_layer4": cfg.unfreeze_layer4,
        },
        model_path,
    )
    metrics_path = cfg.output_dir / cfg.metrics_filename
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("artifacts.saved", model=str(model_path), metrics=str(metrics_path))
