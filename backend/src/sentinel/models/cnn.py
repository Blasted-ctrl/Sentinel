"""ResNet-based CNN for wildfire tile classification (transfer learning).

We start from an ImageNet-pretrained ResNet, freeze the convolutional backbone,
and fine-tune a fresh classification head (optionally also the last residual
block). This is the standard, data-efficient transfer-learning recipe for small
satellite-image datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(slots=True)
class CnnConfig:
    """Configuration for the CNN builder."""

    num_classes: int = 2
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_layer4: bool = True
    dropout: float = 0.3


def build_resnet_cnn(config: CnnConfig | None = None) -> nn.Module:
    """Build a ResNet18 fine-tuning model per ``config``."""
    cfg = config or CnnConfig()
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.resnet18(weights=weights)

    if cfg.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        if cfg.unfreeze_layer4:
            for param in model.layer4.parameters():
                param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(cfg.dropout),
        nn.Linear(in_features, cfg.num_classes),
    )
    return model


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    """Return the parameters that require gradients (for the optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]
