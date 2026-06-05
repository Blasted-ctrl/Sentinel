"""Tests for the ResNet CNN builder (no pretrained download)."""

from __future__ import annotations

import torch

from sentinel.models.cnn import CnnConfig, build_resnet_cnn, trainable_parameters


def test_forward_pass_shape() -> None:
    model = build_resnet_cnn(CnnConfig(pretrained=False))
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


def test_frozen_backbone_only_head_trainable() -> None:
    model = build_resnet_cnn(
        CnnConfig(pretrained=False, freeze_backbone=True, unfreeze_layer4=False)
    )
    # conv1 (backbone) is frozen; fc head is trainable.
    assert not any(p.requires_grad for p in model.conv1.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())


def test_unfreeze_layer4_trains_last_block() -> None:
    model = build_resnet_cnn(
        CnnConfig(pretrained=False, freeze_backbone=True, unfreeze_layer4=True)
    )
    assert any(p.requires_grad for p in model.layer4.parameters())
    assert not any(p.requires_grad for p in model.conv1.parameters())
    # Optimizer should receive only the trainable subset.
    n_trainable = len(trainable_parameters(model))
    n_total = len(list(model.parameters()))
    assert 0 < n_trainable < n_total
