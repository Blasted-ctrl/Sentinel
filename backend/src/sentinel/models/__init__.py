"""Neural network architectures for Sentinel."""

from sentinel.models.cnn import CnnConfig, build_resnet_cnn

__all__ = ["CnnConfig", "build_resnet_cnn"]
