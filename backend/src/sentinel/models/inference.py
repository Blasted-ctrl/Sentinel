"""Load trained models and run inference (shared by the ensemble and the API)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from sentinel.data.wildfire import build_transforms
from sentinel.models.cnn import CnnConfig, build_resnet_cnn


class CnnPredictor:
    """Loads a trained CNN checkpoint and returns wildfire probabilities."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        ckpt = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        self.image_size = int(ckpt.get("image_size", 96))
        self.classes = ckpt.get("classes", ["nowildfire", "wildfire"])
        model = build_resnet_cnn(
            CnnConfig(
                num_classes=len(self.classes),
                pretrained=False,
                freeze_backbone=False,
                unfreeze_layer4=ckpt.get("unfreeze_layer4", True),
            )
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(self.device)
        self.model = model
        self.transform = build_transforms(self.image_size, train=False)

    @torch.no_grad()
    def predict_prob(self, image: Image.Image) -> float:
        """Return P(wildfire) for a single PIL image."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        return float(torch.softmax(logits, dim=1)[0, 1].item())

    @torch.no_grad()
    def predict_prob_batch(self, images: list[Image.Image]) -> list[float]:
        if not images:
            return []
        batch = torch.stack(
            [self.transform(img.convert("RGB")) for img in images]
        ).to(self.device)
        logits = self.model(batch)
        return torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
