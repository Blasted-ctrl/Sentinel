"""Smoke test for the full CNN training pipeline (tiny, CPU, no pretrained).

Exercises discover -> geospatial split -> train -> eval -> artifacts end to end
so the inference/training path stays covered in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.training.train_cnn import TrainConfig, train_cnn


def test_train_cnn_produces_metrics_and_model(
    tiny_wildfire_dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "artifacts"
    cfg = TrainConfig(
        data_root=tiny_wildfire_dataset,
        output_dir=output,
        image_size=32,
        batch_size=8,
        epochs=1,
        cell_size=0.5,
        pretrained=False,
        unfreeze_layer4=False,
        num_workers=0,
        device="cpu",
        seed=42,
    )
    report = train_cnn(cfg)

    # Artifacts written.
    assert (output / "metrics.json").exists()
    assert (output / "cnn_resnet18.pt").exists()

    # metrics.json is valid and matches the returned report.
    saved = json.loads((output / "metrics.json").read_text())
    assert saved["test"]["recall"] == report["test"]["recall"]

    # Honest, well-formed metrics in range.
    test = report["test"]
    for key in ("accuracy", "precision", "recall", "f1", "false_alarm_rate"):
        assert 0.0 <= test[key] <= 1.0

    # Geospatial split recorded with no leakage and all three splits populated.
    assert report["split"]["region_leakage"] is False
    sizes = report["split"]["sizes"]
    assert sizes["train"] > 0 and sizes["val"] > 0 and sizes["test"] > 0
