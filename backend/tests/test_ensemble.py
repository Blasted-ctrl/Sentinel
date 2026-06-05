"""Tests for the CNN+LSTM ensemble combiner and end-to-end build."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sentinel.models.ensemble import EnsembleScorer, fit_meta_model
from sentinel.training.build_ensemble import EnsembleConfig, build_ensemble


def test_weighted_average_scorer() -> None:
    scorer = EnsembleScorer(meta=None, cnn_weight=0.25)
    out = scorer.score(cnn_prob=0.8, lstm_prob=0.4)
    assert out == 0.25 * 0.8 + 0.75 * 0.4


def test_meta_model_separates_classes() -> None:
    rng = np.random.default_rng(0)
    # lstm signal is informative; cnn is noise.
    lstm = np.concatenate([rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.4, 100)])
    cnn = rng.uniform(0, 1, 200)
    y = np.array([1] * 100 + [0] * 100)
    meta = fit_meta_model(cnn, lstm, y, seed=0)
    scorer = EnsembleScorer(meta=meta)
    probs = scorer.score_batch(cnn, lstm)
    # High-lstm samples should score higher on average than low-lstm ones.
    assert probs[:100].mean() > probs[100:].mean()


def test_scorer_save_load_roundtrip(tmp_path: Path) -> None:
    meta = fit_meta_model([0.2, 0.8, 0.3, 0.9], [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
    scorer = EnsembleScorer(meta=meta)
    path = tmp_path / "ens.joblib"
    scorer.save(path)
    loaded = EnsembleScorer.load(path)
    assert np.isclose(loaded.score(0.8, 0.9), scorer.score(0.8, 0.9))


def _make_lstm_inputs(tmp_path: Path) -> Path:
    """Fabricate an ensemble_inputs.npz like train_lstm would write."""
    rng = np.random.default_rng(1)
    regions = [f"r{i}" for i in range(6)]
    groups = np.array([r for r in regions for _ in range(40)])
    n = len(groups)
    y = rng.integers(0, 2, n)
    lstm_prob = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    idx = np.arange(n)
    # Split by region: r0-r3 train, r4 val, r5 test (no region leakage).
    train_idx = idx[np.isin(groups, regions[:4])]
    val_idx = idx[groups == "r4"]
    test_idx = idx[groups == "r5"]
    lstm_dir = tmp_path / "lstm"
    lstm_dir.mkdir()
    np.savez(
        lstm_dir / "ensemble_inputs.npz",
        y=y,
        groups=groups,
        lstm_prob=lstm_prob,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        region_keys=np.array(regions),
        region_bboxes=np.zeros((6, 4)),
    )
    return lstm_dir


def test_build_ensemble_with_injected_cnn_probs(tmp_path: Path) -> None:
    lstm_dir = _make_lstm_inputs(tmp_path)
    cnn_probs = {f"r{i}": 0.5 for i in range(6)}  # uninformative imagery prior
    cfg = EnsembleConfig(
        lstm_dir=lstm_dir,
        cnn_checkpoint=tmp_path / "unused.pt",
        output_dir=tmp_path / "ens",
        seed=0,
    )
    report = build_ensemble(cfg, cnn_region_probs=cnn_probs)

    assert (cfg.output_dir / "metrics.json").exists()
    assert (cfg.output_dir / "ensemble.joblib").exists()
    test = report["test"]
    for variant in ("ensemble", "lstm_only", "cnn_only", "weighted_avg"):
        assert 0.0 <= test[variant]["recall"] <= 1.0
        assert "false_alarm_rate" in test[variant]
