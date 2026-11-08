"""Tests for the P2 experiment code."""

import json
import os

import numpy as np
import pytest

from tests.conftest import requires_data


def test_global_norm_filter_matches_main_pipeline():
    """E4 must use exactly the same band-pass filter as the main pipeline."""
    from src.prepare_global_norm import bandpass
    from src.preprocessing import preprocess_signal

    rng = np.random.default_rng(0)
    t = np.arange(1000) / 100
    sig = (np.sin(2 * np.pi * 1.1 * t)[:, None] * rng.uniform(0.5, 2, 12)
           + 0.3 * t[:, None] + rng.normal(0, 0.05, (1000, 12)))
    f = bandpass(sig)
    z = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
    np.testing.assert_allclose(z.astype(np.float32), preprocess_signal(sig), atol=1e-6)


@requires_data
def test_local_training_rejects_non_training_indices(tmp_path):
    from src.train_multiscale import train
    folds = np.load("data/processed/folds.npy")
    with pytest.raises(ValueError, match="folds 1-8"):
        train(epochs=1, output=str(tmp_path / "x"),
              train_indices=np.where(folds == 9)[0][:10])


@requires_data
def test_global_norm_dataset_alignment():
    d = "data/p2_global_norm"
    if not os.path.exists(f"{d}/X.npy"):
        pytest.skip("E4 dataset not built")
    for name in ("y.npy", "folds.npy", "ecg_ids.npy"):
        np.testing.assert_array_equal(np.load(f"{d}/{name}"), np.load(f"data/processed/{name}"))
    X = np.load(f"{d}/X.npy", mmap_mode="r")
    assert X.shape == np.load("data/processed/X.npy", mmap_mode="r").shape
    folds = np.load(f"{d}/folds.npy")
    train = np.asarray(X[np.where(folds <= 8)[0][:2000]])
    # Train-fold statistics: roughly zero mean / unit variance per lead,
    # but amplitudes now differ between records (not forced to 1).
    assert np.all(np.abs(train.mean(axis=(0, 1))) < 0.05)
    per_record_std = train.std(axis=1)
    assert per_record_std.std() > 0.1
    with open(f"{d}/norm_stats.json") as f:
        stats = json.load(f)
    assert stats["samples_per_lead"] == int((folds <= 8).sum()) * 1000
