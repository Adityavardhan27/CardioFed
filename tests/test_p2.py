"""Tests for the P2 experiment code."""

import json
import os

import numpy as np
import pytest

from tests.conftest import is_real_file, requires_data


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
    if not is_real_file(f"{d}/X.npy"):
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


def test_p2_statistical_methodology_consistency():
    """Verify that p2_analysis does not apply Holm correction to descriptive sign proportions."""
    import inspect
    from src import p2_analysis

    src_code = inspect.getsource(p2_analysis.main)
    # Must NOT apply Holm to p_value or create significance flags
    assert 'comp.loc[m, "p_value_holm"]' not in src_code
    assert "significant_holm_0.05" not in src_code
    # Must compute and output ci_excludes_zero and bootstrap_sign_prob
    assert "ci_excludes_zero" in src_code
    assert "bootstrap_sign_prob" in src_code

    # holm function docstring must explicitly warn against descriptive statistics
    holm_doc = inspect.getdoc(p2_analysis.holm)
    assert "must NOT be applied to descriptive bootstrap" in holm_doc

