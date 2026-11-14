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


def test_p2_corrected_analysis_artifacts_schema_and_values():
    """Verify schema, values, and inference flags of results/p2/analysis_corrected artifacts."""
    import pandas as pd

    comp_path = "results/p2/analysis_corrected/comparisons.csv"
    summary_path = "results/p2/analysis_corrected/summary.json"

    assert os.path.exists(comp_path), f"Corrected comparisons artifact missing: {comp_path}"
    assert os.path.exists(summary_path), f"Corrected summary artifact missing: {summary_path}"

    df = pd.read_csv(comp_path)

    # 1. Exact expected column schema
    expected_cols = [
        "id", "family", "arm_a", "arm_b", "metric", "seeds",
        "difference", "ci_low", "ci_high", "ci_excludes_zero",
        "bootstrap_sign_prob", "p_value"
    ]
    assert list(df.columns) == expected_cols

    # 2. Inadmissible multiple-testing columns must be absent
    assert "p_value_holm" not in df.columns
    assert "significant_holm_0.05" not in df.columns

    # 3. ci_excludes_zero consistency
    computed_excludes = (df["ci_low"] > 0) | (df["ci_high"] < 0)
    assert (df["ci_excludes_zero"].astype(bool) == computed_excludes).all()

    # 4. p_value is strictly identical to bootstrap_sign_prob
    np.testing.assert_allclose(df["p_value"], df["bootstrap_sign_prob"])

    # 5. C2 (FedAvg vs local) strictly excludes zero for all clients and metrics
    c2_rows = df[df["id"].str.startswith("C2")]
    assert len(c2_rows) == 8  # 4 clients x 2 metrics (macro_auroc, macro_f1)
    assert c2_rows["ci_excludes_zero"].all()
    assert (c2_rows["difference"] > 0).all()
    assert (c2_rows["bootstrap_sign_prob"] <= 0.001).all()

    # 6. C1 (FedAvg vs centralized) 95% CI spans zero
    c1_rows = df[df["id"] == "C1"]
    assert len(c1_rows) == 2
    assert not c1_rows["ci_excludes_zero"].any()

    # 7. Metadata summary verification
    with open(summary_path) as f:
        summary = json.load(f)
    assert "paired_inference" in summary["bootstrap"]
    assert "primary_criterion" in summary["bootstrap"]["paired_inference"]
    assert "sign_prob_interpretation" in summary["bootstrap"]["paired_inference"]


def test_p2_historical_analysis_artifacts_preserved():
    """Verify that original historical artifacts in results/p2/analysis/ are preserved untouched."""
    import pandas as pd

    hist_path = "results/p2/analysis/comparisons.csv"
    assert os.path.exists(hist_path), f"Historical comparisons artifact missing: {hist_path}"

    df_hist = pd.read_csv(hist_path)
    # Historical file retained original columns as audit record
    assert "p_value_holm" in df_hist.columns
    assert "significant_holm_0.05" in df_hist.columns


