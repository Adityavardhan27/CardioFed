"""Focused unit and regression tests for robustness evaluation.

Tests verify:
- Corruption implementations (clipping, circular temporal translation, missing leads, flatline).
- Inverse property of circular temporal shifts: shift(shift(x, k), -k) == x.
- Deterministic reproducibility and stochastic seed variation.
- Preprocessing stability under extreme conditions (all-zero flatline, clipped inputs).
- Patient-clustered bootstrap reuse from src.statistical_analysis.
- Frozen validation thresholds.
- Output safety (guard_outputs) and preservation of historical robustness artifacts.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest
import torch

from src.io_utils import guard_outputs, OutputExistsError
from src.model_registry import load_evaluation, MODELS
from src.preprocessing import preprocess_signal, FS, CLASSES
from src.robustness_evaluation import corrupt, repreprocess, is_stochastic
from src.statistical_analysis import (
    patient_bootstrap_indices,
    compute_paired_comparison,
    macro_auc,
    per_class_f1,
)
from src.thresholds import thresholds_from_evaluation


@pytest.fixture
def sample_ecg():
    """Synthetic 12-lead ECG tensor [2, 12, 1000] with known statistics."""
    rng = np.random.default_rng(12345)
    # Generate smooth periodic signal + small noise
    t = np.arange(1000) / FS
    base = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz fundamental
    x = np.zeros((2, 12, 1000), dtype=np.float32)
    for i in range(2):
        for lead in range(12):
            x[i, lead] = (base * (lead + 1) * 0.2 + rng.normal(0, 0.05, 1000)).astype(np.float32)
    return x


def test_corrupt_clipping_bounds(sample_ecg):
    """Verify that clipping bounds values to [-c, c] and leaves in-range values unchanged."""
    c = 1.5
    corrupted = corrupt(sample_ecg, "clipping", {"threshold": c}, None)
    assert corrupted.shape == sample_ecg.shape
    assert np.all(corrupted <= c + 1e-6)
    assert np.all(corrupted >= -c - 1e-6)

    # In-range values must be strictly untouched
    in_range = (sample_ecg >= -c) & (sample_ecg <= c)
    np.testing.assert_array_equal(corrupted[in_range], sample_ecg[in_range])

    # Out-of-range values must be clipped to the boundary
    above = sample_ecg > c
    if np.any(above):
        np.testing.assert_allclose(corrupted[above], c, atol=1e-6)
    below = sample_ecg < -c
    if np.any(below):
        np.testing.assert_allclose(corrupted[below], -c, atol=1e-6)


def test_corrupt_temporal_shift_circular_and_inverse(sample_ecg):
    """Verify circular temporal translation and inverse property: shift(shift(x, k), -k) == x."""
    for k in (-100, -50, 50, 100):
        shifted = corrupt(sample_ecg, "temporal_shift", {"shift": k}, None)
        assert shifted.shape == sample_ecg.shape
        # Inverse shift must return bit-for-bit identical original signal
        restored = corrupt(shifted, "temporal_shift", {"shift": -k}, None)
        np.testing.assert_array_equal(restored, sample_ecg)

        # Check that axis 2 was rolled circularly by k
        expected = np.roll(sample_ecg, k, axis=2)
        np.testing.assert_array_equal(shifted, expected)


def test_corrupt_all_leads_zero_flatline(sample_ecg):
    """Verify all_leads_zero zeroes all 12 leads and repreprocess produces exact zeros without NaNs."""
    corrupted = corrupt(sample_ecg, "missing_leads", {"keep": []}, None)
    assert corrupted.shape == sample_ecg.shape
    np.testing.assert_array_equal(corrupted, 0.0)

    # repreprocess on all-zero array must not fail or produce NaN
    reprocessed = repreprocess(corrupted)
    assert reprocessed.shape == sample_ecg.shape
    assert not np.isnan(reprocessed).any()
    assert not np.isinf(reprocessed).any()
    np.testing.assert_array_equal(reprocessed, 0.0)


def test_corrupt_deterministic_reproducibility(sample_ecg):
    """Verify that deterministic perturbations produce identical arrays regardless of seed."""
    for kind, params in [
        ("missing_leads", {"keep": [0]}),
        ("missing_leads", {"keep": [1]}),
        ("missing_leads", {"keep": list(range(0, 6))}),
        ("missing_leads", {"keep": list(range(6, 12))}),
        ("missing_leads", {"keep": []}),
        ("clipping", {"threshold": 2.0}),
        ("temporal_shift", {"shift": 50}),
    ]:
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(999)
        out1 = corrupt(sample_ecg, kind, params, rng1)
        out2 = corrupt(sample_ecg, kind, params, rng2)
        np.testing.assert_array_equal(out1, out2)


def test_corrupt_stochastic_seed_variation(sample_ecg):
    """Verify that stochastic perturbations vary across seeds and reproduce with same seed."""
    for kind, params in [
        ("gaussian_noise", {"sigma": 0.1}),
        ("baseline_wander", {"amplitude": 1.0}),
        ("random_lead_drop", {"k": 3}),
    ]:
        rng0_a = np.random.default_rng(0)
        rng0_b = np.random.default_rng(0)
        rng1 = np.random.default_rng(1)

        out0_a = corrupt(sample_ecg, kind, params, rng0_a)
        out0_b = corrupt(sample_ecg, kind, params, rng0_b)
        out1 = corrupt(sample_ecg, kind, params, rng1)

        # Same seed must yield identical output
        np.testing.assert_array_equal(out0_a, out0_b)
        # Different seed must yield distinct output
        assert not np.array_equal(out0_a, out1)


def test_repreprocess_preserves_zeroed_leads(sample_ecg):
    """Verify that zeroed leads stay all-zero after 0.5-40 Hz bandpass filtering and z-scoring."""
    # Zero leads 0, 1, 2
    corrupted = sample_ecg.copy()
    corrupted[:, :3, :] = 0.0
    out = repreprocess(corrupted)
    assert out.shape == sample_ecg.shape
    np.testing.assert_array_equal(out[:, :3, :], 0.0)
    # Remaining leads should be normalized (mean ~ 0, std ~ 1)
    for i in range(2):
        for lead in range(3, 12):
            assert abs(out[i, lead].mean()) < 1e-4
            assert abs(out[i, lead].std() - 1.0) < 1e-2


def test_patient_bootstrap_clustering_reuse():
    """Verify that patient_bootstrap_indices preserves patient clusters and draws all records per patient."""
    # 5 patients: p1 has 3 records, p2 has 1, p3 has 2, p4 has 1, p5 has 1 (total 8 records)
    patients = np.array(["p1", "p1", "p1", "p2", "p3", "p3", "p4", "p5"])
    replicates = list(patient_bootstrap_indices(patients, n_boot=20, seed=42))
    assert len(replicates) == 20

    for idx in replicates:
        chosen_patients = patients[idx]
        counts = pd.Series(chosen_patients).value_counts()
        # If p1 is selected, its count must be a multiple of 3
        if "p1" in counts:
            assert counts["p1"] % 3 == 0
        # If p3 is selected, its count must be a multiple of 2
        if "p3" in counts:
            assert counts["p3"] % 2 == 0


def test_frozen_thresholds_validation_derived():
    """Verify that thresholds used for robustness match original validation evaluations and are not re-tuned."""
    for name in ["baseline", "multiscale", "federated_r8"]:
        eval_dict, _ = load_evaluation(name)
        th = thresholds_from_evaluation(eval_dict)
        assert len(th) == 5
        assert all(0.0 < t < 1.0 for t in th)
        # Verify exact stored values
        for i, c in enumerate(CLASSES):
            assert th[i] == pytest.approx(eval_dict["thresholds"][c], abs=1e-6)


def test_guard_outputs_protection(tmp_path):
    """Verify that guard_outputs raises OutputExistsError when target files exist without overwrite."""
    f1 = str(tmp_path / "test1.csv")
    f2 = str(tmp_path / "test2.json")
    with open(f1, "w") as f:
        f.write("content")

    with pytest.raises(OutputExistsError):
        guard_outputs([f1, f2], overwrite=False)

    # With overwrite=True, it should succeed
    paths = guard_outputs([f1, f2], overwrite=True)
    assert paths == [f1, f2]


def test_historical_robustness_artifacts_preserved():
    """Verify that the historical results directory results/revised/robustness/ remains untouched."""
    hist_dir = "results/revised/robustness"
    assert os.path.exists(hist_dir), "Historical robustness directory missing!"
    csv_path = os.path.join(hist_dir, "robustness_metrics.csv")
    json_path = os.path.join(hist_dir, "robustness_summary.json")
    png_path = os.path.join(hist_dir, "robustness_macro_auroc.png")

    assert os.path.isfile(csv_path)
    assert os.path.isfile(json_path)
    assert os.path.isfile(png_path)

    df = pd.read_csv(csv_path)
    assert len(df) == 45, f"Expected exactly 45 rows in historical robustness_metrics.csv, found {len(df)}"
    with open(json_path) as f:
        summary = json.load(f)
    assert "protocol" in summary
    assert summary["seed"] == 0


def test_extended_robustness_artifacts_schema_and_values():
    """Verify that results/revised/robustness_extended/ contains all expected outputs with valid schema."""
    ext_dir = "results/revised/robustness_extended"
    assert os.path.isdir(ext_dir)
    for fname in [
        "robustness_metrics.csv",
        "robustness_paired_ci.csv",
        "robustness_summary.json",
        "robustness_macro_auroc.png",
        "robustness_macro_f1.png",
    ]:
        p = os.path.join(ext_dir, fname)
        assert os.path.isfile(p), f"Missing {p}"

    df_ci = pd.read_csv(os.path.join(ext_dir, "robustness_paired_ci.csv"))
    # 23 conditions * 3 models = 69 rows
    assert len(df_ci) == 69
    assert set(df_ci["model"].unique()) == {"baseline", "multiscale", "federated_r8"}
    assert "all_leads_zero" in df_ci["level"].values
    assert "lead_II_only" in df_ci["level"].values
    assert "clipping" in df_ci["perturbation"].values
    assert "temporal_shift" in df_ci["perturbation"].values

    # Check flatline AUROC is 0.50
    flatline = df_ci[df_ci["level"] == "all_leads_zero"]
    for _, row in flatline.iterrows():
        assert row["macro_auroc_mean"] == pytest.approx(0.5, abs=1e-6)

    # Check temporal shift CI includes zero (shift invariance)
    tshifts = df_ci[df_ci["perturbation"] == "temporal_shift"]
    assert len(tshifts) == 12  # 4 shifts * 3 models
    assert not tshifts["delta_auroc_ci_excludes_zero"].any()

    # Check metrics CSV has full per-class columns
    df_m = pd.read_csv(os.path.join(ext_dir, "robustness_metrics.csv"))
    for c in CLASSES:
        assert f"auroc_{c}" in df_m.columns
        assert f"f1_{c}" in df_m.columns
        assert f"precision_{c}" in df_m.columns
        assert f"recall_{c}" in df_m.columns
        assert f"support_{c}" in df_m.columns

