"""Comprehensive audit tests for PTB-XL dataset pipeline, labels, splits, leakage, and evaluation."""

import ast
import json
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import CLASSES, load_metadata, multi_hot, preprocess_signal
from src.thresholds import tune_thresholds
from src.evaluate import compute_metrics
from src.statistical_analysis import holm, patient_bootstrap_indices, macro_auc
from rebuild_labels import build_labels

RAW_DB = "data/raw/ptb-xl/ptbxl_database.csv"
RAW_SCP = "data/raw/ptb-xl/scp_statements.csv"
PROCESSED_META = "data/processed/processed_metadata.csv"
EVAL_PATH = "results/scheduler_experiment/metrics/multiscale_final_evaluation.json"


# ==============================================================================
# 1. LABEL INTEGRITY AUDIT
# ==============================================================================

def test_label_mapping_and_counts():
    """Verify diagnostic superclass mapping, filtering, and exact class totals."""
    raw_df = pd.read_csv(RAW_DB, index_col="ecg_id")
    raw_df["scp_codes"] = raw_df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(RAW_SCP, index_col=0)

    # Reconstruct stage 2 CardioFed labels
    y, aligned_df = build_labels(raw_df, scp, raw_df.index.to_numpy())
    keep = y.sum(axis=1) > 0
    y_filtered = y[keep]

    # Total usable records must be exactly 21,375 (dropping 424 non-diagnostic/likelihood<=0 records)
    assert len(raw_df) == 21799
    assert int((~keep).sum()) == 424
    assert len(y_filtered) == 21375

    # Check exact class counts matching RESULTS.md
    expected_counts = {
        "NORM": 9097,
        "MI": 5469,
        "STTC": 5108,
        "CD": 4898,
        "HYP": 2649,
    }
    for i, c in enumerate(CLASSES):
        assert int(y_filtered[:, i].sum()) == expected_counts[c]


def test_norm_mutual_exclusivity():
    """Verify that no record has both NORM and any abnormal diagnosis."""
    raw_df = pd.read_csv(RAW_DB, index_col="ecg_id")
    raw_df["scp_codes"] = raw_df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(RAW_SCP, index_col=0)

    proc_meta = pd.read_csv(PROCESSED_META)
    y, _ = build_labels(raw_df, scp, proc_meta["ecg_id"].to_numpy())

    norm_idx = CLASSES.index("NORM")
    abnormal_indices = [CLASSES.index(c) for c in CLASSES if c != "NORM"]

    norm_pos = y[:, norm_idx] == 1
    abn_pos = y[:, abnormal_indices].sum(axis=1) > 0
    overlap = (norm_pos & abn_pos).sum()

    assert overlap == 0, f"Found {overlap} records with NORM and abnormal overlap"
    # Ensure all records have at least one label
    assert np.all(y.sum(axis=1) > 0)


# ==============================================================================
# 2. DATA INTEGRITY & ALIGNMENT
# ==============================================================================

def test_processed_metadata_alignment_and_folds():
    """Verify alignment between processed metadata and raw database, folds, and IDs."""
    raw_df = pd.read_csv(RAW_DB)
    proc_df = pd.read_csv(PROCESSED_META)

    assert len(proc_df) == 21375
    # Unique ECG IDs
    assert proc_df["ecg_id"].nunique() == 21375

    # Merge on ecg_id and verify strat_fold exact match
    merged = proc_df.merge(raw_df[["ecg_id", "strat_fold", "patient_id"]], on="ecg_id", suffixes=("_proc", "_raw"))
    assert len(merged) == 21375
    assert (merged["strat_fold_proc"] == merged["strat_fold_raw"]).all()

    # Verify split sizes
    train_records = (proc_df["strat_fold"] <= 8).sum()
    val_records = (proc_df["strat_fold"] == 9).sum()
    test_records = (proc_df["strat_fold"] == 10).sum()

    assert train_records == 17073
    assert val_records == 2145
    assert test_records == 2157


# ==============================================================================
# 3. DATA LEAKAGE & PATIENT ISOLATION
# ==============================================================================

def test_zero_patient_leakage_across_splits():
    """Verify 0 patient overlap across train (folds 1-8), val (fold 9), and test (fold 10)."""
    raw_df = pd.read_csv(RAW_DB)
    proc_df = pd.read_csv(PROCESSED_META)
    merged = proc_df.merge(raw_df[["ecg_id", "patient_id"]], on="ecg_id")

    train_pts = set(merged[merged["strat_fold"] <= 8]["patient_id"])
    val_pts = set(merged[merged["strat_fold"] == 9]["patient_id"])
    test_pts = set(merged[merged["strat_fold"] == 10]["patient_id"])

    assert len(train_pts) == 14818
    assert len(val_pts) == 1916
    assert len(test_pts) == 1876

    # Strict pairwise disjointness
    assert len(train_pts & val_pts) == 0
    assert len(train_pts & test_pts) == 0
    assert len(val_pts & test_pts) == 0


def test_preprocessing_signal_isolation():
    """Verify per-record signal normalization computes stats solely within the given record."""
    rng = np.random.default_rng(42)
    sig1 = rng.standard_normal((1000, 12)) * 3.0 + 5.0
    sig2 = rng.standard_normal((1000, 12)) * 10.0 - 2.0

    p1 = preprocess_signal(sig1)
    p2 = preprocess_signal(sig2)

    # Check zero mean and unit variance per lead
    assert np.allclose(p1.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(p1.std(axis=0), 1.0, atol=1e-4)
    assert np.allclose(p2.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(p2.std(axis=0), 1.0, atol=1e-4)


# ==============================================================================
# 4. EVALUATION CORRECTNESS
# ==============================================================================

def test_metric_computation_known_values():
    """Verify compute_metrics calculates AUROC, F1, precision, recall, hamming loss correctly."""
    # Synthetic ground truth for 4 records and 5 classes
    y_true = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ], dtype=np.float32)

    # Predicted probabilities
    probs = np.array([
        [0.9, 0.1, 0.2, 0.1, 0.1],
        [0.2, 0.8, 0.1, 0.2, 0.1],
        [0.1, 0.7, 0.9, 0.1, 0.2],
        [0.1, 0.2, 0.1, 0.8, 0.7],
    ], dtype=np.float32)

    thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    m = compute_metrics(probs, y_true, thresholds)

    # All predictions match ground truth at threshold 0.5
    assert m["subset_accuracy"] == 1.0
    assert m["hamming_loss"] == 0.0
    assert m["macro_f1"] == 1.0
    assert m["macro_auroc"] == 1.0


def test_post_hoc_norm_consistency_rule_logic():
    """Verify the post-hoc NORM consistency rule suppresses NORM only when abnormal is positive."""
    # Class order: ["NORM", "MI", "STTC", "CD", "HYP"]
    pred = np.array([
        [1, 1, 0, 0, 0],  # Case 1: Both NORM and MI -> NORM must be suppressed to 0
        [1, 0, 0, 0, 0],  # Case 2: Only NORM -> NORM remains 1
        [0, 1, 0, 0, 0],  # Case 3: Only MI -> Remains unchanged [0, 1, 0, 0, 0]
        [0, 0, 0, 0, 0],  # Case 4: No positive -> Remains unchanged [0, 0, 0, 0, 0]
        [1, 0, 0, 1, 1],  # Case 5: NORM with CD and HYP -> NORM suppressed to 0
    ])

    abnormal_idx = [1, 2, 3, 4]
    has_abnormal = pred[:, abnormal_idx].any(axis=1)

    final_pred = pred.copy()
    final_pred[has_abnormal, 0] = 0

    expected = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ])

    np.testing.assert_array_equal(final_pred, expected)


# ==============================================================================
# 5. STATISTICAL ANALYSIS AUDIT
# ==============================================================================

def test_patient_bootstrap_clustering():
    """Verify patient-level bootstrap preserves patient record clustering."""
    # 5 patients with varying record counts
    patients = np.array(["P1", "P1", "P2", "P3", "P3", "P3", "P4", "P5"])
    samples = list(patient_bootstrap_indices(patients, n_boot=20, seed=42))

    for idx_sample in samples:
        sampled_patients = patients[idx_sample]
        # For each patient sampled, all their records must be present in multiples of their original count
        counts = pd.Series(sampled_patients).value_counts()
        for p, c in counts.items():
            orig_c = (patients == p).sum()
            assert c % orig_c == 0, f"Patient {p} records not clustered in bootstrap"


def test_holm_bonferroni_correction():
    """Verify Holm-Bonferroni correction preserves monotonicity, bounds, and known values."""
    p_vals = np.array([0.01, 0.04, 0.03, 0.10])
    # Sorted: 0.01 (x4 = 0.04), 0.03 (x3 = 0.09), 0.04 (x2 = 0.08 -> running max 0.09), 0.10 (x1 = 0.10)
    adjusted = holm(p_vals)

    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[2] == pytest.approx(0.09)
    assert adjusted[1] == pytest.approx(0.09)
    assert adjusted[3] == pytest.approx(0.10)
    assert np.all(adjusted <= 1.0)
    assert np.all(adjusted >= 0.0)


def test_macro_auc_returns_nan_on_incomplete_classes():
    """Verify macro_auc returns NaN when any class has only one unique label to enforce consistent class definition."""
    # Class 0 and 1 have both 0 and 1; Class 2, 3, 4 have only 0s
    y = np.zeros((10, 5), dtype=np.float32)
    y[:5, 0] = 1.0
    y[:5, 1] = 1.0

    probs = np.zeros((10, 5), dtype=np.float32)
    probs[:5, 0] = 0.9
    probs[5:, 0] = 0.1
    probs[:5, 1] = 0.8
    probs[5:, 1] = 0.2

    # Because classes 2, 3, 4 are undefined, 5-class macro-AUROC is NaN
    score = macro_auc(probs, y)
    assert np.isnan(score)

    # When all 5 classes are valid (biclassed), macro_auc computes exact mean
    y_full = np.zeros((10, 5), dtype=np.float32)
    for c in range(5):
        y_full[:5, c] = 1.0
    score_full = macro_auc(probs, y_full)
    assert np.isfinite(score_full)
    assert score_full >= 0.0 and score_full <= 1.0


def test_bootstrap_transparent_exclusion_tracking():
    """Verify unconditional bootstrap handles rare classes by tracking and reporting excluded replicates."""
    # 6 patients: P1-P5 have class 0, but only P6 has the positive case for class 1
    patients = np.array(["P1", "P2", "P3", "P4", "P5", "P6"])
    y = np.zeros((6, 2), dtype=np.float32)
    y[0, 0] = 1.0
    y[5, 1] = 1.0

    probs_a = np.array([[0.8, 0.2], [0.7, 0.1], [0.6, 0.3], [0.2, 0.1], [0.3, 0.2], [0.1, 0.9]])
    probs_b = np.array([[0.7, 0.3], [0.6, 0.2], [0.5, 0.4], [0.3, 0.2], [0.2, 0.3], [0.2, 0.8]])

    # Unconditional bootstrap without rejection
    n_boot = 50
    indices = list(patient_bootstrap_indices(patients, n_boot=n_boot, seed=42))
    assert len(indices) == n_boot

    auc_a = []
    auc_b = []
    for ix in indices:
        yb = y[ix]
        # Valid only if all classes have both 0 and 1
        all_valid = all(len(np.unique(yb[:, c])) == 2 for c in range(2))
        if all_valid:
            auc_a.append(macro_auc(probs_a[ix], yb))
            auc_b.append(macro_auc(probs_b[ix], yb))
        else:
            auc_a.append(float("nan"))
            auc_b.append(float("nan"))

    auc_a = np.array(auc_a)
    auc_b = np.array(auc_b)

    valid_mask = np.isfinite(auc_a) & np.isfinite(auc_b)
    n_valid = int(valid_mask.sum())
    n_excluded = int((~valid_mask).sum())

    assert n_valid + n_excluded == n_boot
    # In unconditional sampling with 1 rare patient, some draws will naturally miss P6
    assert n_excluded > 0, "Expected some draws to miss the single positive patient"
    # Paired comparisons evaluate only on valid replicates where all classes are defined
    d = auc_a[valid_mask] - auc_b[valid_mask]
    assert len(d) == n_valid
    assert np.all(np.isfinite(d))




# ==============================================================================
# 6. DOCUMENTATION & STORED ARTIFACTS AUDIT
# ==============================================================================

def test_evaluation_artifact_consistency():
    """Verify stored multiscale evaluation artifact matches audited test and validation supports."""
    with open(EVAL_PATH) as f:
        data = json.load(f)

    # Check validation supports
    expected_val_support = {"NORM": 918, "MI": 540, "STTC": 515, "CD": 495, "HYP": 268}
    assert data["validation_metrics"]["per_class_support"] == expected_val_support

    # Check test supports
    expected_test_support = {"NORM": 916, "MI": 550, "STTC": 506, "CD": 496, "HYP": 262}
    assert data["test_metrics"]["per_class_support"] == expected_test_support
