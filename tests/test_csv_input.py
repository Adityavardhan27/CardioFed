"""Tests for CSV input validation in CardioFed Streamlit application (src/app.py)."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.app import LEADS, prepare_ecg_csv

ROOT = Path(__file__).resolve().parent.parent


def _generate_gaussian_fixture_df(n_samples=1000, seed=42):
    """Generate a synthetic Gaussian statistical test fixture DataFrame (zero-mean, unit-variance standardized).

    Note: This is an artificial statistical fixture designed solely for input validation and schema
    testing, NOT a realistic ECG waveform or clinically representative data.
    """
    rng = np.random.default_rng(seed)
    # Generate n_samples Gaussian values with zero mean and unit variance per lead
    raw = rng.standard_normal(size=(n_samples, len(LEADS))).astype(np.float32)
    # Ensure exact zero mean and unit variance for statistical fixture testing
    normalized = (raw - raw.mean(axis=0, keepdims=True)) / (raw.std(axis=0, keepdims=True) + 1e-8)
    return pd.DataFrame(normalized, columns=LEADS)


def _df_to_csv_io(df):
    """Convert DataFrame to an in-memory CSV file-like buffer."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------------
# Valid Input Tests
# -----------------------------------------------------------------------------

def test_prepare_ecg_csv_valid_gaussian_fixture():
    """Verify that a synthetic Gaussian statistical test fixture conforming to shape and normalization requirements is accepted.

    (Note: This is an artificial statistical fixture, NOT a realistic ECG waveform).
    """
    df = _generate_gaussian_fixture_df(1000)
    csv_buf = _df_to_csv_io(df)

    signal, ecg_df = prepare_ecg_csv(csv_buf)

    assert isinstance(signal, torch.Tensor)
    assert signal.shape == (1, 12, 1000)
    assert signal.dtype == torch.float32
    assert isinstance(ecg_df, pd.DataFrame)
    assert list(ecg_df.columns) == LEADS
    assert len(ecg_df) == 1000


def test_prepare_ecg_csv_with_metadata_columns():
    """Verify that standard metadata columns (e.g. sample, time) are permitted alongside valid Gaussian statistical fixture data."""
    df = _generate_gaussian_fixture_df(1000)
    df.insert(0, "ecg_id", 9999)
    df.insert(1, "sample", np.arange(1000))
    df.insert(2, "time_seconds", np.arange(1000) * 0.01)
    csv_buf = _df_to_csv_io(df)

    signal, ecg_df = prepare_ecg_csv(csv_buf)

    assert signal.shape == (1, 12, 1000)
    assert list(ecg_df.columns) == LEADS


def test_prepare_ecg_csv_existing_smoke_tests():
    """Verify that all existing repository smoke test CSVs pass validation."""
    smoke_dir = ROOT / "results" / "app_smoke_tests"
    smoke_files = list(smoke_dir.glob("*.csv"))
    assert len(smoke_files) > 0, "No smoke test CSV files found."

    for file_path in smoke_files:
        signal, ecg_df = prepare_ecg_csv(str(file_path))
        assert signal.shape == (1, 12, 1000)
        assert list(ecg_df.columns) == LEADS
        assert len(ecg_df) == 1000


# -----------------------------------------------------------------------------
# Structural and Format Validation Tests
# -----------------------------------------------------------------------------

def test_reject_empty_csv():
    """Verify that an empty CSV file is rejected."""
    buf = io.StringIO("")
    with pytest.raises(ValueError, match="empty|Could not parse"):
        prepare_ecg_csv(buf)


def test_reject_wrong_sample_count():
    """Verify that non-1000 row counts are rejected when using the Gaussian statistical fixture."""
    # Too short (500 samples)
    df_short = _generate_gaussian_fixture_df(500)
    with pytest.raises(ValueError, match="Expected exactly 1,000 samples"):
        prepare_ecg_csv(_df_to_csv_io(df_short))

    # Too long (1001 samples)
    df_long = _generate_gaussian_fixture_df(1001)
    with pytest.raises(ValueError, match="Expected exactly 1,000 samples"):
        prepare_ecg_csv(_df_to_csv_io(df_long))


def test_reject_missing_leads():
    """Verify that a CSV fixture missing required lead columns is rejected."""
    df = _generate_gaussian_fixture_df(1000).drop(columns=["V6", "Lead_I"])
    with pytest.raises(ValueError, match="Missing required ECG lead column"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_wrong_lead_order():
    """Verify that lead columns arranged in non-standard order in a fixture are rejected."""
    df = _generate_gaussian_fixture_df(1000)
    # Swap V1 and Lead_I
    cols = list(df.columns)
    cols[0], cols[6] = cols[6], cols[0]
    df_swapped = df[cols]

    with pytest.raises(ValueError, match="ECG lead columns appear in the incorrect order"):
        prepare_ecg_csv(_df_to_csv_io(df_swapped))


def test_reject_duplicate_lead_columns():
    """Verify that CSVs with duplicate lead column names are rejected."""
    # Construct a CSV with two V1 columns
    header = "Lead_I,Lead_II,Lead_III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6,V1\n"
    row = "0.0," * 12 + "0.0\n"
    csv_text = header + row * 1000
    with pytest.raises(ValueError, match="Duplicate ECG lead column"):
        prepare_ecg_csv(io.StringIO(csv_text))


# -----------------------------------------------------------------------------
# Numeric and Value Integrity Tests
# -----------------------------------------------------------------------------

def test_reject_nan_values():
    """Verify that missing/NaN values in a statistical fixture are detected and rejected."""
    df = _generate_gaussian_fixture_df(1000)
    df.loc[10, "Lead_II"] = np.nan
    with pytest.raises(ValueError, match="missing \\(NaN\\) or non-numeric"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_non_numeric_strings():
    """Verify that corrupted string entries in a statistical fixture are detected and rejected."""
    df = _generate_gaussian_fixture_df(1000).astype(object)
    df.loc[50, "V3"] = "corrupt_data"
    with pytest.raises(ValueError, match="missing \\(NaN\\) or non-numeric"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_infinite_values():
    """Verify that infinite values (Inf) in a statistical fixture are rejected."""
    df = _generate_gaussian_fixture_df(1000)
    df.loc[25, "aVF"] = np.inf
    with pytest.raises(ValueError, match="infinite \\(Inf\\)"):
        prepare_ecg_csv(_df_to_csv_io(df))


# -----------------------------------------------------------------------------
# Signal Scaling Validation Tests
# -----------------------------------------------------------------------------

def test_reject_flatline_signals():
    """Verify that flatline signals with near-zero variance are rejected."""
    df = pd.DataFrame(np.zeros((1000, len(LEADS)), dtype=np.float32), columns=LEADS)
    with pytest.raises(ValueError, match="zero or near-zero variance across all leads \\(flatline\\)"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_raw_unnormalized_millivolts():
    """Verify that signals with standard deviation far below expected normalization range (e.g. std ~ 0.12) are rejected."""
    rng = np.random.default_rng(0)
    # Generate data with std ~ 0.12, well below normalized unit variance
    raw_mv = (rng.standard_normal((1000, 12)) * 0.12).astype(np.float32)
    df_mv = pd.DataFrame(raw_mv, columns=LEADS)

    with pytest.raises(ValueError, match="outside the expected normalization/scaling range"):
        prepare_ecg_csv(_df_to_csv_io(df_mv))


def test_reject_raw_microvolts_or_large_adc_counts():
    """Verify that signals with amplitudes far above expected normalization range (e.g. std ~ 150) are rejected."""
    rng = np.random.default_rng(0)
    # Generate data with std ~ 150, far above normalized unit variance
    raw_uv = (rng.standard_normal((1000, 12)) * 150.0).astype(np.float32)
    df_uv = pd.DataFrame(raw_uv, columns=LEADS)

    with pytest.raises(ValueError, match="outside the expected normalization/scaling range"):
        prepare_ecg_csv(_df_to_csv_io(df_uv))


def test_reject_severe_dc_baseline_offset():
    """Verify that uncentered signals with huge DC offset outside expected normalization range are rejected."""
    df = _generate_gaussian_fixture_df(1000)
    # Add a massive DC offset to Lead_I
    df["Lead_I"] = df["Lead_I"] + 25.0

    with pytest.raises(ValueError, match="significant DC baseline offset outside the expected normalization/scaling range"):
        prepare_ecg_csv(_df_to_csv_io(df))


# -----------------------------------------------------------------------------
# Independent Per-Lead Validation Tests
# -----------------------------------------------------------------------------

def test_reject_single_flat_lead():
    """Verify that a single flatline lead in a Gaussian statistical fixture cannot pass validation."""
    df = _generate_gaussian_fixture_df(1000)
    df["Lead_II"] = 0.0  # single flat lead

    with pytest.raises(ValueError, match="Lead 'Lead_II' has zero or near-zero variance"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_single_incorrectly_scaled_lead_too_small():
    """Verify that a single lead with reduced scale (std ~ 0.5) in a Gaussian statistical fixture is rejected."""
    df = _generate_gaussian_fixture_df(1000)
    df["V2"] = df["V2"] * 0.5  # under-scaled lead

    with pytest.raises(ValueError, match="Lead 'V2' is outside the expected normalization/scaling range"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_single_incorrectly_scaled_lead_too_large():
    """Verify that a single lead with excessive scaling (std ~ 2.0) in a Gaussian statistical fixture is rejected."""
    df = _generate_gaussian_fixture_df(1000)
    df["Lead_III"] = df["Lead_III"] * 2.0  # over-scaled lead

    with pytest.raises(ValueError, match="Lead 'Lead_III' is outside the expected normalization/scaling range"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_single_lead_unnormalized_millivolts():
    """Verify that a single lead with standard deviation far below expected range (std ~ 0.15) is rejected."""
    df = _generate_gaussian_fixture_df(1000)
    rng = np.random.default_rng(7)
    df["V4"] = (rng.standard_normal(1000) * 0.15).astype(np.float32)

    with pytest.raises(ValueError, match="Lead 'V4' has a standard deviation of .* outside the expected normalization/scaling range"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_reject_single_lead_significant_dc_offset():
    """Verify that a significant DC baseline offset in a single lead of a Gaussian fixture is rejected."""
    df = _generate_gaussian_fixture_df(1000)
    df["aVF"] = df["aVF"] + 1.5  # significant DC offset in one lead

    with pytest.raises(ValueError, match="Lead 'aVF' has a significant DC baseline offset outside the expected normalization/scaling range"):
        prepare_ecg_csv(_df_to_csv_io(df))


def test_accept_gaussian_fixture_csv_with_rounding():
    """Verify that a Gaussian statistical fixture rounded to 2 decimal places passes validation."""
    df = _generate_gaussian_fixture_df(1000)
    df_rounded = df.round(2)
    signal, ecg_df = prepare_ecg_csv(_df_to_csv_io(df_rounded))

    assert signal.shape == (1, 12, 1000)
    assert len(ecg_df) == 1000


def test_reject_wrong_sample_count_clarification():
    """Verify that sample count error clarifies that 1,000 rows alone cannot verify 100 Hz."""
    df_short = _generate_gaussian_fixture_df(900)
    with pytest.raises(ValueError, match="1,000 rows alone cannot verify a 100 Hz sampling rate"):
        prepare_ecg_csv(_df_to_csv_io(df_short))

