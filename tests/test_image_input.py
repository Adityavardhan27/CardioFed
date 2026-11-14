"""Unit tests for ECG image input validation and digitization in CardioFed (src/app.py)."""

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.app import (
    IMAGE_EXPECTED_HEIGHT,
    IMAGE_EXPECTED_WIDTH,
    LEADS,
    digitize_ecg_image,
    prepare_ecg_image,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SYNTHETIC_V2 = ROOT / "results" / "examples" / "ECG_63_synthetic_v2.png"
SAMPLE_SYNTHETIC_V1 = ROOT / "results" / "examples" / "ECG_63_synthetic.png"


class MockUploadedFile:
    """Mock Streamlit UploadedFile interface wrapping in-memory bytes."""

    def __init__(self, data: bytes, name: str = "test_ecg.png"):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


# -----------------------------------------------------------------------------
# Valid Synthetic-Template Tests
# -----------------------------------------------------------------------------

def test_digitize_valid_synthetic_template_image():
    """Verify that a valid CardioFed synthetic template image digitizes successfully."""
    assert SAMPLE_SYNTHETIC_V2.exists(), "Sample synthetic V2 test image not found."

    image = cv2.imread(str(SAMPLE_SYNTHETIC_V2))
    assert image is not None
    assert image.shape == (IMAGE_EXPECTED_HEIGHT, IMAGE_EXPECTED_WIDTH, 3)

    reconstructed_df, debug_image = digitize_ecg_image(image)

    # Output structure and dimensions
    assert isinstance(reconstructed_df, pd.DataFrame)
    assert len(reconstructed_df) == 1000
    assert list(reconstructed_df.columns) == LEADS
    assert isinstance(debug_image, np.ndarray)
    assert debug_image.shape == image.shape

    # Output signal sanity: per-lead values should be finite and within plausible bounds
    arr = reconstructed_df[LEADS].to_numpy(dtype=np.float32)
    assert np.isfinite(arr).all()
    assert np.abs(arr).max() < 25.0
    # Every lead should have active variance
    assert (arr.std(axis=0) > 0.5).all()


def test_prepare_ecg_image_valid_buffer():
    """Verify prepare_ecg_image correctly processes an uploaded file buffer."""
    assert SAMPLE_SYNTHETIC_V2.exists()

    with open(SAMPLE_SYNTHETIC_V2, "rb") as f:
        file_bytes = f.read()

    mock_file = MockUploadedFile(file_bytes)
    signal, reconstructed_df, original_img, debug_img = prepare_ecg_image(mock_file)

    assert isinstance(signal, torch.Tensor)
    assert signal.shape == (1, 12, 1000)
    assert signal.dtype == torch.float32
    assert len(reconstructed_df) == 1000
    assert original_img.shape == (IMAGE_EXPECTED_HEIGHT, IMAGE_EXPECTED_WIDTH, 3)
    assert debug_img.shape == original_img.shape


# -----------------------------------------------------------------------------
# Malformed and Corrupted Image Tests
# -----------------------------------------------------------------------------

def test_reject_none_uploaded_file():
    """Verify that passing None as uploaded_file raises an informative error."""
    with pytest.raises(ValueError, match="No image file was uploaded"):
        prepare_ecg_image(None)


def test_reject_empty_image_file():
    """Verify that an empty image file upload is rejected."""
    empty_file = MockUploadedFile(b"")
    with pytest.raises(ValueError, match="uploaded image file is empty"):
        prepare_ecg_image(empty_file)


def test_reject_corrupted_image_file():
    """Verify that an un-decodable corrupt image file is rejected."""
    corrupted_file = MockUploadedFile(b"not_a_valid_image_header_or_content")
    with pytest.raises(ValueError, match="Could not decode the uploaded image file"):
        prepare_ecg_image(corrupted_file)


def test_reject_non_array_image_input():
    """Verify digitize_ecg_image rejects non-numpy inputs."""
    with pytest.raises(ValueError, match="Invalid image"):
        digitize_ecg_image("not_an_array")

    with pytest.raises(ValueError, match="Invalid image"):
        digitize_ecg_image(None)


def test_reject_grayscale_or_wrong_channel_image():
    """Verify that 2D grayscale or wrong-channel images are rejected."""
    # 2D grayscale
    gray_img = np.full((IMAGE_EXPECTED_HEIGHT, IMAGE_EXPECTED_WIDTH), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected a 3-channel BGR color image"):
        digitize_ecg_image(gray_img)

    # 4-channel image (e.g. RGBA)
    rgba_img = np.full((IMAGE_EXPECTED_HEIGHT, IMAGE_EXPECTED_WIDTH, 4), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected a 3-channel BGR color image"):
        digitize_ecg_image(rgba_img)


def test_reject_blank_solid_color_image():
    """Verify that solid color / blank images with no contrast are rejected."""
    blank_white = np.full((IMAGE_EXPECTED_HEIGHT, IMAGE_EXPECTED_WIDTH, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="blank or solid color"):
        digitize_ecg_image(blank_white)

    blank_black = np.zeros((IMAGE_EXPECTED_HEIGHT, IMAGE_EXPECTED_WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="blank or solid color"):
        digitize_ecg_image(blank_black)


# -----------------------------------------------------------------------------
# Unsupported Layout and Dimension Tests
# -----------------------------------------------------------------------------

def test_reject_unsupported_dimensions_small():
    """Verify that images smaller than expected template dimensions are rejected."""
    small_img = np.full((600, 800, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image dimensions"):
        digitize_ecg_image(small_img)


def test_reject_unsupported_dimensions_arbitrary_photo():
    """Verify that arbitrary photographs/scans (e.g. 4000x3000) are rejected."""
    photo_img = np.full((3000, 4000, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image dimensions"):
        digitize_ecg_image(photo_img)


def test_reject_unsupported_v1_synthetic_template_layout():
    """Verify that older incompatible synthetic layouts (V1, height 1460) are rejected."""
    if SAMPLE_SYNTHETIC_V1.exists():
        v1_img = cv2.imread(str(SAMPLE_SYNTHETIC_V1))
        with pytest.raises(ValueError, match="Unsupported image dimensions"):
            digitize_ecg_image(v1_img)
    else:
        # Create an image with V1 dimensions (1460, 2640, 3)
        v1_mock = np.full((1460, 2640, 3), 255, dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image dimensions"):
            digitize_ecg_image(v1_mock)


# -----------------------------------------------------------------------------
# Waveform Trace Validation and Failure Cases
# -----------------------------------------------------------------------------

def test_reject_missing_lead_trace():
    """Verify that an image with a missing/erased lead trace cannot pass validation."""
    assert SAMPLE_SYNTHETIC_V2.exists()
    img = cv2.imread(str(SAMPLE_SYNTHETIC_V2))

    # Erase lead V2 trace (row index 7: V2) by painting white over the lead ROI
    # Row 7 top = 90 + 7 * 190 = 1420
    row_top = 90 + 7 * 190
    y_start = row_top + 5
    y_end = row_top + 190 - 5
    img[y_start:y_end, 100:2600] = 255  # paint white

    with pytest.raises(ValueError, match="Insufficient waveform detection for lead V2"):
        digitize_ecg_image(img)


def test_reject_discontinuous_waveform_large_gap():
    """Verify that an image with a large broken gap (>100px) in a waveform is rejected."""
    assert SAMPLE_SYNTHETIC_V2.exists()
    img = cv2.imread(str(SAMPLE_SYNTHETIC_V2))

    # Introduce a 200px gap in Lead_I (row 0)
    row_top = 90
    y_start = row_top + 5
    y_end = row_top + 190 - 5
    # Paint white across columns 500 to 750 (gap of 250 pixels)
    img[y_start:y_end, 500:750] = 255

    with pytest.raises(ValueError, match="Excessive waveform discontinuity detected for lead I"):
        digitize_ecg_image(img)
