
"""CardioFed: ECG classification, image digitization and visualization."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import MultiScaleResidualECGNet
from src.preprocessing import CLASSES
from src.model_registry import MODELS, DEFAULT_CENTRALIZED
from src.explainability_v2 import integrated_gradients as ig_v2_core


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Primary (reproducible) centralized multi-scale model and its
# validation-tuned thresholds; see src/model_registry.py.
CHECKPOINT_PATH = ROOT / MODELS[DEFAULT_CENTRALIZED]["checkpoint"]
EVALUATION_PATH = ROOT / MODELS[DEFAULT_CENTRALIZED]["evaluation"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

DISPLAY_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

# These values must match generate_synthetic_ecg_v2.py.
IMAGE_LEFT_MARGIN = 100
IMAGE_TOP_MARGIN = 90
IMAGE_ROW_HEIGHT = 190
IMAGE_PAPER_SPEED = 25
IMAGE_PIXELS_PER_MM = 10
IMAGE_DURATION = 10

IMAGE_VERTICAL_SCALE = 12.0
IMAGE_SAMPLE_COUNT = 1000

IMAGE_EXPECTED_WIDTH = (
    IMAGE_LEFT_MARGIN
    + IMAGE_DURATION * IMAGE_PAPER_SPEED * IMAGE_PIXELS_PER_MM
    + 40
)

IMAGE_EXPECTED_HEIGHT = (
    IMAGE_TOP_MARGIN
    + 12 * IMAGE_ROW_HEIGHT
    + 50
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CardioFed | ECG Analysis",
    page_icon="❤️",
    layout="wide"
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@st.cache_data
def load_thresholds():

    with open(EVALUATION_PATH, "r") as file:
        evaluation = json.load(file)

    return np.array(
        [evaluation["thresholds"][name] for name in CLASSES],
        dtype=np.float32
    )


# --------------------------------------------------
# CSV INPUT
# --------------------------------------------------

def prepare_ecg_csv(uploaded_file):
    """
    Validate and prepare an uploaded 12-lead ECG CSV for CardioFed inference.

    Requirements:
      - Valid CSV format with exactly 1,000 samples. Note: having 1,000 rows satisfies the
        input dimension expected by the model (10 seconds at 100 Hz), but the row count alone
        cannot verify or ensure that the recording was acquired or resampled at a 100 Hz sampling rate.
      - All 12 standard leads must be present in standard order:
        Lead_I, Lead_II, Lead_III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
      - Finite numeric values (no NaN, null, or Inf).
      - Preprocessed signal scaling: signals must be per-lead zero-mean and unit-variance
        standardized consistent with src/preprocessing.py ((signal - mean) / (std + 1e-8)),
        allowing for reasonable CSV rounding tolerance (std within [0.80, 1.20] and |mean| <= 0.10).
        Statistical scaling checks catch signals outside the expected normalization/scaling range
        (such as severe under-scaling, excessive amplitude, flatlines, or uncentered baselines),
        but passing these statistical checks does not prove or guarantee that proper bandpass
        filtering (0.5–40 Hz) or correct preprocessing was performed.
    """

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as error:
        raise ValueError(f"Could not parse uploaded CSV file: {error}")

    if df.empty or len(df.columns) == 0:
        raise ValueError("The uploaded CSV file is empty.")

    # 1. Check for missing lead columns
    missing = [
        lead for lead in LEADS
        if lead not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required ECG lead column(s): {', '.join(missing)}. "
            f"Expected all 12 leads: {', '.join(LEADS)}."
        )

    # 2. Check for duplicate lead columns
    # Note: pandas read_csv mangles duplicate columns to 'Lead.1', 'Lead.2', etc.
    duplicates = [
        lead for lead in LEADS
        if list(df.columns).count(lead) > 1
        or any(
            col == f"{lead}.1"
            or (col.startswith(f"{lead}.") and col[len(lead) + 1:].isdigit())
            for col in df.columns
        )
    ]
    if duplicates:
        raise ValueError(
            f"Duplicate ECG lead column(s) detected: {', '.join(duplicates)}."
        )

    # 3. Validate lead ordering
    lead_cols_in_csv = [col for col in df.columns if col in set(LEADS)]
    if lead_cols_in_csv != LEADS:
        raise ValueError(
            f"ECG lead columns appear in the incorrect order.\n"
            f"Expected standard order: {', '.join(LEADS)}\n"
            f"Found in CSV: {', '.join(lead_cols_in_csv)}"
        )

    # 4. Check sample count (length)
    if len(df) != 1000:
        raise ValueError(
            f"Expected exactly 1,000 samples (10 seconds sampled at 100 Hz), "
            f"but received {len(df)} samples. Note that 1,000 rows alone cannot "
            f"verify a 100 Hz sampling rate."
        )

    # 5. Validate numeric values, NaN, and Inf
    signal_df = df[LEADS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    nan_mask = signal_df.isna()
    if nan_mask.any().any():
        nan_cols = [col for col in LEADS if nan_mask[col].any()]
        nan_count = int(nan_mask.sum().sum())
        raise ValueError(
            f"The ECG contains {nan_count} missing (NaN) or non-numeric value(s) "
            f"in lead(s): {', '.join(nan_cols)}."
        )

    signal_array = signal_df.to_numpy(dtype=np.float32)

    inf_mask = np.isinf(signal_array)
    if inf_mask.any():
        inf_count = int(inf_mask.sum())
        raise ValueError(
            f"The ECG contains {inf_count} infinite (Inf) value(s)."
        )

    # 6. Signal scaling validation
    # CardioFed models expect per-lead zero-mean, unit-variance standardized signals (z-scores)
    # consistent with src/preprocessing.py: (signal - mean) / (std + 1e-8).
    # All 12 leads are validated independently with reasonable CSV rounding tolerance.
    # Note: Statistical scaling checks detect signals outside the expected normalization/scaling range,
    # but passing these checks does NOT prove or guarantee that correct preprocessing
    # (such as 0.5-40 Hz bandpass filtering) was performed.
    lead_stds = signal_array.std(axis=0)
    lead_means = signal_array.mean(axis=0)
    lead_max_abs = np.abs(signal_array).max(axis=0)
    mean_std = float(lead_stds.mean())
    max_abs = float(lead_max_abs.max())

    # 6a. Flatline / zero-variance check (all leads or single lead)
    if (lead_stds < 1e-4).all():
        raise ValueError(
            "The uploaded ECG signal has zero or near-zero variance across all leads (flatline). "
            "Valid ECG signals must contain active physiological waveforms."
        )
    flat_lead_indices = np.where(lead_stds < 1e-4)[0]
    if len(flat_lead_indices) > 0:
        bad_lead = LEADS[flat_lead_indices[0]]
        raise ValueError(
            f"Lead '{bad_lead}' has zero or near-zero variance (std = {lead_stds[flat_lead_indices[0]]:.4f}, flatline). "
            f"Valid ECG signals must contain active physiological waveforms across all 12 leads."
        )

    # 6b. Severe under-scaling / below expected normalization range
    if mean_std < 0.25:
        raise ValueError(
            f"The uploaded ECG has a mean lead standard deviation of {mean_std:.3f}, "
            f"which is outside the expected normalization/scaling range (expected per-lead std ≈ 1.0). "
            f"CardioFed models require per-lead Z-score normalized signals (expected lead std ≈ 1.0). "
            f"Please preprocess your signal with CardioFed's 0.5–40 Hz bandpass filter and z-score standardization."
        )
    low_std_indices = np.where(lead_stds < 0.25)[0]
    if len(low_std_indices) > 0:
        bad_lead = LEADS[low_std_indices[0]]
        raise ValueError(
            f"Lead '{bad_lead}' has a standard deviation of {lead_stds[low_std_indices[0]]:.3f}, "
            f"which is outside the expected normalization/scaling range (expected lead std ≈ 1.0). "
            f"CardioFed models require per-lead Z-score normalized signals (expected lead std ≈ 1.0)."
        )

    # 6c. Excessive amplitude / above expected normalization range
    if mean_std > 20.0 or max_abs > 100.0:
        raise ValueError(
            f"The uploaded ECG contains values outside the expected normalization/scaling range "
            f"(mean lead std = {mean_std:.1f}, max absolute amplitude = {max_abs:.1f}; "
            f"expected lead std ≈ 1.0, typical peaks within ±25). "
            f"CardioFed models require per-lead Z-score normalized signals."
        )
    excessive_lead_indices = np.where((lead_stds > 20.0) | (lead_max_abs > 100.0))[0]
    if len(excessive_lead_indices) > 0:
        bad_lead = LEADS[excessive_lead_indices[0]]
        raise ValueError(
            f"Lead '{bad_lead}' contains values outside the expected normalization/scaling range "
            f"(std = {lead_stds[excessive_lead_indices[0]]:.1f}, "
            f"max absolute amplitude = {lead_max_abs[excessive_lead_indices[0]]:.1f}; "
            f"expected lead std ≈ 1.0). "
            f"CardioFed models require per-lead Z-score normalized signals."
        )

    # 6d. Per-lead incorrect scaling check (outside reasonable CSV rounding tolerance [0.80, 1.20])
    # Signals standardized via src/preprocessing.py have std = 1.0 per lead.
    scale_err_indices = np.where((lead_stds < 0.80) | (lead_stds > 1.20))[0]
    if len(scale_err_indices) > 0:
        bad_lead = LEADS[scale_err_indices[0]]
        raise ValueError(
            f"Lead '{bad_lead}' is outside the expected normalization/scaling range "
            f"(std = {lead_stds[scale_err_indices[0]]:.3f}). "
            f"CardioFed models require per-lead Z-score normalized signals (expected lead std ≈ 1.0, "
            f"acceptable range 0.80–1.20 allowing for CSV rounding tolerance)."
        )

    # 6e. Per-lead DC baseline offset check (allowed |mean| <= 0.10 allowing for CSV rounding tolerance)
    # Signals standardized via src/preprocessing.py have mean = 0.0 per lead.
    offset_err_indices = np.where(np.abs(lead_means) > 0.10)[0]
    if len(offset_err_indices) > 0:
        bad_lead = LEADS[offset_err_indices[0]]
        raise ValueError(
            f"Lead '{bad_lead}' has a significant DC baseline offset outside the expected normalization/scaling range "
            f"(mean = {lead_means[offset_err_indices[0]]:.3f}). "
            f"CardioFed models require zero-mean standardized signals (expected lead mean ≈ 0.0, "
            f"allowed absolute mean ≤ 0.10 allowing for CSV rounding tolerance)."
        )

    # Convert to PyTorch tensor of shape [1, 12, 1000]
    signal = torch.from_numpy(
        signal_array.T.copy()
    ).unsqueeze(0)

    return signal, df[LEADS].copy()


# --------------------------------------------------
# IMAGE DIGITIZATION
# --------------------------------------------------

def create_waveform_mask(roi):
    """Detect dark red waveform pixels using HSV."""

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    lower_red_1 = np.array([0, 100, 40])
    upper_red_1 = np.array([12, 255, 220])

    lower_red_2 = np.array([165, 100, 40])
    upper_red_2 = np.array([180, 255, 220])

    mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    return cv2.bitwise_or(mask_1, mask_2)


def digitize_ecg_image(image):
    """
    Reconstruct 12 ECG leads from the CardioFed synthetic
    12-row image layout.

    Returns:
        reconstructed_df
        debug_image
    """

    height, width = image.shape[:2]

    if (
        width < IMAGE_EXPECTED_WIDTH
        or height < IMAGE_EXPECTED_HEIGHT
    ):
        raise ValueError(
            "Image dimensions do not match the expected "
            "CardioFed synthetic ECG layout. "
            f"Expected at least {IMAGE_EXPECTED_WIDTH}x"
            f"{IMAGE_EXPECTED_HEIGHT}, received {width}x{height}."
        )

    x_start = IMAGE_LEFT_MARGIN

    x_end = (
        IMAGE_LEFT_MARGIN
        + IMAGE_DURATION
        * IMAGE_PAPER_SPEED
        * IMAGE_PIXELS_PER_MM
    )

    reconstructed = {}

    debug_image = image.copy()

    for i, lead in enumerate(LEADS):

        row_top = (
            IMAGE_TOP_MARGIN
            + i * IMAGE_ROW_HEIGHT
        )

        center_y = (
            row_top
            + IMAGE_ROW_HEIGHT // 2
        )

        y_start = row_top + 5
        y_end = row_top + IMAGE_ROW_HEIGHT - 5

        roi = image[
            y_start:y_end,
            x_start:x_end
        ]

        mask = create_waveform_mask(roi)

        detected_columns = []
        median_positions = []

        # Find median waveform position in every column.
        for x in range(mask.shape[1]):

            ys = np.where(mask[:, x] > 0)[0]

            if len(ys) > 0:

                detected_columns.append(x)

                median_positions.append(
                    float(np.median(ys))
                )

        minimum_detected = int(
            mask.shape[1] * 0.80
        )

        if len(detected_columns) < minimum_detected:
            raise ValueError(
                f"Insufficient waveform detection for "
                f"{DISPLAY_LEADS[i]}. "
                "The image may not match the expected template."
            )

        # Fill missing columns through interpolation.
        pixel_positions = np.interp(
            np.arange(mask.shape[1]),
            detected_columns,
            median_positions
        )

        absolute_y = pixel_positions + y_start

        # Convert pixel coordinates into normalized values.
        signal_values = (
            center_y - absolute_y
        ) / IMAGE_VERTICAL_SCALE

        # V5: Sample at exact ECG source sample positions.
        sample_positions = (
            np.arange(IMAGE_SAMPLE_COUNT)
            * IMAGE_PAPER_SPEED
            * IMAGE_PIXELS_PER_MM
            / 100
        )

        reconstructed_signal = np.interp(
            sample_positions,
            np.arange(len(signal_values)),
            signal_values
        )

        reconstructed[lead] = (
            reconstructed_signal.astype(np.float32)
        )

        # Draw the detected path for visual inspection.
        points = np.column_stack((
            np.arange(len(pixel_positions)) + x_start,
            absolute_y
        )).astype(np.int32)

        cv2.polylines(
            debug_image,
            [points.reshape(-1, 1, 2)],
            False,
            (0, 180, 0),
            2
        )

        print(
            f"Image digitization - {DISPLAY_LEADS[i]}: "
            f"{len(reconstructed_signal)} samples"
        )

    reconstructed_df = pd.DataFrame(reconstructed)

    return reconstructed_df, debug_image


def prepare_ecg_image(uploaded_file):

    image_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    reconstructed_df, debug_image = digitize_ecg_image(image)

    signal_array = reconstructed_df[LEADS].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(signal_array).all():
        raise ValueError(
            "Digitization generated invalid signal values."
        )

    signal = torch.from_numpy(
        signal_array.T.copy()
    ).unsqueeze(0)

    return signal, reconstructed_df, image, debug_image


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def flat_lead_names(signal, min_std=1e-3):
    """Return names of leads whose standard deviation is ~0 (flat/missing)."""

    stds = signal[0].std(dim=1).cpu().numpy()

    return [
        LEADS[i] for i, value in enumerate(stds)
        if value < min_std
    ]


def predict_ecg(signal, model, thresholds):

    signal = signal.to(DEVICE)

    with torch.no_grad():

        logits = model(signal)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()[0]

    original_predictions = (
        probabilities >= thresholds
    ).astype(int)

    # Project-specific NORM consistency rule.
    final_predictions = original_predictions.copy()

    abnormal_indices = [
        CLASSES.index(name)
        for name in ["MI", "STTC", "CD", "HYP"]
    ]

    if final_predictions[abnormal_indices].any():
        final_predictions[0] = 0

    return (
        probabilities,
        original_predictions,
        final_predictions
    )


# --------------------------------------------------
# INTEGRATED GRADIENTS
# --------------------------------------------------

def integrated_gradients(
    model,
    signal,
    target_class,
    steps=64
):
    """
    Calculate Integrated Gradients attribution using the validated 64-step midpoint
    Riemann sum from src/explainability_v2.py.

    Returns:
        attribution: Tensor of shape [1, 12, 1000]
        completeness_error: float completeness error fraction
    """
    model.eval()
    x = signal[0].detach()
    attribution, error, _, _ = ig_v2_core(
        model=model,
        x=x,
        target=target_class,
        steps=steps,
        batch=steps
    )
    return attribution.unsqueeze(0), float(error)


def plot_integrated_gradients(
    signal,
    attribution,
    target_name,
    completeness_error=None
):

    signal_np = (
        signal[0].detach().cpu().numpy()
    )

    attribution_np = (
        attribution[0].detach().cpu().numpy()
    )

    time = np.arange(
        signal_np.shape[1]
    ) / 100.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        constrained_layout=True
    )

    # Display leads from top (Lead I) to bottom (V6), matching explainability_v2.
    offset = 5.0
    for lead in range(12):

        axes[0].plot(
            time,
            signal_np[lead] - lead * offset,
            linewidth=0.7,
            label=DISPLAY_LEADS[lead]
        )

    axes[0].set_title("12-Lead ECG Waveforms", fontsize=12)
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("ECG Lead (offset)")
    axes[0].set_xlim(0, 10)
    axes[0].set_yticks([-lead * offset for lead in range(12)])
    axes[0].set_yticklabels(DISPLAY_LEADS)
    axes[0].grid(alpha=0.2)

    # Attribution heatmap with identical lead order: Lead I at top, V6 at bottom.
    lim = float(np.percentile(np.abs(attribution_np), 99.5)) or 1.0
    im = axes[1].imshow(
        attribution_np,
        aspect="auto",
        cmap="seismic",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        interpolation="nearest",
        extent=[0, 10, 11.5, -0.5]
    )

    if completeness_error is not None:
        axes[1].set_title(
            f"Integrated Gradients Attribution for {target_name} "
            f"(64-step midpoint, completeness error: {completeness_error:.1%})",
            fontsize=12
        )
    else:
        axes[1].set_title(
            f"Integrated Gradients Attribution for {target_name} (64-step midpoint)",
            fontsize=12
        )

    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("ECG Lead")
    axes[1].set_xlim(0, 10)
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(DISPLAY_LEADS)

    fig.colorbar(
        im,
        ax=axes[1],
        label="Attribution"
    )

    return fig


# --------------------------------------------------
# ECG VISUALIZATION
# --------------------------------------------------

def plot_ecg(df):

    fig, axes = plt.subplots(
        6,
        2,
        figsize=(15, 12),
        sharex=True
    )

    time = np.arange(len(df)) / 100.0

    for i, lead in enumerate(LEADS):

        ax = axes[i // 2, i % 2]

        ax.plot(
            time,
            df[lead],
            linewidth=0.8
        )

        ax.set_title(
            DISPLAY_LEADS[i],
            fontsize=10
        )

        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    fig.suptitle(
        "12-Lead ECG Waveforms",
        fontsize=16
    )

    fig.tight_layout()

    return fig


# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("❤️ CardioFed")

st.subheader(
    "Multi-Label ECG Classification Using a Multi-Scale Residual CNN"
)

st.markdown(
    """
    Upload a 12-lead ECG recording as a CSV file or a
    supported synthetic ECG image. CardioFed reconstructs
    the signal when an image is provided and analyzes it
    using the trained Multi-Scale Residual CNN.
    """
)

input_mode = st.radio(
    "Choose ECG input format",
    ["CSV File", "ECG Image"],
    horizontal=True
)

uploaded_file = None

if input_mode == "CSV File":

    st.info(
        "**CSV Upload Requirements (Preprocessed, Model-Ready Signals)**\n\n"
        "CardioFed deep-learning models evaluate preprocessed 12-lead ECG signals directly. "
        "Uploaded CSVs must adhere to the following specification:\n\n"
        "- **Signal Dimensions**: Exactly 1,000 samples (10 seconds sampled at 100 Hz). "
        "*(Note: 1,000 rows alone cannot verify that the recording was sampled at 100 Hz; "
        "users must ensure their recordings are properly acquired or resampled to 100 Hz).*\n"
        "- **Required Leads**: All 12 standard leads in standard order: "
        "`Lead_I`, `Lead_II`, `Lead_III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`.\n"
        "- **Preprocessing Status**: Signals **must already be filtered** (0.5–40 Hz bandpass) and "
        "**per-lead Z-score normalized** (zero mean, unit variance per lead as in `src/preprocessing.py`).\n"
        "- **Validation Scope**: All 12 leads are validated independently for mean (≈ 0.0) and "
        "standard deviation (≈ 1.0) allowing for reasonable CSV rounding tolerance. "
        "Statistical scaling checks reject signals outside the expected normalization/scaling range "
        "(such as flatlines, under-scaled signals, excessive amplitudes, and baseline offsets); "
        "however, passing statistical scaling checks does not prove or guarantee "
        "that correct bandpass filtering or preprocessing was performed.\n"
        "- **Note on Raw Signals**: The application does **not** automatically filter, resample, or normalize raw ECG signals "
        "because input sampling frequencies and physical voltage units cannot be reliably inferred. "
        "Signals outside the expected normalization/scaling range will be rejected by input validation."
    )

    with st.expander("ℹ️ Preprocessing & CSV Formatting Guidelines"):
        st.markdown(
            """
            **How to prepare your CSV file for CardioFed:**
            1. **Sampling Rate**: Resample your 10-second recording to 100 Hz (1,000 rows). *Note: Row count alone cannot verify a 100 Hz sampling rate.*
            2. **Filtering**: Apply a 0.5–40 Hz Butterworth bandpass filter (order 3, zero-phase as in `src/preprocessing.py`).
            3. **Normalization**: Standardize each lead independently: `(signal - mean) / (std + 1e-8)`.
            4. **Columns**: Name the 12 columns `Lead_I`, `Lead_II`, `Lead_III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`.
            *(Optional metadata columns such as `ecg_id` or `time_seconds` are permitted as long as the 12 leads appear in standard order).*

            *Important*: Statistical scaling checks confirm per-lead zero mean and unit variance within CSV rounding tolerance, but they do not prove that proper 0.5–40 Hz bandpass filtering was performed.
            """
        )

    uploaded_file = st.file_uploader(
        "Upload ECG CSV",
        type=["csv"],
        key="csv_uploader"
    )

else:

    st.info(
        "Experimental image support: upload a CardioFed synthetic "
        "12-lead ECG image generated using the corrected V2 layout. "
        "Arbitrary clinical ECG photographs and scanned reports "
        "are not yet supported."
    )

    uploaded_file = st.file_uploader(
        "Upload ECG Image",
        type=["png", "jpg", "jpeg"],
        key="image_uploader"
    )


# --------------------------------------------------
# INPUT PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    try:

        if input_mode == "CSV File":

            signal, ecg_df = prepare_ecg_csv(
                uploaded_file
            )

            st.success(
                "ECG CSV uploaded and validated successfully."
            )

        else:

            signal, ecg_df, original_image, debug_image = (
                prepare_ecg_image(uploaded_file)
            )

            st.success(
                "ECG image digitized successfully!"
            )

            st.markdown("### Uploaded ECG Image")

            st.image(
                original_image,
                channels="BGR",
                use_container_width=True
            )

            st.markdown("### Digitization Result")

            st.image(
                debug_image,
                channels="BGR",
                caption=(
                    "Green lines show the extracted ECG waveforms."
                ),
                use_container_width=True
            )

            st.download_button(
                "Download Reconstructed ECG CSV",
                data=ecg_df.to_csv(index=False),
                file_name="cardiofed_reconstructed_ecg.csv",
                mime="text/csv"
            )

        # --------------------------------------------------
        # ECG WAVEFORM
        # --------------------------------------------------

        st.markdown("### Reconstructed ECG Waveforms")

        fig = plot_ecg(ecg_df[LEADS])

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # EXPLAINABILITY SETTINGS
        # --------------------------------------------------

        st.markdown("### Explainability Settings")

        explain_class = st.selectbox(
            "Select class for Integrated Gradients explanation",
            CLASSES,
            index=1
        )

        generate_explanation = st.checkbox(
            "Generate Integrated Gradients explanation",
            value=True
        )

        # --------------------------------------------------
        # ANALYSIS
        # --------------------------------------------------

        if st.button(
            "Analyze ECG",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing ECG..."):

                model = load_model()
                thresholds = load_thresholds()

                probabilities, original, final = predict_ecg(
                    signal,
                    model,
                    thresholds
                )

            st.markdown("## Prediction Results")

            # Input quality check: the models output confident abnormal
            # predictions (e.g. STTC/HYP) for flat-line input, so flat or
            # missing leads make the prediction unreliable.
            flat_leads = flat_lead_names(signal)

            if flat_leads:
                st.warning(
                    "Flat or missing lead(s) detected: "
                    f"{', '.join(flat_leads)}. The models were trained on "
                    "complete 12-lead ECGs and are unreliable when leads are "
                    "missing; a flat-line input is itself classified as "
                    "abnormal. Interpret these predictions with caution."
                )

            # --------------------------------------------------
            # INTEGRATED GRADIENTS
            # --------------------------------------------------

            if generate_explanation:

                st.markdown("## Explainable AI Analysis")

                target_index = CLASSES.index(
                    explain_class
                )

                with st.spinner(
                    f"Calculating Integrated Gradients for {explain_class}..."
                ):

                    attribution, completeness_error = integrated_gradients(
                        model,
                        signal.to(DEVICE),
                        target_class=target_index,
                        steps=64
                    )

                attribution_fig = plot_integrated_gradients(
                    signal.to(DEVICE),
                    attribution,
                    explain_class,
                    completeness_error=completeness_error
                )

                st.pyplot(
                    attribution_fig,
                    use_container_width=True
                )

                plt.close(attribution_fig)

                st.caption(
                    f"Integrated Gradients estimates how input signal "
                    f"features contribute to the selected model output "
                    f"(64-step midpoint Riemann sum, completeness error: {completeness_error:.2%}). "
                    f"Red and blue regions represent positive and negative "
                    f"attribution values respectively. This is a model "
                    f"explanation, not clinical evidence."
                )

            # --------------------------------------------------
            # PREDICTION TABLE
            # --------------------------------------------------

            results = pd.DataFrame({
                "Class": CLASSES,
                "Probability (%)": (probabilities * 100).round(2),
                "Threshold (%)": (thresholds * 100).round(2),
                "Raw Prediction (Thresholds)": [
                    "Positive (1)" if o == 1 else "Negative (0)"
                    for o in original
                ],
                "Final Prediction (NORM Rule)": [
                    "Positive (1)" if f == 1 else "Negative (0)"
                    for f in final
                ],
                "Status": [
                    "NORM Suppressed" if o == 1 and f == 0 else ("Positive" if f == 1 else "Negative")
                    for o, f in zip(original, final)
                ]
            })

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------
            # PROBABILITY CHART
            # --------------------------------------------------

            st.markdown("### Class Probabilities")

            chart_df = pd.DataFrame({
                "Class": CLASSES,
                "Probability (%)": probabilities * 100
            }).set_index("Class")

            st.bar_chart(chart_df)

            # --------------------------------------------------
            # CLASSIFICATION SUMMARY
            # --------------------------------------------------

            raw_predicted_classes = [
                CLASSES[i]
                for i in range(len(CLASSES))
                if original[i] == 1
            ]

            final_predicted_classes = [
                CLASSES[i]
                for i in range(len(CLASSES))
                if final[i] == 1
            ]

            st.markdown("### Classification Summary")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**1. Raw Threshold Predictions** *(Independent decision thresholds)*")
                if raw_predicted_classes:
                    st.write(", ".join(raw_predicted_classes))
                else:
                    st.write("No class crossed its threshold.")

            with col2:
                st.markdown("**2. Final Predictions** *(After post-hoc NORM consistency rule)*")
                if final_predicted_classes:
                    st.write(", ".join(final_predicted_classes))
                else:
                    st.write("No class predicted positive.")

            if not np.array_equal(original, final):
                st.warning(
                    "⚠️ **Post-Hoc NORM Consistency Rule Applied:**\n\n"
                    "The model's raw independent thresholds predicted both **NORM** and abnormal class(es) (**"
                    + ", ".join([CLASSES[i] for i in range(1, 5) if original[i] == 1])
                    + "**). Because CardioFed defines NORM as mutually exclusive with abnormal conditions in its label policy, "
                    "NORM was suppressed (forced to 0) in the final prediction.\n\n"
                    "- **Effect on this record:** NORM was flipped from Positive (1) to Negative (0).\n"
                    "- **Fold-10 Test Benchmark Impact:** Resolves all 72 contradictory predictions in the test fold, "
                    "improving **Subset Accuracy from 57.63% to 58.69%** (+23 exact matches) while slightly lowering **Macro-F1 from 0.7045 to 0.7031** "
                    "(due to suppressing NORM on records with a false-positive abnormal prediction)."
                )

            with st.expander("ℹ️ Model Benchmark Performance: Raw Thresholds vs. NORM Consistency Rule"):
                benchmark_df = pd.DataFrame({
                    "Evaluation Metric": [
                        "Macro-AUROC",
                        "Macro-F1",
                        "Subset Accuracy",
                        "Hamming Loss",
                        "Conflicting Predictions (NORM + Abnormal)"
                    ],
                    "Raw Thresholds": [
                        "0.9061",
                        "0.7045",
                        "57.63% (1,243 / 2,157)",
                        "0.1366",
                        "72 records"
                    ],
                    "With NORM Consistency Rule": [
                        "0.9061",
                        "0.7031",
                        "58.69% (1,266 / 2,157)",
                        "0.1368",
                        "0 records (resolved)"
                    ],
                    "Effect of Rule": [
                        "Unchanged (probabilities unchanged)",
                        "-0.0014 (penalizes NORM recall when abnormal is FP)",
                        "+1.06% (+23 newly exact matches)",
                        "+0.0002 (slight increase)",
                        "Enforces clinical mutually-exclusive NORM policy"
                    ]
                })
                st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Metrics evaluated on the complete fold-10 test set (2,157 records, 1,876 patients) "
                    "using the primary Multi-Scale Residual CNN (results/scheduler_experiment/checkpoints/multiscale_best.pt)."
                )

            st.caption(
                "Probabilities are model outputs, not calibrated "
                "clinical certainty. This research prototype is "
                "not a diagnostic medical device and must not be "
                "used for clinical decisions."
            )

            # --------------------------------------------------
            # DOWNLOAD REPORT
            # --------------------------------------------------

            report = {
                "model": "Multi-Scale Residual CNN",
                "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
                "input_type": input_mode,
                "classes": CLASSES,

                "probabilities": {
                    name: float(probabilities[i])
                    for i, name in enumerate(CLASSES)
                },

                "thresholds": {
                    name: float(thresholds[i])
                    for i, name in enumerate(CLASSES)
                },

                "raw_predictions": {
                    name: int(original[i])
                    for i, name in enumerate(CLASSES)
                },

                "final_predictions": {
                    name: int(final[i])
                    for i, name in enumerate(CLASSES)
                },

                "norm_consistency_rule_applied": bool(not np.array_equal(original, final)),
                "rule_description": "Post-hoc rule suppressing NORM when any abnormal class is predicted positive."
            }

            st.download_button(
                "Download Prediction Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="cardiofed_prediction.json",
                mime="application/json"
            )

    except Exception as error:

        st.error(
            f"Unable to process ECG: {error}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CardioFed | Research Prototype | "
    "PTB-XL Dataset | 12-Lead ECG"
)
