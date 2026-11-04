
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import MultiScaleResidualECGNet
from src.preprocessing import CLASSES


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = ROOT / "results/checkpoints/multiscale_best.pt"
EVALUATION_PATH = ROOT / "results/metrics/multiscale_final_evaluation.json"

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

    df = pd.read_csv(uploaded_file)

    missing = [
        lead for lead in LEADS
        if lead not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ECG leads: {', '.join(missing)}"
        )

    if len(df) != 1000:
        raise ValueError(
            f"Expected 1000 samples, but received {len(df)}."
        )

    signal_array = df[LEADS].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(signal_array).all():
        raise ValueError(
            "The ECG contains missing or invalid signal values."
        )

    # CSV must already be filtered and normalized.
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
    steps=32
):

    model.eval()

    baseline = torch.zeros_like(signal)
    difference = signal - baseline

    total_gradients = torch.zeros_like(signal)

    for alpha in torch.linspace(
        0,
        1,
        steps,
        device=signal.device
    ):

        interpolated = (
            baseline + alpha * difference
        ).detach().requires_grad_(True)

        output = model(interpolated)

        score = output[0, target_class]

        gradients = torch.autograd.grad(
            score,
            interpolated
        )[0]

        total_gradients += gradients.detach()

    average_gradients = (
        total_gradients / steps
    )

    attributions = difference * average_gradients

    return attributions.detach()


def plot_integrated_gradients(
    signal,
    attribution,
    target_name
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

    for lead in range(12):

        axes[0].plot(
            time,
            signal_np[lead] + lead * 5,
            linewidth=0.7,
            label=DISPLAY_LEADS[lead]
        )

    axes[0].set_title("12-Lead ECG Waveforms")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Normalized signal + lead offset")
    axes[0].grid(alpha=0.2)

    im = axes[1].imshow(
        attribution_np,
        aspect="auto",
        cmap="seismic",
        interpolation="nearest",
        extent=[0, 10, 12, 0]
    )

    axes[1].set_title(
        f"Integrated Gradients Attribution for {target_name}"
    )

    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("ECG lead")

    axes[1].set_yticks(
        np.arange(0.5, 12.5, 1)
    )

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
    "Robust and Explainable Multi-Label ECG Classification"
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
        "Upload a CSV containing 1,000 samples and all 12 leads. "
        "The signals must already be filtered and normalized "
        "using the CardioFed preprocessing pipeline."
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

                    attribution = integrated_gradients(
                        model,
                        signal.to(DEVICE),
                        target_class=target_index,
                        steps=32
                    )

                attribution_fig = plot_integrated_gradients(
                    signal.to(DEVICE),
                    attribution,
                    explain_class
                )

                st.pyplot(
                    attribution_fig,
                    use_container_width=True
                )

                plt.close(attribution_fig)

                st.caption(
                    "Integrated Gradients estimates how input signal "
                    "features contribute to the selected model output. "
                    "Red and blue regions represent positive and negative "
                    "attribution values respectively. This is a model "
                    "explanation, not clinical evidence."
                )

            # --------------------------------------------------
            # PREDICTION TABLE
            # --------------------------------------------------

            results = pd.DataFrame({
                "Class": CLASSES,
                "Probability": probabilities,
                "Threshold": thresholds,
                "Raw Prediction": original,
                "Final Prediction": final
            })

            results["Probability"] = (
                results["Probability"] * 100
            ).round(2)

            results["Threshold"] = (
                results["Threshold"] * 100
            ).round(2)

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
            # FINAL CLASSIFICATION
            # --------------------------------------------------

            predicted_classes = [
                CLASSES[i]
                for i in range(len(CLASSES))
                if final[i] == 1
            ]

            st.markdown("### Final Classification")

            if predicted_classes:

                st.write(
                    "Predicted positive classes:",
                    ", ".join(predicted_classes)
                )

            else:

                st.write(
                    "No class crossed its classification threshold."
                )

            if not np.array_equal(original, final):

                st.info(
                    "The NORM consistency rule was applied: "
                    "NORM was suppressed because at least one "
                    "abnormal class was predicted."
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
                }
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
