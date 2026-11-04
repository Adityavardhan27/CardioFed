
"""Compare CardioFed predictions from original CSV, reconstructed CSV,
and a fresh ECG image digitization using the same model and thresholds.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
ORIGINAL_CSV = ROOT / "results/examples/ECG_63.csv"
RECONSTRUCTED_CSV = ROOT / "results/examples/ECG_63_reconstructed_v4.csv"
IMAGE_PATH = ROOT / "results/examples/ECG_63_synthetic_v2.png"

CHECKPOINT_PATH = ROOT / "results/checkpoints/multiscale_best.pt"
EVALUATION_PATH = ROOT / "results/metrics/multiscale_final_evaluation.json"

OUTPUT_DIR = ROOT / "results/examples"
FRESH_IMAGE_CSV = OUTPUT_DIR / "ECG_63_reconstructed_comparison.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

DISPLAY_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Parameters copied from the working Streamlit app and V4 digitizer.
LEFT_MARGIN = 100
TOP_MARGIN = 90
ROW_HEIGHT = 190
PAPER_SPEED = 25
PIXELS_PER_MM = 10
DURATION = 10
SAMPLE_COUNT = 1000
VERTICAL_SCALE = 12.0


def load_signal_csv(path):
    """Load a 12-lead CSV and return [12, 1000] float32 data."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing = [lead for lead in LEADS if lead not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing leads: {missing}")

    if len(df) != SAMPLE_COUNT:
        raise ValueError(
            f"{path.name}: expected 1000 rows, received {len(df)}"
        )

    values = df[LEADS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains invalid values.")

    return values.T.copy()


def create_waveform_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 100, 40])
    upper_red_1 = np.array([12, 255, 220])

    lower_red_2 = np.array([165, 100, 40])
    upper_red_2 = np.array([180, 255, 220])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    return cv2.bitwise_or(mask_1, mask_2)


def digitize_image(image_path):
    """Reproduce the exact column-median extraction used in app.py."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    expected_width = (
        LEFT_MARGIN + DURATION * PAPER_SPEED * PIXELS_PER_MM + 40
    )
    expected_height = TOP_MARGIN + 12 * ROW_HEIGHT + 50

    height, width = image.shape[:2]

    if width < expected_width or height < expected_height:
        raise ValueError(
            f"Unexpected image dimensions: {width}x{height}; "
            f"expected at least {expected_width}x{expected_height}"
        )

    x_start = LEFT_MARGIN
    x_end = LEFT_MARGIN + DURATION * PAPER_SPEED * PIXELS_PER_MM

    reconstructed = {}

    for i, lead in enumerate(LEADS):
        row_top = TOP_MARGIN + i * ROW_HEIGHT
        center_y = row_top + ROW_HEIGHT // 2

        y_start = row_top + 5
        y_end = row_top + ROW_HEIGHT - 5

        roi = image[y_start:y_end, x_start:x_end]
        mask = create_waveform_mask(roi)

        detected_columns = []
        median_positions = []

        for x in range(mask.shape[1]):
            ys = np.where(mask[:, x] > 0)[0]

            if len(ys) > 0:
                detected_columns.append(x)
                median_positions.append(float(np.median(ys)))

        minimum_detected = int(mask.shape[1] * 0.80)

        if len(detected_columns) < minimum_detected:
            raise RuntimeError(
                f"Insufficient waveform detection for {DISPLAY_LEADS[i]}: "
                f"{len(detected_columns)}/{mask.shape[1]} columns"
            )

        pixel_positions = np.interp(
            np.arange(mask.shape[1]),
            detected_columns,
            median_positions
        )

        absolute_y = pixel_positions + y_start

        signal_values = (
            center_y - absolute_y
        ) / VERTICAL_SCALE

        reconstructed_signal = np.interp(
            np.linspace(
                0,
                len(signal_values) - 1,
                SAMPLE_COUNT
            ),
            np.arange(len(signal_values)),
            signal_values
        )

        reconstructed[lead] = reconstructed_signal.astype(np.float32)

    df = pd.DataFrame(reconstructed)
    df.insert(0, "time_seconds", np.arange(SAMPLE_COUNT) / 100.0)
    df.to_csv(FRESH_IMAGE_CSV, index=False)

    return df[LEADS].to_numpy(dtype=np.float32).T.copy()


def load_model_and_thresholds():
    from src.model import MultiScaleResidualECGNet

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

    with open(EVALUATION_PATH, "r") as file:
        evaluation = json.load(file)

    thresholds = np.array(
        [evaluation["thresholds"][name] for name in CLASSES],
        dtype=np.float32
    )

    return model, thresholds


def calculate_signal_metrics(reference, reconstructed):
    """Compare two signals with shape [12, 1000]."""

    rows = []

    for i, lead in enumerate(DISPLAY_LEADS):
        original = reference[i]
        recovered = reconstructed[i]

        mae = np.mean(np.abs(original - recovered))
        rmse = np.sqrt(np.mean((original - recovered) ** 2))
        correlation = np.corrcoef(original, recovered)[0, 1]

        rows.append({
            "Lead": lead,
            "MAE": mae,
            "RMSE": rmse,
            "Correlation": correlation
        })

    return pd.DataFrame(rows)


def predict(model, thresholds, signal):
    tensor = torch.from_numpy(
        signal.astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    raw = (probabilities >= thresholds).astype(int)

    final = raw.copy()
    abnormal_indices = [
        CLASSES.index(name)
        for name in ["MI", "STTC", "CD", "HYP"]
    ]

    if final[abnormal_indices].any():
        final[CLASSES.index("NORM")] = 0

    return probabilities, raw, final


def main():
    print("=" * 75)
    print("CARDIOFED: THREE-WAY INPUT COMPARISON")
    print("=" * 75)
    print("Device:", DEVICE)

    original = load_signal_csv(ORIGINAL_CSV)
    reconstructed_v4 = load_signal_csv(RECONSTRUCTED_CSV)
    reconstructed_image = digitize_image(IMAGE_PATH)

    print("\nFresh image reconstruction saved to:")
    print(FRESH_IMAGE_CSV)

    model, thresholds = load_model_and_thresholds()

    inputs = {
        "Original CSV": original,
        "Saved V4 CSV": reconstructed_v4,
        "Fresh image digitization": reconstructed_image
    }

    # --------------------------------------------------
    # SIGNAL COMPARISON
    # --------------------------------------------------

    print("\n1. SIGNAL RECONSTRUCTION COMPARISON")
    print("-" * 75)

    for name, signal in [
        ("Saved V4 CSV", reconstructed_v4),
        ("Fresh image digitization", reconstructed_image)
    ]:
        metrics = calculate_signal_metrics(original, signal)

        print(f"\nOriginal CSV vs {name}")
        print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

        print(
            f"MEAN: MAE={metrics['MAE'].mean():.5f} | "
            f"RMSE={metrics['RMSE'].mean():.5f} | "
            f"Correlation={metrics['Correlation'].mean():.5f}"
        )

    # Compare the two reconstructed versions directly.
    difference = np.abs(reconstructed_v4 - reconstructed_image)

    print("\nSaved V4 CSV vs fresh image digitization")
    print("Mean absolute difference:", float(difference.mean()))
    print("Maximum absolute difference:", float(difference.max()))

    # --------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------

    print("\n\n2. MODEL PREDICTION COMPARISON")
    print("-" * 75)

    prediction_results = {}

    for name, signal in inputs.items():
        probabilities, raw, final = predict(
            model,
            thresholds,
            signal
        )

        prediction_results[name] = probabilities

        print(f"\n{name}")
        print(
            f"{'Class':<8} {'Probability':>13} "
            f"{'Threshold':>12} {'Raw':>8} {'Final':>8}"
        )

        for i, class_name in enumerate(CLASSES):
            print(
                f"{class_name:<8} "
                f"{probabilities[i] * 100:>12.2f}% "
                f"{thresholds[i] * 100:>11.2f}% "
                f"{raw[i]:>8} "
                f"{final[i]:>8}"
            )

        predicted = [
            CLASSES[i]
            for i in range(len(CLASSES))
            if final[i] == 1
        ]

        print("Final predicted classes:", predicted or "None")

    # --------------------------------------------------
    # PROBABILITY DIFFERENCES
    # --------------------------------------------------

    print("\n\n3. PROBABILITY DIFFERENCES")
    print("-" * 75)

    original_probabilities = prediction_results["Original CSV"]

    for name in ["Saved V4 CSV", "Fresh image digitization"]:
        probabilities = prediction_results[name]

        print(f"\nOriginal CSV vs {name}")

        for i, class_name in enumerate(CLASSES):
            difference = (
                probabilities[i] - original_probabilities[i]
            ) * 100

            print(
                f"{class_name:<8}: "
                f"{difference:+.2f} percentage points"
            )

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()
