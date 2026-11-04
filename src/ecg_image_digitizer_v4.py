
"""Digitize the corrected synthetic 12-lead ECG image using
column-wise median waveform extraction.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT / "results" / "examples" / "ECG_63_synthetic_v2.png"
OUTPUT_DIR = ROOT / "results" / "examples"

OUTPUT_CSV = OUTPUT_DIR / "ECG_63_reconstructed_v4.csv"
OUTPUT_DEBUG = OUTPUT_DIR / "ECG_63_digitization_v4_debug.png"

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]

DISPLAY_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

# Must match generate_synthetic_ecg_v2.py
LEFT_MARGIN = 100
TOP_MARGIN = 90
ROW_HEIGHT = 190
PAPER_SPEED = 25
PIXELS_PER_MM = 10
DURATION = 10

SAMPLE_COUNT = 1000
VERTICAL_SCALE = 12.0


def create_waveform_mask(roi):
    """Detect dark red ECG waveform pixels."""

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 100, 40])
    upper_red_1 = np.array([12, 255, 220])

    lower_red_2 = np.array([165, 100, 40])
    upper_red_2 = np.array([180, 255, 220])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    return cv2.bitwise_or(mask_1, mask_2)


def extract_lead(image, lead_index):

    row_top = TOP_MARGIN + lead_index * ROW_HEIGHT
    center_y = row_top + ROW_HEIGHT // 2

    x_start = LEFT_MARGIN
    x_end = LEFT_MARGIN + DURATION * PAPER_SPEED * PIXELS_PER_MM

    y_start = row_top + 5
    y_end = row_top + ROW_HEIGHT - 5

    roi = image[y_start:y_end, x_start:x_end]

    mask = create_waveform_mask(roi)

    height, width = mask.shape

    detected_columns = []
    median_positions = []

    # Find the median waveform position in each column.
    for x in range(width):

        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) > 0:
            detected_columns.append(x)
            median_positions.append(float(np.median(ys)))

    if len(detected_columns) < width * 0.80:
        raise RuntimeError(
            f"Insufficient waveform detection for "
            f"{DISPLAY_NAMES[lead_index]}: "
            f"{len(detected_columns)}/{width} columns"
        )

    # Interpolate columns where no waveform pixel was detected.
    pixel_positions = np.interp(
        np.arange(width),
        detected_columns,
        median_positions
    )

    absolute_y = pixel_positions + y_start

    # Convert image coordinates to normalized signal values.
    reconstructed = (
        center_y - absolute_y
    ) / VERTICAL_SCALE

    # Resample to exactly 1000 points.
    reconstructed = np.interp(
        np.linspace(0, width - 1, SAMPLE_COUNT),
        np.arange(width),
        reconstructed
    )

    return reconstructed.astype(np.float32), mask, pixel_positions


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError("Could not read ECG image.")

    print("Image dimensions:", image.shape)
    print("Method: Column-wise median extraction")

    reconstructed = {}
    debug_image = image.copy()

    for i, lead in enumerate(LEADS):

        signal, mask, pixel_positions = extract_lead(image, i)

        reconstructed[lead] = signal

        row_top = TOP_MARGIN + i * ROW_HEIGHT
        row_bottom = row_top + ROW_HEIGHT

        # Draw extracted waveform in green.
        points = np.column_stack((
            np.arange(len(pixel_positions)) + LEFT_MARGIN,
            pixel_positions + row_top + 5
        )).astype(np.int32)

        cv2.polylines(
            debug_image,
            [points.reshape(-1, 1, 2)],
            False,
            (0, 180, 0),
            2
        )

        cv2.rectangle(
            debug_image,
            (LEFT_MARGIN, row_top),
            (
                LEFT_MARGIN + mask.shape[1],
                row_bottom
            ),
            (0, 180, 0),
            1
        )

        print(
            f"{DISPLAY_NAMES[i]}: "
            f"{len(signal)} samples | "
            f"min={signal.min():.3f} | "
            f"max={signal.max():.3f}"
        )

    result = pd.DataFrame(reconstructed)

    result.insert(
        0,
        "time_seconds",
        np.arange(SAMPLE_COUNT) / 100.0
    )

    result.to_csv(OUTPUT_CSV, index=False)

    if not cv2.imwrite(str(OUTPUT_DEBUG), debug_image):
        raise RuntimeError("Could not save debug image.")

    print("\nDIGITIZATION V4 COMPLETED!")
    print("Reconstructed shape:", result.shape)
    print("CSV:", OUTPUT_CSV)
    print("Debug image:", OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
