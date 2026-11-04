
"""Experimental V5: Exact horizontal sample mapping.

Keeps V4's median-based extraction but maps image pixels
to the original 1000 ECG sample positions exactly.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT / "results/examples/ECG_63_synthetic_v2.png"
OUTPUT_DIR = ROOT / "results/examples"

OUTPUT_CSV = OUTPUT_DIR / "ECG_63_reconstructed_v5.csv"
OUTPUT_DEBUG = OUTPUT_DIR / "ECG_63_digitization_v5_debug.png"

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

DISPLAY_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

LEFT_MARGIN = 100
TOP_MARGIN = 90
ROW_HEIGHT = 190

PAPER_SPEED = 25
PIXELS_PER_MM = 10

SAMPLE_COUNT = 1000
VERTICAL_SCALE = 12.0


def create_waveform_mask(roi):
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
    width = SAMPLE_COUNT * PAPER_SPEED * PIXELS_PER_MM / 100

    x_end = x_start + int(width)

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

    if len(detected_columns) < mask.shape[1] * 0.80:
        raise RuntimeError(
            f"Insufficient detection for {DISPLAY_NAMES[lead_index]}"
        )

    pixel_positions = np.interp(
        np.arange(mask.shape[1]),
        detected_columns,
        median_positions
    )

    # Exact original sample locations: 2.5 pixels per sample.
    original_sample_x = (
        np.arange(SAMPLE_COUNT)
        * PAPER_SPEED
        * PIXELS_PER_MM
        / 100
    )

    reconstructed_y = np.interp(
        original_sample_x,
        np.arange(mask.shape[1]),
        pixel_positions
    )

    absolute_y = reconstructed_y + y_start

    signal = (
        center_y - absolute_y
    ) / VERTICAL_SCALE

    return signal.astype(np.float32), pixel_positions, y_start


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    reconstructed = {}
    debug_image = image.copy()

    for i, lead in enumerate(LEADS):

        signal, positions, y_start = extract_lead(image, i)

        reconstructed[lead] = signal

        points = np.column_stack((
            np.arange(len(positions)) + LEFT_MARGIN,
            positions + y_start
        )).astype(np.int32)

        cv2.polylines(
            debug_image,
            [points.reshape(-1, 1, 2)],
            False,
            (0, 180, 0),
            2
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

    print("\nV5 EXPERIMENT COMPLETED")
    print("Reconstructed shape:", result.shape)
    print("CSV:", OUTPUT_CSV)
    print("Debug image:", OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
