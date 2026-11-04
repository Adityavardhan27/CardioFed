
"""Generate a synthetic 12-lead ECG paper image from ECG_63.csv."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "results" / "examples" / "ECG_63.csv"
OUTPUT_DIR = ROOT / "results" / "examples"
OUTPUT_PATH = OUTPUT_DIR / "ECG_63_synthetic.png"

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]

DISPLAY_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]

# Image layout
PIXELS_PER_MM = 10
PAPER_SPEED = 25  # mm/second
SECONDS = 10

WIDTH = SECONDS * PAPER_SPEED * PIXELS_PER_MM
ROW_HEIGHT = 110
TOP_MARGIN = 90
BOTTOM_MARGIN = 50
LEFT_MARGIN = 100
RIGHT_MARGIN = 40

HEIGHT = TOP_MARGIN + 12 * ROW_HEIGHT + BOTTOM_MARGIN


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"ECG CSV not found: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    missing = [lead for lead in LEADS if lead not in df.columns]

    if missing:
        raise ValueError(f"Missing leads: {missing}")

    if len(df) != 1000:
        raise ValueError(
            f"Expected 1000 samples, received {len(df)}"
        )

    signals = df[LEADS].to_numpy(dtype=np.float32)

    if not np.isfinite(signals).all():
        raise ValueError("ECG contains invalid values.")

    # White paper background.
    image = np.full(
        (HEIGHT, WIDTH + LEFT_MARGIN + RIGHT_MARGIN, 3),
        255,
        dtype=np.uint8
    )

    # Draw ECG paper grid.
    small_grid = PIXELS_PER_MM
    large_grid = 5 * PIXELS_PER_MM

    # Small grid lines.
    for x in range(LEFT_MARGIN, LEFT_MARGIN + WIDTH + 1, small_grid):
        cv2.line(
            image,
            (x, TOP_MARGIN - 25),
            (x, HEIGHT - BOTTOM_MARGIN),
            (245, 220, 220),
            1
        )

    for y in range(TOP_MARGIN - 25, HEIGHT - BOTTOM_MARGIN + 1, small_grid):
        cv2.line(
            image,
            (LEFT_MARGIN, y),
            (LEFT_MARGIN + WIDTH, y),
            (245, 220, 220),
            1
        )

    # Bold 5 mm grid lines.
    for x in range(LEFT_MARGIN, LEFT_MARGIN + WIDTH + 1, large_grid):
        cv2.line(
            image,
            (x, TOP_MARGIN - 25),
            (x, HEIGHT - BOTTOM_MARGIN),
            (225, 175, 175),
            1
        )

    for y in range(TOP_MARGIN - 25, HEIGHT - BOTTOM_MARGIN + 1, large_grid):
        cv2.line(
            image,
            (LEFT_MARGIN, y),
            (LEFT_MARGIN + WIDTH, y),
            (225, 175, 175),
            1
        )

    # Header.
    cv2.putText(
        image,
        "CARDIOFED - SYNTHETIC 12 LEAD ECG",
        (LEFT_MARGIN, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        "ECG ID: 63 | Paper speed: 25 mm/s | Synthetic display scale",
        (LEFT_MARGIN, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (50, 50, 50),
        1,
        cv2.LINE_AA
    )

    # Each lead is displayed as a complete 10-second trace.
    # Vertical scale is for visualization, not physical mV calibration.
    time = np.arange(len(df)) / 100.0

    x_positions = (
        LEFT_MARGIN + time * PAPER_SPEED * PIXELS_PER_MM
    ).astype(np.int32)

    vertical_scale = 12.0  # pixels per normalized signal unit

    for i, lead in enumerate(LEADS):

        row_top = TOP_MARGIN + i * ROW_HEIGHT
        center_y = row_top + ROW_HEIGHT // 2

        # Lead label.
        cv2.putText(
            image,
            DISPLAY_NAMES[i],
            (20, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA
        )

        y_positions = (
            center_y - signals[:, i] * vertical_scale
        ).astype(np.int32)

        points = np.column_stack(
            (x_positions, y_positions)
        ).reshape(-1, 1, 2)

        cv2.polylines(
            image,
            [points],
            isClosed=False,
            color=(20, 20, 150),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    # Footer.
    cv2.putText(
        image,
        "RESEARCH TEST IMAGE - NOT A CLINICAL ECG REPORT",
        (LEFT_MARGIN, HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (50, 50, 50),
        1,
        cv2.LINE_AA
    )

    success = cv2.imwrite(str(OUTPUT_PATH), image)

    if not success:
        raise RuntimeError("Could not save synthetic ECG image.")

    print("\nSynthetic ECG image generated successfully!")
    print("ECG ID: 63")
    print("Leads:", len(LEADS))
    print("Samples per lead:", len(df))
    print("Duration: 10 seconds")
    print("Image dimensions:", image.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
