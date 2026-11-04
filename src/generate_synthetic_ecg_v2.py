
"""Generate a corrected synthetic 12-lead ECG image."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "results" / "examples" / "ECG_63.csv"
OUTPUT_DIR = ROOT / "results" / "examples"
OUTPUT_PATH = OUTPUT_DIR / "ECG_63_synthetic_v2.png"

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

DISPLAY_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

PIXELS_PER_MM = 10
PAPER_SPEED = 25
SECONDS = 10

WIDTH = SECONDS * PAPER_SPEED * PIXELS_PER_MM

ROW_HEIGHT = 190
TOP_MARGIN = 90
BOTTOM_MARGIN = 50
LEFT_MARGIN = 100
RIGHT_MARGIN = 40

VERTICAL_SCALE = 12.0

HEIGHT = TOP_MARGIN + 12 * ROW_HEIGHT + BOTTOM_MARGIN


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    if len(df) != 1000:
        raise ValueError("Expected exactly 1000 ECG samples.")

    signals = df[LEADS].to_numpy(dtype=np.float32)

    if not np.isfinite(signals).all():
        raise ValueError("ECG contains invalid values.")

    image = np.full(
        (HEIGHT, WIDTH + LEFT_MARGIN + RIGHT_MARGIN, 3),
        255,
        dtype=np.uint8
    )

    # ECG grid
    for x in range(LEFT_MARGIN, LEFT_MARGIN + WIDTH + 1, 10):
        cv2.line(
            image,
            (x, TOP_MARGIN - 25),
            (x, HEIGHT - BOTTOM_MARGIN),
            (245, 220, 220),
            1
        )

    for y in range(TOP_MARGIN - 25, HEIGHT - BOTTOM_MARGIN + 1, 10):
        cv2.line(
            image,
            (LEFT_MARGIN, y),
            (LEFT_MARGIN + WIDTH, y),
            (245, 220, 220),
            1
        )

    # Bold grid
    for x in range(LEFT_MARGIN, LEFT_MARGIN + WIDTH + 1, 50):
        cv2.line(
            image,
            (x, TOP_MARGIN - 25),
            (x, HEIGHT - BOTTOM_MARGIN),
            (225, 175, 175),
            1
        )

    for y in range(TOP_MARGIN - 25, HEIGHT - BOTTOM_MARGIN + 1, 50):
        cv2.line(
            image,
            (LEFT_MARGIN, y),
            (LEFT_MARGIN + WIDTH, y),
            (225, 175, 175),
            1
        )

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

    time = np.arange(len(df)) / 100.0

    x_positions = (
        LEFT_MARGIN + time * PAPER_SPEED * PIXELS_PER_MM
    ).astype(np.int32)

    for i, lead in enumerate(LEADS):

        row_top = TOP_MARGIN + i * ROW_HEIGHT
        center_y = row_top + ROW_HEIGHT // 2

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

        y_positions = np.rint(
            center_y - signals[:, i] * VERTICAL_SCALE
        ).astype(np.int32)

        # Verify the entire trace fits within its row.
        min_y = row_top + 8
        max_y = row_top + ROW_HEIGHT - 8

        if y_positions.min() < min_y or y_positions.max() > max_y:
            raise ValueError(
                f"{lead} exceeds its row boundaries. "
                f"Observed y range: {y_positions.min()}-{y_positions.max()}, "
                f"allowed: {min_y}-{max_y}"
            )

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

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError("Could not save image.")

    print("Corrected synthetic ECG generated.")
    print("Image dimensions:", image.shape)
    print("Row height:", ROW_HEIGHT)
    print("Vertical scale:", VERTICAL_SCALE)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
