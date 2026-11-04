
"""Continuity-aware digitization of a synthetic 12-lead ECG image."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT / "results" / "examples" / "ECG_63_synthetic.png"
OUTPUT_DIR = ROOT / "results" / "examples"
OUTPUT_CSV = OUTPUT_DIR / "ECG_63_reconstructed.csv"
OUTPUT_DEBUG = OUTPUT_DIR / "ECG_63_digitization_debug.png"

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

# These parameters match our synthetic ECG generator.
PIXELS_PER_MM = 10
PAPER_SPEED = 25
DURATION = 10
SAMPLE_COUNT = 1000

LEFT_MARGIN = 100
TOP_MARGIN = 90
ROW_HEIGHT = 110
VERTICAL_SCALE = 12.0

# Tracking parameters.
MAX_VERTICAL_JUMP = 35
SMOOTHNESS_WEIGHT = 0.15
PIXEL_DISTANCE_WEIGHT = 2.0


def create_waveform_mask(roi):
    """Identify dark red waveform pixels."""

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 100, 40])
    upper_red_1 = np.array([12, 255, 220])

    lower_red_2 = np.array([165, 100, 40])
    upper_red_2 = np.array([180, 255, 220])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    return cv2.bitwise_or(mask_1, mask_2)


def track_waveform(mask):
    """
    Track a continuous path through waveform pixels using dynamic programming.

    The algorithm balances:
    1. Distance from detected waveform pixels.
    2. Vertical movement between neighbouring columns.
    """

    height, width = mask.shape

    # Waveform pixels become zero; other pixels become nonzero.
    # Distance transform measures distance to the nearest waveform pixel.
    inverted_mask = cv2.bitwise_not(mask)

    distance_map = cv2.distanceTransform(
        inverted_mask,
        cv2.DIST_L2,
        3
    )

    emission_cost = (
        distance_map * PIXEL_DISTANCE_WEIGHT
    ).astype(np.float32)

    # Dynamic programming storage.
    cumulative_cost = np.full(
        (width, height),
        np.inf,
        dtype=np.float32
    )

    backtrack = np.full(
        (width, height),
        -1,
        dtype=np.int16
    )

    cumulative_cost[0] = emission_cost[:, 0]

    # Track the best path from left to right.
    for x in range(1, width):

        previous = cumulative_cost[x - 1]

        for current_y in range(height):

            best_cost = np.inf
            best_previous_y = -1

            low = max(0, current_y - MAX_VERTICAL_JUMP)
            high = min(height, current_y + MAX_VERTICAL_JUMP + 1)

            previous_positions = np.arange(low, high)

            movement_cost = (
                np.abs(previous_positions - current_y)
                * SMOOTHNESS_WEIGHT
            )

            possible_costs = (
                previous[low:high] + movement_cost
            )

            best_index = int(np.argmin(possible_costs))
            best_cost = possible_costs[best_index]
            best_previous_y = previous_positions[best_index]

            cumulative_cost[x, current_y] = (
                best_cost + emission_cost[current_y, x]
            )

            backtrack[x, current_y] = best_previous_y

    # Find the lowest-cost ending position.
    path = np.zeros(width, dtype=np.int32)

    path[-1] = int(np.argmin(cumulative_cost[-1]))

    # Recover the optimal path backwards.
    for x in range(width - 1, 0, -1):

        previous_y = backtrack[x, path[x]]

        if previous_y < 0:
            raise RuntimeError("Waveform tracking failed.")

        path[x - 1] = previous_y

    return path


def extract_lead(image, lead_index):

    row_top = TOP_MARGIN + lead_index * ROW_HEIGHT
    center_y = row_top + ROW_HEIGHT // 2

    x_start = LEFT_MARGIN
    x_end = (
        LEFT_MARGIN
        + DURATION * PAPER_SPEED * PIXELS_PER_MM
    )

    y_start = row_top + 5
    y_end = row_top + ROW_HEIGHT - 5

    roi = image[y_start:y_end, x_start:x_end]

    mask = create_waveform_mask(roi)

    detected_pixels = np.count_nonzero(mask)

    if detected_pixels < mask.shape[1] * 0.5:
        raise RuntimeError(
            f"Insufficient waveform pixels for {DISPLAY_NAMES[lead_index]}."
        )

    # Follow the waveform instead of taking a median per column.
    tracked_y = track_waveform(mask)

    # Convert local pixel coordinates to absolute image coordinates.
    absolute_y = tracked_y + y_start

    # Convert pixel positions into normalized signal values.
    reconstructed_signal = (
        center_y - absolute_y
    ) / VERTICAL_SCALE

    # Resample to exactly 1000 points.
    pixel_positions = np.arange(len(reconstructed_signal))

    sample_positions = np.linspace(
        0,
        len(reconstructed_signal) - 1,
        SAMPLE_COUNT
    )

    reconstructed_signal = np.interp(
        sample_positions,
        pixel_positions,
        reconstructed_signal
    )

    return reconstructed_signal.astype(np.float32), mask, tracked_y


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic ECG image not found: {IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError("Unable to read ECG image.")

    print("Image dimensions:", image.shape)
    print("Extraction method: continuity-aware dynamic programming")

    reconstructed = {}
    debug_image = image.copy()

    for i, lead in enumerate(LEADS):

        signal, mask, tracked_y = extract_lead(image, i)

        reconstructed[lead] = signal

        row_top = TOP_MARGIN + i * ROW_HEIGHT
        row_bottom = row_top + ROW_HEIGHT

        x_start = LEFT_MARGIN

        # Draw the detected tracking path in green.
        points = np.column_stack((
            np.arange(len(tracked_y)) + x_start,
            tracked_y + row_top + 5
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
            (x_start, row_top),
            (
                x_start + mask.shape[1],
                row_bottom
            ),
            (0, 180, 0),
            1
        )

        print(
            f"{DISPLAY_NAMES[i]}: "
            f"{len(signal)} samples | "
            f"min={signal.min():.3f}, "
            f"max={signal.max():.3f}"
        )

    result = pd.DataFrame(reconstructed)

    result.insert(
        0,
        "time_seconds",
        np.arange(SAMPLE_COUNT) / 100.0
    )

    result.to_csv(OUTPUT_CSV, index=False)

    cv2.imwrite(
        str(OUTPUT_DEBUG),
        debug_image
    )

    print("\nDIGITIZATION COMPLETED!")
    print("Reconstructed shape:", result.shape)
    print("CSV saved:", OUTPUT_CSV)
    print("Debug image saved:", OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
