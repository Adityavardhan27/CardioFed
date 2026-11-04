
"""Batch evaluation of ECG image digitizers V4 and V5.

Uses 15 recordings selected from PTB-XL test fold 10.
Generates synthetic ECG images and compares reconstruction
against the original processed signals.

Existing V2, V4 and V5 scripts are not modified.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
SELECTION_FILE = (
    ROOT / "results" / "examples"
    / "multirecord_test_selection_100.csv"
)

OUTPUT_DIR = ROOT / "results" / "multirecord_digitization_100"
IMAGE_DIR = OUTPUT_DIR / "images"
RECON_DIR = OUTPUT_DIR / "reconstructed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR.mkdir(parents=True, exist_ok=True)

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
SAMPLE_COUNT = 1000
VERTICAL_SCALE = 12.0

TOP_MARGIN = 90
BOTTOM_MARGIN = 50
LEFT_MARGIN = 100
RIGHT_MARGIN = 40

ORIGINAL_ROW_HEIGHT = 190
SECONDS = 10

WIDTH = SECONDS * PAPER_SPEED * PIXELS_PER_MM


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_dataset():

    print("\nLoading processed dataset...")

    X = np.load(DATA_DIR / "X.npy", mmap_mode="r")
    ids = np.load(DATA_DIR / "ecg_ids.npy")
    folds = np.load(DATA_DIR / "folds.npy")

    selection = pd.read_csv(SELECTION_FILE)

    if X.shape[1:] != (1000, 12):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if not np.all(folds[selection["dataset_index"]] == 10):
        raise ValueError("Selected recordings must all belong to fold 10.")

    for _, row in selection.iterrows():
        idx = int(row["dataset_index"])
        expected_id = int(row["ecg_id"])

        if int(ids[idx]) != expected_id:
            raise ValueError(
                f"ECG ID/index mismatch: {expected_id} at index {idx}"
            )

    print("Dataset shape:", X.shape)
    print("Selected recordings:", len(selection))

    return X, ids, selection


# ---------------------------------------------------------
# DETERMINE SAFE IMAGE ROW HEIGHT
# ---------------------------------------------------------

def determine_row_height(X, selection):

    max_abs = 0.0

    for idx in selection["dataset_index"]:
        signal = np.asarray(X[int(idx)], dtype=np.float32)
        max_abs = max(max_abs, float(np.max(np.abs(signal))))

    # Keep the original vertical scale of 12.
    # Add enough room for the largest waveform and 8 px padding.
    required_half_height = int(
        np.ceil(max_abs * VERTICAL_SCALE)
    ) + 8

    row_height = max(
        ORIGINAL_ROW_HEIGHT,
        2 * required_half_height
    )

    if row_height % 2 != 0:
        row_height += 1

    print("\nIMAGE GEOMETRY")
    print("Maximum absolute signal:", round(max_abs, 4))
    print("Vertical scale:", VERTICAL_SCALE)
    print("Selected row height:", row_height)

    return row_height


# ---------------------------------------------------------
# SYNTHETIC ECG IMAGE GENERATION
# ---------------------------------------------------------

def generate_ecg_image(signal, ecg_id, row_height):

    height = (
        TOP_MARGIN
        + 12 * row_height
        + BOTTOM_MARGIN
    )

    image = np.full(
        (
            height,
            WIDTH + LEFT_MARGIN + RIGHT_MARGIN,
            3
        ),
        255,
        dtype=np.uint8
    )

    # Light ECG grid.
    for x in range(
        LEFT_MARGIN,
        LEFT_MARGIN + WIDTH + 1,
        10
    ):
        cv2.line(
            image,
            (x, TOP_MARGIN - 25),
            (x, height - BOTTOM_MARGIN),
            (245, 220, 220),
            1
        )

    for y in range(
        TOP_MARGIN - 25,
        height - BOTTOM_MARGIN + 1,
        10
    ):
        cv2.line(
            image,
            (LEFT_MARGIN, y),
            (LEFT_MARGIN + WIDTH, y),
            (245, 220, 220),
            1
        )

    # Bold grid.
    for x in range(
        LEFT_MARGIN,
        LEFT_MARGIN + WIDTH + 1,
        50
    ):
        cv2.line(
            image,
            (x, TOP_MARGIN - 25),
            (x, height - BOTTOM_MARGIN),
            (225, 175, 175),
            1
        )

    for y in range(
        TOP_MARGIN - 25,
        height - BOTTOM_MARGIN + 1,
        50
    ):
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

    header = (
        f"ECG ID: {ecg_id} | Paper speed: 25 mm/s "
        "| Synthetic display scale"
    )

    cv2.putText(
        image,
        header,
        (LEFT_MARGIN, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (50, 50, 50),
        1,
        cv2.LINE_AA
    )

    time = np.arange(SAMPLE_COUNT) / 100.0

    x_positions = (
        LEFT_MARGIN
        + time * PAPER_SPEED * PIXELS_PER_MM
    ).astype(np.int32)

    for i, lead in enumerate(LEADS):

        row_top = TOP_MARGIN + i * row_height
        center_y = row_top + row_height // 2

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
            center_y - signal[:, i] * VERTICAL_SCALE
        ).astype(np.int32)

        min_y = row_top + 8
        max_y = row_top + row_height - 8

        if (
            y_positions.min() < min_y
            or y_positions.max() > max_y
        ):
            raise ValueError(
                f"Waveform clipping detected in {lead}, "
                f"ECG {ecg_id}"
            )

        points = np.column_stack(
            (x_positions, y_positions)
        ).reshape(-1, 1, 2)

        cv2.polylines(
            image,
            [points],
            False,
            (20, 20, 150),
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        image,
        "RESEARCH TEST IMAGE - NOT A CLINICAL ECG REPORT",
        (LEFT_MARGIN, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (50, 50, 50),
        1,
        cv2.LINE_AA
    )

    return image


# ---------------------------------------------------------
# WAVEFORM DETECTION
# ---------------------------------------------------------

def create_waveform_mask(roi):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

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


# ---------------------------------------------------------
# COMMON MEDIAN-BASED EXTRACTION
# ---------------------------------------------------------

def extract_median_positions(image, lead_index, row_height):

    row_top = TOP_MARGIN + lead_index * row_height
    center_y = row_top + row_height // 2

    x_start = LEFT_MARGIN
    x_end = LEFT_MARGIN + WIDTH

    y_start = row_top + 5
    y_end = row_top + row_height - 5

    roi = image[y_start:y_end, x_start:x_end]

    mask = create_waveform_mask(roi)

    height, width = mask.shape

    detected_columns = []
    median_positions = []

    for x in range(width):

        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) > 0:
            detected_columns.append(x)
            median_positions.append(float(np.median(ys)))

    if len(detected_columns) < width * 0.80:
        raise RuntimeError(
            f"Insufficient waveform detection for "
            f"{DISPLAY_NAMES[lead_index]}: "
            f"{len(detected_columns)}/{width}"
        )

    pixel_positions = np.interp(
        np.arange(width),
        detected_columns,
        median_positions
    )

    return (
        pixel_positions,
        y_start,
        center_y,
        mask
    )


# ---------------------------------------------------------
# V4 RECONSTRUCTION
# ---------------------------------------------------------

def digitize_v4(image, lead_index, row_height):

    positions, y_start, center_y, _ = (
        extract_median_positions(
            image,
            lead_index,
            row_height
        )
    )

    reconstructed = (
        center_y - (positions + y_start)
    ) / VERTICAL_SCALE

    # Original V4 horizontal resampling.
    reconstructed = np.interp(
        np.linspace(0, len(positions) - 1, SAMPLE_COUNT),
        np.arange(len(positions)),
        reconstructed
    )

    return reconstructed.astype(np.float32)


# ---------------------------------------------------------
# V5 RECONSTRUCTION
# ---------------------------------------------------------

def digitize_v5(image, lead_index, row_height):

    positions, y_start, center_y, _ = (
        extract_median_positions(
            image,
            lead_index,
            row_height
        )
    )

    # Exact original sample positions:
    # 2.5 image pixels per ECG sample.
    original_sample_x = (
        np.arange(SAMPLE_COUNT)
        * PAPER_SPEED
        * PIXELS_PER_MM
        / 100
    )

    reconstructed_y = np.interp(
        original_sample_x,
        np.arange(len(positions)),
        positions
    )

    absolute_y = reconstructed_y + y_start

    reconstructed = (
        center_y - absolute_y
    ) / VERTICAL_SCALE

    return reconstructed.astype(np.float32)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(original, reconstructed):

    difference = original - reconstructed

    mae = float(np.mean(np.abs(difference)))

    rmse = float(
        np.sqrt(np.mean(difference ** 2))
    )

    if (
        np.std(original) < 1e-10
        or np.std(reconstructed) < 1e-10
    ):
        correlation = np.nan
    else:
        correlation = float(
            np.corrcoef(original, reconstructed)[0, 1]
        )

    return mae, rmse, correlation


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    X, ids, selection = load_dataset()

    row_height = determine_row_height(
        X,
        selection
    )

    lead_results = []
    record_results = []

    print("\nSTARTING MULTI-RECORD DIGITIZATION")
    print("=" * 65)

    for number, (_, row) in enumerate(
        selection.iterrows(),
        start=1
    ):

        idx = int(row["dataset_index"])
        ecg_id = int(row["ecg_id"])

        original = np.asarray(
            X[idx],
            dtype=np.float32
        )

        if not np.isfinite(original).all():
            raise ValueError(
                f"Invalid signal values in ECG {ecg_id}"
            )

        print(
            f"\n[{number}/{len(selection)}] "
            f"Processing ECG ID {ecg_id}"
        )

        # Generate synthetic image.
        image = generate_ecg_image(
            original,
            ecg_id,
            row_height
        )

        image_path = (
            IMAGE_DIR / f"ECG_{ecg_id}_synthetic.png"
        )

        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(
                f"Could not save ECG image {ecg_id}"
            )

        v4_signals = []
        v5_signals = []

        v4_mae_values = []
        v5_mae_values = []

        v4_rmse_values = []
        v5_rmse_values = []

        v4_corr_values = []
        v5_corr_values = []

        for lead_index, lead in enumerate(LEADS):

            reference = original[:, lead_index]

            reconstructed_v4 = digitize_v4(
                image,
                lead_index,
                row_height
            )

            reconstructed_v5 = digitize_v5(
                image,
                lead_index,
                row_height
            )

            mae4, rmse4, corr4 = calculate_metrics(
                reference,
                reconstructed_v4
            )

            mae5, rmse5, corr5 = calculate_metrics(
                reference,
                reconstructed_v5
            )

            v4_signals.append(reconstructed_v4)
            v5_signals.append(reconstructed_v5)

            v4_mae_values.append(mae4)
            v5_mae_values.append(mae5)

            v4_rmse_values.append(rmse4)
            v5_rmse_values.append(rmse5)

            v4_corr_values.append(corr4)
            v5_corr_values.append(corr5)

            lead_results.append({
                "ecg_id": ecg_id,
                "lead": lead,
                "V4_MAE": mae4,
                "V5_MAE": mae5,
                "V4_RMSE": rmse4,
                "V5_RMSE": rmse5,
                "V4_Correlation": corr4,
                "V5_Correlation": corr5
            })

        # Save reconstructed signals.
        time = np.arange(SAMPLE_COUNT) / 100.0

        v4_df = pd.DataFrame(
            np.column_stack(v4_signals),
            columns=LEADS
        )
        v4_df.insert(0, "time_seconds", time)

        v5_df = pd.DataFrame(
            np.column_stack(v5_signals),
            columns=LEADS
        )
        v5_df.insert(0, "time_seconds", time)

        v4_df.to_csv(
            RECON_DIR / f"ECG_{ecg_id}_V4.csv",
            index=False
        )

        v5_df.to_csv(
            RECON_DIR / f"ECG_{ecg_id}_V5.csv",
            index=False
        )

        record_results.append({
            "ecg_id": ecg_id,
            "dataset_index": idx,
            "fold": int(row["fold"]),
            "V4_Mean_MAE": float(np.mean(v4_mae_values)),
            "V5_Mean_MAE": float(np.mean(v5_mae_values)),
            "V4_Mean_RMSE": float(np.mean(v4_rmse_values)),
            "V5_Mean_RMSE": float(np.mean(v5_rmse_values)),
            "V4_Mean_Correlation": float(np.nanmean(v4_corr_values)),
            "V5_Mean_Correlation": float(np.nanmean(v5_corr_values))
        })

        print(
            f"V4 MAE: {np.mean(v4_mae_values):.5f} | "
            f"V5 MAE: {np.mean(v5_mae_values):.5f}"
        )

        print(
            f"V4 RMSE: {np.mean(v4_rmse_values):.5f} | "
            f"V5 RMSE: {np.mean(v5_rmse_values):.5f}"
        )

        print(
            f"V4 Correlation: {np.nanmean(v4_corr_values):.5f} | "
            f"V5 Correlation: {np.nanmean(v5_corr_values):.5f}"
        )

    # -----------------------------------------------------
    # SAVE REPORTS
    # -----------------------------------------------------

    lead_df = pd.DataFrame(lead_results)
    record_df = pd.DataFrame(record_results)

    lead_df.to_csv(
        OUTPUT_DIR / "per_lead_metrics.csv",
        index=False
    )

    record_df.to_csv(
        OUTPUT_DIR / "per_record_metrics.csv",
        index=False
    )

    summary = {
        "number_of_records": len(selection),
        "fold": 10,
        "row_height": row_height,
        "vertical_scale": VERTICAL_SCALE,
        "V4_mean_MAE": float(lead_df["V4_MAE"].mean()),
        "V5_mean_MAE": float(lead_df["V5_MAE"].mean()),
        "V4_mean_RMSE": float(lead_df["V4_RMSE"].mean()),
        "V5_mean_RMSE": float(lead_df["V5_RMSE"].mean()),
        "V4_mean_correlation": float(
            lead_df["V4_Correlation"].mean()
        ),
        "V5_mean_correlation": float(
            lead_df["V5_Correlation"].mean()
        )
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    # -----------------------------------------------------
    # COMPARISON PLOT
    # -----------------------------------------------------

    x = np.arange(len(record_df))
    width = 0.36

    plt.figure(figsize=(13, 6))

    plt.bar(
        x - width / 2,
        record_df["V4_Mean_MAE"],
        width,
        label="V4"
    )

    plt.bar(
        x + width / 2,
        record_df["V5_Mean_MAE"],
        width,
        label="V5"
    )

    plt.xticks(
        x,
        record_df["ecg_id"].astype(str),
        rotation=45
    )

    plt.xlabel("ECG ID")
    plt.ylabel("Mean MAE across 12 leads")
    plt.title("V4 vs V5 Digitization Error Across 15 ECG Records")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "v4_v5_mae_comparison.png",
        dpi=200
    )

    plt.close()

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("MULTI-RECORD EXPERIMENT COMPLETED")
    print("=" * 65)

    print("\nOVERALL RESULTS")

    for metric in [
        "MAE",
        "RMSE",
        "Correlation"
    ]:
        print(f"\n{metric}:")

        if metric == "MAE":
            print(f"V4: {summary['V4_mean_MAE']:.6f}")
            print(f"V5: {summary['V5_mean_MAE']:.6f}")

        elif metric == "RMSE":
            print(f"V4: {summary['V4_mean_RMSE']:.6f}")
            print(f"V5: {summary['V5_mean_RMSE']:.6f}")

        else:
            print(
                f"V4: "
                f"{summary['V4_mean_correlation']:.6f}"
            )
            print(
                f"V5: "
                f"{summary['V5_mean_correlation']:.6f}"
            )

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- per_lead_metrics.csv")
    print("- per_record_metrics.csv")
    print("- summary.json")
    print("- v4_v5_mae_comparison.png")
    print("- images/ (15 synthetic ECG images)")
    print("- reconstructed/ (30 reconstructed CSV files)")


if __name__ == "__main__":
    main()
