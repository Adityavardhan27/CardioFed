
"""Evaluate V4 ECG image digitization against the original signal."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = ROOT / "results" / "examples" / "ECG_63.csv"
RECONSTRUCTED_PATH = ROOT / "results" / "examples" / "ECG_63_reconstructed_v4.csv"

OUTPUT_DIR = ROOT / "results" / "examples"

METRICS_PATH = OUTPUT_DIR / "digitization_metrics_v4.csv"
PLOT_PATH = OUTPUT_DIR / "digitization_comparison_v4.png"

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

DISPLAY_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def calculate_metrics(original, reconstructed):

    mae = np.mean(np.abs(original - reconstructed))
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))

    if np.std(original) < 1e-12 or np.std(reconstructed) < 1e-12:
        correlation = np.nan
    else:
        correlation = np.corrcoef(original, reconstructed)[0, 1]

    return mae, rmse, correlation


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(ORIGINAL_PATH)
    reconstructed_df = pd.read_csv(RECONSTRUCTED_PATH)

    for lead in LEADS:
        if lead not in original_df.columns:
            raise ValueError(f"Missing original lead: {lead}")

        if lead not in reconstructed_df.columns:
            raise ValueError(f"Missing reconstructed lead: {lead}")

    if len(original_df) != len(reconstructed_df):
        raise ValueError(
            "Original and reconstructed signals have different lengths."
        )

    original = original_df[LEADS].to_numpy(dtype=np.float64)
    reconstructed = reconstructed_df[LEADS].to_numpy(dtype=np.float64)

    if not np.isfinite(original).all() or not np.isfinite(reconstructed).all():
        raise ValueError("Signals contain invalid values.")

    rows = []

    for i, lead in enumerate(LEADS):

        mae, rmse, correlation = calculate_metrics(
            original[:, i],
            reconstructed[:, i]
        )

        rows.append({
            "Lead": DISPLAY_NAMES[i],
            "MAE": mae,
            "RMSE": rmse,
            "Correlation": correlation
        })

    metrics_df = pd.DataFrame(rows)

    mean_row = pd.DataFrame([{
        "Lead": "MEAN",
        "MAE": metrics_df["MAE"].mean(),
        "RMSE": metrics_df["RMSE"].mean(),
        "Correlation": metrics_df["Correlation"].mean()
    }])

    metrics_df = pd.concat(
        [metrics_df, mean_row],
        ignore_index=True
    )

    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\nV4 DIGITIZATION RESULTS")
    print("=" * 65)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Plot original and reconstructed signals.
    fig, axes = plt.subplots(
        12, 1,
        figsize=(16, 24),
        sharex=True
    )

    time = np.arange(len(original)) / 100.0

    for i, ax in enumerate(axes):

        ax.plot(
            time,
            original[:, i],
            label="Original",
            linewidth=1
        )

        ax.plot(
            time,
            reconstructed[:, i],
            label="V4 reconstructed",
            linewidth=1,
            alpha=0.8
        )

        ax.set_ylabel(DISPLAY_NAMES[i])
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        "Original vs V4 Reconstructed ECG Signals",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print("\nSaved metrics:", METRICS_PATH)
    print("Saved comparison plot:", PLOT_PATH)


if __name__ == "__main__":
    main()
