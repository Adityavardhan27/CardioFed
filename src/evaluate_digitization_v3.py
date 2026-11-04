
"""Evaluate Version 3 ECG digitization against the original signal."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = ROOT / "results/examples/ECG_63.csv"
RECONSTRUCTED_PATH = ROOT / "results/examples/ECG_63_reconstructed_v3.csv"

OUTPUT_DIR = ROOT / "results/examples"

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


def main():

    original_df = pd.read_csv(ORIGINAL_PATH)
    reconstructed_df = pd.read_csv(RECONSTRUCTED_PATH)

    original = original_df[LEADS].to_numpy(dtype=np.float64)
    reconstructed = reconstructed_df[LEADS].to_numpy(dtype=np.float64)

    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
        )

    results = []

    for i, lead in enumerate(DISPLAY_NAMES):

        actual = original[:, i]
        predicted = reconstructed[:, i]

        error = predicted - actual

        correlation = np.corrcoef(actual, predicted)[0, 1]

        results.append({
            "Lead": lead,
            "MAE": np.mean(np.abs(error)),
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "Correlation": correlation
        })

    metrics_df = pd.DataFrame(results)

    print("\n========== VERSION 3 DIGITIZATION ACCURACY ==========")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    print("\n========== OVERALL METRICS ==========")
    print("Mean MAE:", round(metrics_df["MAE"].mean(), 5))
    print("Mean RMSE:", round(metrics_df["RMSE"].mean(), 5))
    print(
        "Mean correlation:",
        round(metrics_df["Correlation"].mean(), 5)
    )

    metrics_path = OUTPUT_DIR / "digitization_metrics_v3.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fig, axes = plt.subplots(
        6, 2,
        figsize=(16, 18),
        sharex=True
    )

    time = np.arange(len(original)) / 100.0

    for i, lead in enumerate(DISPLAY_NAMES):

        ax = axes[i // 2, i % 2]

        ax.plot(
            time,
            original[:, i],
            label="Original",
            linewidth=1
        )

        ax.plot(
            time,
            reconstructed[:, i],
            label="Version 3",
            linewidth=1,
            alpha=0.8
        )

        ax.set_title(lead)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Normalized amplitude")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Original vs Version 3 Reconstructed ECG",
        fontsize=16
    )

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "digitization_comparison_v3.png"

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nEvaluation completed!")
    print("Metrics:", metrics_path)
    print("Comparison plot:", plot_path)


if __name__ == "__main__":
    main()
