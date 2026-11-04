
"""Analyze waveform differences between original and reconstructed ECG."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = ROOT / "results/examples/ECG_63.csv"
RECONSTRUCTED_PATH = ROOT / "results/examples/ECG_63_reconstructed_v4.csv"

OUTPUT_DIR = ROOT / "results/examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def load_signals():
    original_df = pd.read_csv(ORIGINAL_PATH)
    reconstructed_df = pd.read_csv(RECONSTRUCTED_PATH)

    original = original_df[LEADS].to_numpy(dtype=np.float64)
    reconstructed = reconstructed_df[LEADS].to_numpy(dtype=np.float64)

    if original.shape != (1000, 12) or reconstructed.shape != (1000, 12):
        raise ValueError("Both ECG files must contain 1000 samples and 12 leads.")

    return original, reconstructed


def calculate_metrics(original, reconstructed):
    rows = []

    for i, lead in enumerate(LEADS):
        a = original[:, i]
        b = reconstructed[:, i]

        mae = np.mean(np.abs(a - b))
        rmse = np.sqrt(np.mean((a - b) ** 2))
        correlation = np.corrcoef(a, b)[0, 1]

        original_range = np.max(a) - np.min(a)
        reconstructed_range = np.max(b) - np.min(b)

        original_peak = np.max(np.abs(a))
        reconstructed_peak = np.max(np.abs(b))

        rows.append({
            "Lead": lead,
            "MAE": mae,
            "RMSE": rmse,
            "Correlation": correlation,
            "Original_Range": original_range,
            "Reconstructed_Range": reconstructed_range,
            "Range_Ratio": reconstructed_range / (original_range + 1e-8),
            "Original_Max_Abs": original_peak,
            "Reconstructed_Max_Abs": reconstructed_peak
        })

    return pd.DataFrame(rows)


def peak_analysis(original, reconstructed):
    rows = []

    for i, lead in enumerate(LEADS):
        a = original[:, i]
        b = reconstructed[:, i]

        original_max_index = np.argmax(a)
        reconstructed_max_index = np.argmax(b)

        original_min_index = np.argmin(a)
        reconstructed_min_index = np.argmin(b)

        rows.append({
            "Lead": lead,
            "Original_Max_Index": original_max_index,
            "Reconstructed_Max_Index": reconstructed_max_index,
            "Max_Peak_Shift": reconstructed_max_index - original_max_index,
            "Original_Min_Index": original_min_index,
            "Reconstructed_Min_Index": reconstructed_min_index,
            "Min_Peak_Shift": reconstructed_min_index - original_min_index
        })

    return pd.DataFrame(rows)


def plot_comparison(original, reconstructed):
    fig, axes = plt.subplots(12, 1, figsize=(16, 24))

    time = np.arange(1000) / 100

    for i, ax in enumerate(axes):
        ax.plot(time, original[:, i], label="Original", linewidth=1.2)
        ax.plot(
            time,
            reconstructed[:, i],
            label="Reconstructed V4",
            linewidth=1.0,
            alpha=0.8
        )

        ax.set_title(LEADS[i], loc="left")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "reconstruction_waveform_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\nWaveform comparison saved to: {output_path}")


def plot_error(original, reconstructed):
    fig, axes = plt.subplots(12, 1, figsize=(16, 24))

    time = np.arange(1000) / 100

    for i, ax in enumerate(axes):
        error = original[:, i] - reconstructed[:, i]

        ax.plot(time, error, linewidth=0.9)
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.set_title(LEADS[i], loc="left")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "reconstruction_error.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Error plot saved to: {output_path}")


def main():
    original, reconstructed = load_signals()

    metrics = calculate_metrics(original, reconstructed)
    peaks = peak_analysis(original, reconstructed)

    print("\nWAVEFORM RECONSTRUCTION METRICS")
    print("=" * 100)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPEAK LOCATION ANALYSIS")
    print("=" * 100)
    print(peaks.to_string(index=False))

    print("\nOVERALL METRICS")
    print("=" * 100)
    print(f"Mean MAE:         {metrics['MAE'].mean():.5f}")
    print(f"Mean RMSE:        {metrics['RMSE'].mean():.5f}")
    print(f"Mean Correlation: {metrics['Correlation'].mean():.5f}")
    print(f"Mean Range Ratio: {metrics['Range_Ratio'].mean():.5f}")

    metrics.to_csv(OUTPUT_DIR / "reconstruction_metrics.csv", index=False)
    peaks.to_csv(OUTPUT_DIR / "reconstruction_peak_analysis.csv", index=False)

    plot_comparison(original, reconstructed)
    plot_error(original, reconstructed)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
