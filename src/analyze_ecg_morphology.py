
"""Beat-level morphology analysis for CardioFed ECG digitization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import find_peaks

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

SAMPLING_RATE = 100

# Search within +/- 0.15 seconds of each reference heartbeat.
SEARCH_WINDOW = 15

# Minimum separation between detected heartbeats: 0.45 seconds.
MIN_PEAK_DISTANCE = 45


def load_signals():
    original_df = pd.read_csv(ORIGINAL_PATH)
    reconstructed_df = pd.read_csv(RECONSTRUCTED_PATH)

    original = original_df[LEADS].to_numpy(dtype=np.float64)
    reconstructed = reconstructed_df[LEADS].to_numpy(dtype=np.float64)

    if original.shape != (1000, 12):
        raise ValueError(f"Unexpected original shape: {original.shape}")

    if reconstructed.shape != (1000, 12):
        raise ValueError(f"Unexpected reconstructed shape: {reconstructed.shape}")

    if not np.isfinite(original).all() or not np.isfinite(reconstructed).all():
        raise ValueError("ECG signals contain invalid values.")

    return original, reconstructed


def detect_reference_beats(original):
    """Detect reference heartbeat peaks using original Lead II."""

    lead_ii = original[:, 1]

    peaks, properties = find_peaks(
        np.abs(lead_ii),
        distance=MIN_PEAK_DISTANCE,
        prominence=0.8
    )

    if len(peaks) < 3:
        print("Trying a lower peak prominence threshold...")
        peaks, properties = find_peaks(
            np.abs(lead_ii),
            distance=MIN_PEAK_DISTANCE,
            prominence=0.4
        )

    if len(peaks) < 2:
        raise RuntimeError(
            "Could not detect enough reference beats in Lead II. "
            "Peak detection needs further inspection."
        )

    return peaks


def find_local_peak(signal, reference_index):
    """Find the strongest absolute peak near a reference heartbeat."""

    start = max(0, reference_index - SEARCH_WINDOW)
    end = min(len(signal), reference_index + SEARCH_WINDOW + 1)

    local_segment = signal[start:end]

    local_index = np.argmax(np.abs(local_segment))
    actual_index = start + local_index

    return actual_index, signal[actual_index]


def analyze_beats(original, reconstructed, reference_peaks):
    rows = []

    for lead_index, lead_name in enumerate(LEADS):

        original_lead = original[:, lead_index]
        reconstructed_lead = reconstructed[:, lead_index]

        for beat_number, reference_peak in enumerate(reference_peaks, start=1):

            original_index, original_amplitude = find_local_peak(
                original_lead, reference_peak
            )

            reconstructed_index, reconstructed_amplitude = find_local_peak(
                reconstructed_lead, reference_peak
            )

            timing_error = reconstructed_index - original_index

            amplitude_difference = (
                reconstructed_amplitude - original_amplitude
            )

            amplitude_ratio = (
                abs(reconstructed_amplitude)
                / (abs(original_amplitude) + 1e-8)
            )

            rows.append({
                "Lead": lead_name,
                "Beat": beat_number,
                "Reference_Time_sec": reference_peak / SAMPLING_RATE,
                "Original_Peak_Time_sec": original_index / SAMPLING_RATE,
                "Reconstructed_Peak_Time_sec": reconstructed_index / SAMPLING_RATE,
                "Original_Amplitude": original_amplitude,
                "Reconstructed_Amplitude": reconstructed_amplitude,
                "Timing_Error_Samples": timing_error,
                "Timing_Error_ms": timing_error * 10,
                "Amplitude_Difference": amplitude_difference,
                "Amplitude_Ratio": amplitude_ratio
            })

    return pd.DataFrame(rows)


def print_summary(results, reference_peaks):

    print("\nREFERENCE HEARTBEATS")
    print("=" * 65)
    print(f"Detected beats: {len(reference_peaks)}")
    print(
        "Reference peak times (seconds):",
        np.round(reference_peaks / SAMPLING_RATE, 3)
    )

    summary = results.groupby("Lead").agg(
        Mean_Absolute_Timing_Error_ms=(
            "Timing_Error_ms",
            lambda x: np.mean(np.abs(x))
        ),
        Mean_Absolute_Amplitude_Difference=(
            "Amplitude_Difference",
            lambda x: np.mean(np.abs(x))
        ),
        Mean_Amplitude_Ratio=("Amplitude_Ratio", "mean"),
        Beat_Count=("Beat", "count")
    ).reset_index()

    print("\nBEAT-LEVEL MORPHOLOGY SUMMARY")
    print("=" * 100)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nOVERALL RESULTS")
    print("=" * 65)
    print(
        "Mean absolute timing error:",
        f"{results['Timing_Error_ms'].abs().mean():.2f} ms"
    )
    print(
        "Mean absolute amplitude difference:",
        f"{results['Amplitude_Difference'].abs().mean():.4f}"
    )
    print(
        "Mean amplitude ratio:",
        f"{results['Amplitude_Ratio'].mean():.4f}"
    )

    return summary


def plot_beat_comparison(original, reconstructed, reference_peaks):

    fig, axes = plt.subplots(12, 1, figsize=(16, 24))

    time = np.arange(1000) / SAMPLING_RATE

    for i, ax in enumerate(axes):

        ax.plot(
            time,
            original[:, i],
            label="Original",
            linewidth=1.1
        )

        ax.plot(
            time,
            reconstructed[:, i],
            label="Reconstructed V4",
            linewidth=1.0,
            alpha=0.8
        )

        for peak in reference_peaks:
            ax.axvline(
                peak / SAMPLING_RATE,
                linestyle=":",
                alpha=0.35,
                linewidth=0.8
            )

        ax.set_title(LEADS[i], loc="left")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "morphology_beat_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\nBeat comparison plot saved: {output_path}")


def main():

    print("Loading ECG signals...")

    original, reconstructed = load_signals()

    print("Detecting reference heartbeats...")

    reference_peaks = detect_reference_beats(original)

    print("Comparing heartbeat morphology across all 12 leads...")

    results = analyze_beats(
        original,
        reconstructed,
        reference_peaks
    )

    summary = print_summary(results, reference_peaks)

    results.to_csv(
        OUTPUT_DIR / "beat_level_morphology.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "beat_level_morphology_summary.csv",
        index=False
    )

    plot_beat_comparison(
        original,
        reconstructed,
        reference_peaks
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
