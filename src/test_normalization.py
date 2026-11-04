
"""Test whether per-lead normalization improves ECG predictions."""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import MultiScaleResidualECGNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

ORIGINAL_PATH = ROOT / "results/examples/ECG_63.csv"
RECONSTRUCTED_PATH = ROOT / "results/examples/ECG_63_reconstructed_v4.csv"

CHECKPOINT_PATH = ROOT / "results/checkpoints/multiscale_best.pt"
EVALUATION_PATH = ROOT / "results/metrics/multiscale_final_evaluation.json"


def load_signal(path):

    df = pd.read_csv(path)

    signal = df[LEADS].to_numpy(dtype=np.float32)

    if signal.shape != (1000, 12):
        raise ValueError(f"Unexpected shape: {signal.shape}")

    if not np.isfinite(signal).all():
        raise ValueError("Invalid signal values.")

    return signal


def normalize_per_lead(signal):

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)

    return ((signal - mean) / (std + 1e-8)).astype(np.float32)


def load_model():

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(EVALUATION_PATH, "r") as f:
        evaluation = json.load(f)

    thresholds = np.array(
        [evaluation["thresholds"][name] for name in CLASSES],
        dtype=np.float32
    )

    return model, thresholds


def predict(model, signal):

    tensor = torch.from_numpy(
        signal.T.copy()
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    return probabilities


def print_statistics(name, signal):

    print(f"\n{name}")

    print(
        pd.DataFrame({
            "Lead": LEADS,
            "Mean": signal.mean(axis=0),
            "Std": signal.std(axis=0, ddof=1)
        }).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )


def main():

    original = load_signal(ORIGINAL_PATH)
    reconstructed = load_signal(RECONSTRUCTED_PATH)

    normalized_reconstructed = normalize_per_lead(
        reconstructed
    )

    print_statistics("ORIGINAL SIGNAL", original)
    print_statistics("RECONSTRUCTED SIGNAL", reconstructed)
    print_statistics("NORMALIZED RECONSTRUCTION", normalized_reconstructed)

    model, thresholds = load_model()

    inputs = {
        "Original": original,
        "Reconstructed V4": reconstructed,
        "Reconstructed V4 + z-score": normalized_reconstructed
    }

    print("\n\nPREDICTION COMPARISON")
    print("=" * 75)

    results = {}

    for name, signal in inputs.items():

        probabilities = predict(model, signal)
        results[name] = probabilities

        print(f"\n{name}")

        for i, class_name in enumerate(CLASSES):

            print(
                f"{class_name:<8} "
                f"Probability: {probabilities[i] * 100:7.2f}% | "
                f"Threshold: {thresholds[i] * 100:7.2f}%"
            )

    print("\n\nPROBABILITY DIFFERENCES FROM ORIGINAL")
    print("=" * 75)

    for name in ["Reconstructed V4", "Reconstructed V4 + z-score"]:

        print(f"\n{name}")

        for i, class_name in enumerate(CLASSES):

            difference = (
                results[name][i] - results["Original"][i]
            ) * 100

            print(f"{class_name:<8}: {difference:+.2f} percentage points")


if __name__ == "__main__":
    main()
