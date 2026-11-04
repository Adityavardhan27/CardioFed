
"""Compare original, V4 and V5 ECG signals using the same trained model."""

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

PATHS = {
    "Original": ROOT / "results/examples/ECG_63.csv",
    "V4": ROOT / "results/examples/ECG_63_reconstructed_v4.csv",
    "V5": ROOT / "results/examples/ECG_63_reconstructed_v5.csv"
}

CHECKPOINT_PATH = ROOT / "results/checkpoints/multiscale_best.pt"
EVALUATION_PATH = ROOT / "results/metrics/multiscale_final_evaluation.json"


def load_signal(path):
    df = pd.read_csv(path)
    signal = df[LEADS].to_numpy(dtype=np.float32)

    if signal.shape != (1000, 12):
        raise ValueError(f"Unexpected signal shape in {path}: {signal.shape}")

    if not np.isfinite(signal).all():
        raise ValueError(f"Invalid values in {path}")

    return signal


def load_model():
    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
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


def main():

    print("Loading ECG signals...")
    signals = {
        name: load_signal(path)
        for name, path in PATHS.items()
    }

    print("Loading trained MultiScale model...")
    model, thresholds = load_model()

    results = {}

    for name, signal in signals.items():
        results[name] = predict(model, signal)

    print("\nPREDICTION COMPARISON")
    print("=" * 85)

    for name, probabilities in results.items():

        print(f"\n{name}")

        for i, class_name in enumerate(CLASSES):
            predicted = int(probabilities[i] >= thresholds[i])

            print(
                f"{class_name:<8} "
                f"Probability: {probabilities[i] * 100:7.2f}% | "
                f"Threshold: {thresholds[i] * 100:7.2f}% | "
                f"Raw prediction: {predicted}"
            )

    print("\nPROBABILITY DIFFERENCES FROM ORIGINAL")
    print("=" * 85)

    for name in ["V4", "V5"]:
        print(f"\n{name}")

        for i, class_name in enumerate(CLASSES):
            difference = (
                results[name][i] - results["Original"][i]
            ) * 100

            print(f"{class_name:<8}: {difference:+.2f} percentage points")

    print("\nEXPERIMENT COMPLETED.")


if __name__ == "__main__":
    main()
