
"""Predict diagnostic classes from a processed CardioFed ECG CSV.

The CSV must contain 1000 samples of the 12 leads that have ALREADY been
band-pass filtered and z-score normalised by the CardioFed pipeline.

Uses the primary (reproducible) multi-scale model and its validation-tuned
thresholds by default. Writes <output-dir>/ECG_<id>_prediction.json and never
overwrites an existing file unless --overwrite is given.
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from src.model import MultiScaleResidualECGNet
from src.preprocessing import CLASSES
from src.io_utils import guard_outputs
from src.model_registry import MODELS, DEFAULT_CENTRALIZED


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = ROOT / MODELS[DEFAULT_CENTRALIZED]["checkpoint"]

EVALUATION_PATH = ROOT / MODELS[DEFAULT_CENTRALIZED]["evaluation"]

DATA_DIR = ROOT / "data/processed"

EXAMPLES_DIR = ROOT / "results/examples"
OUTPUT_DIR = ROOT / "results/revised/examples"


# --------------------------------------------------
# ECG CONFIGURATION
# --------------------------------------------------

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

TARGET_CSV = EXAMPLES_DIR / "ECG_63.csv"


# --------------------------------------------------
# LOAD MODEL AND THRESHOLDS
# --------------------------------------------------

def load_model(device):

    print("\nLoading trained Multi-Scale Residual CNN...")

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(EVALUATION_PATH, "r") as file:
        evaluation = json.load(file)

    thresholds = np.array(
        [evaluation["thresholds"][name] for name in CLASSES],
        dtype=np.float32
    )

    print("Model loaded successfully.")
    print("Checkpoint epoch:", checkpoint["epoch"])

    return model, thresholds


# --------------------------------------------------
# LOAD ECG CSV
# --------------------------------------------------

def load_ecg_csv(csv_path):

    print("\nReading ECG CSV...")

    df = pd.read_csv(csv_path)

    missing = [lead for lead in LEADS if lead not in df.columns]

    if missing:
        raise ValueError(
            f"Missing ECG leads: {missing}"
        )

    if len(df) != 1000:
        raise ValueError(
            f"Expected 1000 samples, received {len(df)}."
        )

    # CSV contains already-preprocessed signals.
    signal = df[LEADS].to_numpy(dtype=np.float32)

    if not np.isfinite(signal).all():
        raise ValueError("ECG contains missing or invalid values.")

    # Convert [1000, 12] into [12, 1000].
    signal = signal.T

    # Add batch dimension: [1, 12, 1000].
    signal = np.expand_dims(signal, axis=0)

    ecg_id = None

    if "ecg_id" in df.columns:
        ecg_id = int(df["ecg_id"].iloc[0])

    print("ECG loaded successfully.")
    print("Input tensor shape:", signal.shape)

    return signal, ecg_id


# --------------------------------------------------
# GET ACTUAL PTB-XL LABELS
# --------------------------------------------------

def get_actual_labels(ecg_id):

    if ecg_id is None:
        return None

    ids = np.load(DATA_DIR / "ecg_ids.npy")
    labels = np.load(DATA_DIR / "y.npy")

    matches = np.where(ids == ecg_id)[0]

    if len(matches) == 0:
        return None

    return labels[int(matches[0])]


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(model, signal, thresholds, device):

    input_tensor = torch.from_numpy(signal).to(device)

    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.cpu().numpy()[0]

    predictions = (probabilities >= thresholds).astype(int)

    return probabilities, predictions


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("CardioFed ECG Prediction System")
    print("Device:", device)

    model, thresholds = load_model(device)

    signal, ecg_id = load_ecg_csv(args.csv)

    output_path = Path(args.output_dir) / f"ECG_{ecg_id}_prediction.json"
    guard_outputs([output_path], overwrite=args.overwrite)

    probabilities, predictions = predict(
        model,
        signal,
        thresholds,
        device
    )

    actual_labels = get_actual_labels(ecg_id)

    print("\n" + "=" * 55)
    print("CARDIOFED PREDICTION RESULTS")
    print("=" * 55)

    print(f"\nECG ID: {ecg_id}")

    results = []

    for i, name in enumerate(CLASSES):

        actual = (
            int(actual_labels[i])
            if actual_labels is not None
            else None
        )

        result = {
            "class": name,
            "probability": float(probabilities[i]),
            "threshold": float(thresholds[i]),
            "predicted_label": int(predictions[i]),
            "actual_label": actual
        }

        results.append(result)

        print(f"\n{name}")
        print(f"  Probability : {probabilities[i]:.4f}")
        print(f"  Threshold   : {thresholds[i]:.4f}")
        print(f"  Prediction  : {'Positive' if predictions[i] else 'Negative'}")

        if actual is not None:
            print(f"  Actual      : {'Positive' if actual else 'Negative'}")

    # Save results.
    output = {
        "ecg_id": ecg_id,
        "model": "Multi-Scale Residual CNN",
        "checkpoint": MODELS[DEFAULT_CENTRALIZED]["checkpoint"],
        "predictions": results
    }

    with open(output_path, "w") as file:
        json.dump(output, file, indent=2)

    print("\n" + "=" * 55)
    print("Prediction saved to:", output_path)
    print("CARDIOFED PREDICTION COMPLETED!")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(TARGET_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
