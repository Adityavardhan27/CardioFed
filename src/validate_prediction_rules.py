
"""Validate the NORM consistency rule on the validation set."""

import argparse
import json
import numpy as np
import torch

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss
)

from src.dataset import make_loaders
from src.model import MultiScaleResidualECGNet
from src.evaluate import predict_probs
from src.preprocessing import CLASSES
from src.io_utils import guard_outputs
from src.model_registry import MODELS, DEFAULT_CENTRALIZED


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = ROOT / MODELS[DEFAULT_CENTRALIZED]["checkpoint"]
EVALUATION_PATH = ROOT / MODELS[DEFAULT_CENTRALIZED]["evaluation"]
OUTPUT_DIR = ROOT / "results/revised/prediction_rules"

OUTPUTS = ['prediction_rule_validation.json']


def calculate_metrics(probabilities, actual, predictions):

    auc_scores = {}

    for i, name in enumerate(CLASSES):
        if len(np.unique(actual[:, i])) < 2:
            auc_scores[name] = None
        else:
            auc_scores[name] = float(
                roc_auc_score(actual[:, i], probabilities[:, i])
            )

    valid_auc = [
        value for value in auc_scores.values()
        if value is not None
    ]

    f1 = f1_score(
        actual, predictions, average=None, zero_division=0
    )

    precision = precision_score(
        actual, predictions, average=None, zero_division=0
    )

    recall = recall_score(
        actual, predictions, average=None, zero_division=0
    )

    return {
        "macro_auroc": float(np.mean(valid_auc)) if valid_auc else None,
        "macro_f1": float(np.mean(f1)),
        "hamming_loss": float(hamming_loss(actual, predictions)),
        "subset_accuracy": float(
            np.mean(np.all(actual == predictions, axis=1))
        ),
        "per_class_f1": dict(zip(CLASSES, map(float, f1))),
        "per_class_precision": dict(zip(CLASSES, map(float, precision))),
        "per_class_recall": dict(zip(CLASSES, map(float, recall)))
    }


def main():

    guard_outputs(
        [OUTPUT_DIR / name for name in OUTPUTS],
        overwrite=ARGS.overwrite
    )

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Thresholds from:", EVALUATION_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("CardioFed Prediction Rule Validation")
    print("Device:", device)

    # Load the existing thresholds.
    with open(EVALUATION_PATH, "r") as file:
        evaluation = json.load(file)

    thresholds = np.array(
        [evaluation["thresholds"][name] for name in CLASSES],
        dtype=np.float32
    )

    print("\nLoaded thresholds:")

    for name, threshold in zip(CLASSES, thresholds):
        print(f"{name}: {threshold:.4f}")

    # Load validation data only.
    _, val_loader, _, _ = make_loaders(
        processed_dir=str(DATA_DIR),
        batch_size=16
    )

    # Load the trained model.
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

    print("\nModel checkpoint loaded.")

    # Generate validation probabilities.
    probabilities, actual = predict_probs(
        model, val_loader, device
    )

    print("Validation recordings:", len(actual))

    # Original predictions.
    original_predictions = (
        probabilities >= thresholds
    ).astype(int)

    # Apply the proposed NORM consistency rule.
    abnormal_indices = [
        CLASSES.index(name)
        for name in ["MI", "STTC", "CD", "HYP"]
    ]

    predicted_abnormal = (
        original_predictions[:, abnormal_indices].any(axis=1)
    )

    consistent_predictions = original_predictions.copy()

    consistent_predictions[predicted_abnormal, 0] = 0

    # Calculate both sets of metrics.
    original_metrics = calculate_metrics(
        probabilities,
        actual,
        original_predictions
    )

    consistent_metrics = calculate_metrics(
        probabilities,
        actual,
        consistent_predictions
    )

    # Count conflicts.
    original_conflicts = int(
        (
            (original_predictions[:, 0] == 1)
            & predicted_abnormal
        ).sum()
    )

    remaining_conflicts = int(
        (
            (consistent_predictions[:, 0] == 1)
            & predicted_abnormal
        ).sum()
    )

    # Display comparison.
    print("\n========== ORIGINAL VALIDATION METRICS ==========")
    print(json.dumps(original_metrics, indent=2))

    print("\n========== AFTER NORM CONSISTENCY RULE ==========")
    print(json.dumps(consistent_metrics, indent=2))

    print("\n========== CONFLICT ANALYSIS ==========")
    print("Original NORM + abnormal conflicts:", original_conflicts)
    print("Remaining conflicts:", remaining_conflicts)

    # Display metric comparison.
    print("\n========== METRIC COMPARISON ==========")

    print(
        f"{'Metric':<20}"
        f"{'Original':>14}"
        f"{'Consistency':>16}"
    )

    for metric in [
        "macro_auroc",
        "macro_f1",
        "hamming_loss",
        "subset_accuracy"
    ]:
        print(
            f"{metric:<20}"
            f"{original_metrics[metric]:>14.6f}"
            f"{consistent_metrics[metric]:>16.6f}"
        )

    # Save results.
    summary = {
        "model": "Multi-Scale Residual CNN",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)) if CHECKPOINT_PATH.is_relative_to(ROOT) else str(CHECKPOINT_PATH),
        "dataset": "Validation set (fold 9)",
        "validation_recordings": int(len(actual)),
        "thresholds": dict(zip(CLASSES, thresholds.tolist())),
        "original_metrics": original_metrics,
        "norm_consistency_metrics": consistent_metrics,
        "original_norm_abnormal_conflicts": original_conflicts,
        "remaining_norm_abnormal_conflicts": remaining_conflicts
    }

    output_path = OUTPUT_DIR / "prediction_rule_validation.json"

    with open(output_path, "w") as file:
        json.dump(summary, file, indent=2)

    print("\nResults saved to:", output_path)
    print("\nVALIDATION COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--evaluation", default=str(EVALUATION_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    ARGS = parser.parse_args()
    CHECKPOINT_PATH = Path(ARGS.checkpoint)
    EVALUATION_PATH = Path(ARGS.evaluation)
    OUTPUT_DIR = Path(ARGS.output_dir)
    main()
