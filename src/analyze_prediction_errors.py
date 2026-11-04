
"""Analyze prediction errors and NORM-abnormal conflicts."""

import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import confusion_matrix

from src.dataset import make_loaders
from src.model import MultiScaleResidualECGNet
from src.evaluate import predict_probs, compute_metrics
from src.preprocessing import CLASSES


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = ROOT / "results/checkpoints/multiscale_best.pt"
EVALUATION_PATH = ROOT / "results/metrics/multiscale_final_evaluation.json"
OUTPUT_DIR = ROOT / "results/metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("CardioFed Prediction Error Analysis")
    print("Device:", device)

    # Load saved validation thresholds.
    with open(EVALUATION_PATH, "r") as file:
        evaluation = json.load(file)

    thresholds = np.array(
        [evaluation["thresholds"][name] for name in CLASSES],
        dtype=np.float32
    )

    print("\nLoaded thresholds:")
    for name, threshold in zip(CLASSES, thresholds):
        print(f"{name}: {threshold:.4f}")

    # Load test data.
    _, _, test_loader, _ = make_loaders(
        processed_dir=str(DATA_DIR),
        batch_size=16
    )

    # Load trained model.
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

    # Obtain test predictions.
    probabilities, actual = predict_probs(
        model, test_loader, device
    )

    original_predictions = (
        probabilities >= thresholds
    ).astype(int)

    # Retrieve test ECG IDs in fold order.
    folds = np.load(DATA_DIR / "folds.npy")
    ecg_ids = np.load(DATA_DIR / "ecg_ids.npy")

    test_indices = np.where(folds == 10)[0]
    test_ids = ecg_ids[test_indices]

    if len(test_ids) != len(actual):
        raise ValueError(
            "Test ECG ID count does not match test predictions. "
            "Check the ordering in src/dataset.py."
        )

    print("\nTest recordings:", len(actual))

    # --------------------------------------------------
    # ORIGINAL PREDICTION ANALYSIS
    # --------------------------------------------------

    original_metrics = compute_metrics(
        probabilities,
        actual,
        thresholds
    )

    print("\n========== ORIGINAL MODEL METRICS ==========")
    print(json.dumps(original_metrics, indent=2))

    # Count class-wise false positives and false negatives.
    error_summary = []

    for i, name in enumerate(CLASSES):

        tn, fp, fn, tp = confusion_matrix(
            actual[:, i],
            original_predictions[:, i],
            labels=[0, 1]
        ).ravel()

        error_summary.append({
            "class": name,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        })

    error_df = pd.DataFrame(error_summary)

    print("\n========== CLASS-WISE ERRORS ==========")
    print(error_df.to_string(index=False))

    # --------------------------------------------------
    # NORM-ABNORMAL CONFLICT ANALYSIS
    # --------------------------------------------------

    abnormal_indices = [
        CLASSES.index(name)
        for name in ["MI", "STTC", "CD", "HYP"]
    ]

    predicted_abnormal = (
        original_predictions[:, abnormal_indices].any(axis=1)
    )

    actual_abnormal = (
        actual[:, abnormal_indices].any(axis=1)
    )

    predicted_norm = original_predictions[:, 0] == 1
    actual_norm = actual[:, 0] == 1

    predicted_conflicts = predicted_norm & predicted_abnormal
    actual_conflicts = actual_norm & actual_abnormal

    print("\n========== LABEL CONFLICT ANALYSIS ==========")

    print(
        "Actual NORM + abnormal conflicts:",
        int(actual_conflicts.sum())
    )

    print(
        "Predicted NORM + abnormal conflicts:",
        int(predicted_conflicts.sum())
    )

    print(
        "NORM false positives:",
        int(
            (
                (original_predictions[:, 0] == 1)
                & (actual[:, 0] == 0)
            ).sum()
        )
    )

    # --------------------------------------------------
    # TEST PROPOSED NORM CONSISTENCY RULE
    # --------------------------------------------------

    consistent_predictions = original_predictions.copy()

    consistent_predictions[
        predicted_abnormal, 0
    ] = 0

    consistent_metrics = compute_metrics(
        probabilities,
        actual,
        np.array(thresholds, copy=True)
    )

    # Compute metrics using the modified predictions.
    from sklearn.metrics import (
        f1_score,
        hamming_loss,
        precision_score,
        recall_score
    )

    consistent_metrics["macro_f1"] = float(
        f1_score(
            actual,
            consistent_predictions,
            average="macro",
            zero_division=0
        )
    )

    consistent_metrics["hamming_loss"] = float(
        hamming_loss(actual, consistent_predictions)
    )

    consistent_metrics["subset_accuracy"] = float(
        np.mean(
            np.all(
                consistent_predictions == actual,
                axis=1
            )
        )
    )

    consistent_metrics["per_class_f1"] = dict(
        zip(
            CLASSES,
            map(
                float,
                f1_score(
                    actual,
                    consistent_predictions,
                    average=None,
                    zero_division=0
                )
            )
        )
    )

    consistent_metrics["per_class_precision"] = dict(
        zip(
            CLASSES,
            map(
                float,
                precision_score(
                    actual,
                    consistent_predictions,
                    average=None,
                    zero_division=0
                )
            )
        )
    )

    consistent_metrics["per_class_recall"] = dict(
        zip(
            CLASSES,
            map(
                float,
                recall_score(
                    actual,
                    consistent_predictions,
                    average=None,
                    zero_division=0
                )
            )
        )
    )

    print("\n========== AFTER NORM CONSISTENCY RULE ==========")
    print(json.dumps(consistent_metrics, indent=2))

    print(
        "\nRemaining predicted NORM + abnormal conflicts:",
        int(
            (
                (consistent_predictions[:, 0] == 1)
                & predicted_abnormal
            ).sum()
        )
    )

    # --------------------------------------------------
    # MISCLASSIFIED ECG RECORDINGS
    # --------------------------------------------------

    wrong = np.any(
        original_predictions != actual,
        axis=1
    )

    error_records = []

    for row in np.where(wrong)[0]:

        record = {
            "ecg_id": int(test_ids[row]),
            "actual_labels": [
                CLASSES[i]
                for i in range(len(CLASSES))
                if actual[row, i] == 1
            ],
            "predicted_labels": [
                CLASSES[i]
                for i in range(len(CLASSES))
                if original_predictions[row, i] == 1
            ],
            "probabilities": {
                name: float(probabilities[row, i])
                for i, name in enumerate(CLASSES)
            }
        }

        error_records.append(record)

    error_records.sort(key=lambda item: item["ecg_id"])

    errors_path = OUTPUT_DIR / "misclassified_ecgs.csv"

    error_rows = []

    for item in error_records:
        error_rows.append({
            "ecg_id": item["ecg_id"],
            "actual_labels": ", ".join(item["actual_labels"]),
            "predicted_labels": ", ".join(item["predicted_labels"]),
            **{
                f"{name}_probability": item["probabilities"][name]
                for name in CLASSES
            }
        })

    pd.DataFrame(error_rows).to_csv(
        errors_path,
        index=False
    )

    # --------------------------------------------------
    # CHECK ECG 63
    # --------------------------------------------------

    matches = np.where(test_ids == 63)[0]

    if len(matches):

        row = int(matches[0])

        print("\n========== ECG 63 CHECK ==========")

        print("Actual labels:", actual[row].astype(int).tolist())
        print(
            "Original predictions:",
            original_predictions[row].tolist()
        )
        print(
            "After consistency rule:",
            consistent_predictions[row].tolist()
        )

    else:
        print("\nECG 63 was not found in the test split.")

    # --------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------

    summary = {
        "model": "Multi-Scale Residual CNN",
        "test_recordings": int(len(actual)),
        "thresholds": dict(zip(CLASSES, thresholds.tolist())),
        "original_metrics": original_metrics,
        "norm_consistency_metrics": consistent_metrics,
        "actual_norm_abnormal_conflicts": int(
            actual_conflicts.sum()
        ),
        "predicted_norm_abnormal_conflicts": int(
            predicted_conflicts.sum()
        ),
        "misclassified_recordings": int(wrong.sum())
    }

    summary_path = OUTPUT_DIR / "prediction_error_analysis.json"

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    error_df.to_csv(
        OUTPUT_DIR / "class_error_counts.csv",
        index=False
    )

    print("\n========== ANALYSIS FILES SAVED ==========")
    print("Summary:", summary_path)
    print("Class errors:", OUTPUT_DIR / "class_error_counts.csv")
    print("Misclassified ECGs:", errors_path)

    print("\nERROR ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
