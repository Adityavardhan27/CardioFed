
"""Final evaluation of the CardioFed FedAvg model."""

import os
import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve

from src.model import MultiScaleResidualECGNet
from src.dataset import ECGDataset
from src.evaluate import predict_probs, compute_metrics
from src.preprocessing import CLASSES


def tune_thresholds(probabilities, targets):
    """Tune one classification threshold per class on validation data."""

    thresholds = []

    for class_index in range(len(CLASSES)):
        precision, recall, threshold_values = precision_recall_curve(
            targets[:, class_index],
            probabilities[:, class_index]
        )

        if len(threshold_values) == 0:
            thresholds.append(0.5)
            continue

        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )

        best_index = np.argmax(f1_scores)
        thresholds.append(float(threshold_values[best_index]))

    return np.array(thresholds)


def main():
    os.makedirs("results/metrics", exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    folds = np.load("data/processed/folds.npy")
    y = np.load("data/processed/y.npy")

    val_indices = np.where(folds == 9)[0]
    test_indices = np.where(folds == 10)[0]

    print("Validation records:", len(val_indices))
    print("Test records:", len(test_indices))

    val_dataset = ECGDataset(
        "data/processed", val_indices
    )

    test_dataset = ECGDataset(
        "data/processed", test_indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(device)

    checkpoint = torch.load(
        "results/checkpoints/federated_best.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nLoaded federated checkpoint.")
    print("Best communication round:", checkpoint["round"])
    print(
        "Best validation AUROC:",
        round(checkpoint["best_val_auroc"], 4)
    )

    # Tune thresholds using validation data only.
    print("\nGenerating validation predictions...")

    val_probabilities, _ = predict_probs(
        model, val_loader, device
    )
    val_targets = y[val_indices]

    thresholds = tune_thresholds(
        val_probabilities,
        val_targets
    )

    print("\nValidation-tuned thresholds:")

    for name, threshold in zip(CLASSES, thresholds):
        print(f"{name}: {threshold:.4f}")

    # Evaluate held-out test data once.
    print("\nEvaluating held-out test set...")

    test_probabilities, _ = predict_probs(
        model, test_loader, device
    )

    test_targets = y[test_indices]


    metrics = compute_metrics(
        test_probabilities,
        test_targets,
        thresholds=thresholds
    )

    results = {
        "model": "Federated Multi-Scale Residual CNN (FedAvg)",
        "best_communication_round": int(checkpoint["round"]),
        "validation_macro_auroc": float(
            checkpoint["best_val_auroc"]
        ),
        "test_records": int(len(test_indices)),
        "thresholds": {
            name: float(value)
            for name, value in zip(CLASSES, thresholds)
        },
        "metrics": metrics
    }

    output_path = (
        "results/metrics/federated_final_evaluation.json"
    )

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 55)
    print("FEDERATED FINAL TEST RESULTS")
    print("=" * 55)

    for key, value in metrics.items():
        print(f"\n{key}:")

        if isinstance(value, dict):
            for class_name, score in value.items():
                print(f"  {class_name}: {score}")
        else:
            print(f"  {value}")

    print("\nThresholds:", results["thresholds"])
    print("\nSaved:", output_path)
    print("\nFEDERATED FINAL EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
