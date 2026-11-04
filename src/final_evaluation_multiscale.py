
"""Final evaluation of CardioFed Multi-Scale Residual CNN."""

import os
import json
import numpy as np
import torch

from sklearn.metrics import precision_recall_curve

from src.dataset import make_loaders
from src.model import MultiScaleResidualECGNet
from src.evaluate import predict_probs, compute_metrics
from src.preprocessing import CLASSES


def tune_thresholds(probs, targets):

    thresholds = []

    for i, name in enumerate(CLASSES):

        precision, recall, values = precision_recall_curve(
            targets[:, i],
            probs[:, i]
        )

        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )

        best_idx = int(np.argmax(f1))
        threshold = float(values[best_idx])

        thresholds.append(threshold)

        print(
            f"{name}: threshold={threshold:.4f}, "
            f"validation F1={f1[best_idx]:.4f}"
        )

    return np.asarray(thresholds, dtype=np.float32)


def main():

    output_dir = "results/metrics"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    train_loader, val_loader, test_loader, _ = make_loaders(
        processed_dir="data/processed",
        batch_size=16
    )

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(device)

    checkpoint = torch.load(
        "results/checkpoints/multiscale_best.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded best multi-scale checkpoint.")
    print("Best validation AUROC:", checkpoint["best_val_auroc"])
    print("Best checkpoint epoch:", checkpoint["epoch"])

    # Tune thresholds using validation data.
    print("\nTuning thresholds on validation set...")

    val_probs, val_targets = predict_probs(
        model, val_loader, device
    )

    thresholds = tune_thresholds(
        val_probs,
        val_targets
    )

    val_metrics = compute_metrics(
        val_probs,
        val_targets,
        thresholds
    )

    # Evaluate test set once.
    print("\nEvaluating held-out test set...")

    test_probs, test_targets = predict_probs(
        model, test_loader, device
    )

    test_metrics = compute_metrics(
        test_probs,
        test_targets,
        thresholds
    )

    results = {
        "model": "Multi-Scale Residual CNN",
        "best_checkpoint_epoch": checkpoint["epoch"],
        "best_validation_auroc": checkpoint["best_val_auroc"],
        "classes": CLASSES,
        "thresholds": dict(
            zip(CLASSES, thresholds.tolist())
        ),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }

    path = os.path.join(
        output_dir,
        "multiscale_final_evaluation.json"
    )

    with open(path, "w") as file:
        json.dump(results, file, indent=2)

    print("\n========== MULTI-SCALE TEST RESULTS ==========")
    print(json.dumps(test_metrics, indent=2))

    print("\nThresholds:", results["thresholds"])
    print("Results saved to:", path)

    print("\nMULTI-SCALE EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
