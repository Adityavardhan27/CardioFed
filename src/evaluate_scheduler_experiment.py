
"""Evaluate CardioFed scheduler experiment."""

import os
import json

import numpy as np
import torch

from sklearn.metrics import precision_recall_curve

from src.dataset import make_loaders
from src.model import MultiScaleResidualECGNet
from src.evaluate import predict_probs, compute_metrics
from src.preprocessing import CLASSES


CHECKPOINT_PATH = (
    "results/scheduler_experiment/checkpoints/multiscale_best.pt"
)

OUTPUT_PATH = (
    "results/scheduler_experiment/metrics/"
    "multiscale_final_evaluation.json"
)


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

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

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
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nLoaded scheduler experiment checkpoint.")
    print("Best validation AUROC:", checkpoint["best_val_auroc"])
    print("Best checkpoint epoch:", checkpoint["epoch"])

    # ---------------------------------
    # Validation: tune thresholds
    # ---------------------------------

    print("\nCollecting validation predictions...")

    val_probs, val_targets = predict_probs(
        model,
        val_loader,
        device
    )

    if not np.isfinite(val_probs).all():
        raise RuntimeError("Non-finite validation probabilities found.")

    print("\nTuning thresholds on validation data...")

    thresholds = tune_thresholds(
        val_probs,
        val_targets
    )

    val_metrics = compute_metrics(
        val_probs,
        val_targets,
        thresholds
    )

    # ---------------------------------
    # Test: evaluate once
    # ---------------------------------

    print("\nEvaluating held-out test data...")

    test_probs, test_targets = predict_probs(
        model,
        test_loader,
        device
    )

    if not np.isfinite(test_probs).all():
        raise RuntimeError("Non-finite test probabilities found.")

    test_metrics = compute_metrics(
        test_probs,
        test_targets,
        thresholds
    )

    results = {
        "experiment": "Learning-rate scheduling",
        "model": "Multi-Scale Residual CNN",
        "checkpoint_path": CHECKPOINT_PATH,
        "best_checkpoint_epoch": checkpoint["epoch"],
        "best_validation_auroc": checkpoint["best_val_auroc"],
        "classes": list(CLASSES),
        "threshold_selection": "Validation set, maximum per-class F1",
        "thresholds": dict(
            zip(CLASSES, thresholds.tolist())
        ),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }

    with open(OUTPUT_PATH, "w") as file:
        json.dump(results, file, indent=2)

    print("\n========== SCHEDULER EXPERIMENT TEST RESULTS ==========")
    print(json.dumps(test_metrics, indent=2))

    print("\nSelected thresholds:")
    print(json.dumps(results["thresholds"], indent=2))

    print("\nResults saved to:", OUTPUT_PATH)
    print("\nSCHEDULER EXPERIMENT EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
