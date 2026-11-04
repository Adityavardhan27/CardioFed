
"""Final baseline evaluation with validation-tuned thresholds."""

import os
import json
import numpy as np
import torch

from sklearn.metrics import precision_recall_curve

from src.dataset import make_loaders
from src.model import ECGNet
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
        best_threshold = float(values[best_idx])

        thresholds.append(best_threshold)

        print(
            f"{name}: threshold={best_threshold:.4f}, "
            f"validation F1={f1[best_idx]:.4f}"
        )

    return np.asarray(thresholds, dtype=np.float32)


def main():

    processed = "data/processed"
    checkpoint_path = "results/checkpoints/centralized_best.pt"
    output_dir = "results/metrics"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    train_loader, val_loader, test_loader, y_train = make_loaders(
        processed_dir=processed,
        batch_size=16
    )

    model = ECGNet(
        n_leads=12,
        n_classes=5,
        use_lstm=False,
        dropout=0.3
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nLoaded best baseline checkpoint.")
    print("Best validation AUROC:", checkpoint["best_val_auroc"])

    # Step 1: Tune thresholds using validation data only.
    print("\nTuning thresholds on validation set...")

    val_probs, val_targets = predict_probs(
        model, val_loader, device
    )

    thresholds = tune_thresholds(val_probs, val_targets)

    # Step 2: Evaluate validation metrics with tuned thresholds.
    val_metrics = compute_metrics(
        val_probs,
        val_targets,
        thresholds
    )

    # Step 3: Evaluate the untouched test set.
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
        "model": "Baseline 1D CNN",
        "best_checkpoint_epoch": checkpoint["epoch"],
        "best_validation_auroc": checkpoint["best_val_auroc"],
        "classes": CLASSES,
        "thresholds": dict(zip(CLASSES, thresholds.tolist())),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }

    output_path = os.path.join(
        output_dir,
        "baseline_final_evaluation.json"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========== FINAL TEST RESULTS ==========")

    print(json.dumps(test_metrics, indent=2))

    print("\nThresholds:", results["thresholds"])

    print("\nResults saved to:", output_path)

    print("\nBASELINE EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
