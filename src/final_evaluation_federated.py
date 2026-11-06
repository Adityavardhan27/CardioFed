
"""Final evaluation of a CardioFed FedAvg checkpoint.

Defaults reproduce results/metrics/federated_final_evaluation.json (8 rounds).
For the 3-round model use:
    --checkpoint results/checkpoints/federated_3round_best.pt
    --output <new path>
Existing outputs are never overwritten unless --overwrite is given.
"""

import argparse
import json
import numpy as np
import torch

from torch.utils.data import DataLoader

from src.model import MultiScaleResidualECGNet
from src.dataset import ECGDataset
from src.evaluate import predict_probs, compute_metrics
from src.preprocessing import CLASSES
from src.thresholds import tune_thresholds
from src.io_utils import guard_outputs


def main(args):
    guard_outputs([args.output], overwrite=args.overwrite)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    folds = np.load(f"{args.processed}/folds.npy")
    y = np.load(f"{args.processed}/y.npy")

    val_indices = np.where(folds == 9)[0]
    test_indices = np.where(folds == 10)[0]

    print("Validation records:", len(val_indices))
    print("Test records:", len(test_indices))

    val_dataset = ECGDataset(
        args.processed, val_indices
    )

    test_dataset = ECGDataset(
        args.processed, test_indices
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
        args.checkpoint,
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

    output_path = args.output

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument(
        "--checkpoint", default="results/checkpoints/federated_best.pt"
    )
    parser.add_argument(
        "--output", default="results/metrics/federated_final_evaluation.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
