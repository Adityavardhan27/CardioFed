"""Final baseline evaluation with validation-tuned thresholds.

Thresholds are tuned on the validation fold (9); the test fold (10) is
evaluated once with those thresholds.

By default this reproduces results/metrics/baseline_final_evaluation.json.
Because that file already exists, the script refuses to run unless you pass
a new --output path (recommended) or --overwrite.
"""

import argparse
import json

import torch

from src.dataset import make_loaders
from src.model import ECGNet
from src.evaluate import predict_probs, compute_metrics
from src.preprocessing import CLASSES
from src.thresholds import tune_thresholds
from src.io_utils import guard_outputs


def main(args):

    guard_outputs([args.output], overwrite=args.overwrite)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    train_loader, val_loader, test_loader, y_train = make_loaders(
        processed_dir=args.processed,
        batch_size=16
    )

    model = ECGNet(
        n_leads=12,
        n_classes=5,
        use_lstm=False,
        dropout=0.3
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nLoaded best baseline checkpoint:", args.checkpoint)
    print("Best validation AUROC:", checkpoint["best_val_auroc"])

    # Step 1: Tune thresholds using validation data only.
    print("\nTuning thresholds on validation set...")

    val_probs, val_targets = predict_probs(
        model, val_loader, device
    )

    thresholds = tune_thresholds(val_probs, val_targets, verbose=True)

    # Step 2: Validation metrics. NOTE: optimistic, because the thresholds
    # were tuned on this same validation data.
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

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========== FINAL TEST RESULTS ==========")

    print(json.dumps(test_metrics, indent=2))

    print("\nThresholds:", results["thresholds"])

    print("\nResults saved to:", args.output)

    print("\nBASELINE EVALUATION COMPLETED!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument(
        "--checkpoint", default="results/checkpoints/centralized_best.pt"
    )
    parser.add_argument(
        "--output", default="results/metrics/baseline_final_evaluation.json"
    )
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())
