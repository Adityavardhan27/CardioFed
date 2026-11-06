"""Final evaluation of a centralized Multi-Scale Residual CNN checkpoint.

Defaults reproduce the LEGACY artifact
results/metrics/multiscale_final_evaluation.json, evaluated from
results/checkpoints/multiscale_best.pt. That checkpoint was trained by an
earlier version of src/train_multiscale.py that is no longer in the
repository. The primary (reproducible) multi-scale model is evaluated by
src/evaluate_scheduler_experiment.py.

Pass --checkpoint/--output to evaluate another multi-scale checkpoint. Existing
output files are never overwritten unless --overwrite is given.
"""

import argparse
import json

import torch

from src.dataset import make_loaders
from src.model import MultiScaleResidualECGNet
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

    train_loader, val_loader, test_loader, _ = make_loaders(
        processed_dir=args.processed,
        batch_size=16
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

    print("Loaded multi-scale checkpoint:", args.checkpoint)
    print("Best validation AUROC:", checkpoint["best_val_auroc"])
    print("Best checkpoint epoch:", checkpoint["epoch"])

    # Tune thresholds using validation data.
    print("\nTuning thresholds on validation set...")

    val_probs, val_targets = predict_probs(
        model, val_loader, device
    )

    thresholds = tune_thresholds(val_probs, val_targets, verbose=True)

    # Optimistic: thresholds were tuned on this same validation data.
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

    with open(args.output, "w") as file:
        json.dump(results, file, indent=2)

    print("\n========== MULTI-SCALE TEST RESULTS ==========")
    print(json.dumps(test_metrics, indent=2))

    print("Results saved to:", args.output)

    print("\nMULTI-SCALE EVALUATION COMPLETED!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument(
        "--checkpoint", default="results/checkpoints/multiscale_best.pt"
    )
    parser.add_argument(
        "--output", default="results/metrics/multiscale_final_evaluation.json"
    )
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())
