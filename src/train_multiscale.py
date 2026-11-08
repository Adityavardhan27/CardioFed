
"""Train CardioFed Multi-Scale Residual CNN with LR scheduling.

Running with the default arguments reproduces the primary multi-scale model
(results/scheduler_experiment/checkpoints/multiscale_best.pt). Training on CPU
with seed 42 is deterministic: an independent run
(results/diagnostic/checkpoints/multiscale_best.pt) has identical weights.

Refuses to start if the checkpoint or history already exists, unless
--overwrite is given. Use --output to train into a new directory.
"""

import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.dataset import ECGDataset, make_loaders
from src.model import MultiScaleResidualECGNet
from src.evaluate import predict_probs, compute_metrics
from src.io_utils import guard_outputs


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pos_weights(y_train):
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives

    weights = negatives / np.maximum(positives, 1)
    weights = np.clip(weights, 1.0, 10.0)

    return torch.tensor(weights, dtype=torch.float32)


def train(
    epochs=20,
    batch_size=16,
    learning_rate=0.001,
    patience=5,
    output="results/scheduler_experiment",
    seed=42,
    processed="data/processed",
    overwrite=False,
    train_indices=None
):
    """Train the multi-scale model.

    train_indices: optional subset of training-record indices (e.g. one
    federated client's data for local-only baselines). When given, only
    these records are used for training and for the positive-class weights.
    Validation (fold 9) is unchanged. Default None = all folds 1-8.
    """
    guard_outputs(
        [
            f"{output}/checkpoints/multiscale_best.pt",
            f"{output}/metrics/multiscale_history.json",
        ],
        overwrite=overwrite
    )

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)
    print("Model: Multi-Scale Residual ECGNet")
    print("Experiment: Learning-rate scheduling")
    print("Random seed:", seed)

    train_loader, val_loader, test_loader, y_train = make_loaders(
        processed_dir=processed,
        batch_size=batch_size
    )

    if train_indices is not None:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        folds = np.load(f"{processed}/folds.npy")

        if not np.all(folds[train_indices] <= 8):
            raise ValueError("train_indices must come from folds 1-8.")

        train_loader = DataLoader(
            ECGDataset(processed, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        y_train = np.array(
            np.load(f"{processed}/y.npy", mmap_mode="r")[train_indices],
            dtype=np.float32
        )

        print("Training subset records:", len(train_indices))

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(device)

    pos_weights = calculate_pos_weights(y_train).to(device)

    print("Positive class weights:", pos_weights.cpu().numpy())

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Reduce learning rate when validation loss stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6
    )

    os.makedirs(f"{output}/checkpoints", exist_ok=True)
    os.makedirs(f"{output}/metrics", exist_ok=True)

    checkpoint_path = (
        f"{output}/checkpoints/multiscale_best.pt"
    )

    history_path = (
        f"{output}/metrics/multiscale_history.json"
    )

    history = []

    best_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        # -------------------------
        # Training
        # -------------------------
        model.train()

        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)

        train_loss = total_loss / total_samples

        # -------------------------
        # Validation loss
        # -------------------------
        model.eval()

        val_total_loss = 0.0
        val_total_samples = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_total_loss += loss.item() * len(y)
                val_total_samples += len(y)

        val_loss = val_total_loss / val_total_samples

        # -------------------------
        # Validation metrics
        # -------------------------
        val_probs, val_targets = predict_probs(
            model,
            val_loader,
            device
        )

        metrics = compute_metrics(
            val_probs,
            val_targets
        )

        val_auc = metrics["macro_auroc"]

        if val_auc is None or not np.isfinite(val_auc):
            raise RuntimeError(
                "Validation Macro-AUROC is invalid."
            )

        # Update scheduler using validation loss.
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auc:.4f} | "
            f"Val Macro-F1: {metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.6f}",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": current_lr,
            "validation_metrics": metrics
        })

        # -------------------------
        # Best checkpoint selection
        # -------------------------
        if val_auc > best_auc:

            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_auroc": best_auc,
                    "best_val_loss": val_loss,
                    "epoch": epoch,
                    "class_weights": pos_weights.cpu(),
                    "seed": seed
                },
                checkpoint_path
            )

            print("Best checkpoint saved.")

        else:

            epochs_without_improvement += 1

            print(
                "No validation AUROC improvement:",
                epochs_without_improvement,
                "/",
                patience
            )

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    # Save experiment history.
    with open(history_path, "w") as file:
        json.dump(history, file, indent=2)

    # Restore the best validation-AUROC checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nTRAINING COMPLETED!")
    print("Best validation AUROC:", round(best_auc, 4))
    print("Best epoch:", best_epoch)
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)

    # Test set intentionally remains unused.
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--output", default="results/scheduler_experiment")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        output=args.output,
        seed=args.seed,
        processed=args.processed,
        overwrite=args.overwrite
    )
