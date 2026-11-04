
"""Train the centralized baseline 1D CNN."""

import os
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from src.dataset import make_loaders
from src.model import ECGNet
from src.evaluate import predict_probs, compute_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_pos_weights(y_train):

    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives

    weights = negatives / np.maximum(positives, 1)

    # Avoid excessively large weights.
    weights = np.clip(weights, 1.0, 10.0)

    return torch.tensor(weights, dtype=torch.float32)


def train(
    processed="data/processed",
    epochs=12,
    batch_size=32,
    learning_rate=0.001,
    patience=4,
    output="results",
    seed=42
):

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training device:", device)

    train_loader, val_loader, test_loader, y_train = make_loaders(
        processed_dir=processed,
        batch_size=batch_size
    )

    model = ECGNet(
        n_leads=12,
        n_classes=5,
        use_lstm=False,
        dropout=0.3
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

    os.makedirs(f"{output}/checkpoints", exist_ok=True)
    os.makedirs(f"{output}/metrics", exist_ok=True)

    checkpoint_path = (
        f"{output}/checkpoints/centralized_best.pt"
    )

    history = []
    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

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
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)

        train_loss = total_loss / total_samples

        # Validation only: test set remains untouched.
        val_probs, val_targets = predict_probs(
            model, val_loader, device
        )

        metrics = compute_metrics(val_probs, val_targets)

        val_auc = metrics["macro_auroc"]

        if val_auc is None:
            raise RuntimeError(
                "Validation macro-AUROC could not be calculated."
            )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUROC: {val_auc:.4f} | "
            f"Val Macro-F1: {metrics['macro_f1']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_metrics": metrics
        })

        if val_auc > best_auc:

            best_auc = val_auc
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_auroc": best_auc,
                    "epoch": epoch,
                    "class_weights": pos_weights.cpu()
                },
                checkpoint_path
            )

            print("Best model checkpoint saved.")

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    with open(
        f"{output}/metrics/baseline_history.json",
        "w"
    ) as file:

        json.dump(history, file, indent=2)

    print("\nBASELINE TRAINING COMPLETED!")
    print("Best validation AUROC:", round(best_auc, 4))
    print("Checkpoint:", checkpoint_path)

    # Test loader is deliberately not evaluated here.
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=4)

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )
