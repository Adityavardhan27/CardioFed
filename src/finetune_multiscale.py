import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from src.dataset import make_loaders
from src.model import MultiScaleResidualECGNet
from src.evaluate import predict_probs, compute_metrics
from src.io_utils import guard_outputs


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validation_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            total_samples += len(y)

    return total_loss / total_samples


def save_history(history, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def train_finetuning(
    checkpoint_path="results/scheduler_experiment/checkpoints/multiscale_best.pt",
    processed="data/processed",
    output="results/fine_tuning_v1",
    epochs=15,
    batch_size=16,
    learning_rate=0.0001,
    weight_decay=0.0001,
    patience=5,
    seed=42,
):
    checkpoint_output = f"{output}/checkpoints/finetuned_best.pt"
    history_output = f"{output}/metrics/finetuning_history.json"

    guard_outputs([checkpoint_output, history_output])

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Fine-tuning from:", checkpoint_path)
    print("Output:", output)

    train_loader, val_loader, _, y_train = make_loaders(
        processed_dir=processed,
        batch_size=batch_size
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "class_weights" not in checkpoint:
        raise KeyError("Original checkpoint has no class_weights.")

    class_weights = torch.as_tensor(
        checkpoint["class_weights"],
        dtype=torch.float32,
        device=device
    )

    if class_weights.numel() != 5:
        raise ValueError("Expected five class weights.")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights
    )

    # Fresh optimizer, initialized with a deliberately small learning rate.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

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

    # Evaluate the original checkpoint on validation data first.
    initial_loss = validation_loss(
        model, val_loader, criterion, device
    )

    initial_probs, initial_targets = predict_probs(
        model, val_loader, device
    )

    initial_metrics = compute_metrics(
        initial_probs, initial_targets
    )

    best_auc = initial_metrics["macro_auroc"]

    if best_auc is None or not np.isfinite(best_auc):
        raise RuntimeError("Initial validation AUROC is invalid.")

    print("\nStarting checkpoint validation:")
    print(f"Validation loss: {initial_loss:.4f}")
    print(f"Validation AUROC: {best_auc:.4f}")
    print(f"Validation Macro-F1: {initial_metrics['macro_f1']:.4f}")

    # Preserve the starting checkpoint as the initial best candidate.
    best_epoch = 0
    epochs_without_improvement = 0

    def save_best(epoch, val_loss, val_auc):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_auroc": float(val_auc),
                "best_val_loss": float(val_loss),
                "epoch": epoch,
                "starting_checkpoint": checkpoint_path,
                "original_checkpoint_epoch": checkpoint.get("epoch"),
                "class_weights": class_weights.cpu(),
                "seed": seed,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            checkpoint_output
        )

    save_best(0, initial_loss, best_auc)

    history = [{
        "epoch": 0,
        "train_loss": None,
        "validation_loss": float(initial_loss),
        "validation_auroc": float(best_auc),
        "validation_macro_f1": float(initial_metrics["macro_f1"]),
        "learning_rate": float(learning_rate),
        "note": "Starting checkpoint"
    }]

    save_history(history, history_output)

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
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)

        train_loss = total_loss / total_samples

        val_loss = validation_loss(
            model, val_loader, criterion, device
        )

        val_probs, val_targets = predict_probs(
            model, val_loader, device
        )

        metrics = compute_metrics(
            val_probs, val_targets
        )

        val_auc = metrics["macro_auroc"]

        if val_auc is None or not np.isfinite(val_auc):
            raise RuntimeError("Validation AUROC is invalid.")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auc:.4f} | "
            f"Val Macro-F1: {metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.7f}",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss),
            "validation_auroc": float(val_auc),
            "validation_macro_f1": float(metrics["macro_f1"]),
            "learning_rate": float(current_lr)
        })

        save_history(history, history_output)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            save_best(epoch, val_loss, val_auc)
            print("New best checkpoint saved.")
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

    print("\nFINE-TUNING COMPLETED")
    print("Best epoch:", best_epoch)
    print("Best validation AUROC:", round(best_auc, 4))
    print("Checkpoint:", checkpoint_output)
    print("History:", history_output)
    print("Test set was not used.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--output", default="results/fine_tuning_v1")
    parser.add_argument(
        "--checkpoint",
        default="results/scheduler_experiment/checkpoints/multiscale_best.pt"
    )

    args = parser.parse_args()

    train_finetuning(
        checkpoint_path=args.checkpoint,
        processed=args.processed,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed
    )
