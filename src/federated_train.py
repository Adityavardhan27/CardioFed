
"""CardioFed: Federated training using FedAvg.

Four simulated clients are formed using PTB-XL recording sites.
All ECG records belonging to one patient are assigned to one client.

Simulation assumptions (state these when reporting results):
  * Positive-class weights are computed from the pooled training labels of
    all clients (a shared aggregate statistic).
  * Model selection and threshold tuning use one central validation set
    (fold 9).
  * BatchNorm statistics are averaged by FedAvg like all other buffers.
  * Each client creates a fresh Adam optimiser every round.
  * No differential privacy or secure aggregation is implemented.

Reported runs (rounds 3 and 8) were produced before outputs were named per
run; they live at results/checkpoints/federated_3round_best.pt and
results/checkpoints/federated_best.pt. New runs write
<output>/checkpoints/<run-name>_best.pt and refuse to overwrite existing files
unless --overwrite is given.
"""

import os
import json
import copy
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.model import MultiScaleResidualECGNet
from src.dataset import ECGDataset
from src.preprocessing import CLASSES
from src.io_utils import guard_outputs


# ---------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# Patient-level client partitioning
# ---------------------------------------------------------

def create_client_partitions(
    metadata_path="data/raw/ptb-xl/ptbxl_database.csv",
    processed="data/processed"
):
    """
    Assign each training patient to one client.

    Clients:
        0: Site 0 dominant
        1: Site 1 dominant
        2: Site 2 dominant
        3: All remaining sites

    If a patient has records from multiple sites, all records
    are assigned according to that patient's most frequent site.
    """

    metadata = pd.read_csv(metadata_path)

    ecg_ids = np.load(f"{processed}/ecg_ids.npy")
    folds = np.load(f"{processed}/folds.npy")

    processed = metadata.set_index("ecg_id").loc[ecg_ids].copy()

    processed["processed_index"] = np.arange(len(ecg_ids))
    processed["fold"] = folds

    # Only folds 1-8 are used for federated client training.
    train_data = processed[processed["fold"] <= 8].copy()

    # Determine each patient's dominant recording site.
    patient_sites = (
        train_data.groupby(["patient_id", "site"])
        .size()
        .reset_index(name="records")
    )

    dominant_sites = (
        patient_sites.sort_values(
            ["patient_id", "records", "site"],
            ascending=[True, False, True]
        )
        .drop_duplicates("patient_id")
        .set_index("patient_id")["site"]
    )

    train_data["dominant_site"] = train_data["patient_id"].map(
        dominant_sites
    )

    def assign_client(site):
        if site == 0:
            return 0
        elif site == 1:
            return 1
        elif site == 2:
            return 2
        return 3

    train_data["client"] = train_data["dominant_site"].apply(
        assign_client
    )

    partitions = {}

    for client_id in range(4):
        indices = train_data.loc[
            train_data["client"] == client_id,
            "processed_index"
        ].to_numpy(dtype=int)

        partitions[client_id] = indices

    # Verify that each training patient belongs to only one client.
    patient_client_counts = train_data.groupby(
        "patient_id"
    )["client"].nunique()

    if patient_client_counts.max() > 1:
        raise RuntimeError(
            "Patient leakage detected between clients."
        )

    # Verify that every training record has been assigned.
    assigned = np.concatenate(list(partitions.values()))

    if len(assigned) != len(train_data):
        raise RuntimeError(
            "Some training records were not assigned to clients."
        )

    if len(np.unique(assigned)) != len(assigned):
        raise RuntimeError(
            "Duplicate ECG records found across clients."
        )

    print("\nCLIENT PARTITION SUMMARY")

    for client_id, indices in partitions.items():
        client_rows = train_data[
            train_data["processed_index"].isin(indices)
        ]

        print(
            f"Client {client_id + 1}: "
            f"{len(indices)} records | "
            f"{client_rows['patient_id'].nunique()} patients"
        )

    print(
        "\nPatients assigned to multiple clients:",
        int((patient_client_counts > 1).sum())
    )

    return partitions


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def evaluate_validation(model, val_loader, device):
    model.eval()

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for signals, targets in val_loader:
            signals = signals.to(device)

            logits = model(signals)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )
            all_targets.append(targets.numpy())

    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets)

    class_aucs = []

    for class_index in range(len(CLASSES)):
        if len(np.unique(targets[:, class_index])) < 2:
            continue

        auc = roc_auc_score(
            targets[:, class_index],
            probabilities[:, class_index]
        )
        class_aucs.append(auc)

    return float(np.mean(class_aucs))


# ---------------------------------------------------------
# Local client training
# ---------------------------------------------------------

def train_client(
    model,
    loader,
    criterion,
    device,
    local_epochs=1,
    learning_rate=0.0005
):
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    total_loss = 0.0
    total_samples = 0

    for epoch in range(local_epochs):
        progress = tqdm(
            loader,
            desc=f"Local epoch {epoch + 1}",
            leave=False
        )

        for signals, targets in progress:
            signals = signals.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(signals)
            loss = criterion(logits, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            batch_size = signals.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------
# Federated averaging
# ---------------------------------------------------------

def fed_avg(client_states, client_sizes):
    """
    Weighted average of client model parameters.

    Weight of each client = its number of training records
                            / total training records.
    """

    total_size = sum(client_sizes)

    global_state = copy.deepcopy(client_states[0])

    for key in global_state:
        if torch.is_floating_point(global_state[key]):
            global_state[key] = sum(
                client_states[i][key] *
                (client_sizes[i] / total_size)
                for i in range(len(client_states))
            )
        else:
            global_state[key] = client_states[0][key]

    return global_state


# ---------------------------------------------------------
# Main federated training
# ---------------------------------------------------------

def main(args):
    set_seed(args.seed)

    torch.set_num_threads(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Classes:", CLASSES)

    run_name = args.run_name or f"federated_{args.rounds}round"
    checkpoint_path = f"{args.output}/checkpoints/{run_name}_best.pt"
    history_path = f"{args.output}/metrics/{run_name}_history.json"
    summary_path = f"{args.output}/metrics/{run_name}_summary.json"

    guard_outputs(
        [checkpoint_path, history_path, summary_path],
        overwrite=args.overwrite
    )

    print("Run name:", run_name)

    # Build patient-disjoint clients.
    partitions = create_client_partitions(args.metadata, args.processed)

    # Load validation data: fold 9 only.
    folds = np.load(f"{args.processed}/folds.npy")
    val_indices = np.where(folds == 9)[0]

    val_dataset = ECGDataset(
        args.processed,
        val_indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Compute positive class weights using training data only.
    y = np.load(f"{args.processed}/y.npy")

    train_indices = np.where(folds <= 8)[0]
    y_train = y[train_indices]

    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives

    pos_weights = negatives / np.maximum(positives, 1)
    pos_weights = np.clip(pos_weights, 1.0, 10.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weights,
            dtype=torch.float32,
            device=device
        )
    )

    print("\nPositive class weights:", pos_weights)

    # Build client data loaders.
    client_loaders = {}
    client_sizes = {}

    for client_id, indices in partitions.items():
        dataset = ECGDataset(
            args.processed,
            indices
        )

        client_loaders[client_id] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

        client_sizes[client_id] = len(dataset)

    # Initialize the global Multi-Scale Residual CNN.
    global_model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(device)

    history = []
    best_val_auc = -1.0

    print("\nStarting Federated Averaging")
    print("Communication rounds:", args.rounds)
    print("Local epochs per client:", args.local_epochs)

    # -----------------------------------------------------
    # Communication rounds
    # -----------------------------------------------------

    for round_number in range(args.rounds):
        print(
            f"\n{'=' * 55}\n"
            f"FEDERATED ROUND {round_number + 1}/{args.rounds}\n"
            f"{'=' * 55}"
        )

        client_states = []
        client_losses = []

        # Each client receives the current global model.
        for client_id in range(4):
            print(f"\nTraining Client {client_id + 1}")

            local_model = MultiScaleResidualECGNet(
                n_leads=12,
                n_classes=5,
                dropout=0.35
            ).to(device)

            local_model.load_state_dict(
                global_model.state_dict()
            )

            local_loss = train_client(
                model=local_model,
                loader=client_loaders[client_id],
                criterion=criterion,
                device=device,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate
            )

            print(
                f"Client {client_id + 1} loss: "
                f"{local_loss:.4f}"
            )

            client_losses.append(local_loss)

            client_states.append({
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            })

            del local_model

        # Server aggregates client weights.
        print("\nServer: Performing FedAvg aggregation...")

        aggregated_state = fed_avg(
            client_states,
            [
                client_sizes[i]
                for i in range(4)
            ]
        )

        global_model.load_state_dict(aggregated_state)
        global_model.to(device)

        # Evaluate global model on centralized validation data.
        val_auc = evaluate_validation(
            global_model,
            val_loader,
            device
        )

        average_loss = float(np.mean(client_losses))

        print(f"\nRound {round_number + 1} results:")
        print(f"Average client loss: {average_loss:.4f}")
        print(f"Validation macro-AUROC: {val_auc:.4f}")

        history.append({
            "round": round_number + 1,
            "average_client_loss": average_loss,
            "validation_macro_auroc": val_auc,
            "client_losses": client_losses
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc

            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "round": round_number + 1,
                    "best_val_auroc": best_val_auc,
                    "classes": CLASSES
                },
                checkpoint_path
            )

            print("Best federated model checkpoint saved.")

        del client_states

    # Save training history.
    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)

    summary = {
        "clients": 4,
        "communication_rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "client_sizes": {
            str(key + 1): value
            for key, value in client_sizes.items()
        },
        "best_validation_macro_auroc": best_val_auc,
        "best_checkpoint": checkpoint_path
    }

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 55)
    print("Best validation macro-AUROC:", round(best_val_auc, 4))
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument(
        "--metadata", default="data/raw/ptb-xl/ptbxl_database.csv"
    )
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--run-name", default=None,
        help="Output name prefix (default: federated_<rounds>round)."
    )
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())
