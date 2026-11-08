"""P2: local-only baselines (one model per federated client, no federation).

Each client trains the multi-scale CNN on its own training records only,
using exactly the client partition of src/federated_train.py and the
centralized multi-scale recipe of src/train_multiscale.py (Adam lr 1e-3,
batch 16, ReduceLROnPlateau on validation loss, early stopping on validation
macro-AUROC with patience 5, at most 20 epochs). Positive-class weights are
computed from the client's own labels.

For comparability with the FedAvg simulation, model selection uses the same
central validation fold (9). A real isolated site would only have its own
validation data; this is a simulation assumption (see results/p2/PROTOCOL.md).

Output: results/p2/local/client<k>_seed<s>/{checkpoints,metrics}/
Refuses to overwrite existing outputs unless --overwrite is given.
"""

import argparse
import io
import contextlib

from src.federated_train import create_client_partitions
from src.train_multiscale import train


def main(args):
    with contextlib.redirect_stdout(io.StringIO()):
        partitions = create_client_partitions(args.metadata, args.processed)

    indices = partitions[args.client - 1]
    output = args.output or f"results/p2/local/client{args.client}_seed{args.seed}"

    print(f"Local-only training: client {args.client}, "
          f"{len(indices)} records, seed {args.seed}, output {output}")

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        output=output,
        seed=args.seed,
        processed=args.processed,
        overwrite=args.overwrite,
        train_indices=indices,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--metadata", default="data/raw/ptb-xl/ptbxl_database.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
