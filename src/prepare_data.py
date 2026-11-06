
"""Stage 1 of data preparation: filter and normalise PTB-XL signals.

Reads the 100 Hz records, applies a 0.5-40 Hz band-pass filter and per-record,
per-lead z-score normalisation, and stores memory-mapped arrays.

Labels written here still contain NORM together with abnormal classes. The
reported CardioFed results additionally require stage 2:
    python rebuild_labels.py --processed <this --out> --out data/processed

Existing outputs are never overwritten unless --overwrite is given.
"""

import argparse
import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from src.preprocessing import (
    load_metadata,
    multi_hot,
    preprocess_signal,
    CLASSES
)
from src.io_utils import guard_outputs

OUTPUT_FILES = [
    "X.npy", "y.npy", "folds.npy", "sites.npy", "ecg_ids.npy",
    "processed_metadata.csv",
]


def main(root, output, overwrite=False):

    guard_outputs(
        [os.path.join(output, f) for f in OUTPUT_FILES],
        overwrite=overwrite
    )

    print("\nLoading PTB-XL metadata...")

    df = load_metadata(root)

    print("Records with recognised diagnostic labels:", len(df))

    y = multi_hot(df["superclasses"].tolist())

    print("\nDiagnostic class counts:")

    for name, count in zip(CLASSES, y.sum(axis=0)):
        print(f"{name}: {int(count)}")

    # Check potentially conflicting NORM labels.
    norm_idx = CLASSES.index("NORM")
    abnormal_indices = [
        CLASSES.index(c) for c in CLASSES if c != "NORM"
    ]

    conflicts = (
        (y[:, norm_idx] == 1)
        & (y[:, abnormal_indices].sum(axis=1) > 0)
    )

    print("NORM/abnormal label conflicts:", int(conflicts.sum()))

    n = len(df)

    x_path = os.path.join(output, "X.npy")

    print("\nCreating memory-mapped signal array...")

    X = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, 1000, 12)
    )

    print("\nReading and preprocessing ECG recordings...")

    for i, filename in enumerate(
        tqdm(df["filename_lr"].tolist(), desc="ECG records")
    ):

        record_path = os.path.join(root, filename)

        signal, _ = wfdb.rdsamp(record_path)

        X[i] = preprocess_signal(signal)

        if (i + 1) % 1000 == 0:
            X.flush()
            print(f"Processed {i + 1}/{n} records")

    X.flush()
    del X

    print("\nSaving labels and metadata...")

    np.save(os.path.join(output, "y.npy"), y)

    np.save(
        os.path.join(output, "folds.npy"),
        df["strat_fold"].to_numpy(dtype=np.int64)
    )

    sites = (
        df["site"].fillna("").astype(str).to_numpy()
        if "site" in df.columns
        else np.full(n, "", dtype=str)
    )

    np.save(os.path.join(output, "sites.npy"), sites)

    np.save(
        os.path.join(output, "ecg_ids.npy"),
        df.index.to_numpy(dtype=np.int64)
    )

    df[["filename_lr", "strat_fold"]].to_csv(
        os.path.join(output, "processed_metadata.csv"),
        index=True
    )

    print("\nChecking saved arrays...")

    X_check = np.load(x_path, mmap_mode="r")
    y_check = np.load(os.path.join(output, "y.npy"))

    assert X_check.shape == (n, 1000, 12)
    assert y_check.shape == (n, 5)

    print("Signal shape:", X_check.shape)
    print("Label shape:", y_check.shape)
    print("Signal dtype:", X_check.dtype)
    print("Label dtype:", y_check.dtype)

    print("\nFold distribution:")
    print(pd.Series(df["strat_fold"]).value_counts().sort_index())

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="data/raw/ptb-xl"
    )

    parser.add_argument(
        "--out",
        default="data/stage1"
    )

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    main(args.root, args.out, args.overwrite)
