"""Stage 2 of data preparation: apply the CardioFed label policy.

Run after `python -m src.prepare_data`. Every reported CardioFed result uses
labels produced by this script (verified: data/processed/y.npy is reproduced
exactly).

Label policy (differs from the standard PTB-XL superclass benchmark):
  1. Only SCP diagnostic statements with likelihood > 0 are used.
  2. NORM is exclusive: if any abnormal superclass (MI, STTC, CD, HYP) is
     present, NORM is removed for that record.
  3. Records left without any label are removed (signals and metadata are
     filtered consistently).

Usage (recommended, non-destructive):
    python -m src.prepare_data --root data/raw/ptb-xl --out data/stage1
    python rebuild_labels.py --processed data/stage1 --out data/processed

By default the result is written to a separate --out directory. Rewriting the
input directory in place (the original behaviour, which replaces X.npy)
requires the explicit --in-place flag. Existing files in --out are never
overwritten unless --overwrite is given.
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

from src.preprocessing import CLASSES
from src.io_utils import guard_outputs

OUTPUT_FILES = [
    "X.npy", "y.npy", "folds.npy", "sites.npy", "ecg_ids.npy",
    "processed_metadata.csv",
]


def build_labels(df, scp, ecg_ids):
    """Return (y, df_aligned) using the CardioFed label policy."""

    diagnostic = scp[
        (pd.to_numeric(scp["diagnostic"], errors="coerce") == 1)
        & scp["diagnostic_class"].isin(CLASSES)
    ]

    mapping = diagnostic["diagnostic_class"].to_dict()

    # Preserve the exact original signal order.
    df = df.loc[ecg_ids]

    y = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    class_index = {name: i for i, name in enumerate(CLASSES)}

    for i, codes in enumerate(df["scp_codes"]):

        labels = {
            mapping[code]
            for code, likelihood in codes.items()
            if code in mapping and likelihood > 0
        }

        # NORM is exclusive: abnormal diagnoses take precedence.
        abnormal = labels - {"NORM"}

        if abnormal:
            labels = abnormal

        for label in labels:
            y[i, class_index[label]] = 1.0

    return y, df


def main(root, processed, out, in_place=False, overwrite=False):

    processed = os.path.normpath(processed)
    out = os.path.normpath(out if out is not None else processed)
    same_dir = os.path.abspath(out) == os.path.abspath(processed)

    if same_dir and not in_place:
        raise SystemExit(
            "Refusing to rewrite the input directory in place. Pass a "
            "different --out directory (recommended) or --in-place."
        )

    if not same_dir:
        guard_outputs(
            [os.path.join(out, f) for f in OUTPUT_FILES],
            overwrite=overwrite,
        )

    print("Loading metadata...")

    df = pd.read_csv(
        os.path.join(root, "ptbxl_database.csv"),
        index_col="ecg_id"
    )

    scp = pd.read_csv(
        os.path.join(root, "scp_statements.csv"),
        index_col=0
    )

    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    ecg_ids = np.load(os.path.join(processed, "ecg_ids.npy"))

    y, df = build_labels(df, scp, ecg_ids)

    print("Aligned records:", len(df))

    # Remove records with no usable target.
    keep = y.sum(axis=1) > 0
    removed = int((~keep).sum())

    print("Records without usable labels:", removed)

    if removed == len(df):
        raise ValueError("No usable records remain.")

    # Open original signals without loading everything into RAM.
    X = np.load(os.path.join(processed, "X.npy"), mmap_mode="r")

    if X.shape[0] != len(df):
        raise ValueError("Signal and metadata alignment error.")

    new_n = int(keep.sum())
    final_x_path = os.path.join(out, "X.npy")
    write_path = (
        os.path.join(out, "X_clean.npy") if same_dir else final_x_path
    )

    print("\nCreating cleaned signal array...")

    X_clean = np.lib.format.open_memmap(
        write_path,
        mode="w+",
        dtype=np.float32,
        shape=(new_n, *X.shape[1:])
    )

    # Copy only valid records in small chunks.
    write_pos = 0
    chunk_size = 256

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        local_keep = keep[start:end]

        count = int(local_keep.sum())

        if count:
            X_clean[write_pos:write_pos + count] = X[start:end][local_keep]
            write_pos += count

        if write_pos and write_pos % 2000 < chunk_size:
            print(f"Copied {write_pos}/{new_n} signals")

    X_clean.flush()
    del X_clean
    del X

    if same_dir:
        # Original in-place behaviour: replace the old array.
        os.replace(write_path, final_x_path)

    # Keep labels and all metadata aligned.
    y = y[keep]
    df = df.iloc[np.flatnonzero(keep)].copy()
    ecg_ids = ecg_ids[keep]

    np.save(os.path.join(out, "y.npy"), y)

    np.save(
        os.path.join(out, "folds.npy"),
        df["strat_fold"].to_numpy(dtype=np.int64)
    )

    sites = (
        df["site"].astype("string").fillna("").astype(str).to_numpy()
        if "site" in df.columns
        else np.full(len(df), "", dtype=str)
    )

    np.save(os.path.join(out, "sites.npy"), sites)
    np.save(os.path.join(out, "ecg_ids.npy"), ecg_ids)

    df[["filename_lr", "strat_fold"]].to_csv(
        os.path.join(out, "processed_metadata.csv"),
        index=True
    )

    # Final validation.
    X_check = np.load(final_x_path, mmap_mode="r")

    y_check = np.load(os.path.join(out, "y.npy"))
    folds = np.load(os.path.join(out, "folds.npy"))
    ids = np.load(os.path.join(out, "ecg_ids.npy"))

    assert X_check.shape[0] == len(y_check)
    assert len(y_check) == len(folds) == len(ids)
    assert X_check.shape[1:] == (1000, 12)
    assert y_check.shape[1] == 5
    assert np.all(y_check.sum(axis=1) > 0)

    print("\nFINAL CLASS COUNTS:")

    for name, count in zip(CLASSES, y_check.sum(axis=0)):
        print(f"{name}: {int(count)}")

    print("\nRemoved records:", removed)
    print("Final signal shape:", X_check.shape)
    print("Final label shape:", y_check.shape)
    print("Final fold counts:")
    print(pd.Series(folds).value_counts().sort_index())

    print("\nDATASET LABELS AND ALIGNMENT VERIFIED!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default="data/raw/ptb-xl")
    parser.add_argument(
        "--processed", required=True,
        help="Directory produced by src.prepare_data (input)."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (must differ from --processed unless --in-place)."
    )
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    main(args.root, args.processed, args.out, args.in_place, args.overwrite)
