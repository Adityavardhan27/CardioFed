"""P2 / E4: alternative normalisation that preserves relative amplitude.

The standard pipeline (src/preprocessing.py) z-scores every lead of every
record individually, which removes absolute voltage. This script builds an
alternative dataset:
  1. same 0.5-40 Hz band-pass filter (order 3, zero-phase) as the standard
     pipeline;
  2. per-lead mean and standard deviation computed on TRAINING folds 1-8
     only, then applied to every record (train, validation, test).

Record order, labels, folds, IDs and sites are copied from the reference
processed directory, so splits and labels are identical to the main
experiments. The normalisation statistics are saved in norm_stats.json.

Output: data/p2_global_norm/ (refuses to overwrite existing files).
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from src.io_utils import guard_outputs
from src.preprocessing import FS

COPIED = ["y.npy", "folds.npy", "ecg_ids.npy", "sites.npy", "processed_metadata.csv"]


def bandpass(signal):
    """Identical filter to src.preprocessing.preprocess_signal, without z-score."""
    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, signal.astype(np.float64), axis=0)


def main(args):
    out = args.out
    guard_outputs([os.path.join(out, f) for f in COPIED + ["X.npy", "norm_stats.json"]],
                  overwrite=args.overwrite)

    ids = np.load(os.path.join(args.reference, "ecg_ids.npy"))
    folds = np.load(os.path.join(args.reference, "folds.npy"))
    meta = pd.read_csv(os.path.join(args.root, "ptbxl_database.csv"), index_col="ecg_id")
    files = meta.loc[ids, "filename_lr"].tolist()

    tmp_path = os.path.join(args.tmp_dir, "X_filtered.npy")
    os.makedirs(args.tmp_dir, exist_ok=True)
    filtered = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32, shape=(len(ids), 1000, 12))

    # Streaming per-lead statistics on training folds only (float64).
    total = np.zeros(12)
    total_sq = np.zeros(12)
    count = 0

    for i, name in enumerate(tqdm(files, desc="Filtering records")):
        signal, _ = wfdb.rdsamp(os.path.join(args.root, name))
        if signal.shape != (1000, 12):
            raise ValueError(f"Unexpected shape {signal.shape} for {name}")
        f = bandpass(signal)
        if not np.isfinite(f).all():
            raise ValueError(f"Non-finite values in {name}")
        filtered[i] = f.astype(np.float32)
        if folds[i] <= 8:
            total += f.sum(axis=0)
            total_sq += (f ** 2).sum(axis=0)
            count += f.shape[0]

    filtered.flush()
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)

    X = np.lib.format.open_memmap(
        os.path.join(out, "X.npy"), mode="w+", dtype=np.float32,
        shape=(len(ids), 1000, 12))
    for start in range(0, len(ids), 1024):
        end = min(start + 1024, len(ids))
        X[start:end] = ((filtered[start:end] - mean) / std).astype(np.float32)
    X.flush()
    del X, filtered
    os.remove(tmp_path)

    for name in COPIED:
        shutil.copyfile(os.path.join(args.reference, name), os.path.join(out, name))

    leads = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    with open(os.path.join(out, "norm_stats.json"), "w") as f:
        json.dump({
            "description": "per-lead statistics of band-pass filtered signals (mV), "
                           "training folds 1-8 only",
            "samples_per_lead": int(count),
            "mean": dict(zip(leads, mean.tolist())),
            "std": dict(zip(leads, std.tolist())),
            "reference": args.reference,
        }, f, indent=2)

    print("Per-lead std (mV):", np.round(std, 4))
    print("Saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/ptb-xl")
    parser.add_argument("--reference", default="data/processed")
    parser.add_argument("--out", default="data/p2_global_norm")
    parser.add_argument("--tmp-dir", required=True,
                        help="Scratch directory for the intermediate filtered array.")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
