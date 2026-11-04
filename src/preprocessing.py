
"""ECG preprocessing and PTB-XL diagnostic label generation."""

import ast
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FS = 100


def load_metadata(root):
    df = pd.read_csv(f"{root}/ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(f"{root}/scp_statements.csv", index_col=0)

    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    diagnostic = scp[
        (pd.to_numeric(scp["diagnostic"], errors="coerce") == 1)
        & scp["diagnostic_class"].isin(CLASSES)
    ]

    superclass_map = diagnostic["diagnostic_class"].to_dict()

    def get_superclasses(codes):
        return sorted({
            superclass_map[code]
            for code, likelihood in codes.items()
            if code in superclass_map and likelihood > 0
        })

    df["superclasses"] = df["scp_codes"].apply(get_superclasses)

    # Keep records with at least one recognised diagnostic superclass.
    df = df[df["superclasses"].map(len) > 0].copy()

    return df


def multi_hot(superclass_lists):
    y = np.zeros((len(superclass_lists), len(CLASSES)), dtype=np.float32)

    class_index = {name: i for i, name in enumerate(CLASSES)}

    for row, labels in enumerate(superclass_lists):
        for label in labels:
            y[row, class_index[label]] = 1.0

    return y


def preprocess_signal(signal):
    """Input and output shape: [1000, 12]."""

    if signal.shape != (1000, 12):
        raise ValueError(f"Unexpected ECG shape: {signal.shape}")

    signal = signal.astype(np.float64)

    sos = butter(
        3,
        [0.5, 40.0],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    filtered = sosfiltfilt(sos, signal, axis=0)

    mean = filtered.mean(axis=0, keepdims=True)
    std = filtered.std(axis=0, keepdims=True)

    normalized = (filtered - mean) / (std + 1e-8)

    return normalized.astype(np.float32)
