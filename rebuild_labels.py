
import ast
import os
import numpy as np
import pandas as pd

from src.preprocessing import CLASSES

ROOT = "data/raw/ptb-xl"
OUT = "data/processed"

print("Loading metadata...")

df = pd.read_csv(
    os.path.join(ROOT, "ptbxl_database.csv"),
    index_col="ecg_id"
)

scp = pd.read_csv(
    os.path.join(ROOT, "scp_statements.csv"),
    index_col=0
)

df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

diagnostic = scp[
    (pd.to_numeric(scp["diagnostic"], errors="coerce") == 1)
    & scp["diagnostic_class"].isin(CLASSES)
]

mapping = diagnostic["diagnostic_class"].to_dict()

# Preserve the exact original signal order.
ecg_ids = np.load(os.path.join(OUT, "ecg_ids.npy"))
df = df.loc[ecg_ids]

print("Aligned records:", len(df))

# Build labels using positive-likelihood diagnostic codes.
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

# Remove records with no usable target.
keep = y.sum(axis=1) > 0
removed = int((~keep).sum())

print("Records without usable labels:", removed)

if removed == len(df):
    raise ValueError("No usable records remain.")

# Open original signals without loading everything into RAM.
X = np.load(os.path.join(OUT, "X.npy"), mmap_mode="r")

if X.shape[0] != len(df):
    raise ValueError("Signal and metadata alignment error.")

new_n = int(keep.sum())
clean_path = os.path.join(OUT, "X_clean.npy")

print("\nCreating cleaned signal array...")

X_clean = np.lib.format.open_memmap(
    clean_path,
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

# Save cleaned signals, replacing the old array.
os.replace(
    clean_path,
    os.path.join(OUT, "X.npy")
)

# Keep labels and all metadata aligned.
y = y[keep]
df = df.iloc[np.flatnonzero(keep)].copy()
ecg_ids = ecg_ids[keep]

np.save(os.path.join(OUT, "y.npy"), y)

np.save(
    os.path.join(OUT, "folds.npy"),
    df["strat_fold"].to_numpy(dtype=np.int64)
)

sites = (
    df["site"].astype("string").fillna("").astype(str).to_numpy()
    if "site" in df.columns
    else np.full(len(df), "", dtype=str)
)

np.save(os.path.join(OUT, "sites.npy"), sites)
np.save(os.path.join(OUT, "ecg_ids.npy"), ecg_ids)

df[["filename_lr", "strat_fold"]].to_csv(
    os.path.join(OUT, "processed_metadata.csv"),
    index=True
)

# Final validation.
X_check = np.load(
    os.path.join(OUT, "X.npy"),
    mmap_mode="r"
)

y_check = np.load(os.path.join(OUT, "y.npy"))
folds = np.load(os.path.join(OUT, "folds.npy"))
ids = np.load(os.path.join(OUT, "ecg_ids.npy"))

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
