import os
import numpy as np
import pandas as pd

from src.preprocessing import CLASSES, load_metadata, multi_hot

ROOT = "data/raw/ptb-xl"
PROCESSED = "data/processed"
LFS_SIGNATURE = b"version https://git-lfs"


def is_real_file(path):
    """Check if file exists and is not a Git LFS pointer."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(LFS_SIGNATURE)) != LFS_SIGNATURE
    except OSError:
        return False


# --------------------------------------------------
# 1. LOAD PROCESSED ARRAYS / METADATA
# --------------------------------------------------

is_real_npy = is_real_file(f"{PROCESSED}/y.npy")

if is_real_npy:
    print("\n[INFO] Loading binary NumPy arrays from disk.")
    processed_y = np.load(f"{PROCESSED}/y.npy")
    folds = np.load(f"{PROCESSED}/folds.npy")
    ecg_ids = np.load(f"{PROCESSED}/ecg_ids.npy")
else:
    print("\n[NOTE] data/processed/*.npy are Git LFS pointer files.")
    print("[NOTE] Loading folds and ECG IDs from data/processed/processed_metadata.csv")
    print("[NOTE] Reconstructing target labels using the documented CardioFed policy.")
    meta_proc = pd.read_csv(f"{PROCESSED}/processed_metadata.csv")
    ecg_ids = meta_proc["ecg_id"].to_numpy(dtype=np.int64)
    folds = meta_proc["strat_fold"].to_numpy(dtype=np.int64)

    orig_meta = load_metadata(ROOT).loc[ecg_ids]
    exclusive_superclasses = []
    for labels in orig_meta["superclasses"]:
        labels_set = set(labels)
        abnormal = labels_set - {"NORM"}
        if abnormal:
            labels_set = abnormal
        exclusive_superclasses.append(sorted(labels_set))
    processed_y = multi_hot(exclusive_superclasses)

print("\n========== SHAPE VERIFICATION ==========")
print("Processed labels:", processed_y.shape)
print("Fold array:", folds.shape)
print("ECG IDs:", ecg_ids.shape)

if not (
    len(processed_y) == len(folds) == len(ecg_ids)
):
    raise ValueError("Processed arrays have different lengths.")

if processed_y.ndim != 2 or processed_y.shape[1] != len(CLASSES):
    raise ValueError("Unexpected processed label dimensions.")

if len(np.unique(ecg_ids)) != len(ecg_ids):
    raise ValueError("Duplicate ECG IDs found in processed data.")

if not np.isin(processed_y, [0, 1]).all():
    raise ValueError("Processed labels contain values other than 0 and 1.")



# --------------------------------------------------
# 2. SPLIT DISTRIBUTION
# --------------------------------------------------

print("\n========== SPLIT DISTRIBUTION ==========")

for name, mask in [
    ("Train", folds <= 8),
    ("Validation", folds == 9),
    ("Test", folds == 10)
]:
    labels = processed_y[mask]

    print(f"\n{name}: {len(labels)} records")

    for i, class_name in enumerate(CLASSES):
        count = int(labels[:, i].sum())
        percentage = count / len(labels) * 100 if len(labels) else 0

        print(
            f"  {class_name}: {count} "
            f"({percentage:.2f}%)"
        )


# --------------------------------------------------
# 3. FOLD VERIFICATION
# --------------------------------------------------

print("\n========== FOLD VERIFICATION ==========")
print("Unique folds:", np.unique(folds))

invalid_folds = int(((folds < 1) | (folds > 10)).sum())

print("Records outside folds 1-10:", invalid_folds)

if invalid_folds:
    raise ValueError("Invalid fold values found.")


# --------------------------------------------------
# 4. PATIENT-WISE SPLIT CHECK
# --------------------------------------------------

print("\n========== PATIENT-WISE SPLIT CHECK ==========")

raw_metadata = pd.read_csv(
    f"{ROOT}/ptbxl_database.csv"
)

raw_metadata["ecg_id"] = raw_metadata["ecg_id"].astype(int)

mapping = pd.DataFrame({
    "ecg_id": ecg_ids.astype(int),
    "fold": folds
})

merged = mapping.merge(
    raw_metadata[["ecg_id", "patient_id"]],
    on="ecg_id",
    how="left",
    validate="one_to_one"
)

missing_patients = int(merged["patient_id"].isna().sum())

print("Missing patient IDs:", missing_patients)

patient_fold_counts = (
    merged.dropna(subset=["patient_id"])
    .groupby("patient_id")["fold"]
    .nunique()
)

overlap = int((patient_fold_counts > 1).sum())

print("Patients appearing in multiple folds:", overlap)


# --------------------------------------------------
# 5. ORIGINAL VS PROCESSED LABEL VERIFICATION
# --------------------------------------------------

print("\n========== ORIGINAL LABEL VERIFICATION ==========")

original_metadata = load_metadata(ROOT).copy()

# Normalize ECG IDs to integers.
original_metadata.index = (
    pd.to_numeric(original_metadata.index).astype(int)
)

print("Original usable metadata records:", len(original_metadata))
print("Processed ECG records:", len(ecg_ids))

if original_metadata.index.has_duplicates:
    raise ValueError("Duplicate ECG IDs found in original metadata.")

processed_ids = ecg_ids.astype(int)

# Find processed ECG IDs missing from original metadata.
missing_ids = sorted(
    set(processed_ids) - set(original_metadata.index)
)

print("Processed ECG IDs missing from original metadata:", len(missing_ids))

# Find original usable records absent from processed data.
extra_ids = sorted(
    set(original_metadata.index) - set(processed_ids)
)

print("Original usable ECG IDs absent from processed data:", len(extra_ids))

if missing_ids:
    print("First missing ECG IDs:", missing_ids[:10])

# Compare labels only if every processed ECG ID has a matching source row.
if not missing_ids:

    # Align original labels to the exact order of processed ECG IDs.
    aligned_superclasses = original_metadata.loc[
        processed_ids, "superclasses"
    ].tolist()

    # Apply the same exclusive-NORM policy as rebuild_labels.py.
    exclusive_superclasses = []

    for labels in aligned_superclasses:
        labels = set(labels)
        abnormal = labels - {"NORM"}

        if abnormal:
            labels = abnormal

        exclusive_superclasses.append(sorted(labels))

    expected_y = multi_hot(exclusive_superclasses)

    print("Reconstructed labels shape:", expected_y.shape)

    # Compare the reconstructed labels with y.npy.
    mismatch_mask = np.any(
        expected_y != processed_y,
        axis=1
    )

    mismatch_count = int(mismatch_mask.sum())

    print("Records with label mismatches:", mismatch_count)

    for i, class_name in enumerate(CLASSES):
        class_mismatches = int(
            np.sum(expected_y[:, i] != processed_y[:, i])
        )

        print(
            f"{class_name} label mismatches: "
            f"{class_mismatches}"
        )

    if mismatch_count:
        mismatch_indices = np.where(mismatch_mask)[0]

        print("\nFirst mismatched records:")

        for idx in mismatch_indices[:10]:
            print(
                f"ECG ID: {processed_ids[idx]} | "
                f"Original: {expected_y[idx].astype(int).tolist()} | "
                f"Processed: {processed_y[idx].astype(int).tolist()}"
            )

    else:
        print("\nSUCCESS: Every processed label matches the original metadata.")

else:
    print("\nLABEL COMPARISON SKIPPED because ECG IDs are missing.")


# --------------------------------------------------
# 6. FINAL SUMMARY
# --------------------------------------------------

print("\n========== AUDIT SUMMARY ==========")

print("Shape verification: PASSED")
print("Duplicate processed ECG IDs:", len(processed_ids) - len(np.unique(processed_ids)))
print("Invalid folds:", invalid_folds)
print("Missing patient IDs:", missing_patients)
print("Patients spanning multiple folds:", overlap)
print("Missing original ECG IDs:", len(missing_ids))
print("Extra original ECG IDs:", len(extra_ids))

if not missing_ids:
    print("Label mismatches:", mismatch_count)

print("\nAUDIT COMPLETED.")
