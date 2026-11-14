
import numpy as np
import pandas as pd

from src.preprocessing import load_metadata, CLASSES, multi_hot

ROOT = "data/raw/ptb-xl"

df = load_metadata(ROOT)
y_stage1 = multi_hot(df["superclasses"].tolist())

# Stage 2: CardioFed NORM-exclusive policy
exclusive_superclasses = []
for labels in df["superclasses"]:
    labels_set = set(labels)
    abnormal = labels_set - {"NORM"}
    if abnormal:
        labels_set = abnormal
    exclusive_superclasses.append(sorted(labels_set))

y_stage2 = multi_hot(exclusive_superclasses)

print("\n==================================================")
print("STAGE 1: RAW PTB-XL MAPPING (Pre-Exclusivity)")
print("==================================================")

for i, name in enumerate(CLASSES):
    positives = int(y_stage1[:, i].sum())
    percentage = positives / len(y_stage1) * 100
    print(f"{name:6s}: {positives:5d} ({percentage:.2f}%)")

print("\nTotal usable records:", len(y_stage1))

norm_index = CLASSES.index("NORM")
abnormal_indices = [
    CLASSES.index(name)
    for name in ["MI", "STTC", "CD", "HYP"]
]

norm_pos1 = y_stage1[:, norm_index] == 1
abn_pos1 = y_stage1[:, abnormal_indices].sum(axis=1) > 0
overlap1 = int((norm_pos1 & abn_pos1).sum())

print("Stage 1 NORM positive:", int(norm_pos1.sum()))
print("Stage 1 Abnormal positive:", int(abn_pos1.sum()))
print("Stage 1 NORM + abnormal overlap:", overlap1)

print("\n==================================================")
print("STAGE 2: CARDIOFED LABELS (NORM-Exclusive Policy)")
print("==================================================")

for i, name in enumerate(CLASSES):
    positives = int(y_stage2[:, i].sum())
    percentage = positives / len(y_stage2) * 100
    print(f"{name:6s}: {positives:5d} ({percentage:.2f}%)")

print("\nTotal usable records:", len(y_stage2))

norm_pos2 = y_stage2[:, norm_index] == 1
abn_pos2 = y_stage2[:, abnormal_indices].sum(axis=1) > 0
overlap2 = int((norm_pos2 & abn_pos2).sum())

print("Stage 2 NORM positive:", int(norm_pos2.sum()))
print("Stage 2 Abnormal positive:", int(abn_pos2.sum()))
print("Stage 2 NORM + abnormal overlap:", overlap2)

print("\n==================================================")
print("STAGE 1 VS STAGE 2 COMPARISON")
print("==================================================")
records_diff = int(np.any(y_stage1 != y_stage2, axis=1).sum())
norm_diff = int((y_stage1[:, norm_index] != y_stage2[:, norm_index]).sum())
abnormal_diff = int(np.any(y_stage1[:, abnormal_indices] != y_stage2[:, abnormal_indices], axis=1).sum())

print("Total records modified by NORM exclusivity:", records_diff)
print("Records where NORM was removed:", norm_diff)
print("Records where abnormal classes were modified:", abnormal_diff)
print("Empty records after exclusivity:", int((y_stage2.sum(axis=1) == 0).sum()))

print("\n========== LABEL COMBINATIONS (STAGE 2) ==========")

combinations = pd.Series(
    [" + ".join(
        name for i, name in enumerate(CLASSES) if row[i] == 1
    ) for row in y_stage2]
)

print(combinations.value_counts().to_string())

print("\n========== LABELS PER RECORD (STAGE 2) ==========")

label_counts = y_stage2.sum(axis=1)

for count in range(1, len(CLASSES) + 1):
    print(
        f"{count} positive label(s):",
        int((label_counts == count).sum())
    )

