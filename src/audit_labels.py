
import numpy as np
import pandas as pd

from src.preprocessing import load_metadata, CLASSES, multi_hot

ROOT = "data/raw/ptb-xl"

df = load_metadata(ROOT)
y = multi_hot(df["superclasses"].tolist())

print("\n========== LABEL DISTRIBUTION ==========")

for i, name in enumerate(CLASSES):
    positives = int(y[:, i].sum())
    percentage = positives / len(y) * 100

    print(
        f"{name:6s}: {positives:5d} "
        f"({percentage:.2f}%)"
    )

print("\nTotal usable records:", len(y))

print("\n========== LABEL COMBINATIONS ==========")

combinations = pd.Series(
    [" + ".join(
        name for i, name in enumerate(CLASSES) if row[i] == 1
    ) for row in y]
)

print(combinations.value_counts().to_string())

print("\n========== NORM AND ABNORMAL OVERLAP ==========")

norm_index = CLASSES.index("NORM")
abnormal_indices = [
    CLASSES.index(name)
    for name in ["MI", "STTC", "CD", "HYP"]
]

norm_positive = y[:, norm_index] == 1
abnormal_positive = y[:, abnormal_indices].sum(axis=1) > 0

print("NORM positive:", int(norm_positive.sum()))
print("Abnormal positive:", int(abnormal_positive.sum()))
print(
    "NORM + abnormal overlap:",
    int((norm_positive & abnormal_positive).sum())
)

print("\n========== LABELS PER RECORD ==========")

label_counts = y.sum(axis=1)

for count in range(1, len(CLASSES) + 1):
    print(
        f"{count} positive label(s):",
        int((label_counts == count).sum())
    )
