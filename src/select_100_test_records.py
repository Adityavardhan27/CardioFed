
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
EXAMPLES_DIR = ROOT / "results" / "examples"

EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = EXAMPLES_DIR / "multirecord_test_selection_100.csv"

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

rng = np.random.default_rng(42)

# Load processed dataset metadata.
y = np.load(DATA_DIR / "y.npy")
ids = np.load(DATA_DIR / "ecg_ids.npy")
folds = np.load(DATA_DIR / "folds.npy")
sites = np.load(DATA_DIR / "sites.npy", allow_pickle=True)

# Load previous 15-record selection to avoid reusing records.
previous_file = EXAMPLES_DIR / "multirecord_test_selection.csv"
previous = pd.read_csv(previous_file)

previous_indices = set(
    previous["dataset_index"].astype(int)
)

# Only use held-out test fold recordings.
test_indices = np.where(folds == 10)[0]

available_indices = np.array([
    idx for idx in test_indices
    if idx not in previous_indices
])

selected = set()

print("SELECTING 20 RECORDINGS PER SUPERCLASS")
print("-" * 55)

for class_index, class_name in enumerate(CLASSES):

    matching = available_indices[
        y[available_indices, class_index] == 1
    ]

    if len(matching) < 20:
        raise ValueError(
            f"Not enough recordings for {class_name}"
        )

    chosen = rng.choice(
        matching,
        size=20,
        replace=False
    )

    selected.update(chosen.tolist())

    print(
        f"{class_name}: "
        f"20 selected from {len(matching)} available"
    )

# Fill remaining places to reach exactly 100 unique records.
selected = list(selected)

remaining_needed = 100 - len(selected)

if remaining_needed > 0:

    remaining = np.array([
        idx for idx in available_indices
        if idx not in set(selected)
    ])

    additional = rng.choice(
        remaining,
        size=remaining_needed,
        replace=False
    )

    selected.extend(additional.tolist())

selected = sorted(set(selected))

if len(selected) != 100:
    raise RuntimeError(
        f"Expected 100 unique records, got {len(selected)}"
    )

# Create selection report.
data = {
    "ecg_id": ids[selected],
    "dataset_index": selected,
    "fold": folds[selected],
    "site": sites[selected]
}

for j, class_name in enumerate(CLASSES):
    data[class_name] = y[selected, j]

result = pd.DataFrame(data)

result.to_csv(OUTPUT_FILE, index=False)

print("\nFINAL SELECTION")
print("-" * 55)
print("Unique recordings:", len(result))
print("Fold distribution:")
print(result["fold"].value_counts())

print("\nActual diagnostic label counts:")
for class_name in CLASSES:
    print(
        f"{class_name}: "
        f"{int(result[class_name].sum())}"
    )

print("\nPrevious 15 recordings reused: 0")
print("Saved:", OUTPUT_FILE)
