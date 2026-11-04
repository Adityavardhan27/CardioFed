
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "examples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ECG diagnostic classes
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# 12 ECG lead names
LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

# Select ECG recording
TARGET_ECG_ID = 63

print("Loading processed ECG dataset...")

# Memory-mapped loading
X = np.load(DATA_DIR / "X.npy", mmap_mode="r")
y = np.load(DATA_DIR / "y.npy")
ecg_ids = np.load(DATA_DIR / "ecg_ids.npy")

# Find the actual row corresponding to ECG ID 63
matches = np.where(ecg_ids == TARGET_ECG_ID)[0]

if len(matches) == 0:
    raise ValueError(f"ECG ID {TARGET_ECG_ID} not found!")

idx = int(matches[0])

# Extract ECG signal and labels
signal = np.asarray(X[idx], dtype=np.float32)
labels = y[idx]

print("\nECG found successfully!")
print("ECG ID:", TARGET_ECG_ID)
print("Array index:", idx)
print("Signal shape:", signal.shape)
print("Number of leads:", signal.shape[1])
print("Samples per lead:", signal.shape[0])

# Create time axis
sampling_rate = 100
time = np.arange(signal.shape[0]) / sampling_rate

# Create CSV
df = pd.DataFrame(signal, columns=LEADS)

df.insert(0, "time_seconds", time)
df.insert(0, "sample", np.arange(1, len(signal) + 1))
df.insert(0, "ecg_id", TARGET_ECG_ID)

csv_path = OUTPUT_DIR / f"ECG_{TARGET_ECG_ID}.csv"
df.to_csv(csv_path, index=False)

print("\nCSV saved:", csv_path)

# Print actual diagnostic labels
print("\nActual diagnostic labels:")

for class_name, label in zip(CLASSES, labels):
    print(f"{class_name}: {int(label)}")

# Plot all 12 ECG leads
fig, axes = plt.subplots(
    12, 1,
    figsize=(15, 20),
    sharex=True
)

for i, ax in enumerate(axes):
    ax.plot(time, signal[:, i], linewidth=0.8)
    ax.set_ylabel(LEADS[i])
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    f"CardioFed - ECG Recording {TARGET_ECG_ID}",
    fontsize=16
)

plt.tight_layout()

plot_path = OUTPUT_DIR / f"ECG_{TARGET_ECG_ID}_waveform.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")

print("\nWaveform saved:", plot_path)

plt.show()
