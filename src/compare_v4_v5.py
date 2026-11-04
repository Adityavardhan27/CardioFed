
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

original = pd.read_csv(ROOT / "results/examples/ECG_63.csv")
v4 = pd.read_csv(ROOT / "results/examples/ECG_63_reconstructed_v4.csv")
v5 = pd.read_csv(ROOT / "results/examples/ECG_63_reconstructed_v5.csv")

rows = []

for lead in LEADS:

    a = original[lead].to_numpy()
    b = v4[lead].to_numpy()
    c = v5[lead].to_numpy()

    rows.append({
        "Lead": lead,
        "V4_MAE": np.mean(np.abs(a - b)),
        "V5_MAE": np.mean(np.abs(a - c)),
        "V4_RMSE": np.sqrt(np.mean((a - b) ** 2)),
        "V5_RMSE": np.sqrt(np.mean((a - c) ** 2)),
        "V4_Correlation": np.corrcoef(a, b)[0, 1],
        "V5_Correlation": np.corrcoef(a, c)[0, 1]
    })

results = pd.DataFrame(rows)

print("\nV4 VS V5 COMPARISON")
print("=" * 100)
print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

print("\nOVERALL")
for metric in ["MAE", "RMSE", "Correlation"]:
    print(
        f"{metric}: "
        f"V4={results[f'V4_{metric}'].mean():.5f} | "
        f"V5={results[f'V5_{metric}'].mean():.5f}"
    )
