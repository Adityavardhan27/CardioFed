"""Builds a tiny fake PTB-XL (same CSV schema) and checks the label/filter logic."""
import os, tempfile, numpy as np, pandas as pd
from src.preprocessing import load_metadata, multi_hot, preprocess_signal, CLASSES

d = tempfile.mkdtemp()
pd.DataFrame({
    "ecg_id": [1, 2, 3, 4],
    "scp_codes": ["{'NORM': 100.0, 'SR': 0.0}", "{'IMI': 80.0, 'LAFB': 100.0}",
                  "{'SR': 0.0}", "{'LVH': 50.0, 'NDT': 100.0}"],
    "strat_fold": [1, 9, 10, 3], "site": [0, 1, 1, 2],
    "filename_lr": ["a", "b", "c", "d"],
}).to_csv(f"{d}/ptbxl_database.csv", index=False)
pd.DataFrame({
    "Unnamed: 0": ["NORM", "IMI", "LAFB", "LVH", "NDT", "SR"],
    "diagnostic": [1, 1, 1, 1, 1, 0],
    "diagnostic_class": ["NORM", "MI", "CD", "HYP", "STTC", None],
}).set_index("Unnamed: 0").to_csv(f"{d}/scp_statements.csv")

df = load_metadata(d)
print("kept ecg_ids:", list(df.index), "(id 3 has only a rhythm code -> dropped)")
y = multi_hot(df["superclasses"].tolist())
print(CLASSES); print(y)
assert y.shape == (3, 5) and y[1].tolist() == [0, 1, 0, 1, 0]   # MI + CD together
assert y[2].tolist() == [0, 0, 1, 0, 1]                          # STTC + HYP

t = np.arange(1000) / 100
sig = np.stack([np.sin(2*np.pi*1.2*t) + 5 + 0.5*t + 0.3*np.sin(2*np.pi*48*t)] * 12, 1)
out = preprocess_signal(sig)
print("shape", out.shape, "mean %.3f std %.3f" % (out.mean(), out.std()))
assert out.shape == (1000, 12) and abs(out.mean()) < 1e-3
print("PREPROCESSING OK")
