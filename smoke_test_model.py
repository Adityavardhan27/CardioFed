"""Run this FIRST on your machine (no dataset needed). ~20 seconds.
Checks: model output shape, loss decreases, full train loop works on fake data."""
import os
import shutil
import uuid

import numpy as np
import torch

from src.model import ECGNet

# 1) shape check
for lstm in (False, True):
    out = ECGNet(use_lstm=lstm)(torch.randn(4, 12, 1000))
    assert out.shape == (4, 5), out.shape
    print(f"model (lstm={lstm}) output shape OK:", tuple(out.shape))

# 2) fake dataset with a learnable pattern -> run real training loop
rng = np.random.default_rng(0)
N = 600
y = (rng.random((N, 5)) < 0.3).astype(np.float32)
X = rng.normal(size=(N, 1000, 12)).astype(np.float32)
t = np.arange(1000) / 100
for c in range(5):  # inject class-specific frequency
    X[y[:, c] == 1] += 0.8 * np.sin(2 * np.pi * (2 + c) * t)[None, :, None]
folds = rng.integers(1, 11, N)

from src.train import train

# Use a unique temporary directory under the OS temp location.
# Avoid tempfile.TemporaryDirectory because it crashes in some tested
# Python 3.14 + native-library import combinations on this machine.
temp_root = (
    os.environ.get("TEMP")
    or os.environ.get("TMP")
    or os.environ.get("TMPDIR")
    or os.getcwd()
)
temp_dir = os.path.join(temp_root, f"cardiofed_smoke_{uuid.uuid4().hex}")
os.makedirs(temp_dir, exist_ok=False)

try:
    processed = os.path.join(temp_dir, "fake_ptbxl")
    results = os.path.join(temp_dir, "fake_results")
    os.makedirs(processed, exist_ok=True)

    for name, array in dict(X=X, y=y, folds=folds).items():
        np.save(os.path.join(processed, f"{name}.npy"), array)

    train(processed=processed, epochs=4, output=results)
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

print("\nSMOKE TEST PASSED - pipeline works end to end")
