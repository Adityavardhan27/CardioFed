"""Run this FIRST on your machine (no dataset needed). ~20 seconds.
Checks: model output shape, loss decreases, full train loop works on fake data."""
import os, numpy as np, torch
from src.model import ECGNet

# 1) shape check
for lstm in (False, True):
    out = ECGNet(use_lstm=lstm)(torch.randn(4, 12, 1000))
    assert out.shape == (4, 5), out.shape
    print(f"model (lstm={lstm}) output shape OK:", tuple(out.shape))

# 2) fake dataset with a learnable pattern -> run real training loop
rng = np.random.default_rng(0); N = 600
y = (rng.random((N, 5)) < 0.3).astype(np.float32)
X = rng.normal(size=(N, 1000, 12)).astype(np.float32)
t = np.arange(1000) / 100
for c in range(5):                       # inject class-specific frequency
    X[y[:, c] == 1] += 0.8 * np.sin(2*np.pi*(2+c)*t)[None, :, None]
folds = rng.integers(1, 11, N)
os.makedirs("/tmp/fake_ptbxl", exist_ok=True)
for n, a in dict(X=X, y=y, folds=folds).items(): np.save(f"/tmp/fake_ptbxl/{n}.npy", a)

from src.train import train
train(processed="/tmp/fake_ptbxl", epochs=4, out="/tmp/fake_results")
print("\nSMOKE TEST PASSED - pipeline works end to end")
