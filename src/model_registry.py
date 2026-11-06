"""Registry of the trained CardioFed models and their original artifacts.

This is the single source of truth for which checkpoint, architecture and
evaluation file make up each reported model. All post-hoc analysis scripts
(statistics, per-site evaluation, robustness, explainability, app) read from
here, so they always refer to the same models.

Primary models are reproducible from the committed training code.
`multiscale_legacy` was trained by an earlier version of the multi-scale
training script that is no longer in the repository; it is kept only for
transparency and is not part of the primary comparison.
"""

import json
import os

import numpy as np
import torch

from src.model import ECGNet, MultiScaleResidualECGNet

MODELS = {
    "baseline": {
        "label": "Baseline 1D CNN (centralized)",
        "arch": "ecgnet",
        "checkpoint": "results/checkpoints/centralized_best.pt",
        "evaluation": "results/metrics/baseline_final_evaluation.json",
        "history": "results/metrics/baseline_history.json",
        "primary": True,
    },
    "multiscale": {
        "label": "Multi-Scale Residual CNN (centralized)",
        "arch": "multiscale",
        "checkpoint": "results/scheduler_experiment/checkpoints/multiscale_best.pt",
        "evaluation": "results/scheduler_experiment/metrics/multiscale_final_evaluation.json",
        "history": "results/scheduler_experiment/metrics/multiscale_history.json",
        "primary": True,
    },
    "federated_r3": {
        "label": "Multi-Scale Residual CNN (FedAvg, 3 rounds)",
        "arch": "multiscale",
        "checkpoint": "results/checkpoints/federated_3round_best.pt",
        "evaluation": "results/metrics/federated_3round_final_evaluation.json",
        "history": "results/metrics/federated_3round_history.json",
        "primary": True,
    },
    "federated_r8": {
        "label": "Multi-Scale Residual CNN (FedAvg, 8 rounds)",
        "arch": "multiscale",
        "checkpoint": "results/checkpoints/federated_best.pt",
        "evaluation": "results/metrics/federated_final_evaluation.json",
        "history": "results/metrics/federated_history.json",
        "primary": True,
    },
    "multiscale_legacy": {
        "label": "Multi-Scale Residual CNN (legacy run, training code not retained)",
        "arch": "multiscale",
        "checkpoint": "results/checkpoints/multiscale_best.pt",
        "evaluation": "results/metrics/multiscale_final_evaluation.json",
        "history": "results/metrics/multiscale_history.json",
        "primary": False,
    },
}

PRIMARY = [k for k, v in MODELS.items() if v["primary"]]

# Model used by the demo app, prediction script and post-hoc analyses that
# need a single centralized model.
DEFAULT_CENTRALIZED = "multiscale"
DEFAULT_FEDERATED = "federated_r8"


def build_model(arch):
    if arch == "ecgnet":
        return ECGNet(n_leads=12, n_classes=5, use_lstm=False, dropout=0.3)
    if arch == "multiscale":
        return MultiScaleResidualECGNet(n_leads=12, n_classes=5, dropout=0.35)
    raise ValueError(f"Unknown architecture: {arch}")


def load_model(name, device="cpu", root="."):
    entry = MODELS[name]
    model = build_model(entry["arch"])
    checkpoint = torch.load(
        os.path.join(root, entry["checkpoint"]), map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def load_evaluation(name, root="."):
    with open(os.path.join(root, MODELS[name]["evaluation"])) as f:
        evaluation = json.load(f)
    test = evaluation.get("test_metrics", evaluation.get("metrics"))
    return evaluation, test


@torch.no_grad()
def predict_array(model, x, batch_size=256, device="cpu"):
    """Sigmoid probabilities for x of shape [N, 12, 1000] (tensor or array)."""
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    out = []
    for i in range(0, len(x), batch_size):
        out.append(torch.sigmoid(model(x[i:i + batch_size].to(device))).cpu().numpy())
    return np.concatenate(out)


def load_split(split, processed="data/processed"):
    """Return (X [N,12,1000] float32 tensor, y [N,5], dataset indices)."""
    folds = np.load(os.path.join(processed, "folds.npy"))
    mask = {"train": folds <= 8, "val": folds == 9, "test": folds == 10}[split]
    idx = np.where(mask)[0]
    X = np.load(os.path.join(processed, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(processed, "y.npy"))
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(X[idx]).transpose(0, 2, 1)))
    return x, y[idx], idx
