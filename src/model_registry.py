"""Registry of the trained CardioFed models and their original artifacts.

This is the single source of truth for which checkpoint, architecture and
evaluation file make up each reported model. All post-hoc analysis scripts
(statistics, per-site evaluation, robustness, explainability, app) read from
here, so they always refer to the same models.

Primary models are reproducible from the committed training code.
`multiscale_legacy` was trained by an earlier version of the multi-scale
training script that is no longer in the repository; it is kept only for
transparency and is not part of the primary comparison.

Final Model Selection
---------------------
The selected centralized model for inference (``DEFAULT_CENTRALIZED``) is
the **multiscale** (scheduler experiment, seed 42, best epoch 10).

Selection rationale (see also RESULTS.md §3 and §10):
  1. Reproducibility — training code (``src/train_multiscale.py``) is retained.
  2. Complete evaluation — both validation and fold-10 test metrics exist and
     are verified by ``tests/test_research_artifacts.py::test_reported_test_metrics_reproduce``.
  3. Architecture consistency — Multi-Scale Residual CNN is the project's core
     contribution, shared with the federated models for a fair comparison.
  4. Statistical indistinguishability — paired patient-level bootstrap
     (2 000 resamples) shows no significant difference in macro-AUROC or
     macro-F1 versus the baseline CNN or federated-8-round models (all
     Holm-corrected p ≥ 0.35).
  5. HYP (the weakest class, F1 ≈ 0.46) is equally weak across all models;
     no checkpoint demonstrably improves it.

Models *not* selected and why:
  - baseline: different (simpler) architecture; would undermine the
    multi-scale residual contribution the project evaluates.
  - multiscale_legacy: training code not retained; not reproducible.
  - federated_r8: had not fully converged (see §10.1); exists mainly to
    demonstrate privacy-preserving viability, not as a recommended
    deployment artefact.
  - finetuned_v1 / finetuned_v2: no test-fold evaluation exists; selecting
    them would be indefensible without test metrics.
  - federated_r20_seed42 (P2): the best overall point estimate
    (test AUROC 0.910, test F1 0.712), but the advantage over centralized
    training is not statistically significant (seed-42 p = 0.055, multi-seed
    Δ = −0.0005) and it is a P2 supplementary experiment, not a primary model.
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
    # ------------------------------------------------------------------
    # Fine-tuning experiments (validation metrics only — no test
    # evaluation performed).  Included for transparency, NOT selectable
    # as a deployment checkpoint.
    # ------------------------------------------------------------------
    "finetuned_v1": {
        "label": "Multi-Scale Residual CNN (fine-tuned V1, lr=1e-4, from multiscale)",
        "arch": "multiscale",
        "checkpoint": "results/fine_tuning_v1/checkpoints/finetuned_best.pt",
        "evaluation": None,
        "history": "results/fine_tuning_v1/metrics/finetuning_history.json",
        "primary": False,
        "note": (
            "Fine-tuned from the multiscale scheduler-experiment checkpoint with "
            "lr=1e-4 for 6 epochs (best epoch 1, val AUROC 0.9125).  No test-fold "
            "evaluation was performed; validation loss diverged while AUROC rose, "
            "suggesting overfitting.  Not suitable as a deployment checkpoint."
        ),
    },
    "finetuned_v2": {
        "label": "Multi-Scale Residual CNN (fine-tuned V2, lr=1e-5, from multiscale)",
        "arch": "multiscale",
        "checkpoint": "results/fine_tuning_v2/checkpoints/finetuned_best.pt",
        "evaluation": None,
        "history": "results/fine_tuning_v2/metrics/finetuning_history.json",
        "primary": False,
        "note": (
            "Fine-tuned from the multiscale scheduler-experiment checkpoint with "
            "lr=1e-5 for 9 epochs (best epoch 6, val AUROC 0.9145).  No test-fold "
            "evaluation was performed.  Not suitable as a deployment checkpoint."
        ),
    },
    # ------------------------------------------------------------------
    # P2 supplementary experiment models.  These have full test
    # evaluations but are not part of the original primary comparison.
    # ------------------------------------------------------------------
    "federated_r20_seed42": {
        "label": "Multi-Scale Residual CNN (FedAvg, 20 rounds, seed 42)",
        "arch": "multiscale",
        "checkpoint": "results/p2/federated/r20_seed42/checkpoints/federated_20round_best.pt",
        "evaluation": "results/p2/federated/r20_seed42/evaluation/evaluation.json",
        "history": "results/p2/federated/r20_seed42/metrics/federated_20round_history.json",
        "primary": False,
        "note": (
            "P2 experiment — fully converged FedAvg (best round 14 of 20, seed 42).  "
            "Test AUROC 0.910, test F1 0.712.  Difference from centralized training is "
            "not statistically significant (Holm p = 0.055)."
        ),
    },
}

PRIMARY = [k for k, v in MODELS.items() if v["primary"]]

# Model used by the demo app, prediction script and post-hoc analyses that
# need a single centralized model.  See the module docstring for the full
# selection rationale.
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
    entry = MODELS[name]
    if entry.get("evaluation") is None:
        raise ValueError(
            f"Model '{name}' has no test evaluation file.  "
            f"Only models with stored test-fold evaluations can be loaded.  "
            f"Note: {entry.get('note', '')}"
        )
    with open(os.path.join(root, entry["evaluation"])) as f:
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
