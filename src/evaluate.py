
"""Evaluation utilities for multi-label ECG classification."""

import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss
)

from src.preprocessing import CLASSES


@torch.no_grad()
def predict_probs(model, loader, device):

    model.eval()

    probabilities = []
    targets = []

    for x, y in loader:

        x = x.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)

        probabilities.append(probs.cpu().numpy())
        targets.append(y.numpy())

    return (
        np.concatenate(probabilities),
        np.concatenate(targets)
    )


def compute_metrics(probs, y, thresholds=0.5):

    pred = (probs >= thresholds).astype(int)

    auc_scores = {}

    for i, name in enumerate(CLASSES):

        if len(np.unique(y[:, i])) < 2:
            auc_scores[name] = None
        else:
            auc_scores[name] = float(
                roc_auc_score(y[:, i], probs[:, i])
            )

    valid_auc = [
        value for value in auc_scores.values()
        if value is not None
    ]

    f1 = f1_score(
        y, pred, average=None, zero_division=0
    )

    precision = precision_score(
        y, pred, average=None, zero_division=0
    )

    recall = recall_score(
        y, pred, average=None, zero_division=0
    )

    return {
        "macro_auroc": (
            float(np.mean(valid_auc)) if valid_auc else None
        ),
        "macro_f1": float(np.mean(f1)),
        "hamming_loss": float(hamming_loss(y, pred)),
        "subset_accuracy": float(np.mean(np.all(pred == y, axis=1))),
        "per_class_auroc": auc_scores,
        "per_class_f1": dict(zip(CLASSES, map(float, f1))),
        "per_class_precision": dict(zip(CLASSES, map(float, precision))),
        "per_class_recall": dict(zip(CLASSES, map(float, recall))),
        "per_class_support": dict(
            zip(CLASSES, map(int, y.sum(axis=0)))
        )
    }
