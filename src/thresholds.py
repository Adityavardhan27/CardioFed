"""Per-class decision-threshold selection (validation data only).

Single shared implementation of the rule used for every reported result:
for each class, choose the threshold that maximises F1 on the validation
set (fold 9). Previously this function was copy-pasted into four scripts.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.preprocessing import CLASSES


def tune_thresholds(probs, targets, verbose=False):
    """Return one F1-maximising threshold per class as float32.

    Thresholds are probability values taken from `probs` itself, so the
    float32 representation is exact.
    """
    thresholds = []

    for i, name in enumerate(CLASSES):
        precision, recall, values = precision_recall_curve(
            targets[:, i], probs[:, i]
        )

        if len(values) == 0:
            thresholds.append(0.5)
            continue

        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )

        best = int(np.argmax(f1))
        thresholds.append(float(values[best]))

        if verbose:
            print(
                f"{name}: threshold={values[best]:.4f}, "
                f"validation F1={f1[best]:.4f}"
            )

    return np.asarray(thresholds, dtype=np.float32)


def thresholds_from_evaluation(evaluation):
    """Read the saved thresholds from an evaluation JSON dictionary."""
    return np.asarray(
        [evaluation["thresholds"][name] for name in CLASSES],
        dtype=np.float32,
    )
