"""Checks against the real PTB-XL data and the stored research artifacts.

Skipped automatically when data/checkpoints are not present.
"""

import ast
import json
import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import requires_data
from src.preprocessing import CLASSES

P = "data/processed"
META = "data/raw/ptb-xl/ptbxl_database.csv"


@pytest.fixture(scope="module")
def arrays():
    ids = np.load(f"{P}/ecg_ids.npy")
    meta = pd.read_csv(META, index_col="ecg_id").loc[ids]
    return {
        "y": np.load(f"{P}/y.npy"),
        "folds": np.load(f"{P}/folds.npy"),
        "ids": ids,
        "meta": meta,
    }


@requires_data
def test_arrays_aligned(arrays):
    X = np.load(f"{P}/X.npy", mmap_mode="r")
    n = len(arrays["ids"])
    assert X.shape == (n, 1000, 12)
    assert arrays["y"].shape == (n, 5)
    assert len(np.unique(arrays["ids"])) == n
    assert (arrays["meta"]["strat_fold"].to_numpy() == arrays["folds"]).all()


@requires_data
def test_no_patient_in_two_splits(arrays):
    f = arrays["folds"]
    split = np.where(f <= 8, "train", np.where(f == 9, "val", "test"))
    per_patient = pd.Series(split).groupby(
        arrays["meta"]["patient_id"].to_numpy()).nunique()
    assert per_patient.max() == 1


@requires_data
def test_labels_follow_documented_policy(arrays):
    scp = pd.read_csv("data/raw/ptb-xl/scp_statements.csv", index_col=0)
    diag = scp[(pd.to_numeric(scp["diagnostic"], errors="coerce") == 1)
               & scp["diagnostic_class"].isin(CLASSES)]["diagnostic_class"].to_dict()
    expected = np.zeros_like(arrays["y"])
    for i, codes in enumerate(arrays["meta"]["scp_codes"].apply(ast.literal_eval)):
        labels = {diag[c] for c, v in codes.items() if c in diag and v > 0}
        if labels - {"NORM"}:
            labels -= {"NORM"}
        for c in labels:
            expected[i, CLASSES.index(c)] = 1
    np.testing.assert_array_equal(arrays["y"], expected)
    assert ((arrays["y"][:, 0] == 1) & (arrays["y"][:, 1:].sum(1) > 0)).sum() == 0


@requires_data
def test_client_partitions_match_reported_sizes():
    from src.federated_train import create_client_partitions
    parts = create_client_partitions()
    with open("results/metrics/federated_summary.json") as f:
        reported = json.load(f)["client_sizes"]
    assert {str(k + 1): len(v) for k, v in parts.items()} == reported
    assigned = np.concatenate(list(parts.values()))
    assert len(assigned) == len(np.unique(assigned))


@requires_data
@pytest.mark.parametrize("name", [
    "baseline", "multiscale", "federated_r3", "federated_r8", "multiscale_legacy"])
def test_reported_test_metrics_reproduce(name):
    from src.model_registry import (
        MODELS, load_model, load_evaluation, load_split, predict_array)
    from src.evaluate import compute_metrics
    from src.thresholds import thresholds_from_evaluation, tune_thresholds

    if not os.path.exists(MODELS[name]["checkpoint"]):
        pytest.skip("checkpoint not available")
    model, _ = load_model(name)
    evaluation, saved = load_evaluation(name)

    xv, yv, _ = load_split("val")
    th = tune_thresholds(predict_array(model, xv), yv)
    np.testing.assert_array_equal(th, thresholds_from_evaluation(evaluation))

    xt, yt, _ = load_split("test")
    m = compute_metrics(predict_array(model, xt), yt, th)
    for key in ("macro_auroc", "macro_f1", "hamming_loss", "subset_accuracy"):
        assert m[key] == pytest.approx(saved[key], abs=1e-9), key
