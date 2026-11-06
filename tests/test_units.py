"""Fast unit tests (no PTB-XL data required)."""

import ast
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import precision_recall_curve

from src.io_utils import guard_outputs, OutputExistsError
from src.thresholds import tune_thresholds
from src.federated_train import fed_avg
from src.preprocessing import CLASSES

ROOT = os.getcwd()


# ---------------------------------------------------------------- io_utils
def test_guard_refuses_existing(tmp_path):
    existing = tmp_path / "a.json"
    existing.write_text("{}")
    with pytest.raises(OutputExistsError):
        guard_outputs([existing])
    assert existing.read_text() == "{}"


def test_guard_allows_new_and_creates_parents(tmp_path):
    target = tmp_path / "new" / "dir" / "b.json"
    guard_outputs([target])
    assert target.parent.is_dir() and not target.exists()


def test_guard_overwrite_flag(tmp_path):
    existing = tmp_path / "a.json"
    existing.write_text("{}")
    guard_outputs([existing], overwrite=True)  # must not raise


# -------------------------------------------------------------- thresholds
def _original_tune(probs, targets):
    """Verbatim logic of the four former copies in the evaluation scripts."""
    out = []
    for i in range(len(CLASSES)):
        p, r, v = precision_recall_curve(targets[:, i], probs[:, i])
        f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-8)
        out.append(float(v[int(np.argmax(f1))]))
    return np.asarray(out)


def test_shared_thresholds_match_original_logic():
    rng = np.random.default_rng(0)
    y = (rng.random((500, 5)) < 0.3).astype(np.float32)
    probs = np.clip(y * 0.4 + rng.random((500, 5)) * 0.6, 0, 1).astype(np.float32)
    new = tune_thresholds(probs, y)
    old = _original_tune(probs, y)
    assert new.dtype == np.float32
    np.testing.assert_array_equal(new, old.astype(np.float32))
    # float32 vs float64 thresholds yield identical predictions
    np.testing.assert_array_equal(probs >= new, probs >= old)


# ----------------------------------------------------------------- FedAvg
def test_fed_avg_is_size_weighted_mean():
    s1 = {"w": torch.tensor([1.0, 2.0]), "n": torch.tensor(5)}
    s2 = {"w": torch.tensor([3.0, 6.0]), "n": torch.tensor(9)}
    out = fed_avg([s1, s2], [1, 3])
    torch.testing.assert_close(out["w"], torch.tensor([2.5, 5.0]))
    assert out["n"].item() == 5  # integer buffers copied from client 0


# -------------------------------------------------------- label rebuilding
def _make_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "ecg_id": [1, 2, 3, 4],
        "scp_codes": [
            "{'NORM': 100.0}",                 # NORM only
            "{'NORM': 80.0, 'IMI': 100.0}",    # NORM + MI -> MI only
            "{'IMI': 0.0}",                    # likelihood 0 -> unlabeled
            "{'LVH': 50.0, 'NDT': 100.0}",     # HYP + STTC
        ],
        "strat_fold": [1, 9, 10, 3], "site": [0, 1, 1, 2],
        "filename_lr": ["a", "b", "c", "d"],
    }).to_csv(raw / "ptbxl_database.csv", index=False)
    pd.DataFrame({
        "code": ["NORM", "IMI", "LVH", "NDT"],
        "diagnostic": [1, 1, 1, 1],
        "diagnostic_class": ["NORM", "MI", "HYP", "STTC"],
    }).set_index("code").to_csv(raw / "scp_statements.csv")

    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    X = np.arange(4 * 1000 * 12, dtype=np.float32).reshape(4, 1000, 12)
    np.save(stage1 / "X.npy", X)
    np.save(stage1 / "ecg_ids.npy", np.array([1, 2, 3, 4], dtype=np.int64))
    return raw, stage1, X


def _run_rebuild(*args):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "rebuild_labels.py"), *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_rebuild_labels_policy_and_non_destructive(tmp_path):
    raw, stage1, X = _make_fixture(tmp_path)
    before = (stage1 / "X.npy").read_bytes()
    out = tmp_path / "processed"

    r = _run_rebuild("--root", str(raw), "--processed", str(stage1), "--out", str(out))
    assert r.returncode == 0, r.stderr

    y = np.load(out / "y.npy")
    ids = np.load(out / "ecg_ids.npy")
    assert ids.tolist() == [1, 2, 4]                       # record 3 removed
    assert y.tolist() == [[1, 0, 0, 0, 0],                 # NORM
                          [0, 1, 0, 0, 0],                 # NORM dropped
                          [0, 0, 1, 0, 1]]                 # STTC + HYP
    np.testing.assert_array_equal(np.load(out / "X.npy"), X[[0, 1, 3]])
    assert np.load(out / "folds.npy").tolist() == [1, 9, 3]
    assert (stage1 / "X.npy").read_bytes() == before       # input untouched

    # Second run must refuse to overwrite the outputs.
    r2 = _run_rebuild("--root", str(raw), "--processed", str(stage1), "--out", str(out))
    assert r2.returncode != 0 and "Refusing to overwrite" in r2.stderr


def test_rebuild_labels_refuses_in_place_without_flag(tmp_path):
    raw, stage1, _ = _make_fixture(tmp_path)
    before = (stage1 / "X.npy").read_bytes()
    r = _run_rebuild("--root", str(raw), "--processed", str(stage1))
    assert r.returncode != 0
    assert (stage1 / "X.npy").read_bytes() == before


# -------------------------------------------------- federated smoke (fake)
def test_federated_training_smoke(tmp_path):
    """One FedAvg round on tiny synthetic data; checks naming and guards."""
    rng = np.random.default_rng(0)
    n = 160
    processed = tmp_path / "processed"
    processed.mkdir()
    y = (rng.random((n, 5)) < 0.4).astype(np.float32)
    folds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * (n // 10))
    # Validation rows: alternating 0/1 so every class has both labels.
    y[folds == 9] = np.array(
        [[(i + c) % 2 for c in range(5)] for i in range(16)], dtype=np.float32)
    np.save(processed / "X.npy", rng.normal(size=(n, 1000, 12)).astype(np.float32))
    np.save(processed / "y.npy", y)
    np.save(processed / "folds.npy", folds)
    np.save(processed / "ecg_ids.npy", np.arange(1, n + 1))
    meta = tmp_path / "meta.csv"
    pd.DataFrame({
        "ecg_id": np.arange(1, n + 1),
        "patient_id": np.arange(n) // 2,
        "site": np.array([0, 0, 1, 1, 2, 2, 5, 5] * (n // 8)),
    }).to_csv(meta, index=False)

    cmd = [sys.executable, "-m", "src.federated_train", "--rounds", "1",
           "--batch-size", "8", "--processed", str(processed),
           "--metadata", str(meta), "--output", str(tmp_path / "out")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]

    ckpt = tmp_path / "out" / "checkpoints" / "federated_1round_best.pt"
    summary = tmp_path / "out" / "metrics" / "federated_1round_summary.json"
    assert ckpt.exists() and summary.exists()
    s = pd.read_json(summary, typ="series")
    assert s["best_checkpoint"].endswith("federated_1round_best.pt")
    assert s["learning_rate"] == 0.0005 and s["seed"] == 42

    r2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode != 0 and "Refusing to overwrite" in r2.stderr
