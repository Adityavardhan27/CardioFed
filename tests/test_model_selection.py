"""Tests for model selection consistency and registry integrity (Step 3).

These tests do NOT require the PTB-XL dataset or real checkpoint weights.
They verify structural invariants of the model registry, label ordering,
evaluation JSON integrity, and selection consistency.
"""

import json
import os

import numpy as np
import pytest

from src.model_registry import (
    MODELS,
    PRIMARY,
    DEFAULT_CENTRALIZED,
    DEFAULT_FEDERATED,
    build_model,
    load_evaluation,
)
from src.preprocessing import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Registry structure checks
# ---------------------------------------------------------------------------

def test_default_centralized_is_in_primary():
    """DEFAULT_CENTRALIZED must be a primary model."""
    assert DEFAULT_CENTRALIZED in PRIMARY, (
        f"DEFAULT_CENTRALIZED='{DEFAULT_CENTRALIZED}' is not a primary model. "
        f"Primary models: {PRIMARY}"
    )


def test_default_federated_is_in_primary():
    """DEFAULT_FEDERATED must be a primary model."""
    assert DEFAULT_FEDERATED in PRIMARY, (
        f"DEFAULT_FEDERATED='{DEFAULT_FEDERATED}' is not a primary model. "
        f"Primary models: {PRIMARY}"
    )


def test_all_models_have_required_keys():
    """Every registry entry must have label, arch, checkpoint, and primary fields."""
    required_keys = {"label", "arch", "checkpoint", "primary"}
    for name, entry in MODELS.items():
        missing = required_keys - set(entry.keys())
        assert not missing, f"Model '{name}' is missing keys: {missing}"


def test_arch_values_are_valid():
    """Every model arch must be one of the known architectures."""
    valid_archs = {"ecgnet", "multiscale"}
    for name, entry in MODELS.items():
        assert entry["arch"] in valid_archs, (
            f"Model '{name}' has unknown arch '{entry['arch']}'. "
            f"Valid: {valid_archs}"
        )


def test_models_without_evaluation_are_non_primary():
    """Models without an evaluation file must be non-primary."""
    for name, entry in MODELS.items():
        if entry.get("evaluation") is None:
            assert not entry["primary"], (
                f"Model '{name}' has no evaluation file but is marked primary. "
                "Primary models must have a stored test-fold evaluation."
            )


def test_load_evaluation_raises_for_no_evaluation_model():
    """load_evaluation must raise ValueError for models without evaluation files."""
    # Find a model without an evaluation (e.g. finetuned_v1 or finetuned_v2)
    no_eval_models = [
        name for name, entry in MODELS.items()
        if entry.get("evaluation") is None
    ]
    if not no_eval_models:
        pytest.skip("No models without evaluation files in registry")
    with pytest.raises(ValueError, match="no test evaluation file"):
        load_evaluation(no_eval_models[0])


# ---------------------------------------------------------------------------
# Label order consistency
# ---------------------------------------------------------------------------

def test_classes_order_is_norm_mi_sttc_cd_hyp():
    """CLASSES order must be NORM, MI, STTC, CD, HYP — consistent with all JSONs."""
    assert CLASSES == ["NORM", "MI", "STTC", "CD", "HYP"], (
        f"CLASSES order has changed: {CLASSES}. "
        "This would silently break all threshold and prediction lookups."
    )


def test_evaluation_json_classes_match_preprocessing_classes():
    """Evaluation JSONs for primary models must list CLASSES in the same order."""
    for name, entry in MODELS.items():
        if not entry["primary"] or entry.get("evaluation") is None:
            continue
        eval_path = os.path.join(ROOT, entry["evaluation"])
        if not os.path.isfile(eval_path):
            continue  # LFS pointer or absent — skip
        with open(eval_path) as f:
            data = json.load(f)
        json_classes = data.get("classes")
        if json_classes is not None:
            assert json_classes == CLASSES, (
                f"Model '{name}' evaluation JSON has classes {json_classes}, "
                f"expected {CLASSES}."
            )


def test_thresholds_cover_all_classes():
    """Evaluation JSONs for primary models must have thresholds for every class."""
    for name, entry in MODELS.items():
        if not entry["primary"] or entry.get("evaluation") is None:
            continue
        eval_path = os.path.join(ROOT, entry["evaluation"])
        if not os.path.isfile(eval_path):
            continue
        with open(eval_path) as f:
            data = json.load(f)
        thresholds = data.get("thresholds", {})
        missing = [c for c in CLASSES if c not in thresholds]
        assert not missing, (
            f"Model '{name}' evaluation JSON missing thresholds for: {missing}"
        )


def test_thresholds_are_in_valid_range():
    """All stored thresholds must be in (0, 1)."""
    for name, entry in MODELS.items():
        if entry.get("evaluation") is None:
            continue
        eval_path = os.path.join(ROOT, entry["evaluation"])
        if not os.path.isfile(eval_path):
            continue
        with open(eval_path) as f:
            data = json.load(f)
        thresholds = data.get("thresholds", {})
        for cls, thr in thresholds.items():
            assert 0.0 < thr < 1.0, (
                f"Model '{name}', class '{cls}': threshold {thr} not in (0, 1)"
            )


# ---------------------------------------------------------------------------
# Architecture consistency
# ---------------------------------------------------------------------------

def test_build_model_produces_correct_output_shape():
    """build_model must produce a model with output shape (batch, 5) for both archs."""
    import torch
    for arch in ("ecgnet", "multiscale"):
        model = build_model(arch)
        x = torch.randn(2, 12, 1000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5), (
            f"arch '{arch}' produced output shape {tuple(out.shape)}, expected (2, 5)"
        )


def test_default_model_arch_is_multiscale():
    """The default centralized model must use the multiscale architecture."""
    assert MODELS[DEFAULT_CENTRALIZED]["arch"] == "multiscale", (
        f"DEFAULT_CENTRALIZED='{DEFAULT_CENTRALIZED}' has arch "
        f"'{MODELS[DEFAULT_CENTRALIZED]['arch']}', expected 'multiscale'."
    )


# ---------------------------------------------------------------------------
# Non-primary fine-tuning entries: guard against accidental selection
# ---------------------------------------------------------------------------

def test_finetuned_models_are_non_primary():
    """Fine-tuned checkpoint entries must NOT be marked primary (no test eval)."""
    for name in ("finetuned_v1", "finetuned_v2"):
        if name in MODELS:
            assert not MODELS[name]["primary"], (
                f"'{name}' must be non-primary: no fold-10 test evaluation exists."
            )


def test_finetuned_models_have_no_evaluation_path():
    """Fine-tuned checkpoint entries must have evaluation=None (no test eval)."""
    for name in ("finetuned_v1", "finetuned_v2"):
        if name in MODELS:
            assert MODELS[name].get("evaluation") is None, (
                f"'{name}' must have evaluation=None: "
                "only models with test-fold evaluations may be deployed."
            )


# ---------------------------------------------------------------------------
# Default model evaluation JSON accessible
# ---------------------------------------------------------------------------

def test_default_model_evaluation_file_exists_or_lfs():
    """DEFAULT_CENTRALIZED's evaluation JSON must exist on disk (may be LFS)."""
    eval_path = os.path.join(ROOT, MODELS[DEFAULT_CENTRALIZED]["evaluation"])
    assert os.path.isfile(eval_path), (
        f"Evaluation file for DEFAULT_CENTRALIZED ('{DEFAULT_CENTRALIZED}') "
        f"not found at: {eval_path}"
    )
