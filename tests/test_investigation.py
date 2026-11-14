"""Tests for model performance investigation artifacts and error analysis integrity."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.investigate_errors import run_investigation
from src.preprocessing import CLASSES

ROOT = Path(__file__).resolve().parent.parent


def test_investigation_outputs_consistency():
    """Verify that run_investigation computes correct error matrices matching evaluation."""
    data = run_investigation(root=ROOT)
    class_summary = data["class_summary"]

    assert set(class_summary.keys()) == set(CLASSES)

    # Load stored final evaluation
    eval_path = ROOT / "results" / "scheduler_experiment" / "metrics" / "multiscale_final_evaluation.json"
    assert eval_path.exists()
    with open(eval_path) as f:
        eval_data = json.load(f)

    test_metrics = eval_data["test_metrics"]

    # Verify per-class support and F1 match within numerical precision
    for c in CLASSES:
        summary_c = class_summary[c]
        expected_support = test_metrics["per_class_support"][c]
        expected_f1 = test_metrics["per_class_f1"][c]

        assert summary_c["support"] == expected_support, f"Support mismatch for {c}"
        assert pytest.approx(summary_c["f1"], abs=1e-4) == expected_f1, f"F1 mismatch for {c}"


def test_hyp_error_profile_invariants():
    """Verify specific HYP performance characteristics documented in investigation report."""
    data = run_investigation(root=ROOT)
    hyp = data["class_summary"]["HYP"]

    # HYP has the lowest F1 score across all classes
    f1_scores = {c: data["class_summary"][c]["f1"] for c in CLASSES}
    assert min(f1_scores, key=f1_scores.get) == "HYP"

    # HYP false positives outnumber true positives (precision < 50%)
    assert hyp["FP"] > hyp["TP"]
    assert hyp["precision"] < 0.50

    # HYP threshold is the highest among all classes
    thresholds = {c: data["class_summary"][c]["threshold"] for c in CLASSES}
    assert max(thresholds, key=thresholds.get) == "HYP"


def test_mi_subcode_breakdown_invariants():
    """Verify MI anatomical breakdown invariants (e.g. ASMI vs IMI error disparity)."""
    data = run_investigation(root=ROOT)
    mi_sub = data["mi_subcodes"]

    assert "ASMI" in mi_sub
    assert "IMI" in mi_sub

    # Anteroseptal MI (ASMI) is much better detected than Inferior MI (IMI)
    assert mi_sub["ASMI"]["fn_rate"] < 0.10  # Less than 10% missed
    assert mi_sub["IMI"]["fn_rate"] > 0.20   # Greater than 20% missed
    assert mi_sub["IMI"]["fn"] > mi_sub["ASMI"]["fn"]


def test_hyp_subcode_breakdown_invariants():
    """Verify HYP sub-statement invariants (atrial vs ventricular hypertrophy)."""
    data = run_investigation(root=ROOT)
    hyp_sub = data["hyp_subcodes"]

    assert "LVH" in hyp_sub
    assert "LAO/LAE" in hyp_sub

    # Atrial overload has a substantially higher FN rate than ventricular hypertrophy
    assert hyp_sub["LAO/LAE"]["fn_rate"] > hyp_sub["LVH"]["fn_rate"]
    assert hyp_sub["LAO/LAE"]["fn_rate"] > 0.50  # Over half of atrial overload missed


def test_investigation_report_document_exists():
    """Verify that the investigation markdown report is present and comprehensive."""
    doc_path = ROOT / "docs" / "MODEL_PERFORMANCE_INVESTIGATION.md"
    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")

    # Check key sections are discussed
    assert "HYP (Hypertrophy)" in content
    assert "MI (Myocardial Infarction)" in content
    assert "Z-Scoring" in content or "z-score" in content
    assert "ASMI" in content
    assert "IMI" in content
    assert "Recommendations" in content
