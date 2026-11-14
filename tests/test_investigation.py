"""Tests for model performance investigation artifacts, calculations, and report consistency."""

import json
from pathlib import Path

import pytest

from src.investigate_errors import run_investigation
from src.preprocessing import CLASSES

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def investigation_data():
    """Run investigation calculations once for test assertions."""
    return run_investigation(root=ROOT)


def test_investigation_outputs_consistency(investigation_data):
    """Verify that run_investigation computes correct error matrices matching evaluation."""
    class_summary = investigation_data["class_summary"]
    assert set(class_summary.keys()) == set(CLASSES)

    eval_path = ROOT / "results" / "scheduler_experiment" / "metrics" / "multiscale_final_evaluation.json"
    assert eval_path.exists()
    with open(eval_path) as f:
        eval_data = json.load(f)

    test_metrics = eval_data["test_metrics"]

    for c in CLASSES:
        summary_c = class_summary[c]
        expected_support = test_metrics["per_class_support"][c]
        expected_f1 = test_metrics["per_class_f1"][c]

        assert summary_c["support"] == expected_support, f"Support mismatch for {c}"
        assert pytest.approx(summary_c["f1"], abs=1e-4) == expected_f1, f"F1 mismatch for {c}"


def test_hyp_error_profile_invariants(investigation_data):
    """Verify specific HYP performance characteristics documented in investigation report."""
    hyp = investigation_data["class_summary"]["HYP"]

    # HYP has the lowest F1 score across all classes
    f1_scores = {c: investigation_data["class_summary"][c]["f1"] for c in CLASSES}
    assert min(f1_scores, key=f1_scores.get) == "HYP"

    # HYP false positives outnumber true positives (precision < 50%)
    assert hyp["FP"] > hyp["TP"]
    assert hyp["FP"] == 197
    assert hyp["TP"] == 136
    assert hyp["FN"] == 126
    assert hyp["precision"] < 0.50

    # HYP threshold is the highest among all classes
    thresholds = {c: investigation_data["class_summary"][c]["threshold"] for c in CLASSES}
    assert max(thresholds, key=thresholds.get) == "HYP"


def test_hyp_fp_sources(investigation_data):
    """Verify exact sources of the 197 HYP False Positives."""
    hyp_fp = investigation_data["hyp_fp_sources"]
    exact_labels = hyp_fp["exact_label_sets"]

    # Total must equal 197
    assert sum(exact_labels.values()) == 197

    # Exact counts matching verified labels
    assert exact_labels["STTC"] == 49
    assert exact_labels["CD"] == 38
    assert exact_labels["CD, MI"] == 28
    assert exact_labels["MI"] == 25
    assert exact_labels["NORM"] == 22
    assert exact_labels["MI, STTC"] == 16
    assert exact_labels["CD, STTC"] == 12
    assert exact_labels["CD, MI, STTC"] == 7

    # Superclass incidence
    incidence = hyp_fp["superclass_incidence"]
    assert incidence["CD_present"] == 85
    assert incidence["STTC_present"] == 84
    assert incidence["MI_present"] == 76
    assert incidence["NORM_present"] == 22
    assert incidence["abnormal_present"] == 175
    assert pytest.approx(incidence["abnormal_pct"], abs=1e-3) == 0.888


def test_mi_subcode_breakdown_invariants(investigation_data):
    """Verify MI anatomical breakdown invariants (ASMI vs IMI error disparity)."""
    mi_sub = investigation_data["mi_subcodes"]

    assert "ASMI" in mi_sub
    assert "IMI" in mi_sub
    assert "LMI" in mi_sub

    # Anteroseptal MI (ASMI) is well detected (<10% FN rate)
    assert mi_sub["ASMI"]["total"] == 234
    assert mi_sub["ASMI"]["fn"] == 15
    assert mi_sub["ASMI"]["fn_rate"] < 0.10

    # Inferior MI (IMI) represents >50% of all missed MIs (65/119)
    assert mi_sub["IMI"]["total"] == 267
    assert mi_sub["IMI"]["fn"] == 65
    assert mi_sub["IMI"]["fn"] / 119 > 0.50
    assert mi_sub["IMI"]["fn_rate"] > 0.20

    # Lateral MI (LMI) has >50% miss rate
    assert mi_sub["LMI"]["total"] == 20
    assert mi_sub["LMI"]["fn"] == 11
    assert mi_sub["LMI"]["fn_rate"] == 0.55


def test_mi_fn_predictions(investigation_data):
    """Verify the predicted labels assigned to the 119 missed MI cases."""
    preds = investigation_data["mi_fn_predictions"]
    raw = preds["raw"]
    final = preds["final"]

    assert sum(raw.values()) == 119
    assert sum(final.values()) == 119

    # Raw predictions
    assert raw["NORM"] == 54
    assert raw["STTC, HYP"] == 16
    assert raw["STTC"] == 16
    assert raw["CD"] == 12
    assert raw["NORM, CD"] == 5
    assert raw["None"] == 5  # Exactly 5 have no class predicted

    # Final predictions (with NORM consistency rule suppressing NORM on NORM, CD)
    assert final["NORM"] == 54
    assert final["CD"] == 17  # 12 + 5
    assert final["None"] == 5


def test_demographics_calculations(investigation_data):
    """Verify age (including sentinel 300 handling) and sex (0=Male, 1=Female) calculations."""
    mi_demo = investigation_data["mi_demographics"]
    hyp_demo = investigation_data["hyp_demographics"]

    # MI Demographics
    assert mi_demo["TP_age"]["median"] == 73.0
    assert mi_demo["FN_age"]["median"] == 67.0
    assert mi_demo["TP_age"]["sentinel_count"] == 14
    assert mi_demo["FN_age"]["sentinel_count"] == 1
    assert pytest.approx(mi_demo["TP_age"]["valid_mean"], abs=0.1) == 70.8
    assert pytest.approx(mi_demo["FN_age"]["valid_mean"], abs=0.1) == 66.2

    # MI Sex (1 = Female, 0 = Male)
    assert mi_demo["TP_sex"]["female_count"] == 189
    assert mi_demo["FN_sex"]["female_count"] == 64
    assert pytest.approx(mi_demo["TP_sex"]["female_pct"], abs=1e-3) == 0.439
    assert pytest.approx(mi_demo["FN_sex"]["female_pct"], abs=1e-3) == 0.538

    # HYP Demographics
    assert hyp_demo["TP_age"]["median"] == 74.0
    assert hyp_demo["FN_age"]["median"] == 66.0
    assert hyp_demo["FP_age"]["median"] == 71.0
    assert hyp_demo["TP_age"]["sentinel_count"] == 4
    assert hyp_demo["FN_age"]["sentinel_count"] == 1
    assert hyp_demo["FP_age"]["sentinel_count"] == 8

    # HYP Sex
    assert hyp_demo["TP_sex"]["female_count"] == 72
    assert hyp_demo["FN_sex"]["female_count"] == 55
    assert pytest.approx(hyp_demo["TP_sex"]["female_pct"], abs=1e-3) == 0.529
    assert pytest.approx(hyp_demo["FN_sex"]["female_pct"], abs=1e-3) == 0.437


def test_site_groups_prevalence(investigation_data):
    """Verify test-set site-wise NORM prevalence and error rates."""
    sites = investigation_data["site_groups"]

    assert sites["site 0"]["records"] == 483
    assert pytest.approx(sites["site 0"]["NORM_prevalence"], abs=1e-3) == 0.642
    assert sites["site 0"]["MI_FN"] == 30
    assert sites["site 0"]["MI_total"] == 78
    assert pytest.approx(sites["site 0"]["MI_FN_rate"], abs=1e-3) == 0.385

    assert sites["site 1"]["records"] == 597
    assert pytest.approx(sites["site 1"]["NORM_prevalence"], abs=1e-3) == 0.444

    assert sites["site 2"]["records"] == 867
    assert pytest.approx(sites["site 2"]["NORM_prevalence"], abs=1e-3) == 0.326

    assert sites["other sites"]["records"] == 210
    assert pytest.approx(sites["other sites"]["NORM_prevalence"], abs=1e-3) == 0.276


def test_report_text_tone_and_evidence():
    """Verify report text does not use ungrounded causal assertions."""
    doc_path = ROOT / "docs" / "MODEL_PERFORMANCE_INVESTIGATION.md"
    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")

    # Ensure objective, qualified terminology
    assert "The Core Flaw" not in content
    assert "Voltage Erasure" not in content
    assert "Root Cause Matrix" not in content

    # Ensure key qualified sections are present
    assert "Amplitude Normalisation as a Plausible Factor" in content
    assert "Summary of Investigated Contributing Factors" in content
    assert "Sentinel Age 300" in content
    assert "hyp_mi_investigation.json" in content
