"""Investigate model performance, focusing on HYP and MI false negatives.

This script parses existing test-fold predictions, metadata and diagnostic statements
to provide detailed empirical breakdowns of classification errors without retraining
or modifying models.
"""

import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.preprocessing import CLASSES


def _calc_age_stats(series):
    """Compute age statistics explicitly documenting the PTB-XL age > 89 sentinel (300)."""
    valid = series[series <= 100]
    sentinel_count = int((series == 300).sum())
    return {
        "median": float(series.median()),
        "valid_mean": float(valid.mean()) if len(valid) > 0 else 0.0,
        "valid_std": float(valid.std()) if len(valid) > 1 else 0.0,
        "valid_median": float(valid.median()) if len(valid) > 0 else 0.0,
        "valid_count": int(len(valid)),
        "sentinel_count": sentinel_count,
        "sentinel_pct": float(sentinel_count / len(series)) if len(series) > 0 else 0.0,
        "raw_mean": float(series.mean()),
        "raw_std": float(series.std()),
    }


def _calc_sex_stats(series):
    """Compute sex statistics in PTB-XL: 0 = Male, 1 = Female."""
    female_count = int((series == 1).sum())
    male_count = int((series == 0).sum())
    total = len(series)
    return {
        "female_count": female_count,
        "female_pct": float(female_count / total) if total > 0 else 0.0,
        "male_count": male_count,
        "male_pct": float(male_count / total) if total > 0 else 0.0,
        "total": total,
    }


def run_investigation(root=ROOT):
    """Compute detailed empirical metrics on HYP and MI error distributions."""
    raw_csv = root / "data" / "raw" / "ptb-xl" / "ptbxl_database.csv"
    scp_csv = root / "data" / "raw" / "ptb-xl" / "scp_statements.csv"
    prob_csv = root / "results" / "revised" / "statistics" / "test_probabilities_multiscale.csv"
    eval_json = root / "results" / "scheduler_experiment" / "metrics" / "multiscale_final_evaluation.json"

    with open(eval_json) as f:
        eval_data = json.load(f)
    thresholds = eval_data["thresholds"]

    probs = pd.read_csv(prob_csv, index_col="ecg_id")
    meta = pd.read_csv(raw_csv, index_col="ecg_id")
    scp_df = pd.read_csv(scp_csv, index_col=0)

    diag_scp = scp_df[pd.to_numeric(scp_df["diagnostic"], errors="coerce") == 1]
    code_to_class = diag_scp["diagnostic_class"].to_dict()

    test_df = meta.loc[probs.index].copy()

    # Label extraction following CardioFed policy
    def get_labels(scp_str):
        codes = ast.literal_eval(scp_str)
        classes = set()
        for c, v in codes.items():
            if c in code_to_class and v > 0:
                classes.add(code_to_class[c])
        if classes - {"NORM"}:
            classes -= {"NORM"}
        return classes

    test_df["labels"] = test_df["scp_codes"].apply(get_labels)
    test_df["actual_str"] = test_df["labels"].apply(lambda s: ", ".join(sorted(s)))

    for c in CLASSES:
        test_df[f"true_{c}"] = test_df["labels"].apply(lambda s: int(c in s))
        test_df[f"prob_{c}"] = probs[f"{c}_probability"]
        test_df[f"pred_{c}"] = (test_df[f"prob_{c}"] >= thresholds[c]).astype(int)

    # Post-hoc NORM consistency rule
    has_abnormal = (test_df[["pred_MI", "pred_STTC", "pred_CD", "pred_HYP"]].sum(axis=1) > 0)
    test_df["final_pred_NORM"] = test_df["pred_NORM"].copy()
    test_df.loc[has_abnormal, "final_pred_NORM"] = 0

    def format_pred_raw(row):
        active = [c for c in CLASSES if row[f"pred_{c}"] == 1]
        return ", ".join(active) if active else "None"

    def format_pred_final(row):
        active = []
        if row["final_pred_NORM"] == 1:
            active.append("NORM")
        for c in ["MI", "STTC", "CD", "HYP"]:
            if row[f"pred_{c}"] == 1:
                active.append(c)
        return ", ".join(active) if active else "None"

    test_df["pred_raw_str"] = test_df.apply(format_pred_raw, axis=1)
    test_df["pred_final_str"] = test_df.apply(format_pred_final, axis=1)

    # 1. Error counts summary
    summary = {}
    for c in CLASSES:
        tp = int(((test_df[f"true_{c}"] == 1) & (test_df[f"pred_{c}"] == 1)).sum())
        fp = int(((test_df[f"true_{c}"] == 0) & (test_df[f"pred_{c}"] == 1)).sum())
        fn = int(((test_df[f"true_{c}"] == 1) & (test_df[f"pred_{c}"] == 0)).sum())
        tn = int(((test_df[f"true_{c}"] == 0) & (test_df[f"pred_{c}"] == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        summary[c] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "support": tp + fn,
            "prevalence": (tp + fn) / len(test_df),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "threshold": thresholds[c],
        }

    # 2. MI Detailed Breakdown
    mi_records = test_df[test_df["true_MI"] == 1].copy()
    mi_tp = mi_records[mi_records["pred_MI"] == 1]
    mi_fn = mi_records[mi_records["pred_MI"] == 0]

    def get_mi_subcodes(scp_str):
        codes = ast.literal_eval(scp_str)
        return [c for c, v in codes.items() if c in code_to_class and code_to_class[c] == "MI" and v > 0]

    mi_records["subcodes"] = mi_records["scp_codes"].apply(get_mi_subcodes)
    mi_breakdown = {}
    for _, r in mi_records.iterrows():
        is_fn = (r["pred_MI"] == 0)
        for code in r["subcodes"]:
            if code not in mi_breakdown:
                mi_breakdown[code] = {"total": 0, "fn": 0, "tp": 0}
            mi_breakdown[code]["total"] += 1
            if is_fn:
                mi_breakdown[code]["fn"] += 1
            else:
                mi_breakdown[code]["tp"] += 1

    for code, stats in mi_breakdown.items():
        stats["fn_rate"] = stats["fn"] / stats["total"] if stats["total"] > 0 else 0.0

    # MI missed predictions distributions
    mi_fn_pred_raw = mi_fn["pred_raw_str"].value_counts().to_dict()
    mi_fn_pred_final = mi_fn["pred_final_str"].value_counts().to_dict()

    # MI demographics
    mi_demographics = {
        "TP_age": _calc_age_stats(mi_tp["age"]),
        "FN_age": _calc_age_stats(mi_fn["age"]),
        "TP_sex": _calc_sex_stats(mi_tp["sex"]),
        "FN_sex": _calc_sex_stats(mi_fn["sex"]),
    }

    # MI co-occurring conditions
    mi_co_occurring = {
        "FN_sole_MI_count": int(((mi_fn["true_CD"] == 0) & (mi_fn["true_STTC"] == 0) & (mi_fn["true_HYP"] == 0)).sum()),
        "FN_sole_MI_pct": float(((mi_fn["true_CD"] == 0) & (mi_fn["true_STTC"] == 0) & (mi_fn["true_HYP"] == 0)).mean()),
        "TP_sole_MI_count": int(((mi_tp["true_CD"] == 0) & (mi_tp["true_STTC"] == 0) & (mi_tp["true_HYP"] == 0)).sum()),
        "TP_sole_MI_pct": float(((mi_tp["true_CD"] == 0) & (mi_tp["true_STTC"] == 0) & (mi_tp["true_HYP"] == 0)).mean()),
        "FN_with_CD_count": int((mi_fn["true_CD"] == 1).sum()),
        "FN_with_CD_pct": float((mi_fn["true_CD"] == 1).mean()),
        "TP_with_CD_count": int((mi_tp["true_CD"] == 1).sum()),
        "TP_with_CD_pct": float((mi_tp["true_CD"] == 1).mean()),
        "FN_with_STTC_count": int((mi_fn["true_STTC"] == 1).sum()),
        "FN_with_STTC_pct": float((mi_fn["true_STTC"] == 1).mean()),
        "TP_with_STTC_count": int((mi_tp["true_STTC"] == 1).sum()),
        "TP_with_STTC_pct": float((mi_tp["true_STTC"] == 1).mean()),
    }

    # 3. HYP Detailed Breakdown
    hyp_records = test_df[test_df["true_HYP"] == 1].copy()
    hyp_tp = hyp_records[hyp_records["pred_HYP"] == 1]
    hyp_fn = hyp_records[hyp_records["pred_HYP"] == 0]
    hyp_fp = test_df[(test_df["true_HYP"] == 0) & (test_df["pred_HYP"] == 1)].copy()

    def get_hyp_subcodes(scp_str):
        codes = ast.literal_eval(scp_str)
        return [c for c, v in codes.items() if c in code_to_class and code_to_class[c] == "HYP" and v > 0]

    hyp_records["subcodes"] = hyp_records["scp_codes"].apply(get_hyp_subcodes)
    hyp_breakdown = {}
    for _, r in hyp_records.iterrows():
        is_fn = (r["pred_HYP"] == 0)
        for code in r["subcodes"]:
            if code not in hyp_breakdown:
                hyp_breakdown[code] = {"total": 0, "fn": 0, "tp": 0}
            hyp_breakdown[code]["total"] += 1
            if is_fn:
                hyp_breakdown[code]["fn"] += 1
            else:
                hyp_breakdown[code]["tp"] += 1

    for code, stats in hyp_breakdown.items():
        stats["fn_rate"] = stats["fn"] / stats["total"] if stats["total"] > 0 else 0.0

    # HYP FP sources
    hyp_fp_exact_labels = hyp_fp["actual_str"].value_counts().to_dict()
    hyp_fp_superclasses = {
        "STTC_present": int((hyp_fp["true_STTC"] == 1).sum()),
        "STTC_pct": float((hyp_fp["true_STTC"] == 1).mean()),
        "CD_present": int((hyp_fp["true_CD"] == 1).sum()),
        "CD_pct": float((hyp_fp["true_CD"] == 1).mean()),
        "MI_present": int((hyp_fp["true_MI"] == 1).sum()),
        "MI_pct": float((hyp_fp["true_MI"] == 1).mean()),
        "NORM_present": int((hyp_fp["true_NORM"] == 1).sum()),
        "NORM_pct": float((hyp_fp["true_NORM"] == 1).mean()),
        "abnormal_present": int(((hyp_fp["true_STTC"] == 1) | (hyp_fp["true_CD"] == 1) | (hyp_fp["true_MI"] == 1)).sum()),
        "abnormal_pct": float(((hyp_fp["true_STTC"] == 1) | (hyp_fp["true_CD"] == 1) | (hyp_fp["true_MI"] == 1)).mean()),
    }

    # HYP demographics
    hyp_demographics = {
        "TP_age": _calc_age_stats(hyp_tp["age"]),
        "FN_age": _calc_age_stats(hyp_fn["age"]),
        "FP_age": _calc_age_stats(hyp_fp["age"]),
        "TP_sex": _calc_sex_stats(hyp_tp["sex"]),
        "FN_sex": _calc_sex_stats(hyp_fn["sex"]),
        "FP_sex": _calc_sex_stats(hyp_fp["sex"]),
    }

    # 4. Site-level analysis for MI and HYP
    # Compute 4 primary site groups: site 0, site 1, site 2, other sites
    site_groups = {}
    for name, site_filter in [
        ("site 0", test_df["site"] == 0),
        ("site 1", test_df["site"] == 1),
        ("site 2", test_df["site"] == 2),
        ("other sites", ~test_df["site"].isin([0, 1, 2])),
    ]:
        grp = test_df[site_filter]
        n_rec = len(grp)
        mi_tot = int((grp["true_MI"] == 1).sum())
        mi_fn = int(((grp["true_MI"] == 1) & (grp["pred_MI"] == 0)).sum())
        hyp_tot = int((grp["true_HYP"] == 1).sum())
        hyp_fn = int(((grp["true_HYP"] == 1) & (grp["pred_HYP"] == 0)).sum())
        norm_tot = int((grp["true_NORM"] == 1).sum())

        site_groups[name] = {
            "records": n_rec,
            "NORM_count": norm_tot,
            "NORM_prevalence": norm_tot / n_rec if n_rec > 0 else 0.0,
            "MI_total": mi_tot,
            "MI_FN": mi_fn,
            "MI_FN_rate": mi_fn / mi_tot if mi_tot > 0 else 0.0,
            "HYP_total": hyp_tot,
            "HYP_FN": hyp_fn,
            "HYP_FN_rate": hyp_fn / hyp_tot if hyp_tot > 0 else 0.0,
        }

    return {
        "class_summary": summary,
        "mi_subcodes": mi_breakdown,
        "mi_demographics": mi_demographics,
        "mi_fn_predictions": {
            "raw": mi_fn_pred_raw,
            "final": mi_fn_pred_final,
        },
        "mi_co_occurring": mi_co_occurring,
        "hyp_subcodes": hyp_breakdown,
        "hyp_fp_sources": {
            "exact_label_sets": hyp_fp_exact_labels,
            "superclass_incidence": hyp_fp_superclasses,
        },
        "hyp_demographics": hyp_demographics,
        "site_groups": site_groups,
    }


if __name__ == "__main__":
    results = run_investigation()
    out_dir = ROOT / "results" / "revised" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hyp_mi_investigation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Investigation results saved to {out_file}")
