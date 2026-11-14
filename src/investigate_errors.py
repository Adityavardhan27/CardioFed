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
    for c in CLASSES:
        test_df[f"true_{c}"] = test_df["labels"].apply(lambda s: int(c in s))
        test_df[f"prob_{c}"] = probs[f"{c}_probability"]
        test_df[f"pred_{c}"] = (test_df[f"prob_{c}"] >= thresholds[c]).astype(int)

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
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "threshold": thresholds[c],
        }

    # 2. MI Sub-statement breakdown
    mi_records = test_df[test_df["true_MI"] == 1].copy()
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

    # 3. HYP Sub-statement breakdown
    hyp_records = test_df[test_df["true_HYP"] == 1].copy()
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

    # 4. Site-level analysis for MI and HYP
    site_analysis = {}
    for site, grp in test_df.groupby("site"):
        site_key = f"site_{int(site)}"
        mi_tot = int((grp["true_MI"] == 1).sum())
        mi_fn = int(((grp["true_MI"] == 1) & (grp["pred_MI"] == 0)).sum())
        hyp_tot = int((grp["true_HYP"] == 1).sum())
        hyp_fn = int(((grp["true_HYP"] == 1) & (grp["pred_HYP"] == 0)).sum())
        site_analysis[site_key] = {
            "records": len(grp),
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
        "hyp_subcodes": hyp_breakdown,
        "site_analysis": site_analysis,
    }


if __name__ == "__main__":
    results = run_investigation()
    out_dir = ROOT / "results" / "revised" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hyp_mi_investigation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Investigation results saved to {out_file}")
