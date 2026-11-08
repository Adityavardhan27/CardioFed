"""Evaluate any CardioFed checkpoint with the standard protocol (P2 runs).

Protocol (identical to the original evaluations):
  1. thresholds = per-class F1-maximising thresholds on validation (fold 9);
  2. test fold 10 evaluated once with those thresholds;
  3. additionally: per-site test metrics and saved test probabilities
     (used for patient-level bootstrap comparisons).

Writes <output-dir>/evaluation.json and <output-dir>/test_probabilities.csv.
Refuses to overwrite existing outputs unless --overwrite is given.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.evaluate import compute_metrics
from src.io_utils import guard_outputs
from src.model_registry import build_model, load_split, predict_array
from src.per_site_evaluation import site_group
from src.preprocessing import CLASSES
from src.thresholds import tune_thresholds


def evaluate(checkpoint_path, arch="multiscale", processed="data/processed",
             metadata="data/raw/ptb-xl/ptbxl_database.csv"):
    model = build_model(arch)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    xv, yv, _ = load_split("val", processed)
    pv = predict_array(model, xv)
    thresholds = tune_thresholds(pv, yv)

    xt, yt, test_idx = load_split("test", processed)
    pt = predict_array(model, xt)

    ecg_ids = np.load(os.path.join(processed, "ecg_ids.npy"))[test_idx]
    meta = pd.read_csv(metadata, index_col="ecg_id").loc[ecg_ids]
    groups = meta["site"].map(site_group).to_numpy()

    per_site = {}
    for g in ("site 0", "site 1", "site 2", "other sites"):
        m = groups == g
        sm = compute_metrics(pt[m], yt[m], thresholds)
        per_site[g] = {"records": int(m.sum()),
                       "macro_auroc": sm["macro_auroc"],
                       "macro_f1": sm["macro_f1"]}

    info = {k: v for k, v in checkpoint.items()
            if k in ("epoch", "round", "best_val_auroc", "seed")}

    result = {
        "checkpoint": checkpoint_path.replace("\\", "/"),
        "arch": arch,
        "processed": processed,
        "checkpoint_info": info,
        "threshold_selection": "validation fold 9, per-class max F1",
        "thresholds": dict(zip(CLASSES, thresholds.tolist())),
        "validation_metrics": compute_metrics(pv, yv, thresholds),
        "test_metrics": compute_metrics(pt, yt, thresholds),
        "per_site_test": per_site,
    }
    probs = pd.DataFrame(pt, columns=[f"{c}_probability" for c in CLASSES])
    probs.insert(0, "ecg_id", ecg_ids)
    probs.insert(1, "patient_id", meta["patient_id"].to_numpy())
    probs.insert(2, "site_group", groups)
    return result, probs


def main(args):
    out_json = os.path.join(args.output_dir, "evaluation.json")
    out_csv = os.path.join(args.output_dir, "test_probabilities.csv")
    guard_outputs([out_json, out_csv], overwrite=args.overwrite)

    result, probs = evaluate(args.checkpoint, args.arch, args.processed, args.metadata)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    probs.to_csv(out_csv, index=False)

    t = result["test_metrics"]
    print(f"{args.checkpoint}: test macro-AUROC {t['macro_auroc']:.4f} "
          f"macro-F1 {t['macro_f1']:.4f} -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="multiscale", choices=["multiscale", "ecgnet"])
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--metadata", default="data/raw/ptb-xl/ptbxl_database.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
