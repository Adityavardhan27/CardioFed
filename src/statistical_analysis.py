"""Uncertainty estimates and paired model comparisons on the test fold.

For every model in src/model_registry.py this script:
  1. recomputes test probabilities from the checkpoint and verifies that,
     with the thresholds saved in the original evaluation JSON, they reproduce
     the reported test metrics (aborts otherwise);
  2. computes 95% percentile bootstrap confidence intervals for macro- and
     per-class AUROC/F1, resampling PATIENTS (some test patients have several
     ECGs, so records are not independent);
  3. compares pre-specified model pairs with a paired bootstrap and reports
     two-sided p-values with Holm correction.

Thresholds are never re-tuned here: each model keeps its validation-tuned
thresholds from the original evaluation.

Outputs (default results/revised/statistics/):
  overall_with_ci.csv, per_class_with_ci.csv, paired_differences.csv,
  bootstrap_summary.json, test_probabilities_<model>.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import compute_metrics
from src.io_utils import guard_outputs
from src.model_registry import (
    MODELS, PRIMARY, load_model, load_evaluation, load_split, predict_array
)
from src.preprocessing import CLASSES
from src.thresholds import thresholds_from_evaluation

# Pre-specified comparisons: (model A, model B, question answered).
COMPARISONS = [
    ("federated_r8", "multiscale",
     "Federated vs centralized training, same architecture"),
    ("baseline", "multiscale",
     "Baseline CNN vs multi-scale CNN, both centralized"),
    ("baseline", "federated_r8",
     "Centralized baseline vs federated multi-scale"),
    ("federated_r8", "federated_r3",
     "Effect of 8 vs 3 communication rounds"),
]


def macro_auc(probs, y):
    """Compute macro-AUROC across all classes in y if all are biclassed, else NaN.

    Enforces a consistent complete-class definition across all replicates.
    """
    n_classes = y.shape[1]
    for i in range(n_classes):
        if len(np.unique(y[:, i])) < 2:
            return float("nan")
    return float(np.mean([roc_auc_score(y[:, i], probs[:, i]) for i in range(n_classes)]))


def per_class_f1(pred, y):
    tp = ((pred == 1) & (y == 1)).sum(0)
    fp = ((pred == 1) & (y == 0)).sum(0)
    fn = ((pred == 0) & (y == 1)).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def holm(pvalues):
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def patient_bootstrap_indices(patients, n_boot, seed):
    """Yield record indices for each unconditional bootstrap sample of patients.

    Resamples unique patient IDs with replacement without conditioning or
    rejection, preserving patient clustering and true empirical variance.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(patients)
    groups = {p: np.where(patients == p)[0] for p in unique}
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([groups[p] for p in chosen])




def main(args):
    names = PRIMARY + (["multiscale_legacy"] if args.include_legacy else [])
    out = args.output_dir

    outputs = [os.path.join(out, f) for f in (
        "overall_with_ci.csv", "per_class_with_ci.csv",
        "paired_differences.csv", "bootstrap_summary.json")]
    outputs += [os.path.join(out, f"test_probabilities_{n}.csv") for n in names]
    guard_outputs(outputs, overwrite=args.overwrite)

    x_test, y_test, test_idx = load_split("test", args.processed)
    ecg_ids = np.load(os.path.join(args.processed, "ecg_ids.npy"))[test_idx]
    meta = pd.read_csv(args.metadata, index_col="ecg_id")
    patients = meta.loc[ecg_ids, "patient_id"].to_numpy()

    probs, thresholds, reproduced = {}, {}, {}

    for name in names:
        model, _ = load_model(name)
        p = predict_array(model, x_test)
        evaluation, saved = load_evaluation(name)
        th = thresholds_from_evaluation(evaluation)
        recomputed = compute_metrics(p, y_test, th)

        diffs = {k: abs(recomputed[k] - saved[k]) for k in
                 ("macro_auroc", "macro_f1", "hamming_loss", "subset_accuracy")}
        if max(diffs.values()) > 1e-6:
            raise RuntimeError(
                f"{name}: recomputed metrics do not match "
                f"{MODELS[name]['evaluation']}: {diffs}"
            )
        reproduced[name] = max(diffs.values())
        probs[name], thresholds[name] = p, th

        pd.DataFrame(
            p, columns=[f"{c}_probability" for c in CLASSES]
        ).assign(ecg_id=ecg_ids, patient_id=patients).to_csv(
            os.path.join(out, f"test_probabilities_{name}.csv"), index=False
        )
        print(f"{name}: reported test metrics reproduced "
              f"(max abs diff {reproduced[name]:.2e})")

    # Bootstrap.
    auc_s = {n: [] for n in names}
    f1_s = {n: [] for n in names}
    cls_auc = {n: [] for n in names}
    cls_f1 = {n: [] for n in names}
    preds = {n: (probs[n] >= thresholds[n]).astype(int) for n in names}

    for ix in patient_bootstrap_indices(patients, args.n_boot, args.seed):
        yb = y_test[ix]
        class_valid = [bool(len(np.unique(yb[:, i])) == 2) for i in range(5)]
        all_classes_valid = all(class_valid)

        for n in names:
            pc_auc = [
                float(roc_auc_score(yb[:, i], probs[n][ix, i]))
                if class_valid[i]
                else float("nan")
                for i in range(5)
            ]
            pc_f1 = per_class_f1(preds[n][ix], yb)
            cls_auc[n].append(pc_auc)
            cls_f1[n].append(pc_f1)
            # Consistent 5-class macro-AUROC across replicates
            if all_classes_valid:
                auc_s[n].append(float(np.mean(pc_auc)))
            else:
                auc_s[n].append(float("nan"))
            f1_s[n].append(float(np.mean(pc_f1)))

    def ci_and_counts(a):
        a = np.asarray(a)
        valid = a[np.isfinite(a)]
        n_total = len(a)
        n_valid = int(len(valid))
        n_excluded = int(n_total - n_valid)
        if n_valid == 0:
            return float("nan"), float("nan"), n_valid, n_excluded
        return (
            float(np.percentile(valid, 2.5)),
            float(np.percentile(valid, 97.5)),
            n_valid,
            n_excluded,
        )

    overall, per_class = [], []
    for n in names:
        point = compute_metrics(probs[n], y_test, thresholds[n])
        lo_a, hi_a, val_a, exc_a = ci_and_counts(auc_s[n])
        lo_f, hi_f, val_f, exc_f = ci_and_counts(f1_s[n])
        overall.append({
            "model": n, "label": MODELS[n]["label"],
            "macro_auroc": point["macro_auroc"],
            "macro_auroc_ci_low": lo_a, "macro_auroc_ci_high": hi_a,
            "macro_f1": point["macro_f1"],
            "macro_f1_ci_low": lo_f, "macro_f1_ci_high": hi_f,
            "hamming_loss": point["hamming_loss"],
            "subset_accuracy": point["subset_accuracy"],
            "valid_bootstrap_samples": val_a,
            "excluded_bootstrap_samples": exc_a,
        })
        ca, cf = np.array(cls_auc[n]), np.array(cls_f1[n])
        for i, c in enumerate(CLASSES):
            lo_ca, hi_ca, val_ca, exc_ca = ci_and_counts(ca[:, i])
            lo_cf, hi_cf, val_cf, exc_cf = ci_and_counts(cf[:, i])
            per_class.append({
                "model": n, "class": c,
                "auroc": point["per_class_auroc"][c],
                "auroc_ci_low": lo_ca,
                "auroc_ci_high": hi_ca,
                "f1": point["per_class_f1"][c],
                "f1_ci_low": lo_cf,
                "f1_ci_high": hi_cf,
                "precision": point["per_class_precision"][c],
                "recall": point["per_class_recall"][c],
                "support": point["per_class_support"][c],
                "valid_bootstrap_samples": val_ca,
                "excluded_bootstrap_samples": exc_ca,
            })

    rows = []
    for a, b, question in COMPARISONS:
        if a not in names or b not in names:
            continue
        for metric, samples in (("macro_auroc", auc_s), ("macro_f1", f1_s)):
            sa, sb = np.array(samples[a]), np.array(samples[b])
            valid_mask = np.isfinite(sa) & np.isfinite(sb)
            valid_count = int(valid_mask.sum())
            excluded_count = int((~valid_mask).sum())
            if valid_count == 0:
                ci_low, ci_high, p = float("nan"), float("nan"), float("nan")
            else:
                d = sa[valid_mask] - sb[valid_mask]
                ci_low = float(np.percentile(d, 2.5))
                ci_high = float(np.percentile(d, 97.5))
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                p = float(min(1.0, max(p, 1.0 / valid_count)))

            point_a = [r for r in overall if r["model"] == a][0][metric]
            point_b = [r for r in overall if r["model"] == b][0][metric]
            rows.append({
                "model_a": a, "model_b": b, "question": question,
                "metric": metric, "difference": point_a - point_b,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p,
                "valid_bootstrap_samples": valid_count,
                "excluded_bootstrap_samples": excluded_count,
            })
    diffs = pd.DataFrame(rows)
    for metric in ("macro_auroc", "macro_f1"):
        mask = diffs["metric"] == metric
        diffs.loc[mask, "p_value_holm"] = holm(diffs.loc[mask, "p_value"].to_numpy())
    diffs["significant_at_0.05_holm"] = diffs["p_value_holm"] < 0.05

    pd.DataFrame(overall).to_csv(outputs[0], index=False)
    pd.DataFrame(per_class).to_csv(outputs[1], index=False)
    diffs.to_csv(outputs[2], index=False)

    summary = {
        "test_records": int(len(y_test)),
        "test_patients": int(len(np.unique(patients))),
        "bootstrap": {
            "resampling_unit": "patient",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "interval": "95% percentile",
            "policy": "unconditional cluster bootstrap with complete-class consistency for macro-AUROC and transparent reporting of undefined replicates",
        },
        "thresholds": "validation-tuned thresholds from each original evaluation JSON",
        "p_value_floor": f"p-values are floored at 1/n_boot = {1.0 / args.n_boot}",
        "reproduction_max_abs_diff": reproduced,
        "overall": overall,
        "paired_differences": diffs.to_dict(orient="records"),
    }
    with open(outputs[3], "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + pd.DataFrame(overall)[[
        "model", "macro_auroc", "macro_auroc_ci_low", "macro_auroc_ci_high",
        "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]].round(4).to_string(index=False))
    print("\n" + diffs[["model_a", "model_b", "metric", "difference", "ci_low",
                        "ci_high", "p_value", "p_value_holm"]].round(4).to_string(index=False))
    print("\nSaved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--metadata", default="data/raw/ptb-xl/ptbxl_database.csv")
    parser.add_argument("--output-dir", default="results/revised/statistics")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
