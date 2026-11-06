"""CardioFed: comparison of the primary centralized and federated models.

Reads the original evaluation JSONs (via src/model_registry.py) and, for
confidence intervals and significance, the outputs of
`python -m src.statistical_analysis` (run that first).

Writes to results/revised/comparison/ by default. The original
results/comparison/ outputs are left untouched; note that their
comparison_summary.json names "best" models, which the statistical analysis
does not support (no pairwise difference among the 8-round federated,
centralized multi-scale and baseline models is significant).

Existing outputs are never overwritten unless --overwrite is given.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io_utils import guard_outputs
from src.model_registry import MODELS, PRIMARY, load_evaluation
from src.preprocessing import CLASSES

OUTPUT_FILES = [
    "overall_comparison.csv", "per_class_comparison.csv",
    "comparison_summary.json", "macro_auroc.png", "macro_f1.png",
    "hamming_loss.png", "subset_accuracy.png", "per_class_f1.png",
    "per_class_auroc.png",
]


def load_results(statistics_dir):
    overall_ci = pd.read_csv(
        os.path.join(statistics_dir, "overall_with_ci.csv")).set_index("model")
    class_ci = pd.read_csv(
        os.path.join(statistics_dir, "per_class_with_ci.csv")
    ).set_index(["model", "class"])

    overall, per_class = [], []
    for name in PRIMARY:
        _, m = load_evaluation(name)
        ci = overall_ci.loc[name]
        overall.append({
            "model": name, "label": MODELS[name]["label"],
            "macro_auroc": m["macro_auroc"],
            "macro_auroc_ci_low": ci.macro_auroc_ci_low,
            "macro_auroc_ci_high": ci.macro_auroc_ci_high,
            "macro_f1": m["macro_f1"],
            "macro_f1_ci_low": ci.macro_f1_ci_low,
            "macro_f1_ci_high": ci.macro_f1_ci_high,
            "hamming_loss": m["hamming_loss"],
            "subset_accuracy": m["subset_accuracy"],
        })
        for c in CLASSES:
            cc = class_ci.loc[(name, c)]
            per_class.append({
                "model": name, "label": MODELS[name]["label"], "class": c,
                "auroc": m["per_class_auroc"][c],
                "auroc_ci_low": cc.auroc_ci_low, "auroc_ci_high": cc.auroc_ci_high,
                "f1": m["per_class_f1"][c],
                "f1_ci_low": cc.f1_ci_low, "f1_ci_high": cc.f1_ci_high,
                "precision": m["per_class_precision"][c],
                "recall": m["per_class_recall"][c],
                "support": m["per_class_support"][c],
            })
    return pd.DataFrame(overall), pd.DataFrame(per_class)


def bar(df, column, title, path, ci=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    err = None
    if ci:
        err = [df[column] - df[f"{column}_ci_low"], df[f"{column}_ci_high"] - df[column]]
    bars = ax.bar(df["label"], df[column], yerr=err, capsize=5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title + (" (95% patient-bootstrap CI)" if ci else ""))
    ax.set_ylabel(column.replace("_", " "))
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def grouped(per_class, column, title, path):
    pivot = per_class.pivot(index="class", columns="label", values=column).loc[CLASSES]
    lo = per_class.pivot(index="class", columns="label", values=f"{column}_ci_low").loc[CLASSES]
    hi = per_class.pivot(index="class", columns="label", values=f"{column}_ci_high").loc[CLASSES]
    fig, ax = plt.subplots(figsize=(12, 7))
    n = len(pivot.columns)
    width = 0.8 / n
    x = np.arange(len(CLASSES))
    for k, col in enumerate(pivot.columns):
        ax.bar(x + (k - (n - 1) / 2) * width, pivot[col], width,
               yerr=[pivot[col] - lo[col], hi[col] - pivot[col]], capsize=2, label=col)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1)
    ax.set_title(title + " (95% patient-bootstrap CI)")
    ax.set_xlabel("Diagnostic class")
    ax.set_ylabel(column.upper() if column == "auroc" else "F1 score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(args):
    out = args.output_dir
    paths = dict(zip(OUTPUT_FILES, guard_outputs(
        [os.path.join(out, f) for f in OUTPUT_FILES], overwrite=args.overwrite)))

    overall, per_class = load_results(args.statistics_dir)
    overall.to_csv(paths["overall_comparison.csv"], index=False)
    per_class.to_csv(paths["per_class_comparison.csv"], index=False)

    bar(overall, "macro_auroc", "Test macro-AUROC", paths["macro_auroc.png"])
    bar(overall, "macro_f1", "Test macro-F1", paths["macro_f1.png"])
    bar(overall, "hamming_loss", "Test Hamming loss", paths["hamming_loss.png"], ci=False)
    bar(overall, "subset_accuracy", "Test subset accuracy", paths["subset_accuracy.png"], ci=False)
    grouped(per_class, "f1", "Per-class test F1", paths["per_class_f1.png"])
    grouped(per_class, "auroc", "Per-class test AUROC", paths["per_class_auroc.png"])

    diffs = pd.read_csv(os.path.join(args.statistics_dir, "paired_differences.csv"))
    summary = {
        "models_compared": overall["model"].tolist(),
        "note": ("Point estimates only rank models; use the paired comparisons "
                 "for claims. Thresholds are validation-tuned per model."),
        "paired_comparisons": diffs[[
            "model_a", "model_b", "metric", "difference", "ci_low", "ci_high",
            "p_value", "p_value_holm", "significant_at_0.05_holm"]].to_dict(orient="records"),
        "significant_differences": diffs[diffs["significant_at_0.05_holm"]][
            ["model_a", "model_b", "metric"]].to_dict(orient="records"),
    }
    with open(paths["comparison_summary.json"], "w") as f:
        json.dump(summary, f, indent=2)

    print(overall.drop(columns="label").round(4).to_string(index=False))
    print("\nSignificant differences (Holm, 0.05):", summary["significant_differences"])
    print("Saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", default="results/revised/statistics")
    parser.add_argument("--output-dir", default="results/revised/comparison")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
