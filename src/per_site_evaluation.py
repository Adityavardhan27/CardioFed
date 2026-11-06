"""Per-site evaluation of the trained models on the test fold.

PTB-XL records carry the recording site. Test records are grouped as
site 0, site 1, site 2 and "other" (all remaining sites, including missing),
mirroring the four federated clients (client 1 = site 0, client 2 = site 1,
client 3 = site 2, client 4 = other sites). Note that the clients were formed
from TRAINING patients only; test patients are disjoint from all clients.

Each model keeps the validation-tuned thresholds from its original
evaluation JSON. Macro-AUROC confidence intervals use a patient-level
bootstrap within each site group. Small groups give wide intervals; differences
between sites can reflect label mix as well as site/device effects.

Outputs (default results/revised/per_site/):
  per_site_metrics.csv, per_site_per_class.csv, site_label_prevalence.csv,
  per_site_summary.json, per_site_macro_auroc.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

SITE_GROUPS = ["site 0", "site 1", "site 2", "other sites"]


def site_group(site):
    if pd.isna(site):
        return "other sites"
    site = int(site)
    return f"site {site}" if site in (0, 1, 2) else "other sites"


def bootstrap_macro_auc(probs, y, patients, n_boot, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(patients)
    groups = {p: np.where(patients == p)[0] for p in unique}
    values = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[p] for p in rng.choice(unique, len(unique))])
        aucs = [roc_auc_score(y[ix, i], probs[ix, i])
                for i in range(5) if len(np.unique(y[ix, i])) == 2]
        values.append(np.mean(aucs))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main(args):
    out = args.output_dir
    files = ["per_site_metrics.csv", "per_site_per_class.csv",
             "site_label_prevalence.csv", "per_site_summary.json",
             "per_site_macro_auroc.png"]
    paths = guard_outputs([os.path.join(out, f) for f in files],
                          overwrite=args.overwrite)

    x_test, y_test, test_idx = load_split("test", args.processed)
    ecg_ids = np.load(os.path.join(args.processed, "ecg_ids.npy"))[test_idx]
    meta = pd.read_csv(args.metadata, index_col="ecg_id").loc[ecg_ids]
    groups = meta["site"].map(site_group).to_numpy()
    patients = meta["patient_id"].to_numpy()

    prevalence = []
    for g in SITE_GROUPS + ["all"]:
        m = np.ones(len(groups), bool) if g == "all" else groups == g
        prevalence.append({
            "site_group": g, "records": int(m.sum()),
            "patients": int(len(np.unique(patients[m]))),
            **{c: round(float(y_test[m, i].mean()), 4) for i, c in enumerate(CLASSES)},
        })

    overall, per_class = [], []
    for name in PRIMARY:
        model, _ = load_model(name)
        probs = predict_array(model, x_test)
        evaluation, _ = load_evaluation(name)
        th = thresholds_from_evaluation(evaluation)

        for g in SITE_GROUPS + ["all"]:
            m = np.ones(len(groups), bool) if g == "all" else groups == g
            met = compute_metrics(probs[m], y_test[m], th)
            lo, hi = bootstrap_macro_auc(
                probs[m], y_test[m], patients[m], args.n_boot, args.seed)
            undefined = [c for c, v in met["per_class_auroc"].items() if v is None]
            overall.append({
                "model": name, "site_group": g, "records": int(m.sum()),
                "macro_auroc": met["macro_auroc"],
                "macro_auroc_ci_low": lo, "macro_auroc_ci_high": hi,
                "macro_f1": met["macro_f1"],
                "hamming_loss": met["hamming_loss"],
                "subset_accuracy": met["subset_accuracy"],
                "classes_without_auroc": ",".join(undefined),
            })
            for c in CLASSES:
                per_class.append({
                    "model": name, "site_group": g, "class": c,
                    "support": met["per_class_support"][c],
                    "auroc": met["per_class_auroc"][c],
                    "f1": met["per_class_f1"][c],
                    "precision": met["per_class_precision"][c],
                    "recall": met["per_class_recall"][c],
                })
        print(f"{name}: done")

    overall_df = pd.DataFrame(overall)
    overall_df.to_csv(paths[0], index=False)
    pd.DataFrame(per_class).to_csv(paths[1], index=False)
    pd.DataFrame(prevalence).to_csv(paths[2], index=False)

    with open(paths[3], "w") as f:
        json.dump({
            "site_groups": "site 0/1/2 as recorded in PTB-XL; 'other sites' = all others incl. missing",
            "client_mapping": {"client 1": "site 0", "client 2": "site 1",
                               "client 3": "site 2", "client 4": "other sites"},
            "thresholds": "validation-tuned thresholds from each original evaluation JSON",
            "bootstrap": {"unit": "patient", "n_boot": args.n_boot, "seed": args.seed},
            "label_prevalence": prevalence,
            "metrics": overall,
        }, f, indent=2)

    # Plot: macro-AUROC per site with 95% CI.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / len(PRIMARY)
    xs = np.arange(len(SITE_GROUPS) + 1)
    for k, name in enumerate(PRIMARY):
        d = overall_df[overall_df.model == name].set_index("site_group").loc[SITE_GROUPS + ["all"]]
        err = [d.macro_auroc - d.macro_auroc_ci_low, d.macro_auroc_ci_high - d.macro_auroc]
        ax.bar(xs + (k - (len(PRIMARY) - 1) / 2) * width, d.macro_auroc, width,
               yerr=err, capsize=3, label=MODELS[name]["label"])
    n_per_group = {p["site_group"]: p["records"] for p in prevalence}
    ax.set_xticks(xs, [f"{g}\n(n={n_per_group[g]})" for g in SITE_GROUPS + ["all"]])
    ax.set_ylim(0.8, 0.96)
    ax.set_ylabel("Test macro-AUROC (95% patient-bootstrap CI)")
    ax.set_title("Test performance by recording site")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(paths[4], dpi=200)
    plt.close(fig)

    print("\n" + pd.DataFrame(prevalence).to_string(index=False))
    print("\n" + overall_df[["model", "site_group", "records", "macro_auroc",
                             "macro_auroc_ci_low", "macro_auroc_ci_high", "macro_f1"]]
          .round(4).to_string(index=False))
    print("\nSaved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--metadata", default="data/raw/ptb-xl/ptbxl_database.csv")
    parser.add_argument("--output-dir", default="results/revised/per_site")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
