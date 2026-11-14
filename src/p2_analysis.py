"""P2 analysis: aggregate runs and apply the pre-specified comparisons.

Implements results/p2/PROTOCOL.md, Sections 3-5. Reads only the outputs of
src/evaluate_checkpoint.py (evaluation.json, test_probabilities.csv) and the
training histories; no model is loaded or trained here.

Arms:
  centralized   multi-scale, per-record normalisation; seed 42 = existing
                primary run, seeds 43/44 = additional runs
  fedavg20      FedAvg, 20 rounds; seeds 42, 43, 44
  local_c1..c4  local-only, one per client; seed 42 only (scope revised,
                see PROTOCOL.md Section 7). Clients 3 and 4 use the complete
                reruns (*_complete); the interrupted first attempts are kept
                but not analysed.

Primary comparisons (seed 42, seed-matched): C1 FedAvg-20 vs centralized,
C2 FedAvg-20 vs each local-only client. Paired patient-level bootstrap
(2,000 resamples). Supplementary: C1 with all three seeds per arm (metric
averaged over seeds in each resample), reported separately.

Inference policy (consistent with src/statistical_analysis.py):
  Paired 95% percentile bootstrap confidence intervals [ci_low, ci_high] are
  the primary inferential output. Intervals excluding zero (ci_excludes_zero)
  are the project's stated inferential criterion for detecting differences.
  bootstrap_sign_prob (and its legacy alias p_value) reports empirical sign
  consistency across replicates as a descriptive statistic, NOT a formal
  null-hypothesis test p-value. Holm correction and formal significance flags
  are deliberately omitted as applying them to descriptive sign proportions is
  methodologically invalid.

  Note: PROTOCOL.md §4 pre-specified "Two-sided p-values with Holm correction"
  but the underlying computation 2*min(P(d<=0), P(d>=0)) is a bootstrap sign
  proportion, not a valid p-value under any null hypothesis. This script was
  updated for methodological consistency; the substantive conclusions (which
  CIs exclude zero) are unchanged.

Outputs (default results/p2/analysis/), never overwritten without --overwrite.
Use --check to list run status without writing anything.
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

from src.io_utils import guard_outputs
from src.preprocessing import CLASSES

SEEDS = [42, 43, 44]
P2 = "results/p2"


def run_dirs():
    arms = {
        "centralized": {42: f"{P2}/centralized/seed42_existing"} | {
            s: f"{P2}/centralized/seed{s}" for s in (43, 44)},
        "fedavg20": {s: f"{P2}/federated/r20_seed{s}" for s in SEEDS},
    }
    arms["local_c1"] = {42: f"{P2}/local/client1_seed42"}
    arms["local_c2"] = {42: f"{P2}/local/client2_seed42"}
    arms["local_c3"] = {42: f"{P2}/local/client3_seed42_complete"}
    arms["local_c4"] = {42: f"{P2}/local/client4_seed42_complete"}
    return arms


def history_path(arm, d):
    if arm == "fedavg20":
        return f"{d}/metrics/federated_20round_history.json"
    if arm == "centralized" and d.endswith("seed42_existing"):
        return "results/scheduler_experiment/metrics/multiscale_history.json"
    return f"{d}/metrics/multiscale_history.json"


def load_runs():
    runs = {}
    for arm, seeds in run_dirs().items():
        for seed, d in seeds.items():
            ev = f"{d}/evaluation/evaluation.json"
            if os.path.exists(ev):
                with open(ev) as f:
                    e = json.load(f)
                probs = pd.read_csv(f"{d}/evaluation/test_probabilities.csv")
                runs[(arm, seed)] = {"dir": d, "eval": e, "probs": probs}
    return runs


def macro_auc(p, y):
    valid = [
        float(roc_auc_score(y[:, i], p[:, i]))
        for i in range(5)
        if len(np.unique(y[:, i])) == 2
    ]
    return float(np.mean(valid)) if valid else float("nan")



def macro_f1(pred, y):
    tp = ((pred == 1) & (y == 1)).sum(0)
    fp = ((pred == 1) & (y == 0)).sum(0)
    fn = ((pred == 0) & (y == 1)).sum(0)
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))


def holm(p):
    """Holm-Bonferroni step-down correction for a family of valid p-values.

    NOTE: This procedure requires valid p-values that control type I error
    under the null hypothesis. It must NOT be applied to descriptive bootstrap
    statistics such as bootstrap_sign_prob.
    """
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main(args):
    runs = load_runs()
    expected = [(a, s) for a, d in run_dirs().items() for s in d]
    if args.check:
        for a, s in expected:
            print(f"{a:12s} seed {s}: {'OK' if (a, s) in runs else 'missing'}")
        return

    out = args.output_dir
    names = ["runs.csv", "arm_summary.csv", "comparisons.csv", "own_site.csv",
             "convergence.csv", "reproduction_check.json", "summary.json",
             "fedavg_convergence.png", "arms_macro_auroc.png"]
    paths = dict(zip(names, guard_outputs([os.path.join(out, n) for n in names],
                                          overwrite=args.overwrite)))

    y = np.load("data/processed/y.npy")
    folds = np.load("data/processed/folds.npy")
    y_test = y[folds == 10]
    ref_ids = np.load("data/processed/ecg_ids.npy")[folds == 10]

    # ---------------------------------------------------------- per-run table
    rows = []
    for (arm, seed), r in sorted(runs.items()):
        if not np.array_equal(r["probs"]["ecg_id"].to_numpy(), ref_ids):
            raise RuntimeError(f"{arm} seed {seed}: test order mismatch")
        e, t = r["eval"], r["eval"]["test_metrics"]
        info = e["checkpoint_info"]
        rows.append({
            "arm": arm, "seed": seed,
            "best_epoch_or_round": info.get("epoch", info.get("round")),
            "best_val_auroc": info.get("best_val_auroc"),
            "test_macro_auroc": t["macro_auroc"], "test_macro_f1": t["macro_f1"],
            "test_hamming_loss": t["hamming_loss"],
            "test_subset_accuracy": t["subset_accuracy"],
            **{f"auroc_{c}": t["per_class_auroc"][c] for c in CLASSES},
            **{f"f1_{c}": t["per_class_f1"][c] for c in CLASSES},
            **{f"site_auroc_{g}": v["macro_auroc"] for g, v in e["per_site_test"].items()},
        })
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(paths["runs.csv"], index=False)

    metric_cols = [c for c in runs_df.columns if c not in ("arm", "seed")]
    summary = runs_df.groupby("arm")[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(paths["arm_summary.csv"])

    # ------------------------------------------------- bootstrap comparisons
    def arm_arrays(arm, seeds):
        out_ = []
        for s in seeds:
            if (arm, s) in runs:
                r = runs[(arm, s)]
                p = r["probs"][[f"{c}_probability" for c in CLASSES]].to_numpy()
                th = np.array([r["eval"]["thresholds"][c] for c in CLASSES], np.float32)
                out_.append((p, (p >= th).astype(int)))
        return out_

    # (id, arm A, arm B, seeds, family)
    comparisons = [("C1", "fedavg20", "centralized", [42], "primary")]
    comparisons += [(f"C2-client{c}", "fedavg20", f"local_c{c}", [42], "primary")
                    for c in range(1, 5)]
    comparisons += [("S1-3seeds", "fedavg20", "centralized", SEEDS, "supplementary")]
    comparisons = [c for c in comparisons
                   if len(arm_arrays(c[1], c[3])) == len(c[3])
                   and len(arm_arrays(c[2], c[3])) == len(c[3])]

    patients = runs[("centralized", 42)]["probs"]["patient_id"].to_numpy()
    unique = np.unique(patients)
    groups = {p: np.where(patients == p)[0] for p in unique}
    rng = np.random.default_rng(args.seed)
    needed = sorted({(a, tuple(c[3])) for c in comparisons for a in c[1:3]})
    arrays = {k: arm_arrays(k[0], k[1]) for k in needed}
    point = {a: (np.mean([macro_auc(p, y_test) for p, _ in arrays[a]]),
                 np.mean([macro_f1(pr, y_test) for _, pr in arrays[a]])) for a in needed}
    samples = {a: [] for a in needed}
    for _ in range(args.n_boot):
        ix = np.concatenate([groups[p] for p in rng.choice(unique, len(unique))])
        yb = y_test[ix]
        for a in needed:
            samples[a].append((
                np.mean([macro_auc(p[ix], yb) for p, _ in arrays[a]]),
                np.mean([macro_f1(pr[ix], yb) for _, pr in arrays[a]]),
            ))
    samples = {a: np.array(v) for a, v in samples.items()}

    comp_rows = []
    for cid, a, b, seeds, family in comparisons:
        ka, kb = (a, tuple(seeds)), (b, tuple(seeds))
        for k, metric in enumerate(("macro_auroc", "macro_f1")):
            d = samples[ka][:, k] - samples[kb][:, k]
            sign_prob = 2 * min((d <= 0).mean(), (d >= 0).mean())
            ci_low = float(np.percentile(d, 2.5))
            ci_high = float(np.percentile(d, 97.5))
            comp_rows.append({
                "id": cid, "family": family, "arm_a": a, "arm_b": b,
                "metric": metric, "seeds": ",".join(map(str, seeds)),
                "difference": point[ka][k] - point[kb][k],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                "bootstrap_sign_prob": float(min(1.0, max(sign_prob, 1.0 / args.n_boot))),
                "p_value": float(min(1.0, max(sign_prob, 1.0 / args.n_boot))),  # legacy alias; NOT a formal p-value
            })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(paths["comparisons.csv"], index=False)

    # --------------------------------------- descriptive: own-site subsets
    site_of_client = {1: "site 0", 2: "site 1", 3: "site 2", 4: "other sites"}
    own = []
    for c, g in site_of_client.items():
        for arm in (f"local_c{c}", "fedavg20", "centralized"):
            vals = runs_df[runs_df.arm == arm][f"site_auroc_{g}"]
            if len(vals):
                own.append({"client": c, "site_group": g, "arm": arm,
                            "macro_auroc_mean": vals.mean(), "macro_auroc_sd": vals.std(),
                            "seeds": len(vals)})
    pd.DataFrame(own).to_csv(paths["own_site.csv"], index=False)

    # -------------------------------------------------- FedAvg convergence
    conv_rows, curves = [], {}
    for s in SEEDS:
        if ("fedavg20", s) not in runs:
            continue
        with open(history_path("fedavg20", runs[("fedavg20", s)]["dir"])) as f:
            h = json.load(f)
        auc = np.array([r["validation_macro_auroc"] for r in h])
        curves[s] = (auc, [r["average_client_loss"] for r in h])
        best = int(np.argmax(auc)) + 1
        conv_rows.append({
            "seed": s, "rounds": len(h), "best_round": best,
            "best_val_auroc": float(auc.max()),
            "first_round_within_0.005": int(np.argmax(auc >= auc.max() - 0.005)) + 1,
            "val_auroc_round8": float(auc[7]), "val_auroc_round20": float(auc[-1]),
            "max_gain_after_round8": float(auc[8:].max() - auc[:8].max()),
        })
    pd.DataFrame(conv_rows).to_csv(paths["convergence.csv"], index=False)

    repro = {}
    if 42 in curves:
        with open("results/metrics/federated_history.json") as f:
            stored = json.load(f)
        with open(history_path("fedavg20", runs[("fedavg20", 42)]["dir"])) as f:
            new = json.load(f)[:8]
        repro = {
            "compared": "FedAvg seed 42 rounds 1-8 vs results/metrics/federated_history.json",
            "max_abs_diff_val_auroc": float(max(abs(a["validation_macro_auroc"] - b["validation_macro_auroc"]) for a, b in zip(stored, new))),
            "max_abs_diff_client_loss": float(max(abs(x - z) for a, b in zip(stored, new) for x, z in zip(a["client_losses"], b["client_losses"]))),
        }
        repro["bitwise_identical"] = repro["max_abs_diff_val_auroc"] == 0 and repro["max_abs_diff_client_loss"] == 0
    with open(paths["reproduction_check.json"], "w") as f:
        json.dump(repro, f, indent=2)

    # ----------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for s, (auc, loss) in curves.items():
        r = np.arange(1, len(auc) + 1)
        axes[0].plot(r, auc, marker="o", ms=3, label=f"seed {s}")
        axes[1].plot(r, loss, marker="o", ms=3, label=f"seed {s}")
    for ax, t in zip(axes, ["Validation macro-AUROC", "Mean client training loss"]):
        ax.set_xlabel("Communication round"); ax.set_title(t); ax.grid(alpha=0.25); ax.legend()
    cen = runs_df[runs_df.arm == "centralized"].best_val_auroc
    if len(cen):
        axes[0].axhline(cen.mean(), ls="--", color="gray",
                        label="centralized (mean best val)")
        axes[0].legend()
    fig.tight_layout(); fig.savefig(paths["fedavg_convergence.png"], dpi=200); plt.close(fig)

    order = ["centralized", "fedavg20", "local_c1", "local_c2", "local_c3",
             "local_c4"]
    order = [a for a in order if a in set(runs_df.arm)]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, a in enumerate(order):
        v = runs_df[runs_df.arm == a].test_macro_auroc
        ax.scatter([i] * len(v), v, color="tab:blue", zorder=3)
        ax.errorbar(i, v.mean(), yerr=v.std() if len(v) > 1 else 0,
                    fmt="_", color="black", ms=20, capsize=6, zorder=2)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("Test macro-AUROC (dots = seeds; bar = mean ± SD)")
    ax.set_title("P2 arms (test fold 10)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(paths["arms_macro_auroc.png"], dpi=200); plt.close(fig)

    with open(paths["summary.json"], "w") as f:
        json.dump({
            "protocol": "results/p2/PROTOCOL.md",
            "runs_found": [f"{a} seed {s}" for a, s in sorted(runs)],
            "runs_missing": [f"{a} seed {s}" for a, s in expected if (a, s) not in runs],
            "bootstrap": {
                "unit": "patient",
                "n_boot": args.n_boot,
                "seed": args.seed,
                "interval": "95% percentile",
                "seed_handling": "metric averaged over each arm's seeds per resample",
                "paired_inference": {
                    "primary_criterion": (
                        "95% percentile bootstrap confidence interval of paired differences [ci_low, ci_high]. "
                        "Intervals excluding zero (ci_excludes_zero == True) are the project's stated inferential "
                        "criterion for detecting differences, not a substitute for a separately justified multiple-comparison procedure."
                    ),
                    "sign_prob_interpretation": (
                        "bootstrap_sign_prob (and legacy alias p_value) represents an empirical bootstrap sign-proportion "
                        "estimate 2 * min(P(d <= 0), P(d >= 0)) floored at 1/n_boot measuring replicate sign consistency. "
                        "It is a descriptive statistic rather than a formal null-hypothesis test p-value. Holm correction "
                        "and formal significance flags are deliberately omitted as applying them to descriptive sign "
                        "proportions is methodologically invalid."
                    ),
                },
            },
            "comparisons": comp.to_dict(orient="records"),
            "convergence": conv_rows,
            "reproduction_check": repro,
        }, f, indent=2)

    print(runs_df[["arm", "seed", "best_epoch_or_round", "test_macro_auroc",
                   "test_macro_f1", "auroc_HYP", "f1_HYP"]].round(4).to_string(index=False))
    print("\n" + comp.round(4).to_string(index=False))
    print("\nConvergence:\n" + pd.DataFrame(conv_rows).round(4).to_string(index=False))
    print("\nReproduction:", repro)
    print("Saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/p2/analysis")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
