"""Inference-only robustness evaluation on the test fold.

No model is retrained. Each test ECG (already band-pass filtered and z-scored)
is corrupted and then passed through the project's preprocessing again
(src.preprocessing.preprocess_signal: 0.5-40 Hz band-pass + per-lead z-score),
which approximates corruption of the recording before the pipeline. Noise
amplitudes are therefore relative to each lead's standard deviation.

Perturbations
  gaussian_noise    additive white noise, sigma in {0.05, 0.1, 0.2, 0.5} (5 seeds)
  baseline_wander   0.1-0.5 Hz sinusoid, amplitude in {0.5, 1.0, 2.0} (5 seeds)
                    (largely attenuated by the 0.5 Hz high-pass filter)
  missing_leads     limb leads only (V1-V6 zeroed), precordial only (I-aVF zeroed),
                    lead I only, lead II only, all leads zero (flatline)
  random_lead_drop  k in {1, 3, 6} randomly chosen leads zeroed per record (5 seeds)
  clipping          pointwise dynamic range limit at +/- c in {1.5, 2.0, 2.5} lead std
  temporal_shift    controlled circular temporal translations of +/- s in {50, 100} samples
                    (time-shift invariance check; does not simulate realistic window shift)

A zeroed lead stays all-zero after preprocessing. Thresholds are each model's
validation-tuned thresholds from the original evaluation; they are NOT
re-tuned for corrupted inputs, so F1 under missing leads also reflects
threshold miscalibration.

Outputs (default results/revised/robustness_extended/):
  robustness_metrics.csv, robustness_paired_ci.csv, robustness_summary.json,
  robustness_macro_auroc.png, robustness_macro_f1.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# PyTorch 2.6 compatibility: checkpoints contain full state dicts with non-weight metadata
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, "weights_only": False})

from src.evaluate import compute_metrics
from src.io_utils import guard_outputs
from src.model_registry import (
    MODELS, load_model, load_evaluation, load_split, predict_array
)
from src.preprocessing import preprocess_signal, FS, CLASSES
from src.thresholds import thresholds_from_evaluation
from src.statistical_analysis import (
    patient_bootstrap_indices,
    compute_paired_comparison,
    macro_auc,
    per_class_f1,
)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_MODELS = ["baseline", "multiscale", "federated_r8"]
STOCHASTIC_SEEDS = [0, 1, 2, 3, 4]


def is_stochastic(kind):
    return kind in ("gaussian_noise", "baseline_wander", "random_lead_drop")


def conditions():
    yield "clean", None, {}
    for s in (0.05, 0.1, 0.2, 0.5):
        yield "gaussian_noise", s, {"sigma": s}
    for a in (0.5, 1.0, 2.0):
        yield "baseline_wander", a, {"amplitude": a}
    yield "missing_leads", "limb_only", {"keep": list(range(0, 6))}
    yield "missing_leads", "precordial_only", {"keep": list(range(6, 12))}
    yield "missing_leads", "lead_I_only", {"keep": [0]}
    yield "missing_leads", "lead_II_only", {"keep": [1]}
    yield "missing_leads", "all_leads_zero", {"keep": []}
    for k in (1, 3, 6):
        yield "random_lead_drop", k, {"k": k}
    for c in (1.5, 2.0, 2.5):
        yield "clipping", c, {"threshold": c}
    for s in (-100, -50, 50, 100):
        yield "temporal_shift", s, {"shift": s}


def corrupt(x, kind, params, rng):
    """x: [N, 12, 1000] float32 numpy array (copy is returned)."""
    x = x.copy()
    n = x.shape[0]
    t = np.arange(x.shape[2]) / FS

    if kind == "gaussian_noise":
        x += rng.normal(0, params["sigma"], x.shape).astype(np.float32)
    elif kind == "baseline_wander":
        f = rng.uniform(0.1, 0.5, (n, 1, 1))
        phase = rng.uniform(0, 2 * np.pi, (n, 1, 1))
        x += (params["amplitude"] * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)
    elif kind == "missing_leads":
        mask = np.zeros(12, bool)
        mask[params["keep"]] = True
        x[:, ~mask, :] = 0.0
    elif kind == "random_lead_drop":
        for i in range(n):
            x[i, rng.choice(12, params["k"], replace=False), :] = 0.0
    elif kind == "clipping":
        c = float(params["threshold"])
        x = np.clip(x, -c, c)
    elif kind == "temporal_shift":
        s = int(params["shift"])
        # Circular temporal translation along time axis (axis=2)
        x = np.roll(x, s, axis=2)
    return x


def repreprocess(x):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = preprocess_signal(x[i].T.astype(np.float64)).T
    return out


def main(args):
    out = args.output_dir
    paths = guard_outputs(
        [os.path.join(out, f) for f in (
            "robustness_metrics.csv", "robustness_paired_ci.csv",
            "robustness_summary.json", "robustness_macro_auroc.png",
            "robustness_macro_f1.png")],
        overwrite=args.overwrite,
    )

    x_test, y_test, test_idx = load_split("test", args.processed)
    x_np = x_test.numpy()

    # Load patient IDs for patient-clustered paired bootstrap
    ecg_ids = np.load(os.path.join(args.processed, "ecg_ids.npy"))[test_idx]
    meta = pd.read_csv(args.metadata, index_col="ecg_id")
    patients = meta.loc[ecg_ids, "patient_id"].to_numpy()

    device = args.device
    print(f"Using device: {device}", flush=True)

    models = {}
    for name in args.models:
        model, _ = load_model(name, device=device)
        evaluation, _ = load_evaluation(name)
        models[name] = (model, thresholds_from_evaluation(evaluation))

    # 1. Clean predictions (computed ONCE per model)
    print("Computing clean baseline predictions...", flush=True)
    clean_probs = {}
    clean_metrics = {}
    preds_clean = {}
    for name in args.models:
        model, th = models[name]
        cp = predict_array(model, x_test, device=device)
        clean_probs[name] = cp
        clean_metrics[name] = compute_metrics(cp, y_test, th)
        preds_clean[name] = (cp >= th).astype(int)

    # 2. Pre-generate patient bootstrap indices (for paired difference CIs)
    print(f"Generating {args.n_boot} patient-clustered bootstrap resamples...", flush=True)
    boot_indices = list(patient_bootstrap_indices(patients, args.n_boot, args.bootstrap_seed))
    y_test_boots = [y_test[ix] for ix in boot_indices]

    # Precompute clean bootstrap metric samples for each model
    clean_boot_auc = {name: [] for name in args.models}
    clean_boot_f1 = {name: [] for name in args.models}
    for b_idx, ix in enumerate(boot_indices):
        yb = y_test_boots[b_idx]
        for name in args.models:
            cp = clean_probs[name][ix]
            pred = preds_clean[name][ix]
            clean_boot_auc[name].append(macro_auc(cp, yb))
            clean_boot_f1[name].append(float(np.mean(per_class_f1(pred, yb))))

    # 3. Main perturbation evaluation loop
    rows = []
    paired_ci_rows = []
    cond_list = list(conditions())
    total_conds = len(cond_list)

    import time
    for cond_idx, (kind, level, params) in enumerate(cond_list):
        t_start = time.time()
        seeds = args.seeds if is_stochastic(kind) else [0]
        stochastic = is_stochastic(kind)

        # Store corrupted probabilities for each model across seeds: {name: [probs_seed0, ...]}
        cond_probs = {name: [] for name in args.models}
        cond_metrics = {name: [] for name in args.models}

        for seed in seeds:
            rng = np.random.default_rng(seed)
            if kind == "clean":
                xc = x_test
            else:
                xc_np = corrupt(x_np, kind, params, rng)
                xc = torch.from_numpy(repreprocess(xc_np))

            for name in args.models:
                model, th = models[name]
                if kind == "clean":
                    probs = clean_probs[name]
                    m = clean_metrics[name]
                else:
                    probs = predict_array(model, xc, device=device)
                    m = compute_metrics(probs, y_test, th)

                cond_probs[name].append(probs)
                cond_metrics[name].append(m)

                clean_m = clean_metrics[name]
                delta_auroc = m["macro_auroc"] - clean_m["macro_auroc"] if m["macro_auroc"] is not None else np.nan
                delta_f1 = m["macro_f1"] - clean_m["macro_f1"]

                row = {
                    "model": name,
                    "perturbation": kind,
                    "level": str(level),
                    "seed": seed,
                    "stochastic": stochastic,
                    "macro_auroc": m["macro_auroc"],
                    "macro_f1": m["macro_f1"],
                    "hamming_loss": m["hamming_loss"],
                    "subset_accuracy": m["subset_accuracy"],
                    "delta_macro_auroc": delta_auroc,
                    "delta_macro_f1": delta_f1,
                    **{f"auroc_{c}": m["per_class_auroc"].get(c) for c in CLASSES},
                    **{f"f1_{c}": m["per_class_f1"][c] for c in CLASSES},
                    **{f"precision_{c}": m["per_class_precision"][c] for c in CLASSES},
                    **{f"recall_{c}": m["per_class_recall"][c] for c in CLASSES},
                    **{f"support_{c}": m["per_class_support"][c] for c in CLASSES},
                }
                rows.append(row)

        # 4. Paired patient-clustered bootstrap across seeds
        # For each replicate b, compute paired difference averaged across corruption seeds
        for name in args.models:
            th = models[name][1]
            n_seeds = len(seeds)
            seed_aurocs = [cond_metrics[name][s]["macro_auroc"] for s in range(n_seeds)
                           if cond_metrics[name][s]["macro_auroc"] is not None]
            seed_f1s = [cond_metrics[name][s]["macro_f1"] for s in range(n_seeds)]

            mean_auroc = float(np.mean(seed_aurocs)) if seed_aurocs else np.nan
            std_auroc = float(np.std(seed_aurocs, ddof=1)) if len(seed_aurocs) > 1 else 0.0
            mean_f1 = float(np.mean(seed_f1s))
            std_f1 = float(np.std(seed_f1s, ddof=1)) if len(seed_f1s) > 1 else 0.0

            clean_base_auroc = clean_metrics[name]["macro_auroc"]
            clean_base_f1 = clean_metrics[name]["macro_f1"]
            point_delta_auroc = mean_auroc - clean_base_auroc if not np.isnan(mean_auroc) else np.nan
            point_delta_f1 = mean_f1 - clean_base_f1

            # Precompute thresholded binary predictions once per seed
            preds_s = [(cond_probs[name][s] >= th).astype(int) for s in range(n_seeds)]

            # Paired bootstrap across replicates
            paired_diff_auc = []
            paired_diff_f1 = []

            for b_idx, ix in enumerate(boot_indices):
                yb = y_test_boots[b_idx]
                c_auc = clean_boot_auc[name][b_idx]
                c_f1 = clean_boot_f1[name][b_idx]

                # Average corrupted metric across seeds for this replicate
                rep_auc_diffs = []
                rep_f1_diffs = []
                for s in range(n_seeds):
                    p_s = cond_probs[name][s][ix]
                    pred_s = preds_s[s][ix]
                    m_auc = macro_auc(p_s, yb)
                    m_f1 = float(np.mean(per_class_f1(pred_s, yb)))
                    if np.isfinite(m_auc) and np.isfinite(c_auc):
                        rep_auc_diffs.append(m_auc - c_auc)
                    rep_f1_diffs.append(m_f1 - c_f1)

                paired_diff_auc.append(np.mean(rep_auc_diffs) if rep_auc_diffs else np.nan)
                paired_diff_f1.append(np.mean(rep_f1_diffs))

            # Inferential stats using shared compute_paired_comparison
            stat_auc = compute_paired_comparison(
                np.array(paired_diff_auc) + clean_base_auroc,
                np.full_like(paired_diff_auc, clean_base_auroc),
                point_diff=point_delta_auroc
            )
            stat_f1 = compute_paired_comparison(
                np.array(paired_diff_f1) + clean_base_f1,
                np.full_like(paired_diff_f1, clean_base_f1),
                point_diff=point_delta_f1
            )

            ci_row = {
                "model": name,
                "perturbation": kind,
                "level": str(level),
                "n_seeds": n_seeds,
                "stochastic": stochastic,
                "macro_auroc_mean": mean_auroc,
                "macro_auroc_std_seed": std_auroc,
                "macro_f1_mean": mean_f1,
                "macro_f1_std_seed": std_f1,
                "delta_auroc_point": point_delta_auroc,
                "delta_auroc_ci_low": stat_auc["ci_low"],
                "delta_auroc_ci_high": stat_auc["ci_high"],
                "delta_auroc_ci_excludes_zero": stat_auc["ci_excludes_zero"],
                "delta_auroc_sign_prob": stat_auc["bootstrap_sign_prob"],
                "delta_f1_point": point_delta_f1,
                "delta_f1_ci_low": stat_f1["ci_low"],
                "delta_f1_ci_high": stat_f1["ci_high"],
                "delta_f1_ci_excludes_zero": stat_f1["ci_excludes_zero"],
                "delta_f1_sign_prob": stat_f1["bootstrap_sign_prob"],
            }
            paired_ci_rows.append(ci_row)

        elapsed = time.time() - t_start
        print(f"[{cond_idx + 1}/{total_conds}] {kind} {level} (seeds={len(seeds)}): done in {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(paths[0], index=False)

    df_ci = pd.DataFrame(paired_ci_rows)
    df_ci.to_csv(paths[1], index=False)

    # Summary JSON
    summary_data = {
        "protocol": "corrupt processed test signal -> re-apply preprocess_signal (0.5-40 Hz band-pass + per-lead z-score) -> predict with frozen validation thresholds",
        "thresholds": "validation-tuned thresholds from original evaluations (not re-tuned)",
        "models": {n: MODELS[n]["label"] for n in args.models},
        "stochastic_seeds": args.seeds,
        "n_boot": args.n_boot,
        "bootstrap_seed": args.bootstrap_seed,
        "clean_metrics": {n: {k: v for k, v in clean_metrics[n].items() if k != "per_class_support"} for n in args.models},
        "paired_ci_summary": df_ci.to_dict(orient="records"),
    }
    with open(paths[2], "w") as f:
        json.dump(summary_data, f, indent=2)

    # 5. Visualizations: Macro-AUROC and Macro-F1
    # Plot AUROC
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    plot_kinds = [
        ("gaussian_noise", "noise sigma (x lead std)"),
        ("baseline_wander", "wander amplitude (x lead std)"),
        ("random_lead_drop", "leads removed (k)"),
        ("clipping", "clipping threshold (+/- lead std)"),
        ("temporal_shift", "shift (samples, fs=100 Hz)"),
        ("missing_leads", "lead configuration"),
    ]

    for ax, (kind, xlabel) in zip(axes.flat, plot_kinds):
        for name in args.models:
            d = df_ci[(df_ci.model == name) & (df_ci.perturbation == kind)]
            if kind == "missing_leads":
                # Bar chart for structured missing leads
                x_pos = np.arange(len(d))
                width = 0.25
                offset = {"baseline": -width, "multiscale": 0.0, "federated_r8": width}[name]
                ax.bar(x_pos + offset, d.macro_auroc_mean, width=width,
                       label=MODELS[name]["label"], alpha=0.85)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(d.level, rotation=30, ha="right", fontsize=8)
            else:
                base = clean_metrics[name]["macro_auroc"]
                levels = list(d.level)
                vals = list(d.macro_auroc_mean)
                if kind in ("gaussian_noise", "baseline_wander", "random_lead_drop"):
                    levels = ["0"] + levels
                    vals = [base] + vals
                ax.plot(levels, vals, marker="o", label=MODELS[name]["label"])
        ax.set_title(kind.replace("_", " "))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test macro-AUROC")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[3], dpi=200)
    plt.close(fig)

    # Plot F1
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    for ax, (kind, xlabel) in zip(axes.flat, plot_kinds):
        for name in args.models:
            d = df_ci[(df_ci.model == name) & (df_ci.perturbation == kind)]
            if kind == "missing_leads":
                x_pos = np.arange(len(d))
                width = 0.25
                offset = {"baseline": -width, "multiscale": 0.0, "federated_r8": width}[name]
                ax.bar(x_pos + offset, d.macro_f1_mean, width=width,
                       label=MODELS[name]["label"], alpha=0.85)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(d.level, rotation=30, ha="right", fontsize=8)
            else:
                base = clean_metrics[name]["macro_f1"]
                levels = list(d.level)
                vals = list(d.macro_f1_mean)
                if kind in ("gaussian_noise", "baseline_wander", "random_lead_drop"):
                    levels = ["0"] + levels
                    vals = [base] + vals
                ax.plot(levels, vals, marker="o", label=MODELS[name]["label"])
        ax.set_title(kind.replace("_", " ") + " (Macro-F1)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test macro-F1")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[4], dpi=200)
    plt.close(fig)

    print(f"\nCompleted robustness evaluation. Artifacts saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--metadata", default="data/raw/ptb-xl/ptbxl_database.csv")
    parser.add_argument("--output-dir", default="results/revised/robustness_extended")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=STOCHASTIC_SEEDS)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    default_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())

