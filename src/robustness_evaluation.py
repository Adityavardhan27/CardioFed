"""Inference-only robustness evaluation on the test fold.

No model is retrained. Each test ECG (already band-pass filtered and z-scored)
is corrupted and then passed through the project's preprocessing again
(src.preprocessing.preprocess_signal: 0.5-40 Hz band-pass + per-lead z-score),
which approximates corruption of the recording before the pipeline. Noise
amplitudes are therefore relative to each lead's standard deviation.

Perturbations
  gaussian_noise    additive white noise, sigma in {0.05, 0.1, 0.2, 0.5}
  baseline_wander   0.1-0.5 Hz sinusoid, amplitude in {0.5, 1.0, 2.0}
                    (largely attenuated by the 0.5 Hz high-pass filter)
  missing_leads     limb leads only (V1-V6 zeroed), precordial only
                    (I-aVF zeroed), lead I only, lead II only
  random_lead_drop  k in {1, 3, 6} randomly chosen leads zeroed per record

A zeroed lead stays all-zero after preprocessing. Thresholds are each model's
validation-tuned thresholds from the original evaluation; they are NOT
re-tuned for corrupted inputs, so F1 under missing leads also reflects
threshold miscalibration. Results are single realisations with a fixed seed.

Outputs (default results/revised/robustness/):
  robustness_metrics.csv, robustness_summary.json, robustness_macro_auroc.png
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

from src.evaluate import compute_metrics
from src.io_utils import guard_outputs
from src.model_registry import (
    MODELS, load_model, load_evaluation, load_split, predict_array
)
from src.preprocessing import preprocess_signal, FS
from src.thresholds import thresholds_from_evaluation

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_MODELS = ["baseline", "multiscale", "federated_r8"]


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
    for k in (1, 3, 6):
        yield "random_lead_drop", k, {"k": k}


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
            "robustness_metrics.csv", "robustness_summary.json",
            "robustness_macro_auroc.png")],
        overwrite=args.overwrite,
    )

    x_test, y_test, _ = load_split("test", args.processed)
    x_np = x_test.numpy()

    models = {}
    for name in args.models:
        model, _ = load_model(name)
        evaluation, _ = load_evaluation(name)
        models[name] = (model, thresholds_from_evaluation(evaluation))

    rows = []
    for kind, level, params in conditions():
        rng = np.random.default_rng(args.seed)
        if kind == "clean":
            xc = x_test
        else:
            xc = torch.from_numpy(repreprocess(corrupt(x_np, kind, params, rng)))
        for name, (model, th) in models.items():
            m = compute_metrics(predict_array(model, xc), y_test, th)
            rows.append({
                "model": name, "perturbation": kind, "level": str(level),
                "macro_auroc": m["macro_auroc"], "macro_f1": m["macro_f1"],
                **{f"auroc_{c}": v for c, v in m["per_class_auroc"].items()},
            })
        print(f"{kind} {level}: done", flush=True)

    df = pd.DataFrame(rows)
    clean = df[df.perturbation == "clean"].set_index("model")
    df["delta_macro_auroc"] = df.apply(
        lambda r: r.macro_auroc - clean.loc[r.model, "macro_auroc"], axis=1)
    df["delta_macro_f1"] = df.apply(
        lambda r: r.macro_f1 - clean.loc[r.model, "macro_f1"], axis=1)
    df.to_csv(paths[0], index=False)

    with open(paths[1], "w") as f:
        json.dump({
            "protocol": "corrupt processed test signal -> re-apply preprocess_signal -> predict",
            "thresholds": "validation-tuned thresholds from original evaluations (not re-tuned)",
            "seed": args.seed,
            "models": {n: MODELS[n]["label"] for n in args.models},
            "results": df.to_dict(orient="records"),
        }, f, indent=2)

    # Plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, kind in zip(axes, ["gaussian_noise", "baseline_wander", "random_lead_drop"]):
        for name in args.models:
            d = df[(df.model == name) & (df.perturbation == kind)]
            base = clean.loc[name, "macro_auroc"]
            levels = ["0"] + list(d.level)
            ax.plot(levels, [base] + list(d.macro_auroc), marker="o",
                    label=MODELS[name]["label"])
        ax.set_title(kind.replace("_", " "))
        ax.set_xlabel({"gaussian_noise": "noise sigma (x lead std)",
                       "baseline_wander": "wander amplitude (x lead std)",
                       "random_lead_drop": "leads removed"}[kind])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Test macro-AUROC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=200)
    plt.close(fig)

    print("\n" + df[["model", "perturbation", "level", "macro_auroc",
                     "delta_macro_auroc", "macro_f1", "delta_macro_f1"]]
          .round(4).to_string(index=False))
    print("\nSaved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--output-dir", default="results/revised/robustness")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
