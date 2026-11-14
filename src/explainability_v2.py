"""Integrated Gradients explanations with sanity checks (revised).

Improvements over src/explainability.py (whose output is kept unchanged):
  * Explains the reproducible models: centralized multi-scale ("multiscale")
    and federated FedAvg-8 ("federated_r8").
  * Example selection is rule-based, not hand-picked: for each class, test
    ECGs that are positive for the class AND whose full predicted label set
    equals the true label set for BOTH models; then a seeded random sample.
  * Integrated Gradients uses a zero baseline (the per-lead mean of the
    z-scored signal), the midpoint Riemann rule, and reports the completeness
    error |sum(attr) - (f(x) - f(baseline))| / |f(x) - f(baseline)| on the
    class logit.
  * Faithfulness check (deletion test): setting the top-k% most positively
    attributed samples to the baseline should lower the class logit more than
    setting k% randomly chosen samples to the baseline.
  * Figures use a colour scale centred at zero and lead names.

The lead-importance summary is descriptive model behaviour on a limited
sample; it is not clinical validation.

Outputs (default results/revised/explainability/):
  figures/*.png, selected_examples.csv, ig_quality.csv,
  deletion_test.csv, lead_importance.csv, lead_importance.png, summary.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch

from src.io_utils import guard_outputs
from src.model_registry import (
    MODELS, load_model, load_evaluation, load_split, predict_array
)
from src.preprocessing import CLASSES, FS
from src.thresholds import thresholds_from_evaluation

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
EXPLAINED_MODELS = ["multiscale", "federated_r8"]
DELETION_FRACTIONS = [0.01, 0.02, 0.05, 0.10]


def integrated_gradients(model, x, target, steps=64, batch=64, baseline=None):
    """x: [12, 1000] tensor. Returns (attribution [12,1000], completeness error, fx, fb)."""
    was_training = model.training
    model.eval()
    try:
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            if baseline.shape != x.shape:
                raise ValueError(
                    f"baseline shape {tuple(baseline.shape)} must match input shape {tuple(x.shape)}"
                )
            if not torch.isfinite(baseline).all():
                raise ValueError("baseline tensor must contain only finite values (no NaN or Inf)")
            baseline = baseline.to(device=x.device, dtype=x.dtype)
        alphas = (torch.arange(steps, dtype=torch.float32, device=x.device) + 0.5) / steps
        grads = torch.zeros_like(x)
        for i in range(0, steps, batch):
            a = alphas[i:i + batch].view(-1, 1, 1)
            path = (baseline + a * (x - baseline)).requires_grad_(True)
            logits = model(path)[:, target].sum()
            grads += torch.autograd.grad(logits, path)[0].sum(0)
        attribution = (x - baseline) * grads / steps
        with torch.no_grad():
            fx = model(x[None])[0, target].item()
            fb = model(baseline[None])[0, target].item()
        gap = fx - fb
        error = abs(attribution.sum().item() - gap) / max(abs(gap), 1e-8)
        return attribution, error, fx, fb
    finally:
        model.train(was_training)


def deletion_scores(model, x, attribution, target, rng, n_random=5):
    """Logit drop when top-k% attributed vs random k% samples are zeroed."""
    flat = attribution.flatten()
    order = torch.argsort(flat, descending=True)
    with torch.no_grad():
        base_logit = model(x[None])[0, target].item()
    out = []
    for frac in DELETION_FRACTIONS:
        k = int(frac * flat.numel())
        xs = [x.flatten().clone()]
        xs[0][order[:k]] = 0.0
        for _ in range(n_random):
            xr = x.flatten().clone()
            xr[torch.from_numpy(rng.choice(flat.numel(), k, replace=False))] = 0.0
            xs.append(xr)
        with torch.no_grad():
            logits = model(torch.stack(xs).view(-1, 12, 1000))[:, target].numpy()
        out.append({
            "fraction": frac,
            "drop_top": base_logit - float(logits[0]),
            "drop_random": base_logit - float(logits[1:].mean()),
        })
    return out


def plot_example(path, x, attribution, info):
    x = x.numpy()
    a = attribution.numpy()
    lim = float(np.abs(a).max()) or 1.0
    t = np.arange(x.shape[1]) / FS

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 10), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1]})

    offset = 5.0
    for lead in range(12):
        axes[0].plot(t, x[lead] - lead * offset, lw=0.7, color="black")
    axes[0].set_yticks([-i * offset for i in range(12)], LEADS)
    axes[0].set_xlim(0, t[-1])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title(info["title"], fontsize=10)
    axes[0].grid(alpha=0.2)

    im = axes[1].imshow(
        a, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        interpolation="nearest", extent=[0, t[-1] + 1 / FS, 11.5, -0.5])
    axes[1].set_yticks(range(12), LEADS)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(
        f"Integrated Gradients for {info['class']} logit "
        f"(red = supports {info['class']}, blue = opposes; "
        f"completeness error {info['completeness_error']:.1%})", fontsize=10)
    fig.colorbar(im, ax=axes[1], label="Attribution")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    out = args.output_dir
    fig_dir = os.path.join(out, "figures")
    names = ["selected_examples.csv", "ig_quality.csv", "deletion_test.csv",
             "lead_importance.csv", "lead_importance.png", "summary.json"]
    paths = guard_outputs([os.path.join(out, n) for n in names],
                          overwrite=args.overwrite)
    if os.path.isdir(fig_dir) and os.listdir(fig_dir) and not args.overwrite:
        raise SystemExit(f"{fig_dir} is not empty; use --overwrite or a new --output-dir")
    os.makedirs(fig_dir, exist_ok=True)

    x_test, y_test, test_idx = load_split("test", args.processed)
    ecg_ids = np.load(os.path.join(args.processed, "ecg_ids.npy"))[test_idx]

    models, probs, preds = {}, {}, {}
    for name in EXPLAINED_MODELS:
        model, _ = load_model(name)
        evaluation, _ = load_evaluation(name)
        th = thresholds_from_evaluation(evaluation)
        models[name] = (model, th)
        probs[name] = predict_array(model, x_test)
        preds[name] = (probs[name] >= th).astype(int)

    fully_correct = np.all(
        [np.all(preds[n] == y_test, axis=1) for n in EXPLAINED_MODELS], axis=0)

    rng = np.random.default_rng(args.seed)
    selection, figure_set = [], set()
    for ci, c in enumerate(CLASSES):
        candidates = np.where((y_test[:, ci] == 1) & fully_correct)[0]
        n_agg = min(args.aggregate_per_class, len(candidates))
        chosen = rng.choice(candidates, n_agg, replace=False)
        for rank, pos in enumerate(chosen):
            selection.append({"class": c, "position": int(pos),
                              "ecg_id": int(ecg_ids[pos]),
                              "plotted": rank < args.figures_per_class,
                              "candidates_available": int(len(candidates))})
            if rank < args.figures_per_class:
                figure_set.add((c, int(pos)))
    sel_df = pd.DataFrame(selection)
    sel_df.to_csv(paths[0], index=False)

    quality, deletion, lead_rows = [], [], []
    del_rng = np.random.default_rng(args.seed + 1)
    for name in EXPLAINED_MODELS:
        model, th = models[name]
        for row in selection:
            c, pos = row["class"], row["position"]
            ci = CLASSES.index(c)
            x = x_test[pos]
            attr, err, fx, fb = integrated_gradients(model, x, ci, steps=args.steps)
            quality.append({"model": name, "class": c, "ecg_id": row["ecg_id"],
                            "logit": fx, "baseline_logit": fb,
                            "completeness_error": err})
            pos_attr = attr.clamp(min=0).sum(dim=1).numpy()
            share = pos_attr / max(pos_attr.sum(), 1e-12)
            lead_rows.append({"model": name, "class": c, "ecg_id": row["ecg_id"],
                              **dict(zip(LEADS, share))})
            for d in deletion_scores(model, x, attr, ci, del_rng):
                deletion.append({"model": name, "class": c,
                                 "ecg_id": row["ecg_id"], **d})
            if (c, pos) in figure_set:
                true = [k for k, v in zip(CLASSES, y_test[pos]) if v]
                p = probs[name][pos]
                plot_example(
                    os.path.join(fig_dir, f"{name}_{c}_ecg{row['ecg_id']}.png"),
                    x, attr, {
                        "class": c, "completeness_error": err,
                        "title": (
                            f"{MODELS[name]['label']} | ECG {row['ecg_id']} | "
                            f"true: {', '.join(true)} | "
                            f"P({c}) = {p[ci]:.2f} (threshold {th[ci]:.2f}) | "
                            "all predicted labels correct"),
                    })
        print(f"{name}: explained {len(selection)} ECGs", flush=True)

    q = pd.DataFrame(quality)
    q.to_csv(paths[1], index=False)
    dl = pd.DataFrame(deletion)
    dl.to_csv(paths[2], index=False)
    li = pd.DataFrame(lead_rows)
    li.to_csv(paths[3], index=False)

    mean_share = li.groupby(["model", "class"])[LEADS].mean()
    fig, axes = plt.subplots(1, len(EXPLAINED_MODELS), figsize=(14, 4.5), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), EXPLAINED_MODELS):
        m = mean_share.loc[name].loc[CLASSES]
        im = ax.imshow(m.values, cmap="viridis", aspect="auto", vmin=0)
        ax.set_xticks(range(12), LEADS)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_title(MODELS[name]["label"], fontsize=9)
        fig.colorbar(im, ax=ax, label="Mean share of positive attribution")
    fig.suptitle("Lead-level attribution share (descriptive; correctly classified test ECGs)")
    fig.tight_layout()
    fig.savefig(paths[4], dpi=200)
    plt.close(fig)

    del_summary = (dl.groupby(["model", "fraction"])[["drop_top", "drop_random"]]
                   .mean().reset_index())
    top_beats_random = (dl.drop_top > dl.drop_random).groupby(
        [dl.model, dl.fraction]).mean().rename("fraction_top_beats_random").reset_index()
    del_summary = del_summary.merge(top_beats_random, on=["model", "fraction"])

    summary = {
        "models": {n: MODELS[n]["label"] for n in EXPLAINED_MODELS},
        "selection_rule": ("test ECGs positive for the class whose full predicted label set "
                           "equals the true set for both models; seeded random sample"),
        "seed": args.seed, "ig_steps": args.steps, "baseline": "zeros",
        "examples_per_class_aggregate": args.aggregate_per_class,
        "figures_per_class": args.figures_per_class,
        "candidates_per_class": sel_df.groupby("class").candidates_available.first().to_dict(),
        "completeness_error": q.groupby("model").completeness_error.describe()[
            ["mean", "50%", "max"]].to_dict(orient="index"),
        "deletion_test": del_summary.to_dict(orient="records"),
    }
    with open(paths[5], "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCompleteness error by model:\n",
          q.groupby("model").completeness_error.describe()[["mean", "50%", "max"]].round(4))
    print("\nDeletion test (mean logit drop):\n", del_summary.round(3).to_string(index=False))
    print("\nMean lead share:\n", mean_share.round(3).to_string())
    print("\nSaved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--output-dir", default="results/revised/explainability")
    parser.add_argument("--figures-per-class", type=int, default=3)
    parser.add_argument("--aggregate-per-class", type=int, default=30)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
