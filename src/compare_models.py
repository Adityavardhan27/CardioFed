
"""CardioFed: Final comparison of centralized and federated models."""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RESULT_FILES = {
    "Baseline CNN": "results/metrics/baseline_final_evaluation.json",
    "Multi-Scale Residual CNN": "results/metrics/multiscale_final_evaluation.json",
    "Federated Multi-Scale CNN (8 rounds)": (
        "results/metrics/federated_final_evaluation.json"
    )
}

OUTPUT_DIR = "results/comparison"

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------------------------------------------------
# Load evaluation results
# ---------------------------------------------------------


def load_results():
    results = {}

    for model_name, filepath in RESULT_FILES.items():

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing evaluation file: {filepath}"
            )

        with open(filepath, "r") as file:
            data = json.load(file)

        # Baseline and Multi-Scale use test_metrics.
        # Federated evaluation uses metrics.
        if "test_metrics" in data:
            metrics = data["test_metrics"]
        elif "metrics" in data:
            metrics = data["metrics"]
        else:
            raise KeyError(
                f"Test metrics not found in {filepath}"
            )

        results[model_name] = metrics

        print(f"Loaded: {model_name}")

    return results



# ---------------------------------------------------------
# Create overall comparison
# ---------------------------------------------------------

def create_overall_comparison(results):

    rows = []

    for model_name, metrics in results.items():

        rows.append({
            "Model": model_name,
            "Macro AUROC": metrics["macro_auroc"],
            "Macro F1": metrics["macro_f1"],
            "Hamming Loss": metrics["hamming_loss"],
            "Subset Accuracy": metrics["subset_accuracy"]
        })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        f"{OUTPUT_DIR}/overall_comparison.csv",
        index=False
    )

    print("\nOVERALL MODEL COMPARISON")
    print(comparison.to_string(index=False))

    return comparison


# ---------------------------------------------------------
# Create per-class comparison
# ---------------------------------------------------------

def create_per_class_comparison(results):

    rows = []

    for model_name, metrics in results.items():

        for class_name in CLASSES:

            rows.append({
                "Model": model_name,
                "Class": class_name,
                "AUROC": metrics["per_class_auroc"][class_name],
                "F1": metrics["per_class_f1"][class_name],
                "Precision": metrics["per_class_precision"][class_name],
                "Recall": metrics["per_class_recall"][class_name],
                "Support": metrics["per_class_support"][class_name]
            })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        f"{OUTPUT_DIR}/per_class_comparison.csv",
        index=False
    )

    print("\nPER-CLASS COMPARISON")
    print(comparison.to_string(index=False))

    return comparison


# ---------------------------------------------------------
# Plot overall performance
# ---------------------------------------------------------

def plot_overall_comparison(comparison):

    metrics_to_plot = [
        ("Macro AUROC", "Macro AUROC Comparison"),
        ("Macro F1", "Macro F1 Comparison"),
        ("Hamming Loss", "Hamming Loss Comparison"),
        ("Subset Accuracy", "Subset Accuracy Comparison")
    ]

    for column, title in metrics_to_plot:

        plt.figure(figsize=(10, 6))

        bars = plt.bar(
            comparison["Model"],
            comparison[column]
        )

        plt.title(title)
        plt.ylabel(column)
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.25)

        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        filename = column.lower().replace(" ", "_") + ".png"

        plt.savefig(
            f"{OUTPUT_DIR}/{filename}",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ---------------------------------------------------------
# Plot per-class F1
# ---------------------------------------------------------

def plot_per_class_f1(per_class):

    pivot = per_class.pivot(
        index="Class",
        columns="Model",
        values="F1"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7),
        width=0.8
    )

    ax.set_title("Per-Class F1 Score Comparison")
    ax.set_xlabel("Diagnostic Class")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=0)
    plt.legend(title="Model")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/per_class_f1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------------
# Plot per-class AUROC
# ---------------------------------------------------------

def plot_per_class_auroc(per_class):

    pivot = per_class.pivot(
        index="Class",
        columns="Model",
        values="AUROC"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7),
        width=0.8
    )

    ax.set_title("Per-Class AUROC Comparison")
    ax.set_xlabel("Diagnostic Class")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=0)
    plt.legend(title="Model")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/per_class_auroc.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------------
# Generate summary JSON
# ---------------------------------------------------------

def save_summary(comparison):

    summary = {
        "models_compared": len(comparison),
        "best_macro_auroc": comparison.loc[
            comparison["Macro AUROC"].idxmax()
        ]["Model"],
        "best_macro_f1": comparison.loc[
            comparison["Macro F1"].idxmax()
        ]["Model"],
        "lowest_hamming_loss": comparison.loc[
            comparison["Hamming Loss"].idxmin()
        ]["Model"],
        "best_subset_accuracy": comparison.loc[
            comparison["Subset Accuracy"].idxmax()
        ]["Model"]
    }

    with open(
        f"{OUTPUT_DIR}/comparison_summary.json",
        "w"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nBEST MODEL SUMMARY")

    for key, value in summary.items():
        print(f"{key}: {value}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CARDIOFED - FINAL MODEL COMPARISON")
    print("=" * 60)

    results = load_results()

    overall = create_overall_comparison(results)

    per_class = create_per_class_comparison(results)

    print("\nGenerating comparison plots...")

    plot_overall_comparison(overall)
    plot_per_class_f1(per_class)
    plot_per_class_auroc(per_class)

    save_summary(overall)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nAll files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
