
import json

path = "results/scheduler_experiment/metrics/multiscale_history.json"

with open(path, "r") as file:
    history = json.load(file)

print(
    "\nEPOCH | AUROC | MACRO-F1 | HYP-F1 | "
    "HYP-PRECISION | HYP-RECALL | VAL-LOSS | LR"
)

print("-" * 100)

for epoch in history:
    metrics = epoch["validation_metrics"]

    print(
        f"{epoch['epoch']:5d} | "
        f"{metrics['macro_auroc']:.4f} | "
        f"{metrics['macro_f1']:.4f} | "
        f"{metrics['per_class_f1']['HYP']:.4f} | "
        f"{metrics['per_class_precision']['HYP']:.4f} | "
        f"{metrics['per_class_recall']['HYP']:.4f} | "
        f"{epoch['validation_loss']:.4f} | "
        f"{epoch['learning_rate']:.6f}"
    )
