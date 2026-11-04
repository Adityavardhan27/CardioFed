
"""CardioFed explainability using Integrated Gradients."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model import MultiScaleResidualECGNet
from src.dataset import ECGDataset
from src.preprocessing import CLASSES


def integrated_gradients(model, signal, target_class, steps=32):

    """
    signal shape: [1, 12, 1000]
    Returns attribution with the same shape.
    """

    model.eval()

    baseline = torch.zeros_like(signal)

    difference = signal - baseline

    total_gradients = torch.zeros_like(signal)

    for alpha in torch.linspace(0, 1, steps, device=signal.device):

        interpolated = (
            baseline + alpha * difference
        ).detach().requires_grad_(True)

        output = model(interpolated)

        score = output[0, target_class]

        gradients = torch.autograd.grad(
            score,
            interpolated
        )[0]

        total_gradients += gradients.detach()

    average_gradients = total_gradients / steps

    attributions = difference * average_gradients

    return attributions.detach()


def main():

    os.makedirs("results/plots", exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load the trained multi-scale model.
    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(device)

    checkpoint = torch.load(
        "results/checkpoints/multiscale_best.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test records.
    folds = np.load("data/processed/folds.npy")
    y = np.load("data/processed/y.npy")

    test_indices = np.where(folds == 10)[0]

    # Choose the first test ECG with a positive MI label.
    mi_index = CLASSES.index("MI")

    candidates = [
        idx for idx in test_indices
        if y[idx, mi_index] == 1
    ]

    if not candidates:
        raise RuntimeError("No MI-positive test ECG found.")

    sample_idx = candidates[0]

    dataset = ECGDataset(
        "data/processed",
        [sample_idx]
    )

    signal, target = dataset[0]

    signal = signal.unsqueeze(0).to(device)

    # Calculate prediction.
    with torch.no_grad():
        logits = model(signal)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    # Calculate MI attribution.
    print("\nCalculating Integrated Gradients for MI...")

    attribution = integrated_gradients(
        model,
        signal,
        target_class=mi_index,
        steps=32
    )

    attribution = attribution[0].cpu().numpy()
    signal_np = signal[0].cpu().numpy()

    # Aggregate attribution across leads for each time point.
    temporal_importance = np.mean(
        np.abs(attribution),
        axis=0
    )

    # Normalize for visualization only.
    temporal_importance /= (
        temporal_importance.max() + 1e-8
    )

    # Plot ECG signals and attribution.
    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 9),
        constrained_layout=True
    )

    time = np.arange(signal_np.shape[1]) / 100.0

    # ECG waveform: all 12 leads, vertically offset.
    for lead in range(12):

        axes[0].plot(
            time,
            signal_np[lead] + lead * 5,
            linewidth=0.7,
            label=f"Lead {lead + 1}"
        )

    axes[0].set_title(
        f"12-Lead ECG | ECG ID: "
        f"{int(np.load('data/processed/ecg_ids.npy')[sample_idx])}"
    )

    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Normalized signal + lead offset")
    axes[0].grid(alpha=0.2)

    # Attribution heatmap across leads and time.
    im = axes[1].imshow(
        attribution,
        aspect="auto",
        cmap="seismic",
        interpolation="nearest",
        extent=[0, 10, 12, 0]
    )

    axes[1].set_title(
        "Integrated Gradients Attribution for MI"
    )

    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("ECG lead")

    fig.colorbar(
        im,
        ax=axes[1],
        label="Attribution"
    )

    output_path = (
        "results/plots/integrated_gradients_MI.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nECG ID:", int(
        np.load("data/processed/ecg_ids.npy")[sample_idx]
    ))

    print("Actual MI label:", int(target[mi_index]))
    print("Predicted MI probability:", round(
        float(probabilities[mi_index]), 4
    ))

    print("\nAttribution visualization saved:")
    print(output_path)

    print("\nINTEGRATED GRADIENTS COMPLETED!")


if __name__ == "__main__":
    main()
