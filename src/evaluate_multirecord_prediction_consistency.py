
"""Evaluate CNN prediction consistency for original, V4 and V5 ECGs."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import MultiScaleResidualECGNet
from src.preprocessing import CLASSES


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = ROOT / "data" / "processed"

SELECTION_FILE = ROOT / "results" / "examples" / "multirecord_test_selection_100.csv"
RECON_DIR = ROOT / "results" / "multirecord_digitization_100" / "reconstructed"
OUTPUT_DIR = ROOT / "results" / "multirecord_prediction_100"



CHECKPOINT_PATH = (
    ROOT / "results" / "checkpoints"
    / "multiscale_best.pt"
)

EVALUATION_PATH = (
    ROOT / "results" / "metrics"
    / "multiscale_final_evaluation.json"
)

OUTPUT_DIR = (
    ROOT / "results" / "multirecord_prediction_100"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEADS = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():

    print("\nLoading MultiScale CNN...")

    model = MultiScaleResidualECGNet(
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model


# --------------------------------------------------
# LOAD THRESHOLDS
# --------------------------------------------------

def load_thresholds():

    with open(EVALUATION_PATH, "r") as file:
        evaluation = json.load(file)

    thresholds = np.array(
        [
            evaluation["thresholds"][name]
            for name in CLASSES
        ],
        dtype=np.float32
    )

    print("\nCLASSIFICATION THRESHOLDS")

    for name, threshold in zip(CLASSES, thresholds):
        print(f"{name}: {threshold:.4f}")

    return thresholds


# --------------------------------------------------
# CNN PREDICTION
# --------------------------------------------------

def predict_ecg(signal, model, thresholds):

    # Input shape: (1000, 12)
    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if signal.shape != (1000, 12):
        raise ValueError(
            f"Unexpected ECG shape: {signal.shape}"
        )

    if not np.isfinite(signal).all():
        raise ValueError("ECG contains invalid values.")

    # Model input shape: (1, 12, 1000)
    tensor = torch.from_numpy(
        signal.T.copy()
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()[0]

    # Raw threshold predictions.
    raw_predictions = (
        probabilities >= thresholds
    ).astype(int)

    # Same NORM consistency rule as app.py.
    final_predictions = raw_predictions.copy()

    abnormal_indices = [
        CLASSES.index(name)
        for name in ["MI", "STTC", "CD", "HYP"]
    ]

    if final_predictions[abnormal_indices].any():
        final_predictions[0] = 0

    return (
        probabilities,
        raw_predictions,
        final_predictions
    )


# --------------------------------------------------
# LOAD RECONSTRUCTED SIGNALS
# --------------------------------------------------

def load_reconstructed(ecg_id, version):

    path = RECON_DIR / f"ECG_{ecg_id}_{version}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing reconstructed ECG: {path}"
        )

    df = pd.read_csv(path)

    missing = [
        lead for lead in LEADS
        if lead not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing leads in {path}: {missing}"
        )

    signal = df[LEADS].to_numpy(
        dtype=np.float32
    )

    if signal.shape != (1000, 12):
        raise ValueError(
            f"Unexpected reconstructed shape: {signal.shape}"
        )

    return signal


# --------------------------------------------------
# PREDICTION COMPARISON
# --------------------------------------------------

def main():

    print("=" * 65)
    print("CARDIOFED: MULTI-RECORD PREDICTION CONSISTENCY")
    print("=" * 65)

    # Load dataset without loading the full X array into RAM.
    X = np.load(
        DATA_DIR / "X.npy",
        mmap_mode="r"
    )

    ids = np.load(
        DATA_DIR / "ecg_ids.npy"
    )

    folds = np.load(
        DATA_DIR / "folds.npy"
    )

    selection = pd.read_csv(SELECTION_FILE)

    model = load_model()
    thresholds = load_thresholds()

    records = []
    class_results = []

    print("\nSTARTING PREDICTION EVALUATION")
    print("=" * 65)

    for number, (_, row) in enumerate(
        selection.iterrows(),
        start=1
    ):

        idx = int(row["dataset_index"])
        ecg_id = int(row["ecg_id"])

        if int(ids[idx]) != ecg_id:
            raise ValueError(
                f"ECG ID mismatch for index {idx}"
            )

        if int(folds[idx]) != 10:
            raise ValueError(
                f"ECG {ecg_id} is not in test fold 10"
            )

        print(
            f"\n[{number}/{len(selection)}] "
            f"ECG ID: {ecg_id}"
        )

        # Original numerical ECG.
        original_signal = np.asarray(
            X[idx],
            dtype=np.float32
        )

        # Reconstructed ECGs.
        v4_signal = load_reconstructed(
            ecg_id,
            "V4"
        )

        v5_signal = load_reconstructed(
            ecg_id,
            "V5"
        )

        # Obtain predictions using the same model.
        original = predict_ecg(
            original_signal,
            model,
            thresholds
        )

        v4 = predict_ecg(
            v4_signal,
            model,
            thresholds
        )

        v5 = predict_ecg(
            v5_signal,
            model,
            thresholds
        )

        original_prob, original_raw, original_final = original
        v4_prob, v4_raw, v4_final = v4
        v5_prob, v5_raw, v5_final = v5

        # Probability MAE relative to original ECG.
        v4_probability_mae = float(
            np.mean(
                np.abs(v4_prob - original_prob)
            )
        )

        v5_probability_mae = float(
            np.mean(
                np.abs(v5_prob - original_prob)
            )
        )

        # Exact multi-label agreement.
        v4_raw_agreement = int(
            np.array_equal(v4_raw, original_raw)
        )

        v5_raw_agreement = int(
            np.array_equal(v5_raw, original_raw)
        )

        v4_final_agreement = int(
            np.array_equal(v4_final, original_final)
        )

        v5_final_agreement = int(
            np.array_equal(v5_final, original_final)
        )

        # Per-record Hamming agreement.
        v4_raw_label_agreement = float(
            np.mean(v4_raw == original_raw)
        )

        v5_raw_label_agreement = float(
            np.mean(v5_raw == original_raw)
        )

        v4_final_label_agreement = float(
            np.mean(v4_final == original_final)
        )

        v5_final_label_agreement = float(
            np.mean(v5_final == original_final)
        )

        records.append({
            "ecg_id": ecg_id,

            "V4_Probability_MAE": v4_probability_mae,
            "V5_Probability_MAE": v5_probability_mae,

            "V4_Raw_Exact_Agreement": v4_raw_agreement,
            "V5_Raw_Exact_Agreement": v5_raw_agreement,

            "V4_Final_Exact_Agreement": v4_final_agreement,
            "V5_Final_Exact_Agreement": v5_final_agreement,

            "V4_Raw_Label_Agreement": v4_raw_label_agreement,
            "V5_Raw_Label_Agreement": v5_raw_label_agreement,

            "V4_Final_Label_Agreement": v4_final_label_agreement,
            "V5_Final_Label_Agreement": v5_final_label_agreement
        })

        # Per-class probability comparisons.
        for j, class_name in enumerate(CLASSES):

            class_results.append({
                "ecg_id": ecg_id,
                "class": class_name,

                "Original_Probability": float(
                    original_prob[j]
                ),

                "V4_Probability": float(
                    v4_prob[j]
                ),

                "V5_Probability": float(
                    v5_prob[j]
                ),

                "V4_Absolute_Error": float(
                    abs(v4_prob[j] - original_prob[j])
                ),

                "V5_Absolute_Error": float(
                    abs(v5_prob[j] - original_prob[j])
                ),

                "Original_Final_Label": int(
                    original_final[j]
                ),

                "V4_Final_Label": int(
                    v4_final[j]
                ),

                "V5_Final_Label": int(
                    v5_final[j]
                )
            })

        original_labels = [
            CLASSES[j]
            for j in range(5)
            if original_final[j] == 1
        ]

        v4_labels = [
            CLASSES[j]
            for j in range(5)
            if v4_final[j] == 1
        ]

        v5_labels = [
            CLASSES[j]
            for j in range(5)
            if v5_final[j] == 1
        ]

        print("Original prediction:", original_labels)
        print("V4 prediction:      ", v4_labels)
        print("V5 prediction:      ", v5_labels)

        print(
            f"Probability MAE | "
            f"V4: {v4_probability_mae:.5f} | "
            f"V5: {v5_probability_mae:.5f}"
        )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    record_df = pd.DataFrame(records)
    class_df = pd.DataFrame(class_results)

    record_df.to_csv(
        OUTPUT_DIR / "per_record_prediction_metrics.csv",
        index=False
    )

    class_df.to_csv(
        OUTPUT_DIR / "per_class_prediction_metrics.csv",
        index=False
    )

    # --------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------

    summary = {
        "number_of_records": len(record_df),

        "V4_mean_probability_MAE": float(
            record_df["V4_Probability_MAE"].mean()
        ),

        "V5_mean_probability_MAE": float(
            record_df["V5_Probability_MAE"].mean()
        ),

        "V4_raw_exact_agreement_count": int(
            record_df["V4_Raw_Exact_Agreement"].sum()
        ),

        "V5_raw_exact_agreement_count": int(
            record_df["V5_Raw_Exact_Agreement"].sum()
        ),

        "V4_final_exact_agreement_count": int(
            record_df["V4_Final_Exact_Agreement"].sum()
        ),

        "V5_final_exact_agreement_count": int(
            record_df["V5_Final_Exact_Agreement"].sum()
        ),

        "V4_mean_raw_label_agreement": float(
            record_df["V4_Raw_Label_Agreement"].mean()
        ),

        "V5_mean_raw_label_agreement": float(
            record_df["V5_Raw_Label_Agreement"].mean()
        ),

        "V4_mean_final_label_agreement": float(
            record_df["V4_Final_Label_Agreement"].mean()
        ),

        "V5_mean_final_label_agreement": float(
            record_df["V5_Final_Label_Agreement"].mean()
        )
    }

    # Per-class aggregate probability errors.
    per_class_summary = (
        class_df.groupby("class")
        .agg(
            V4_Mean_Probability_Error=(
                "V4_Absolute_Error", "mean"
            ),
            V5_Mean_Probability_Error=(
                "V5_Absolute_Error", "mean"
            )
        )
        .reset_index()
    )

    per_class_summary.to_csv(
        OUTPUT_DIR / "per_class_summary.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("PREDICTION CONSISTENCY RESULTS")
    print("=" * 65)

    print(
        "\nMean probability MAE:"
    )

    print(
        f"V4: {summary['V4_mean_probability_MAE']:.6f}"
    )

    print(
        f"V5: {summary['V5_mean_probability_MAE']:.6f}"
    )

    print("\nExact final multi-label agreement:")

    print(
        f"V4: {summary['V4_final_exact_agreement_count']}"
        f"/{len(record_df)}"
    )

    print(
        f"V5: {summary['V5_final_exact_agreement_count']}"
        f"/{len(record_df)}"
    )

    print("\nMean final label agreement:")

    print(
        f"V4: "
        f"{summary['V4_mean_final_label_agreement']:.4f}"
    )

    print(
        f"V5: "
        f"{summary['V5_mean_final_label_agreement']:.4f}"
    )

    print("\nPER-CLASS PROBABILITY ERROR")
    print(per_class_summary.to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- per_record_prediction_metrics.csv")
    print("- per_class_prediction_metrics.csv")
    print("- per_class_summary.csv")
    print("- summary.json")


if __name__ == "__main__":
    main()
