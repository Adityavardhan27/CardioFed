# CardioFed: Multi-Label ECG Classification (PTB-XL) and Federated Learning

CardioFed is a research project for multi-label classification of 12-lead ECG recordings using the PTB-XL dataset. The repository includes preprocessing, centralized training, multi-scale training, federated training, evaluation, and a Streamlit application.

> **Research-use notice:** This project is for research and educational use. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## 1. Set up the environment

From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the listed dependencies:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the lightweight checks:

```bash
python smoke_test_preprocessing.py
python smoke_test_model.py
```

The preprocessing smoke test creates a small temporary metadata fixture. The model smoke test uses synthetic arrays and runs the centralized training loop; it does not require PTB-XL. It creates an isolated temporary directory using Python's `tempfile` module and automatically cleans it up when the test finishes.

## 2. Obtain PTB-XL

Download PTB-XL version 1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and extract the dataset while preserving its directory structure.

For the commands below, the dataset root is assumed to be:

```text
data/raw/ptb-xl/
```

That directory must contain the PTB-XL metadata files `ptbxl_database.csv` and `scp_statements.csv`, along with the signal files/directories referenced by the metadata. If you extracted the dataset elsewhere, pass that location to `--root`.

## 3. Prepare the processed dataset

From the repository root:

```bash
python -m src.prepare_data --root data/raw/ptb-xl --out data/processed
```

The preparation script writes the processed arrays and metadata into the output directory, including `X.npy`, `y.npy`, `folds.npy`, `sites.npy`, `ecg_ids.npy`, and `processed_metadata.csv`.

**Important:** `src.prepare_data` creates `X.npy` using write mode and writes the other output files as well. Running it again with an existing output directory can overwrite existing processed data. Back up the directory or choose a new `--out` location before rerunning it.

## 4. Label-rebuilding script: read before use

The repository also contains `rebuild_labels.py`. It is **not** part of the normal preparation command above.

This script currently:
- Uses hardcoded paths: `data/raw/ptb-xl` and `data/processed`.
- Executes its work immediately when the script is run (there is no `if __name__ == "__main__"` guard).
- Rebuilds labels using an exclusive-NORM policy: if an abnormal diagnostic class is present, NORM is removed for that record.
- Rewrites the processed signal array and aligned labels/metadata, replacing the existing `X.npy` as part of the process.

Do not run `python rebuild_labels.py` on an existing processed dataset unless you have intentionally chosen this label policy and have a verified backup. Its behavior and label policy should be reviewed before use.

## 5. Train the centralized baseline

The current centralized training entry point is:

```bash
python -m src.train
```

Available command-line options are:

```text
--epochs
--batch-size
--lr
--patience
```

For example:

```bash
python -m src.train --epochs 12 --batch-size 32 --lr 0.001 --patience 4
```

The current `src.train` CLI does **not** expose an `--lstm` option. The training function constructs the baseline ECGNet with `use_lstm=False`.

Training reads the processed data from `data/processed` and writes the best checkpoint and training history under `results/checkpoints` and `results/metrics`. Running training again can overwrite those output files.

## 6. Other project components

The source tree also contains the multi-scale training implementation, federated training, evaluation utilities, explainability code, and the Streamlit application. Their data and checkpoint prerequisites should be checked in the corresponding scripts before running them; a fresh Git clone does not include gitignored datasets or checkpoints.

