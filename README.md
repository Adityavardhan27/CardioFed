# CardioFed: Multi-Label ECG Classification (PTB-XL) with Federated Learning

CardioFed classifies 12-lead ECGs from PTB-XL into five diagnostic
superclasses (NORM, MI, STTC, CD, HYP). It compares centralized training with
Federated Averaging (FedAvg) across four simulated, site-based clients, and
adds statistical analysis, per-site evaluation, robustness tests,
Integrated-Gradients explanations, and a Streamlit demo.

> **Research-use notice:** This project is for research and education. It is
> not a medical device and must not be used for clinical diagnosis or
> treatment decisions.

**Verified results, limitations and supported conclusions: see
[RESULTS.md](RESULTS.md).** In short: FedAvg after 8 rounds is statistically
indistinguishable from centralized training of the same model
(test macro-AUROC 0.906 vs 0.906), and the multi-scale CNN shows no measurable
advantage over the baseline CNN.

---

## 1. Environment

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements.txt` pins the exact versions with which all reported metrics
were re-verified (Python 3.10, CPU). The versions used for the original
training runs were not recorded. On CPU-only machines, install the CPU PyTorch
wheel first (see the comment in `requirements.txt`).

Run the tests:

```bash
python smoke_test_preprocessing.py
python smoke_test_model.py
python -m pytest tests
```

The `tests/` suite covers:
- unit tests: output guard, thresholds, FedAvg weighting, label policy;
- a federated smoke test on synthetic data;
- data and artifact checks: patient-disjoint splits, label reproduction,
  client partition sizes, and every reported test metric recomputed from its
  checkpoint.

The data-dependent tests skip automatically when the data are absent.

## 2. Data pipeline

All reported results use `data/processed/`. The two stages below reproduce it
from raw PTB-XL (verified: labels, folds, IDs, sites and metadata are
byte-identical; signals agree to within 1.2e-7 float32 rounding).

```bash
# 0. Download PTB-XL 1.0.3 (100 Hz records + metadata; resumable)
python download_ptbxl.py                      # into data/raw/ptb-xl

# 1. Filter (0.5-40 Hz) and z-score each record/lead
python -m src.prepare_data --root data/raw/ptb-xl --out data/stage1

# 2. Apply the CardioFed label policy (required for all reported results)
python rebuild_labels.py --processed data/stage1 --out data/processed_new
```

**Label policy** (`rebuild_labels.py`):
- Only diagnostic statements with likelihood > 0 are used.
- NORM is removed when any abnormal superclass is present.
- Unlabeled records are dropped.

This differs from the usual PTB-XL superclass benchmark for 528 records
(2.5%); see RESULTS.md, Section 1.

Splits use the official `strat_fold`: folds 1–8 train, fold 9 validation,
fold 10 test. They are patient-disjoint.

`data/processed_verified_backup/` is a byte-identical copy of `data/processed/`.

## 3. Output safety

Every training, evaluation and analysis script **refuses to overwrite an
existing output file**. Pass a new output path, or `--overwrite` if replacement
is intentional. `rebuild_labels.py` writes to a separate directory and only
rewrites its input with an explicit `--in-place` flag.

## 4. Models and how each artifact was produced

| Model (registry key) | Training command | Evaluation command → artifact |
|---|---|---|
| `baseline` | `python -m src.train --epochs 10`¹ | `python -m src.final_evaluation` → `results/metrics/baseline_final_evaluation.json` |
| `multiscale` (primary) | `python -m src.train_multiscale` (defaults; deterministic on CPU) | `python -m src.evaluate_scheduler_experiment` → `results/scheduler_experiment/metrics/multiscale_final_evaluation.json` |
| `federated_r3` | `python -m src.federated_train --rounds 3`² | `python -m src.final_evaluation_federated --checkpoint results/checkpoints/federated_3round_best.pt --output ...` → `results/metrics/federated_3round_final_evaluation.json` |
| `federated_r8` | `python -m src.federated_train --rounds 8`² | `python -m src.final_evaluation_federated` → `results/metrics/federated_final_evaluation.json` |

Notes:
- ¹ Epoch count taken from `baseline_history.json`. Batch size, learning rate
  and patience were not recorded; the script defaults are assumed.
- ² Batch size 16 and 1 local epoch are recorded in the summaries; learning
  rate and seed are assumed to be the defaults. Current runs write
  `results/checkpoints/federated_<R>round_best.pt`; the 8-round model predates
  per-run naming and is stored as `federated_best.pt`.
- Because the artifacts already exist, the default commands now refuse to run.
  Re-running an evaluation with `--output <new path>` reproduces the original
  JSON exactly. This was verified for all five models
  (`results/revised/reproduced/`).
- `src/model_registry.py` is the single place that maps each model to its
  checkpoint and evaluation file. All analyses and the app use it.

### Legacy and duplicate artifacts

| Artifact | Status |
|---|---|
| `results/checkpoints/multiscale_best.pt`, `results/metrics/multiscale_*.json` | Legacy multi-scale run (best epoch 7) trained by code no longer in the repository; not reproducible. Kept for transparency, excluded from the primary comparison. |
| `results/metrics/prediction_error_analysis.json`, `class_error_counts.csv`, `misclassified_ecgs.csv`, `prediction_rule_validation.json`, `results/plots/integrated_gradients_MI.png`, `results/examples/ECG_63_prediction.json` | Produced with the legacy model. Revised equivalents for the primary model are in `results/revised/`. |
| `results/comparison/` | Original comparison using the legacy multi-scale model. `comparison_summary.json` names "best" models by point estimate, which the statistical analysis does not support. Use `results/revised/comparison/`. |
| `results/metrics/federated_3round_summary.json` | Its `best_checkpoint` field says `federated_best.pt`; the 3-round checkpoint is actually `federated_3round_best.pt`. |
| `results/diagnostic/` | Independent rerun of the primary multi-scale training; weights identical to `results/scheduler_experiment/`, which confirms determinism. |

## 5. Revised analyses (no retraining)

Each script writes only to `results/revised/<name>/`.

```bash
python -m src.statistical_analysis --include-legacy  # bootstrap CIs, paired tests
python -m src.compare_models                         # tables/plots with CIs
python -m src.per_site_evaluation                    # test metrics by site
python -m src.robustness_evaluation                  # noise, wander, missing leads
python -m src.explainability_v2                      # IG + sanity checks
python -m src.validate_prediction_rules              # NORM rule on validation
python -m src.analyze_prediction_errors              # test error analysis
python -m src.predict_ecg                            # single-ECG demo prediction
```

Run `statistical_analysis` before `compare_models`.

## 6. Federated learning setup

`src/federated_train.py` forms 4 clients from training patients by their
dominant recording site: site 0, site 1, site 2, and all other sites. Every
patient belongs to exactly one client.

FedAvg weights clients by record count. Simulation assumptions (also in the
script header):
- class weights come from pooled labels;
- one central validation fold is used for selection and thresholds;
- BatchNorm statistics are averaged;
- each client uses a fresh Adam optimiser per round;
- there is no differential privacy or secure aggregation.

## 7. Demo app

```bash
streamlit run src/app.py
```

The app uses the primary multi-scale model and its validation-tuned
thresholds. It accepts a CSV of **already preprocessed** 12-lead signals
(1000 samples), or an image in the project's own synthetic layout. It warns
when a lead is flat or missing, because the models classify flat-line input
as abnormal. It applies the post-hoc NORM consistency rule and shows the
predictions both before and after the rule.

## 8. Exploratory components (not part of the reported evaluation)

These scripts are kept as the record of development:
- digitizer versions `src/ecg_image_digitizer*.py` (v1–v5);
- `src/generate_synthetic_ecg*.py`, `src/batch_digitization_experiment.py`;
- `src/evaluate_digitization*.py`, `src/compare_*`,
  `src/evaluate_multirecord_prediction_consistency.py`;
- `src/analyze_reconstruction.py`, `src/analyze_ecg_morphology.py`,
  `src/test_normalization.py`.

They still refer to the legacy checkpoint where applicable.

The digitization experiments render already-preprocessed test signals into
synthetic images and read them back, using the same layout constants as the
renderer. Their results measure round-trip consistency only, not performance
on real ECG images (see RESULTS.md, Section 9).

## 9. Repository layout

```
download_ptbxl.py, rebuild_labels.py   data acquisition and label policy
src/prepare_data.py, preprocessing.py  signal preprocessing
src/dataset.py, model.py               data loading and architectures
src/train*.py, federated_train.py      training (centralized / FedAvg)
src/final_evaluation*.py, evaluate_scheduler_experiment.py
src/model_registry.py, thresholds.py, io_utils.py, evaluate.py
src/statistical_analysis.py, per_site_evaluation.py,
    robustness_evaluation.py, explainability_v2.py, compare_models.py
src/app.py                             Streamlit demo
tests/                                 pytest suite
results/                               original artifacts (unchanged)
results/revised/                       outputs of the revised analyses
```

Dataset attribution: see [DATASET_ATTRIBUTION.md](DATASET_ATTRIBUTION.md)
(PTB-XL, CC BY 4.0).
