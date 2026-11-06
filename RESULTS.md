# CardioFed: Verified Results

All numbers below were recomputed from the stored checkpoints on 2026-11-07
(`python -m pytest tests` re-checks them). Every reported test metric
reproduces exactly from its checkpoint. Generated tables and figures are in
`results/revised/`; the original artifacts in `results/` are unchanged.

## 1. Setup

| Item | Value |
|---|---|
| Dataset | PTB-XL 1.0.3, 100 Hz, 12 leads, 10 s (1000 samples) |
| Task | Multi-label: NORM, MI, STTC, CD, HYP |
| Records used | 21,375 of 21,799 |
| Splits | Official `strat_fold`: folds 1–8 train (17,073), fold 9 validation (2,145), fold 10 test (2,157 records, 1,876 patients) |
| Patient overlap between splits | None (verified) |
| Preprocessing | 0.5–40 Hz Butterworth band-pass (order 3, zero-phase); z-score per record and per lead |
| Thresholds | Per class, maximising F1 on validation; applied unchanged to test |
| Model selection | Best validation macro-AUROC (epoch or round) |

### Label policy: differs from the PTB-XL benchmark

1. Only diagnostic SCP statements with likelihood > 0 are used.
2. NORM is removed when any abnormal superclass is present (NORM-exclusive).
3. Records without a remaining label are dropped.

Compared with the common superclass mapping (all likelihoods, no NORM
exclusion), 528 records (2.5%) have a different label set: 417 lose NORM, and
13 records are excluded. Class counts are NORM 9,097, MI 5,469, STTC 5,108,
CD 4,898, HYP 2,649. **Results are therefore not directly comparable with
published PTB-XL superclass benchmarks.**

## 2. Models

| Key | Model | Training | Checkpoint |
|---|---|---|---|
| baseline | 3-block 1D CNN | centralized, 10 epochs (best = epoch 10) | `results/checkpoints/centralized_best.pt` |
| multiscale | Multi-scale residual CNN (kernels 3/7/15) | centralized, ReduceLROnPlateau, early stop at epoch 15 (best = epoch 10) | `results/scheduler_experiment/checkpoints/multiscale_best.pt` |
| federated_r3 | Multi-scale residual CNN | FedAvg, 4 clients, 3 rounds × 1 local epoch | `results/checkpoints/federated_3round_best.pt` |
| federated_r8 | Multi-scale residual CNN | FedAvg, 4 clients, 8 rounds × 1 local epoch (best = round 8) | `results/checkpoints/federated_best.pt` |

`results/checkpoints/multiscale_best.pt` (legacy run, best epoch 7) was trained
by code that is no longer in the repository. It is excluded from the primary
comparison (its test macro-AUROC is 0.905, macro-F1 0.712).

## 3. Main comparison (test fold)

95% CIs come from a patient-level bootstrap (2,000 resamples, seed 0).

| Model | Macro-AUROC [95% CI] | Macro-F1 [95% CI] | Hamming loss | Subset accuracy |
|---|---|---|---|---|
| baseline | 0.908 [0.899, 0.916] | 0.710 [0.692, 0.727] | 0.131 | 0.582 |
| multiscale | 0.906 [0.897, 0.915] | 0.705 [0.688, 0.721] | 0.137 | 0.576 |
| federated_r3 | 0.894 [0.884, 0.903] | 0.694 [0.676, 0.711] | 0.142 | 0.558 |
| federated_r8 | 0.906 [0.897, 0.914] | 0.706 [0.689, 0.723] | 0.131 | 0.590 |

Pre-specified paired comparisons (paired patient bootstrap, Holm-corrected
within each metric):

| Comparison | Δ macro-AUROC [95% CI] | p (Holm) | Δ macro-F1 [95% CI] | p (Holm) |
|---|---|---|---|---|
| federated_r8 − multiscale | −0.0004 [−0.0050, 0.0044] | 1.00 | +0.0017 [−0.0098, 0.0126] | 1.00 |
| baseline − multiscale | +0.0015 [−0.0025, 0.0055] | 1.00 | +0.0052 [−0.0057, 0.0163] | 1.00 |
| baseline − federated_r8 | +0.0019 [−0.0022, 0.0057] | 1.00 | +0.0035 [−0.0081, 0.0155] | 1.00 |
| federated_r8 − federated_r3 | +0.0121 [0.0092, 0.0151] | 0.002 | +0.0126 [0.0040, 0.0209] | 0.012 |

**Supported conclusions**
- After 8 rounds, FedAvg across four site-based clients reaches test
  performance statistically indistinguishable from centralized training of
  the same architecture (Δ macro-AUROC −0.0004; 95% CI −0.005 to +0.004).
- 8 rounds improve significantly over 3 rounds.
- The multi-scale architecture shows no measurable advantage over the
  simpler baseline CNN.

**Not supported:** any claim that one of baseline, multiscale or
federated_r8 is "best". (The original `results/comparison/comparison_summary.json`
names best models by point estimate; disregard it.)

### Per-class test AUROC [95% CI]

| Class | baseline | multiscale | federated_r3 | federated_r8 |
|---|---|---|---|---|
| NORM | 0.935 [0.924, 0.944] | 0.933 [0.922, 0.942] | 0.925 [0.914, 0.936] | 0.935 [0.925, 0.945] |
| MI | 0.924 [0.911, 0.936] | 0.921 [0.908, 0.934] | 0.900 [0.884, 0.914] | 0.915 [0.902, 0.928] |
| STTC | 0.933 [0.921, 0.945] | 0.932 [0.919, 0.943] | 0.924 [0.912, 0.937] | 0.932 [0.920, 0.944] |
| CD | 0.913 [0.895, 0.929] | 0.920 [0.904, 0.935] | 0.909 [0.891, 0.924] | 0.917 [0.901, 0.932] |
| HYP | 0.833 [0.806, 0.858] | 0.825 [0.796, 0.851] | 0.810 [0.780, 0.839] | 0.829 [0.802, 0.855] |

HYP is the weakest class for every model (F1 0.46–0.47). A plausible
contributor, not tested here, is that per-record z-scoring removes absolute
voltage, which is used in hypertrophy criteria.

## 4. Federated setting

| Client | Training records | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|
| 1 (site 0) | 7,692 | 0.421 | 0.291 | 0.223 | 0.210 | 0.123 |
| 2 (site 1) | 5,060 | 0.437 | 0.209 | 0.268 | 0.225 | 0.154 |
| 3 (site 2) | 3,318 | 0.420 | 0.250 | 0.231 | 0.269 | 0.078 |
| 4 (other sites) | 1,003 | 0.419 | 0.250 | 0.247 | 0.258 | 0.132 |

Heterogeneity is mainly in quantity (7.7× between largest and smallest
client) and recording site. Label skew is mild: NORM prevalence is 42–44% on
every client.

Simulation assumptions:
- Positive-class weights come from pooled training labels.
- Model selection and threshold tuning use one central validation fold.
- BatchNorm statistics are averaged.
- Each client uses a fresh Adam optimiser per round.
- No differential privacy or secure aggregation is implemented; no formal
  privacy guarantee is claimed.

Compute is not exactly matched: federated training performed 8 passes over
the training data, versus 15 epochs (best at 10) for the centralized
multi-scale model.

## 5. Per-site test performance

Each model uses its own validation-tuned thresholds. CIs come from a
patient-level bootstrap within each site.

| Site group (n) | baseline | multiscale | federated_r3 | federated_r8 |
|---|---|---|---|---|
| site 0 (483) | 0.923 [0.903, 0.942] | 0.919 [0.899, 0.938] | 0.913 [0.893, 0.932] | 0.926 [0.907, 0.943] |
| site 1 (597) | 0.917 [0.902, 0.930] | 0.915 [0.902, 0.929] | 0.902 [0.886, 0.917] | 0.916 [0.901, 0.929] |
| site 2 (867) | 0.879 [0.864, 0.895] | 0.875 [0.858, 0.890] | 0.867 [0.849, 0.883] | 0.877 [0.860, 0.892] |
| other (210) | 0.920 [0.895, 0.943] | 0.927 [0.903, 0.948] | 0.903 [0.874, 0.929] | 0.915 [0.889, 0.937] |

Every model performs worst on site 2. Label composition differs strongly
between sites in the test fold (NORM 64% at site 0, 33% at site 2, 28% at
other sites), whereas training clients have similar composition. Site and
label-mix effects are therefore confounded; no device or site effect is
claimed. Source: `results/revised/per_site/`.

## 6. Robustness (inference only)

Corruptions are applied to the processed test signals, which are then passed
through the pipeline's own preprocessing again. Thresholds are not re-tuned.
Each entry is a single seeded realisation; no CIs.

| Perturbation | baseline | multiscale | federated_r8 |
|---|---|---|---|
| clean | 0.908 | 0.906 | 0.906 |
| Gaussian noise σ = 0.2 (× lead std) | 0.907 | 0.904 | 0.905 |
| Gaussian noise σ = 0.5 | 0.901 | 0.894 | 0.897 |
| Baseline wander, amplitude 2.0 | 0.900 | 0.896 | 0.901 |
| 1 random lead removed | 0.896 | 0.894 | 0.895 |
| 3 random leads removed | 0.867 | 0.870 | 0.870 |
| 6 random leads removed | 0.806 | 0.817 | 0.820 |
| Limb leads only (V1–V6 removed) | 0.818 | 0.805 | 0.836 |
| Precordial leads only | 0.836 | 0.819 | 0.847 |
| Lead I only | 0.658 | 0.664 | 0.606 |

Values are macro-AUROC.
- All models tolerate moderate noise and baseline wander; low-frequency
  wander is largely removed by the 0.5 Hz high-pass filter.
- Missing leads cause large drops, and macro-F1 falls further because the
  thresholds are calibrated for complete ECGs (e.g. limb-only macro-F1
  0.36–0.38).
- All models classify a flat-line (all-zero) ECG as abnormal: STTC
  probability 0.71–0.97, HYP 0.76–0.97. The demo app now warns when a lead
  is flat or missing.

Source: `results/revised/robustness/`.

## 7. Explainability (Integrated Gradients)

| Item | Value |
|---|---|
| Models | multiscale and federated_r8 |
| Selection | For each class, test ECGs positive for that class whose full predicted label set is correct for both models; random sample of 30 per class (seed 0), 3 per class plotted |
| Eligible ECGs per class | NORM 761, MI 148, STTC 134, CD 134, HYP 50 |
| Method | zero baseline, 64-step midpoint Riemann sum, target = class logit |

Sanity checks (300 explanations):
- **Completeness error:** median 0.6% (multiscale) and 0.3% (federated_r8);
  12 of 300 explanations exceed 5%. All of these are cases where the
  baseline and input logits are almost equal.
- **Deletion test:** setting the top 1% most positively attributed samples to
  the baseline lowers the class logit by 3.1 (multiscale) and 2.6
  (federated_r8) on average, versus 0.01–0.03 for removing 1% of samples at
  random. Top-attributed removal had the larger effect in 100% of cases at
  every tested fraction (1–10%).

Limitations:
- The zero baseline is not neutral for STTC and HYP, because the models
  already predict these classes for a flat-line input. Attributions for those
  classes are relative to an input the model scores as positive.
- The lead-level attribution summary (`lead_importance.png`) describes model
  behaviour on a limited sample. It is not clinical validation, and no
  clinical interpretation is claimed.

The original single-example figure (`results/plots/integrated_gradients_MI.png`)
explains ECG 63 using the legacy model, which misclassified that ECG (false
NORM). Use the revised figures instead. Source: `results/revised/explainability/`.

## 8. NORM consistency rule (post-hoc)

The rule removes NORM whenever an abnormal class is predicted. On the primary
multi-scale model:
- **Validation:** subset accuracy 0.581 → 0.596, macro-F1 0.727 → 0.725,
  Hamming loss 0.1275 → 0.1286.
- **Test:** subset accuracy 0.576 → 0.587, macro-F1 0.705 → 0.703,
  Hamming loss 0.1366 → 0.1368; 72 predicted NORM + abnormal conflicts
  removed. Source: `results/revised/error_analysis/`.

The rule largely mirrors the NORM-exclusive labelling. It is applied in the
demo app but not in any reported metric.

## 9. ECG image digitization: proof of concept only

The digitization experiments do not evaluate real ECG images:
- The images are rendered from already-preprocessed test signals with a fixed
  synthetic layout.
- The digitizer uses the same geometry constants and trace colour as the
  renderer.
- No paper grid noise, scanning artefacts, rotation or real photographs are
  involved, and amplitudes are in z-score units rather than millivolts.

The reported reconstruction quality (mean correlation 0.979, MAE 0.077 for V5)
and prediction agreement (93/100 identical label sets) therefore measure
render–digitize round-trip consistency only, not real-world digitization
performance.

## 10. Limitations

1. Single training seed per model; seed variance is unknown.
2. The baseline (best = last epoch) and FedAvg-8 (best = last round) may not
   have converged.
3. No local-only (per-client) baselines, so the benefit of federation over
   isolated training is not established.
4. The test fold was used for several models and analyses. All results are
   reported; none were selected by test performance.
5. Per-record normalisation discards absolute amplitude.
6. Non-standard label policy (Section 1).
7. Hyper-parameters of the baseline and federated runs other than those
   recorded in history and summary files are assumed to be script defaults.
