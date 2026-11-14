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

FedAvg-8 had not fully converged. The converged 20-round FedAvg, three
seeds per arm, and local-only baselines are in Section 10.

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

Corruptions are applied to processed test signals, which are then passed through
the pipeline's standard preprocessing (`preprocess_signal`: 0.5–40 Hz band-pass
filter and per-lead z-score normalisation). Thresholds are strictly frozen from
validation tuning and never re-calibrated on corrupted test data.

Authoritative extended results are reported in `results/revised/robustness_extended/`
(`robustness_metrics.csv`, `robustness_paired_ci.csv`, `robustness_summary.json`,
`robustness_macro_auroc.png`, `robustness_macro_f1.png`). The original single-seed
45-row outputs in `results/revised/robustness/` are preserved byte-for-byte as
historical artifacts.

### 6.1 Evaluation methodology

- **Seed variability vs patient uncertainty:** For stochastic perturbations
  (Gaussian noise, baseline wander, random lead drop), 5 independent corruption
  realisation seeds (0–4) are evaluated across all models. Seed variability is
  reported as the standard deviation across seeds ($\sigma_{\text{seed}}$).
  Patient-sampling uncertainty is evaluated via paired patient-clustered bootstrap
  resampling ($B = 2,000$ replicates clustered across 1,876 unique patients /
  2,157 ECG recordings in fold 10). Within each bootstrap replicate, paired
  differences ($\Delta = \text{corrupted} - \text{clean}$) are calculated on the
  identical patient sample and averaged across realization seeds, keeping seed
  variance and patient sampling error strictly separated.
- **Extended conditions:** In addition to the original 10 conditions, the
  extended protocol includes: Lead II only, all 12 leads zero (flatline), signal
  amplitude clipping ($\pm 1.5, \pm 2.0, \pm 2.5$ standard deviations), and controlled
  circular temporal translations ($\pm 50, \pm 100$ samples, i.e., $\pm 0.5\text{s}, \pm 1.0\text{s}$).

### 6.2 Macro-AUROC under corruptions

Values are mean across 5 seeds for stochastic conditions (mean $\pm \sigma_{\text{seed}}$)
or point estimates for deterministic conditions (seed 0).

| Perturbation | baseline | multiscale | federated_r8 |
|---|---|---|---|
| Clean | 0.908 | 0.906 | 0.906 |
| Gaussian noise $\sigma = 0.05$ | 0.908 ± 0.000 | 0.906 ± 0.000 | 0.906 ± 0.000 |
| Gaussian noise $\sigma = 0.1$ | 0.908 ± 0.000 | 0.906 ± 0.000 | 0.906 ± 0.000 |
| Gaussian noise $\sigma = 0.2$ | 0.907 ± 0.000 | 0.905 ± 0.000 | 0.905 ± 0.000 |
| Gaussian noise $\sigma = 0.5$ | 0.902 ± 0.000 | 0.894 ± 0.001 | 0.897 ± 0.001 |
| Baseline wander amp = 0.5 | 0.908 ± 0.000 | 0.906 ± 0.000 | 0.906 ± 0.000 |
| Baseline wander amp = 1.0 | 0.906 ± 0.001 | 0.904 ± 0.000 | 0.905 ± 0.000 |
| Baseline wander amp = 2.0 | 0.900 ± 0.002 | 0.895 ± 0.001 | 0.901 ± 0.001 |
| 1 random lead removed | 0.898 ± 0.001 | 0.896 ± 0.002 | 0.896 ± 0.001 |
| 3 random leads removed | 0.870 ± 0.003 | 0.870 ± 0.003 | 0.873 ± 0.003 |
| 6 random leads removed | 0.801 ± 0.005 | 0.814 ± 0.004 | 0.818 ± 0.002 |
| Limb leads only (V1–V6 removed) | 0.818 | 0.805 | 0.836 |
| Precordial leads only | 0.836 | 0.819 | 0.847 |
| Lead I only | 0.658 | 0.664 | 0.606 |
| Lead II only | 0.592 | 0.637 | 0.684 |
| All leads zero (flatline) | 0.500 | 0.500 | 0.500 |
| Clipping at $\pm 2.5\,\text{std}$ | 0.901 | 0.897 | 0.900 |
| Clipping at $\pm 2.0\,\text{std}$ | 0.895 | 0.890 | 0.895 |
| Clipping at $\pm 1.5\,\text{std}$ | 0.885 | 0.877 | 0.887 |
| Circular translation $-100$ samples ($-1.0\text{s}$) | 0.908 | 0.906 | 0.906 |
| Circular translation $-50$ samples ($-0.5\text{s}$) | 0.908 | 0.905 | 0.905 |
| Circular translation $+50$ samples ($+0.5\text{s}$) | 0.908 | 0.906 | 0.906 |
| Circular translation $+100$ samples ($+1.0\text{s}$) | 0.908 | 0.906 | 0.905 |

### 6.3 Macro-F1 under corruptions (frozen validation thresholds)

| Perturbation | baseline | multiscale | federated_r8 |
|---|---|---|---|
| Clean | 0.710 | 0.705 | 0.706 |
| Gaussian noise $\sigma = 0.2$ | 0.708 ± 0.001 | 0.704 ± 0.001 | 0.694 ± 0.002 |
| Gaussian noise $\sigma = 0.5$ | 0.628 ± 0.002 | 0.643 ± 0.002 | 0.605 ± 0.003 |
| Baseline wander amp = 2.0 | 0.689 ± 0.003 | 0.679 ± 0.006 | 0.692 ± 0.005 |
| 6 random leads removed | 0.454 ± 0.007 | 0.475 ± 0.004 | 0.505 ± 0.004 |
| Limb leads only | 0.363 | 0.377 | 0.377 |
| Precordial leads only | 0.507 | 0.425 | 0.541 |
| Lead I only | 0.275 | 0.267 | 0.170 |
| Lead II only | 0.276 | 0.308 | 0.190 |
| All leads zero (flatline) | 0.275 | 0.119 | 0.119 |
| Clipping at $\pm 1.5\,\text{std}$ | 0.517 | 0.595 | 0.609 |
| Circular translations ($-100$ to $+100$) | 0.709–0.711 | 0.706–0.709 | 0.704–0.709 |

### 6.4 Key findings and statistical conclusions

1. **Noise and Baseline Wander:**
   - Realisation seed variability is minimal ($\sigma_{\text{seed}} \le 0.00075$ for noise, $\le 0.002$ for wander), indicating that single-realisation estimates were not anomalous.
   - For mild baseline wander (amplitude 0.5), 95% paired bootstrap CIs for $\Delta_{\text{AUROC}}$ span zero across all models (e.g. multiscale: $-0.0003$ [$-0.0007, +0.0001$], $p = 0.144$), confirming that the 0.5 Hz high-pass Butterworth filter effectively removes low-frequency wander.
   - At high amplitude wander (2.0) and severe noise ($\sigma = 0.5$), AUROC decreases by $-0.006$ to $-0.012$ (95% CIs exclude zero). Macro-F1 degrades substantially (dropping to $0.605 - 0.643$) because fixed validation thresholds become miscalibrated under elevated variance.
2. **Missing Lead Subsets:**
   - Reducing to limb leads only drops AUROC to $0.805 - 0.836$ ($\Delta_{\text{AUROC}} \in [-0.102, -0.070]$, 95% CIs strictly exclude zero) and collapses macro-F1 to $0.363 - 0.377$. Precordial leads preserve slightly more diagnostic information (AUROC $0.819 - 0.847$, F1 $0.425 - 0.541$).
   - Single-lead evaluation demonstrates severe degradation: Lead I yields macro-AUROC $0.606 - 0.664$; Lead II yields $0.592 - 0.684$. In both single-lead scenarios, macro-F1 collapses to $0.170 - 0.308$.
3. **Flatline (All-12-Leads-Zero):**
   - Setting all 12 leads to zero yields macro-AUROC of exactly $0.5000$ across all models ($\Delta_{\text{AUROC}} \approx -0.407$, 95% CIs $[-0.416, -0.397]$).
   - Because the zero baseline produces constant positive output logits exceeding frozen thresholds:
     - The Baseline 1D CNN predicts MI, STTC, CD, and HYP positive for 100% of flatline records (4 out of 5 classes).
     - Multi-Scale CNN and FedAvg-8 predict STTC and HYP positive for 100% of flatline records (2 out of 5 classes).
     - This broader false-positive pattern accounts for the baseline's higher flatline macro-F1 (0.275 vs 0.119 for multiscale and FedAvg-8). The app flatline warning remains clinically essential.
4. **Non-linear Amplitude Clipping:**
   - Mild clipping at $\pm 2.5\,\text{std}$ produces modest AUROC drops ($-0.006$ to $-0.009$, 95% CIs strictly negative).
   - Severe clipping at $\pm 1.5\,\text{std}$ significantly distorts waveform morphology, degrading AUROC by $-0.019$ to $-0.029$ and collapsing macro-F1 to $0.517 - 0.609$.
5. **Circular Temporal Shifts:**
   - Controlled circular translations of $\pm 50$ and $\pm 100$ samples ($\pm 0.5\text{s}$ and $\pm 1.0\text{s}$) produce negligible change in performance ($|\Delta_{\text{AUROC}}| \le 0.00067$).
   - All 95% paired bootstrap CIs for temporal shift span zero ($p \in [0.32, 0.97]$), demonstrating empirical shift-invariance due to 1D global average pooling before the classification head. Note: circular translations wrap signal boundaries; they do not simulate realistic acquisition-window entry/exit truncations.

Source: `results/revised/robustness_extended/`. Historical baseline: `results/revised/robustness/`.

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

## 10. P2 experiments: local-only baselines, longer FedAvg, extra seeds

The protocol was fixed and committed before training
(`results/p2/PROTOCOL.md`, including all deviations). Authoritative corrected
analysis outputs are in `results/p2/analysis_corrected/` (`comparisons.csv`,
`summary.json`, etc.); the original outputs in `results/p2/analysis/` are
retained as historical artifacts (see PROTOCOL.md §7 deviation 5).

Common settings:
- All models use the multi-scale CNN with the standard data and labels.
- Thresholds are tuned on the validation fold; the test fold is evaluated
  once per model.
- Local-only models use only their client's training records (folds 1–8,
  verified in code) and class weights from their own labels.
- Model selection uses the central validation fold, the same simulation
  assumption as FedAvg.

### 10.1 FedAvg convergence (20 rounds, 3 seeds)

| Seed | Best round | Best val AUROC | Val AUROC round 8 | Round 20 | Gain after round 8 |
|---|---|---|---|---|---|
| 42 | 14 | 0.9142 | 0.9096 | 0.9120 | +0.0046 |
| 43 | 13 | 0.9139 | 0.9083 | 0.9117 | +0.0056 |
| 44 | 14 | 0.9139 | 0.9068 | 0.9100 | +0.0067 |

- Validation macro-AUROC rises until about round 13–14, then plateaus or
  declines slightly while training loss keeps falling. Running more rounds
  is therefore not expected to help.
- The 8-round model reported earlier had not fully converged.
- Seed 42, rounds 1–8, reproduces the original 8-round run bit for bit
  (validation AUROC and all client losses identical).

Figure: `results/p2/analysis_corrected/fedavg_convergence.png` (historical: `results/p2/analysis/`).

### 10.2 Test results (fold 10)

| Model | Seed | Best epoch/round | Macro-AUROC | Macro-F1 | Hamming | Subset acc. |
|---|---|---|---|---|---|---|
| Centralized | 42 | 10 | 0.906 | 0.705 | 0.137 | 0.576 |
| Centralized | 43 | 9 | 0.912 | 0.715 | 0.132 | 0.573 |
| Centralized | 44 | 8 | 0.913 | 0.720 | 0.128 | 0.594 |
| FedAvg-20 | 42 | 14 | 0.910 | 0.712 | 0.127 | 0.596 |
| FedAvg-20 | 43 | 13 | 0.909 | 0.709 | 0.133 | 0.586 |
| FedAvg-20 | 44 | 14 | 0.910 | 0.719 | 0.131 | 0.574 |
| Local-only client 1 (7,692 rec.) | 42 | 10 | 0.896 | 0.693 | 0.143 | 0.543 |
| Local-only client 2 (5,060 rec.) | 42 | 8 | 0.889 | 0.685 | 0.146 | 0.552 |
| Local-only client 3 (3,318 rec.) | 42 | 8 | 0.873 | 0.666 | 0.167 | 0.489 |
| Local-only client 4 (1,003 rec.) | 42 | 10 | 0.858 | 0.644 | 0.167 | 0.492 |

Mean ± SD across seeds:

| Arm | Macro-AUROC | Macro-F1 |
|---|---|---|
| Centralized | 0.910 ± 0.004 | 0.713 ± 0.008 |
| FedAvg-20 | 0.910 ± 0.001 | 0.714 ± 0.005 |

Seed-42 centralized (the primary model in Sections 3–9) was the weakest of
the three centralized seeds. Seed-to-seed variation (up to 0.007 AUROC) is
larger than several of the between-model differences in Section 3.

Per-class test AUROC (seed 42):

| Model | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| Centralized | 0.933 | 0.921 | 0.932 | 0.920 | 0.825 |
| FedAvg-20 | 0.937 | 0.924 | 0.933 | 0.926 | 0.832 |
| Local client 1 | 0.930 | 0.917 | 0.927 | 0.915 | 0.790 |
| Local client 2 | 0.922 | 0.890 | 0.922 | 0.900 | 0.812 |
| Local client 3 | 0.913 | 0.876 | 0.915 | 0.904 | 0.758 |
| Local client 4 | 0.900 | 0.865 | 0.899 | 0.866 | 0.762 |

Per-class test F1 (seed 42):

| Model | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| Centralized | 0.838 | 0.735 | 0.741 | 0.751 | 0.457 |
| FedAvg-20 | 0.852 | 0.733 | 0.752 | 0.754 | 0.472 |
| Local client 1 | 0.849 | 0.734 | 0.739 | 0.724 | 0.419 |
| Local client 2 | 0.822 | 0.700 | 0.731 | 0.717 | 0.456 |
| Local client 3 | 0.824 | 0.674 | 0.718 | 0.732 | 0.383 |
| Local client 4 | 0.800 | 0.670 | 0.706 | 0.665 | 0.381 |

### 10.3 Pre-specified comparisons

Paired patient-level bootstrap with 2,000 resamples. Results below are from the
authoritative corrected analysis (`results/p2/analysis_corrected/comparisons.csv`);
the original files in `results/p2/analysis/` containing invalid Holm adjustments
on descriptive sign proportions are retained as historical artifacts (see
`results/p2/PROTOCOL.md` §7 deviation 5).

Primary uncertainty estimates are unadjusted paired 95% percentile confidence
intervals [ci_low, ci_high] with `ci_excludes_zero` indicating whether an interval
excludes zero. `bootstrap_sign_prob` (and legacy column alias `p_value`) reports
empirical replicate sign consistency as a descriptive statistic, not a formal
null-hypothesis test p-value. In accordance with `docs/DATASET_EVALUATION_AUDIT.md` §6,
Holm adjustment is omitted from sign proportions. Crucially, unadjusted paired
confidence intervals do not provide family-wise error rate (FWER) control across
multiple comparisons, and must not be interpreted as family-wise controlled
hypothesis tests.

| Comparison | Δ macro-AUROC [95% CI] | Sign prob | Δ macro-F1 [95% CI] | Sign prob |
|---|---|---|---|---|
| C1: FedAvg-20 − centralized | +0.0041 [−0.0001, 0.0084] | 0.055 | +0.0079 [−0.0026, 0.0187] | 0.145 |
| C2: FedAvg-20 − local client 1 | +0.0145 [0.0104, 0.0190] | 0.0005 | +0.0193 [0.0086, 0.0301] | 0.0010 |
| C2: FedAvg-20 − local client 2 | +0.0208 [0.0157, 0.0258] | 0.0005 | +0.0273 [0.0152, 0.0393] | 0.0005 |
| C2: FedAvg-20 − local client 3 | +0.0370 [0.0305, 0.0433] | 0.0005 | +0.0464 [0.0336, 0.0588] | 0.0005 |
| C2: FedAvg-20 − local client 4 | +0.0518 [0.0428, 0.0600] | 0.0005 | +0.0681 [0.0523, 0.0833] | 0.0005 |
| S1 (supplementary): FedAvg-20 − centralized, 3 seeds each | −0.0005 [−0.0031, 0.0021] | 0.698 | +0.0008 [−0.0061, 0.0076] | 0.856 |

The empirical sign-proportion floor is 1/2,000 = 0.0005.

Descriptive: each local model on its own site's test subset, compared with
FedAvg-20 and centralized on the same subset (mean of 3 seeds for FedAvg-20
and centralized):

| Client (site) | Local-only | FedAvg-20 | Centralized |
|---|---|---|---|
| 1 (site 0, n = 483) | 0.925 | 0.928 | 0.925 |
| 2 (site 1, n = 597) | 0.903 | 0.918 | 0.918 |
| 3 (site 2, n = 867) | 0.859 | 0.883 | 0.881 |
| 4 (other sites, n = 210) | 0.874 | 0.919 | 0.927 |

**Supported conclusions**
- Unadjusted paired 95% bootstrap confidence intervals for FedAvg-20 minus each
  local-only model strictly exclude zero for both macro-AUROC and macro-F1
  (descriptive sign prob <= 0.001), indicating consistent observed test-fold
  performance advantages when federating across all clients.
- The observed gain grows as the client gets smaller, from +0.015 AUROC for the
  largest client to +0.052 for the smallest.
- On each client's own-site test data, FedAvg-20 is at least as good as the
  local model (descriptive, single local seed).
- No detectable difference between FedAvg-20 and centralized training:
  seed 42 95% CI spans zero [−0.0001, +0.0084] (sign prob 0.055);
  three seeds Δ = −0.0005 [−0.0031, +0.0021] (sign prob 0.698).

**Not supported**
- Formal family-wise statistical significance across the family of C2
  comparisons, as unadjusted confidence intervals do not control family-wise
  error rate and no valid multiple-testing procedure was established for sign
  proportions.
- That FedAvg is better than centralized training.
- Any privacy guarantee.
- Local-only seed variability, since only one seed was run per client.

## 11. Limitations

1. Local-only baselines have one seed per client. Centralized and FedAvg-20
   have three seeds each. The baseline CNN and FedAvg-3/8 have one.
2. The baseline CNN (best = last epoch) may not have converged. FedAvg-8 had
   not converged (see 10.1).
3. Local-only and FedAvg models select checkpoints and thresholds on the
   central validation fold, which a real isolated site would not have.
4. The test fold was used for several models and analyses. All results are
   reported; none were selected by test performance.
5. Per-record normalisation discards absolute amplitude. An alternative
   (training-set per-lead normalisation, `src/prepare_global_norm.py`) was
   prepared but not trained or evaluated.
6. Non-standard label policy (Section 1).
7. Hyper-parameters of the baseline and federated runs other than those
   recorded in history and summary files are assumed to be script defaults.
