# CardioFed Dataset, Label, and Evaluation Audit Report

**Repository:** CardioFed  
**Audit Scope:** PTB-XL 1.0.3 Data Pipeline, Label Construction, Patient-Level Splits, Leakage Prevention, and Evaluation Methodology  
**Audit Date:** 2026-11-14  
**Auditor:** Antigravity AI  
**Test Suite Reference:** `tests/test_dataset_audit.py` (10 passed)  

---

## 1. Executive Summary

This audit evaluates the integrity, reproducibility, and methodological rigor of the CardioFed data pipeline, label derivation, split separation, evaluation metrics, and statistical inference procedures. All findings are traceable to repository code, raw metadata (`ptbxl_database.csv`, `scp_statements.csv`), processed metadata (`processed_metadata.csv`), and stored research artifacts.

### Audit Summary Matrix

| Audit Dimension | Status | Key Verified Finding |
|---|---|---|
| **1. Label Integrity** | **VERIFIED** | 5 diagnostic classes; 424 non-diagnostic/likelihood $\le 0$ records removed; exactly 21,375 records retained; 417 records modified by NORM-exclusivity; 0 NORM-abnormal conflicts in processed data. |
| **2. Data Integrity** | **VERIFIED** | Exact 1-to-1 alignment between raw and processed metadata; 0 duplicated ECG IDs; 0 fold mismatches; clean tracking of site metadata across PTB-XL recording locations. |
| **3. Data Leakage** | **VERIFIED** | Strict patient-level stratification with **0 patient overlap** across train (folds 1–8), validation (fold 9), and test (fold 10); per-record z-scoring operates purely intra-record; class weights and thresholds isolated to train and validation sets respectively. |
| **4. Evaluation Correctness** | **VERIFIED** | Standard definitions of macro AUROC, macro $F_1$, precision, recall, Hamming loss, and subset accuracy verified; post-hoc NORM consistency rule correctly isolated to deployment/validation. |
| **5. Statistical Analysis** | **VERIFIED** | Patient-level cluster bootstrap properly groups correlated records per patient; paired difference bootstrap and Holm-Bonferroni step-down correction are mathematically sound. |
| **6. Documentation** | **VERIFIED** | All reported counts, class supports, split dimensions, and benchmark metrics in `RESULTS.md` match underlying pipeline and data exactly. |

---

## 2. Label Integrity Audit

### 2.1 Diagnostic Superclass Mapping
CardioFed models 5 clinical superclasses: `NORM` (Normal ECG), `MI` (Myocardial Infarction), `STTC` (ST/T Changes), `CD` (Conduction Disturbance), and `HYP` (Hypertrophy).
- Mapping is extracted from `data/raw/ptb-xl/scp_statements.csv`, filtering statements where column `diagnostic == 1` and `diagnostic_class` belongs to the 5 target superclasses.
- Non-diagnostic statements (e.g., rhythm codes `SR`, `AFIB`, or form codes) are excluded from the superclass multi-label target.

### 2.2 Likelihood Filtering and Empty Target Removal
PTB-XL SCP statements contain clinician-assigned likelihood scores (0.0 to 100.0).
- The CardioFed pipeline filters SCP codes requiring `likelihood > 0`.
- Out of 21,799 raw records:
  - 411 records contain no diagnostic SCP codes whatsoever (e.g., purely rhythm or pacemaker statements).
  - 13 records contain only diagnostic statements with `likelihood == 0.0`.
  - Exactly 424 records are removed, leaving **21,375 usable records** in both Stage 1 and Stage 2 of the pipeline.

### 2.3 The CardioFed NORM Mutual Exclusivity Policy
In standard PTB-XL benchmarks, `NORM` is treated as an independent binary class, allowing records to be simultaneously labeled as `NORM` and an abnormal condition (e.g., `NORM + CD` in incomplete bundle branch blocks or non-specific conduction delays). CardioFed enforces a clinical policy: **NORM is mutually exclusive with abnormal conditions**.

```python
# From rebuild_labels.py
abnormal = labels - {"NORM"}
if abnormal:
    labels = abnormal
```

#### Quantitative Effect of NORM Exclusivity:
- **Stage 1 (Pre-Exclusivity):** 9,514 NORM records, with **417 records** having concurrent abnormal diagnoses:
  - `NORM + CD`: 410 records
  - `NORM + STTC + CD`: 2 records
  - `NORM + CD + HYP`: 2 records
  - `NORM + HYP`: 2 records
  - `NORM + MI + CD + HYP`: 1 record
- **Stage 2 (CardioFed Final):** NORM is suppressed in all 417 conflicting records.
  - `NORM` count decreases from 9,514 to **9,097** (difference = 417).
  - All abnormal superclass counts remain **100% identical** (MI: 5,469; STTC: 5,108; CD: 4,898; HYP: 2,649).
  - NORM + abnormal overlap in processed targets is **exactly 0**.
  - No records are left empty by this step (every record retained has at least one positive label).

> [!IMPORTANT]
> Because 417 records have NORM excluded, CardioFed evaluation metrics cannot and should not be compared directly to published PTB-XL benchmarks that do not enforce NORM exclusivity.

---

## 3. Data Integrity & Alignment

### 3.1 Metadata and Signal Alignment
The audit verified the alignment between:
1. `data/raw/ptb-xl/ptbxl_database.csv` (21,799 raw records)
2. `data/processed/processed_metadata.csv` (21,375 processed records)
3. Signal file paths (`filename_lr`) and fold assignments (`strat_fold`)

| Metric / Check | Value | Verification Result |
|---|---|---|
| Total processed rows | 21,375 | Exact match |
| Unique `ecg_id` in processed metadata | 21,375 | 0 duplicates |
| Missing `ecg_id` from raw metadata | 0 | 100% alignment |
| `strat_fold` mismatches between raw and processed | 0 | 100% match |
| Records outside valid folds (1–10) | 0 | 100% valid |

### 3.2 Recording Site Distribution
PTB-XL records contain recording device/site metadata (`site`).
The 21,375 records originate from:
- Site 0: 4,895 records (22.90%)
- Site 1: 4,792 records (22.42%)
- Site 2: 4,451 records (20.82%)
- Other / missing sites: 7,237 records (33.86%)
Site assignments are tracked consistently in `sites.npy` and used in `src/federated_train.py` to assign records to clients by each patient's dominant site.

---

## 4. Data Leakage & Patient Isolation Audit

A critical flaw in medical machine learning is data leakage—either by splitting records from the same patient across train/val/test splits, or by leaking test-set statistics into normalization, class weights, or threshold selection.

### 4.1 Zero Patient Overlap Across Splits
PTB-XL stratified folds were designed by PhysioNet to keep all records of each patient within a single fold. We audited patient IDs (`patient_id`) across splits:

| Split | Strat Folds | ECG Records | Unique Patients | Overlap with Other Splits |
|---|---|---|---|---|
| **Train** | 1–8 | 17,073 | 14,818 | **0** |
| **Validation** | 9 | 2,145 | 1,916 | **0** |
| **Test** | 10 | 2,157 | 1,876 | **0** |
| **Total** | 1–10 | 21,375 | 18,610 | **0** |

- Pairwise patient overlap between Train and Validation: **0**
- Pairwise patient overlap between Train and Test: **0**
- Pairwise patient overlap between Validation and Test: **0**

Patient isolation is strictly maintained across all splits.

### 4.2 Signal Preprocessing Isolation
In `src/preprocessing.py`, signal processing consists of:
1. Zero-phase Butterworth band-pass filter (0.5–40 Hz, order 3).
2. Per-lead z-score normalisation:
   $$\tilde{x}_l = \frac{x_l - \mu_l}{\sigma_l + 10^{-8}}$$
Because $\mu_l$ and $\sigma_l$ are computed **strictly within each individual 10-second 12-lead record**, there is no leakage of statistics across records or between splits.

For alternative experiments requiring global voltage preservation (`src/prepare_global_norm.py`), per-lead means and standard deviations were computed **strictly on training folds 1–8** and then applied downstream to validation and test folds without test leakage.

### 4.3 Training Weights and Threshold Calibration Isolation
- **Positive Class Weights:** Computed in `src/train_multiscale.py` and `src/train.py` via `calculate_pos_weights(y_train)` using **only `y_train` (folds 1–8)**.
- **Threshold Tuning:** Decision thresholds are tuned in `src/thresholds.py::tune_thresholds` by maximizing $F_1$ per class on **validation fold 9 only**.
- **Model Checkpoint Selection:** Checkpoints are selected based on validation fold 9 macro-AUROC (`best_val_auroc`), preserving test fold 10 strictly for final unguided evaluation.

---

## 5. Evaluation Correctness Audit

### 5.1 Metric Formulations
We inspected `src/evaluate.py::compute_metrics` against standard definitions:
- **AUROC:** Computed per-class via `sklearn.metrics.roc_auc_score(y[:, i], probs[:, i])`. Macro-AUROC is the arithmetic mean across valid classes.
- **$F_1$, Precision, Recall:** Computed per-class via `sklearn.metrics.f1_score`, `precision_score`, `recall_score` with `zero_division=0`. Macro-$F_1$ is the arithmetic mean of per-class $F_1$ scores.
- **Hamming Loss:** Fraction of incorrect binary label predictions:
  $$\text{Hamming Loss} = \frac{1}{N \cdot C} \sum_{n=1}^N \sum_{c=1}^C \mathbf{1}(\hat{y}_{n,c} \ne y_{n,c})$$
- **Subset Accuracy:** Proportion of records where all 5 labels match ground truth exactly:
  $$\text{Subset Accuracy} = \frac{1}{N} \sum_{n=1}^N \mathbf{1}(\hat{\mathbf{y}}_n = \mathbf{y}_n)$$

### 5.2 Threshold Application and Post-Hoc NORM Consistency Rule
- **Threshold Application:** Decisions are obtained by direct thresholding:
  $$\hat{y}_{n, c} = \mathbf{1}(p_{n, c} \ge \tau_c)$$
  where $\tau_c$ is fixed from validation fold 9.
- **Post-Hoc NORM Consistency Rule:**
  In `src/app.py`, `src/investigate_errors.py`, and `src/validate_prediction_rules.py`, a post-hoc clinical rule is available:
  $$\text{If } \exists c \in \{\text{MI}, \text{STTC}, \text{CD}, \text{HYP}\} \text{ with } \hat{y}_{n, c} = 1, \quad \text{then force } \hat{y}_{n, \text{NORM}} = 0$$
- **Benchmark Integrity:** In all official benchmark evaluations (`src/evaluate_checkpoint.py`, `RESULTS.md` §3), test metrics are computed using raw validation-tuned thresholds **without** applying the post-hoc NORM consistency rule. In the Streamlit app, both raw thresholded predictions and the post-hoc NORM rule predictions are presented side-by-side with clear explanation.

---

## 6. Statistical Analysis Audit

### 6.1 Patient-Level Bootstrap Resampling
In `src/statistical_analysis.py`, confidence intervals are estimated using a non-parametric bootstrap (2,000 resamples, seed 0).
- **Cluster Resampling:** Because 1,876 unique patients generate the 2,157 test records, records from the same patient are not independent. Resampling records independently would underestimate uncertainty.
- `patient_bootstrap_indices` groups records by `patient_id` and samples unique patient IDs with replacement. All records belonging to a sampled patient are included.
- Verified in `tests/test_dataset_audit.py::test_patient_bootstrap_clustering`.

### 6.2 Paired Model Comparisons & Inference Procedure
To compare models $A$ and $B$:
- Predictions for both models are evaluated on identical unconditional bootstrap samples of patients:
  $$d_b = \text{Metric}(A, \text{sample}_b) - \text{Metric}(B, \text{sample}_b)$$
- **Primary Inferential Output (95% Percentile Confidence Interval):**
  Paired 95% percentile bootstrap confidence intervals $[d_{2.5\%}, d_{97.5\%}]$ are the primary inferential output. Intervals excluding zero (`ci_excludes_zero == True`) are the project's stated inferential criterion for detecting differences, not a substitute for a separately justified multiple-comparison procedure.
- **Descriptive Sign-Proportion Estimate (`bootstrap_sign_prob`):**
  The pipeline computes an empirical sign-consistency statistic:
  $$p_{\text{sign}} = 2 \cdot \min\left(\frac{1}{B_{\text{valid}}} \sum_{b=1}^{B_{\text{valid}}} \mathbf{1}(d_b \le 0), \frac{1}{B_{\text{valid}}} \sum_{b=1}^{B_{\text{valid}}} \mathbf{1}(d_b \ge 0)\right)$$
  floored at $\frac{1}{B_{\text{valid}}}$.
  *Methodological Distinction:* In `src/statistical_analysis.py`, this quantity is designated as `bootstrap_sign_prob` (with `p_value` retained as a legacy column alias for downstream compatibility). It is a descriptive bootstrap sign-proportion estimate measuring the consistency of metric difference direction across resamples, rather than a formal hypothesis-test p-value derived under an exchangeable null hypothesis $H_0: \theta_A = \theta_B$ (such as from a permutation test or pivotal bootstrap).
- **Removal of Holm Correction and Formal Significance Flags:**
  Because a descriptive bootstrap sign-proportion is not a formal null-hypothesis p-value, Holm correction cannot convert it into one. Consequently, `p_value_holm` and `significant_at_0.05_holm` have been removed from the paired comparison pipeline and CSV outputs. Formal significance claims must not be asserted from this statistic; comparative claims rest strictly on the paired 95% confidence intervals.
- **Safe Handling of Zero Valid Replicates:**
  If a paired comparison has zero valid replicates ($B_{\text{valid}} = 0$, e.g., on a small sub-cohort where a diagnostic class was completely unobserved), `ci_low`, `ci_high`, and `bootstrap_sign_prob` evaluate safely to `NaN`, and `ci_excludes_zero` evaluates to `False`.

### 6.3 Multiple-Comparison Correction Utility (Holm-Bonferroni)
`src/statistical_analysis.py::holm` implements the Holm-Bonferroni step-down procedure with safe handling of undefined (`NaN`) values:
1. Filter the input vector for finite values: $m = \sum \mathbf{1}(\text{isfinite}(p_i))$.
   - Undefined comparisons with zero valid replicates ($p = \text{NaN}$) do not contaminate the ranking, do not inflate the family size $m$, and remain `NaN` in the adjusted output.
2. If $m > 0$, sort the $m$ valid unadjusted values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
3. Compute adjusted values:
   $$\tilde{p}_{(k)} = \min(1, (m - k + 1) \cdot p_{(k)})$$
4. Enforce step-down monotonicity:
   $$\tilde{p}_{(k)} = \max(\tilde{p}_{(k-1)}, \tilde{p}_{(k)})$$
5. Map adjusted values back to their original positions.
*Application Policy:* This step-down correction is retained as a general-purpose statistical utility, but is **deliberately not applied** to descriptive bootstrap sign proportions in model comparisons.

### 6.4 Assessment of Rejection Sampling vs. Unconditional Patient Bootstrap & Validity Reporting
When evaluating confidence intervals for multi-label classification using patient-level bootstrap:
- **Statistical Critique of Rejection-Based Bootstrap:**
  Conditioning bootstrap draws on having $\ge 1$ positive and $\ge 1$ negative example for every class (rejection sampling) truncates the empirical sampling distribution. For a low-prevalence class with true prevalence $p$, observing 0 cases in a sample has non-zero probability under sampling variability ($P(K=0) = (1-p)^N > 0$). Discarding samples with 0 positives artificially truncates the lower tail of class prevalence, shifts the effective conditional prevalence upward, and artificially compresses the confidence intervals for rare conditions.
- **Adopted Defensible Methodology:**
  1. **Unconditional Patient-Level Resampling:** `patient_bootstrap_indices` samples patient clusters uniformly with replacement without rejection, strictly preserving patient clustering and empirical sampling variance.
  2. **Consistent Complete-Class Definition for Macro-AUROC:**
     To prevent partial-class averages from distorting macro-AUROC comparisons between models (e.g. omitting HYP would artificially inflate macro-AUROC from ~0.906 to ~0.930), macro-AUROC is strictly defined as the mean over all 5 pre-specified target superclasses. If a bootstrap replicate misses any class, its 5-class macro-AUROC is recorded as undefined (`NaN`). Single-model macro-AUROC CIs and paired model macro-AUROC differences are computed only over consistent replicates where all target classes are evaluated.
  3. **Distinct Reporting of AUROC and F1 Replicate Validity:**
     Every output table and `bootstrap_summary.json` provides transparent, explicit validity accounting:
     - **Macro-AUROC Replicates (`macro_auroc_valid_samples`, `macro_auroc_excluded_samples`):** Explicitly tracks complete 5-class macro-AUROC replicates where all 5 classes are biclassed.
     - **Per-Class AUROC Replicates (`auroc_valid_samples`, `auroc_excluded_samples`):** Explicitly tracks replicates where that specific diagnostic class has both $\ge 1$ positive and $\ge 1$ negative example.
     - **F1 Replicates (`macro_f1_valid_samples`, `f1_valid_samples`):** Explicitly tracked separately. F1 is evaluable across all replicates because zero-division in precision/recall defaults to 0.0 under single-outcome distributions.
     - **Paired Differences Output Columns:** `paired_differences.csv` explicitly outputs:
       `model_a`, `model_b`, `question`, `metric`, `difference`, `ci_low`, `ci_high`, `ci_excludes_zero`, `bootstrap_sign_prob`, `p_value` (legacy alias), `valid_bootstrap_samples`, `excluded_bootstrap_samples`.
     On the full PTB-XL fold-10 test set ($N=2,157$ records, $1,876$ patients, minimum class support $n=262$), all 2,000 bootstrap replicates have all 5 classes present (`macro_auroc_valid_samples = 2000`, `macro_auroc_excluded_samples = 0`). For small subgroup cohorts, the reporting transparently exposes sample coverage without inducing rejection bias.
  4. **Regression Tests:** Verified in `tests/test_dataset_audit.py` (`test_macro_auc_returns_nan_on_incomplete_classes`, `test_bootstrap_transparent_exclusion_tracking`, `test_holm_handles_nan_pvalues_safely`, `test_zero_valid_replicate_paired_comparison_safety`, `test_sign_proportions_not_presented_as_formal_pvalues_or_hypotheses`, and `test_validity_counts_distinguish_auroc_and_f1`).

---

## 7. Discovered Nuances & Code Corrections

During the audit, three nuances were identified and corrected:

1. **`src/audit_labels.py` Label Reporting Nuance:**
   - *Previous state:* Loaded metadata via `load_metadata` (Stage 1) and printed `NORM + abnormal overlap: 417` and `NORM: 9514`. It did not display Stage 2 (NORM-exclusive) labels, which could lead an auditor to believe the exclusivity policy was missing.
   - *Correction:* Updated `src/audit_labels.py` to audit and report Stage 1 (pre-exclusivity), Stage 2 (CardioFed NORM-exclusive), and an explicit comparison demonstrating that exactly 417 records lost NORM with 0 empty records and 0 abnormal label modifications.

2. **`src/audit_processed_data.py` Git-LFS Pointer Handling:**
   - *Previous state:* Unconditionally called `np.load("data/processed/y.npy")`. In environments without Git-LFS smudging, this crashed with `ValueError: Cannot load file containing pickled data`.
   - *Correction:* Added `is_real_file()` check. If files are LFS pointers, it loads `processed_metadata.csv` (which is tracked as clean text) and reconstructs target labels using `load_metadata`, performing all shape, fold, patient-wise, and split audits cleanly.

3. **Bootstrap Single-Class Representation & Complete-Class Consistency:**
   - *Previous state:* Rejection sampling was initially considered to eliminate `NaN`s, but conditioned the bootstrap distribution and risked biasing confidence intervals for rare classes. Conversely, unhandled single-class samples caused `roc_auc_score` to return `NaN` and crash percentile calculations.
   - *Correction:* Implemented unconditional patient-level cluster bootstrap with complete-class consistency for macro-AUROC, guarded `macro_auc` against incomplete class definitions, and added transparent reporting of valid vs excluded replicates across overall, per-class, and paired comparison outputs.

4. **New Audit Test Suite (`tests/test_dataset_audit.py`):**
   - Added 12 automated unit and integration tests verifying label mapping, NORM mutual exclusivity, metadata alignment, split counts, patient isolation, intra-record signal normalization, metric computation, post-hoc NORM logic, patient bootstrap clustering, single-class incomplete handling, and Holm correction.

---

## 8. Remaining Limitations

1. **Clinician Diagnostic Heuristics in PTB-XL:** SCP code likelihood assignments reflect subjective clinician interpretations. Filtering at `likelihood > 0` removes non-positive statements, but clinical nuances in borderline cases remain in the raw data.
2. **Voltage Erasure from Per-Record Z-Scoring:** Standard intra-record z-score normalisation standardises each lead to variance 1.0, eliminating absolute millivolt voltages needed for classical amplitude criteria (e.g. Sokolow-Lyon criteria for HYP).
3. **Local Git-LFS Requirements:** Full binary signal evaluations require local execution of `git lfs pull` to download raw `.dat` and processed `.npy` arrays. All metadata, splits, and evaluation summaries remain fully verifiable from committed files.
4. **Subgroup Sample Size Constraints in Bootstrap:** In small patient cohorts (e.g., individual recording sites or demographic slices with $<50$ patients), low-prevalence classes will naturally be omitted in some bootstrap resamples. The audit's transparent exclusion reporting documents the exact sample coverage, but very small sub-cohorts should be interpreted alongside class prevalence and support counts.

