# Investigation of Model Performance: HYP and MI False Negatives

**Repository:** CardioFed  
**Model Under Investigation:** Primary Centralized Multi-Scale Residual CNN  
**Checkpoint:** `results/scheduler_experiment/checkpoints/multiscale_best.pt`  
**Evaluation Set:** PTB-XL Fold-10 Test Split (2,157 records, 1,876 patients)  
**Threshold Calibration:** Fold-9 Validation Split (per-class $F_1$ maximization)  
**Supporting Artifact:** `results/revised/error_analysis/hyp_mi_investigation.json`  

---

## 1. Overview and Evaluation Metrics

This report documents an empirical error analysis of the primary multi-scale model on the fold-10 test set. All metrics are computed using stored test probabilities and metadata, preserving the patient-level split. In accordance with project constraints, no models were retrained and thresholds were not adjusted on the test set.

### 1.1 Test Split Class Performance Summary

| Class | Support | Prevalence | TP | FP | FN | TN | Precision | Recall | $F_1$ Score | Validation-Tuned Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| **NORM** | 916 | 42.47% | 817 | 217 | 99 | 1,024 | 0.7901 | 0.8919 | **0.8379** | 0.5239 |
| **MI** | 550 | 25.50% | 431 | 191 | 119 | 1,416 | 0.6929 | 0.7836 | **0.7355** | 0.5296 |
| **STTC** | 506 | 23.46% | 397 | 168 | 109 | 1,483 | 0.7027 | 0.7846 | **0.7414** | 0.6722 |
| **CD** | 496 | 22.99% | 372 | 123 | 124 | 1,538 | 0.7515 | 0.7500 | **0.7508** | 0.5753 |
| **HYP** | 262 | 12.15% | 136 | 197 | 126 | 1,698 | 0.4084 | 0.5191 | **0.4571** | 0.7273 |

---

## 2. HYP (Hypertrophy) Performance Analysis

HYP has the lowest $F_1$ score ($0.4571$) among all evaluated classes. Analysis of the test set reveals several contributing factors:

### 2.1 Class Imbalance and False-Positive Distribution
- **Prevalence:** HYP is the least prevalent superclass in PTB-XL (12.15% in fold 10; 12.39% across the full dataset).
- **FP / TP Asymmetry:** False positives exceed true positives (197 FP vs 136 TP), resulting in low precision ($0.4084$).
- **Sources of HYP False Positives ($n = 197$):**
  - **By exact true label set:**
    - `STTC`: 49 records (24.87%)
    - `CD`: 38 records (19.29%)
    - `CD, MI`: 28 records (14.21%)
    - `MI`: 25 records (12.69%)
    - `NORM`: 22 records (11.17%)
    - `MI, STTC`: 16 records (8.12%)
    - `CD, STTC`: 12 records (6.09%)
    - `CD, MI, STTC`: 7 records (3.55%)
  - **By superclass presence:** 175 of the 197 false positives (88.83%) have concurrent true cardiac abnormalities:
    - Conduction Defects (`CD`): present in 85 records (43.15%)
    - ST/T Changes (`STTC`): present in 84 records (42.64%)
    - Myocardial Infarction (`MI`): present in 76 records (38.58%)
    - Normal (`NORM`): present in 22 records (11.17%)
  Repolarization abnormalities and QRS widening present in STTC, CD, and MI frequently produce secondary ECG morphological changes that overlap with hypertrophy patterns.

### 2.2 Preprocessing: Amplitude Normalisation as a Plausible Factor
- **Observation:** The preprocessing pipeline standardises each lead per record to zero mean and unit variance: $\tilde{x} = \frac{x - \mu}{\sigma + 10^{-8}}$.
- **Hypothesis:** Established clinical diagnostic criteria for Left Ventricular Hypertrophy (e.g. Sokolow-Lyon: $S_{V1} + R_{V5/V6} \ge 3.5\text{ mV}$; Cornell voltage: $R_{aVL} + S_{V3} > 2.8\text{ mV}$) depend on absolute voltage amplitudes. Because per-record z-scoring normalises every lead to variance 1.0, absolute voltage information is eliminated. The model must rely on relative wave shapes and secondary repolarization signs.
- **Evidence Status:** This represents a plausible contributing factor. Confirming its exact effect would require comparative training with amplitude-preserving normalisation (such as the training-fold statistics implemented in `src/prepare_global_norm.py`).

### 2.3 Diagnostic Sub-Statement Breakdown
PTB-XL assigns diagnostic statements within the HYP superclass. In the test fold:

| Statement | Clinical Meaning | Total in Test | Missed (FN) | Detected (TP) | FN Rate |
|---|---|---|---|---|---|
| **LVH** | Left Ventricular Hypertrophy | 214 | 91 | 123 | **42.52%** |
| **LAO/LAE** | Left Atrial Overload / Enlargement | 42 | 27 | 15 | **64.29%** |
| **RVH** | Right Ventricular Hypertrophy | 12 | 4 | 8 | **33.33%** |
| **RAO/RAE** | Right Atrial Overload / Enlargement | 10 | 9 | 1 | **90.00%** |
| **SEHYP** | Septal Hypertrophy | 2 | 1 | 1 | **50.00%** |

*Note: Individual records may contain more than one sub-statement.*

- Atrial enlargement statements (`LAO/LAE`, `RAO/RAE`) have higher false-negative rates ($64.29\%$ and $90.00\%$) than ventricular hypertrophy (`LVH`, $42.52\%$). Atrial P-wave alterations are subtle at 100 Hz sampling relative to the larger QRS complex.

### 2.4 Threshold Selection Trade-Off
- The validation-tuned threshold for HYP is **0.7273** (compared to $0.5239$–$0.6722$ for other classes).
- Because 197 false positives occur even at $0.7273$, lowering the threshold on validation data would have sharply increased false positives from STTC and CD, reducing $F_1$. Consequently, the threshold optimization favoured precision over recall, leaving test recall at $51.91\%$ (126 missed out of 262).

### 2.5 Age and Sex Distributions
In PTB-XL metadata, `sex` is coded as 0 = Male and 1 = Female. Patients aged >89 years are assigned the sentinel value `age = 300` for anonymization (accounting for 32 test records):

| Subgroup | Records | Female (%) | Male (%) | Median Age (yr) | Valid Mean Age (age $\le 100$, yr) | Sentinel Age 300 Count (%) |
|---|---|---|---|---|---|---|
| **HYP TP** | 136 | 52.94% (72) | 47.06% (64) | 74.0 | 72.33 ± 10.38 ($n=132$) | 4 (2.94%) |
| **HYP FN** | 126 | 43.65% (55) | 56.35% (71) | 66.0 | 63.61 ± 15.33 ($n=125$) | 1 (0.79%) |
| **HYP FP** | 197 | 47.21% (93) | 52.79% (104) | 71.0 | 68.76 ± 14.33 ($n=189$) | 8 (4.06%) |

- Missed HYP cases (FN) have a lower median age (66.0 vs 74.0 years; valid mean $63.61$ vs $72.33$ years) and a higher male proportion (56.35% vs 47.06%) compared to detected cases (TP).

---

## 3. MI (Myocardial Infarction) False-Negative Analysis

The model detects 431 MI records ($78.36\%$ recall) and misses **119 MI records** ($21.64\%$ FN rate).

### 3.1 Anatomical Location and Sub-Statement Breakdown

| Statement | Infarct Location / Type | Total in Test | Missed (FN) | Detected (TP) | FN Rate | Share of All Missed MIs ($n=119$) |
|---|---|---|---|---|---|---|
| **ASMI** | Anteroseptal MI | 234 | 15 | 219 | **6.41%** | 12.61% |
| **ILMI** | Inferolateral MI | 48 | 4 | 44 | **8.33%** | 3.36% |
| **INJAS** | Subendocardial Anteroseptal Injury | 22 | 2 | 20 | **9.09%** | 1.68% |
| **IMI** | Inferior MI | 267 | 65 | 202 | **24.34%** | **54.62%** |
| **ALMI** | Anterolateral MI | 27 | 8 | 19 | **29.63%** | 6.72% |
| **AMI** | Anterior MI | 35 | 14 | 21 | **40.00%** | 11.76% |
| **LMI** | Lateral MI | 20 | 11 | 9 | **55.00%** | 9.24% |
| **IPMI** | Inferoposterior MI | 3 | 1 | 2 | **33.33%** | 0.84% |
| **PMI** | Posterior MI | 2 | 1 | 1 | **50.00%** | 0.84% |
| **IPLMI** | Inferoposterolateral MI | 5 | 1 | 4 | **20.00%** | 0.84% |
| **INJAL / INJIN / INJLA** | Other injury statements | 18 | 0 | 18 | **0.00%** | 0.00% |

#### Key Observations:
1. **Low error in Anteroseptal MI (`ASMI`):** $6.41\%$ FN rate. Precordial leads (V1–V4) show prominent QRS changes that are well captured.
2. **High concentration in Inferior MI (`IMI`):** $24.34\%$ FN rate. Inferior MIs represent **65 of the 119 false negatives (54.62%)**.
3. **Elevated error in Lateral MI (`LMI`):** $55.00\%$ FN rate (11 of 20 missed). Localized limb and high-lateral changes (leads I, aVL) are less consistently detected.

### 3.2 Predictions Assigned to Missed MI Cases ($n = 119$)

#### Raw Threshold Predictions:
- `NORM`: 54 records (45.38%) — pathology missed entirely.
- `STTC, HYP`: 16 records (13.45%)
- `STTC`: 16 records (13.45%)
- `CD`: 12 records (10.08%)
- `NORM, CD`: 5 records (4.20%) — raw threshold conflict.
- `None`: 5 records (4.20%) — no class met its decision threshold.
- `STTC, CD`: 5 records (4.20%)
- `STTC, CD, HYP`: 3 records (2.52%)
- `HYP`: 2 records (1.68%)
- `CD, HYP`: 1 record (0.84%)

#### Final Predictions (After Post-Hoc NORM Consistency Rule):
Under the project's rule (suppressing NORM when an abnormal class is predicted positive), the 5 `NORM, CD` conflicts become `CD`:
- `NORM`: 54 records (45.38%)
- `CD`: 17 records (14.29%)
- `STTC, HYP`: 16 records (13.45%)
- `STTC`: 16 records (13.45%)
- `None` (unclassified): 5 records (4.20%)
- `STTC, CD`: 5 records (4.20%)
- `STTC, CD, HYP`: 3 records (2.52%)
- `HYP`: 2 records (1.68%)
- `CD, HYP`: 1 record (0.84%)

In total, 60 of the 119 missed MI cases (50.42%) were flagged as having another abnormality (CD, STTC, or HYP), 54 (45.38%) were classified as NORM, and 5 (4.20%) had no class predicted.

### 3.3 Clinical Presentation and Demographics

| Subgroup | Records | Female (%) | Male (%) | Median Age (yr) | Valid Mean Age (age $\le 100$, yr) | Sentinel Age 300 Count (%) |
|---|---|---|---|---|---|---|
| **MI TP** | 431 | 43.85% (189) | 56.15% (242) | 72.0 | 70.77 ± 12.20 ($n=417$) | 14 (3.25%) |
| **MI FN** | 119 | 53.78% (64) | 46.22% (55) | 67.0 | 66.15 ± 13.07 ($n=118$) | 1 (0.84%) |

- Missed MI cases (FN) have a lower median age (67.0 vs 72.0 years; valid mean $66.15$ vs $70.77$ years) and a higher female proportion (53.78% vs 43.85%) compared to detected MI cases (TP).
- **Sole MI vs Multi-Class MI:**
  - Among missed MIs (FN), 68 of 119 (**57.14%**) are sole MI records (no concurrent true CD, STTC, or HYP).
  - Among detected MIs (TP), 191 of 431 (**44.32%**) are sole MI records.
  When MI presents without co-occurring conduction defects or ST/T abnormalities, the model misses it more frequently.

---

## 4. Site-Level Observations (Fold-10 Test Split)

| Site Group | Test Records | NORM Prevalence | MI Support | MI Missed (FN) | MI FN Rate | HYP Support | HYP Missed (FN) | HYP FN Rate |
|---|---|---|---|---|---|---|---|---|
| **Site 0** | 483 | **64.18%** (310) | 78 | 30 | **38.46%** | 34 | 19 | **55.88%** |
| **Site 1** | 597 | **44.39%** (265) | 124 | 21 | **16.94%** | 95 | 51 | **53.68%** |
| **Site 2** | 867 | **32.64%** (283) | 283 | 57 | **20.14%** | 83 | 46 | **55.42%** |
| **Other sites** | 210 | **27.62%** (58) | 65 | 11 | **16.92%** | 50 | 10 | **20.00%** |

- **Site 0:** Characterised by high NORM prevalence (64.18% in the test split) and an elevated MI false-negative rate (38.46%, 30/78).
- **Confounding Caution:** In PTB-XL, recording sites, patient populations, and diagnostic prevalence are confounded. As noted in `RESULTS.md` §5, differences across site groups cannot be attributed strictly to device or recording variations.

---

## 5. Summary of Investigated Contributing Factors

| Factor | Evidence Status | Impact on HYP | Impact on MI |
|---|---|---|---|
| **Class Imbalance** | Empirically verified | Severe (12.15% prevalence); contributes to low precision (0.4084). | Moderate (25.50% prevalence); second-largest class. |
| **Signal Normalisation** | Plausible hypothesis | Per-lead z-scoring removes absolute amplitude used in clinical LVH criteria. | Minor; bandpass filtering (0.5–40 Hz) preserves pathological Q waves. |
| **Spatial / Lead Distribution** | Empirically verified | Morphological changes distributed across precordial/limb leads. | Pronounced: ASMI (V1–V4) FN rate is 6.41%; IMI (II, III, aVF) FN rate is 24.34%; LMI (I, aVL) FN rate is 55.00%. |
| **Sub-statement Composition** | Empirically verified | Atrial enlargement (`LAO/LAE`, `RAO/RAE`) missed in 64–90% of cases. | Inferior and lateral infarctions account for 63.86% of all missed MIs. |
| **Threshold Tuning** | Empirically verified | Validation-tuned threshold of 0.7273 limits false positives at the expense of recall (51.91%). | Validation-tuned threshold of 0.5296 provides balanced recall (78.36%) and precision (69.29%). |
| **Demographics** | Empirically verified | Younger patients overrepresented among false negatives (median age 66.0 vs 74.0). | Younger (median age 67.0 vs 72.0) and female (53.78% vs 43.85%) cohorts overrepresented in false negatives. |

---

## 6. Recommendations and Next Steps

The following evidence-backed recommendations are identified for future phases:

1. **Evaluate Amplitude-Preserving Normalisation:**  
   Assess whether training-set per-lead standardisation (e.g. `src/prepare_global_norm.py`) improves HYP detection relative to per-record z-scoring by preserving QRS amplitude variations.
2. **Investigate Regional Lead-Group Weighting:**  
   Given the high error rate in inferior (`IMI`, 24.34%) and lateral (`LMI`, 55.00%) infarctions relative to anteroseptal (`ASMI`, 6.41%), investigate whether anatomical lead groupings (inferior leads II/III/aVF vs precordial leads) can improve sensitivity to non-anterior infarcts.
3. **Consider Separate Modeling for Atrial Enlargement:**  
   Atrial overload statements have 64%–90% FN rates. Separating atrial from ventricular hypertrophy or providing targeted P-wave representations could reduce dilution of the HYP superclass.
4. **Maintain Protocol Compliance:**  
   Thresholds must not be tuned on the test set. Future adjustments should rely on cross-validation or pre-specified optimization criteria.

---

## 7. Limitations

- **Research Scope:** CardioFed is a machine-learning research prototype evaluated on PTB-XL, not a clinical diagnostic tool.
- **Confounded Demographics:** Patient age, sex, recording site, and disease severity co-vary in observational data; observed associations do not imply causation.
- **Label Uncertainty:** PTB-XL labels derive from automated median-beat analysis with variable human review.
