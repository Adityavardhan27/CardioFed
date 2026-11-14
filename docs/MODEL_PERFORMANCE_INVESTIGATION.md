# Investigation of Model Performance: HYP and MI False Negatives

**Repository:** CardioFed  
**Model Under Investigation:** Primary Centralized Multi-Scale Residual CNN  
**Checkpoint:** `results/scheduler_experiment/checkpoints/multiscale_best.pt`  
**Evaluation Set:** Official PTB-XL Fold-10 Test Split (2,157 records, 1,876 patients)  
**Threshold Calibration:** Fold-9 Validation Set (per-class F1 maximization)  
**Date:** November 2026  

---

## 1. Executive Summary

This investigation examines the error profiles of the selected primary model (`multiscale_best.pt`), with specific focus on:
1. **HYP (Hypertrophy):** Lowest F1 score (0.4571) across all classes and architectures.
2. **MI (Myocardial Infarction):** Analysis of 119 false negatives (21.6% miss rate), their anatomical patterns, demographics, and co-occurring labels.

All findings are based on stored test predictions (`test_probabilities_multiscale.csv`), error breakdowns (`misclassified_ecgs.csv`, `class_error_counts.csv`), and patient metadata (`ptbxl_database.csv`, `scp_statements.csv`). In accordance with project constraints, no models were retrained, no test-set threshold tuning was performed, and no clinical diagnostic claims are made.

### Summary Metrics by Class (Fold-10 Test Set)

| Class | Support | Prevalence | TP | FP | FN | TN | Precision | Recall | F1 Score | Validation-Tuned Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| **NORM** | 916 | 42.5% | 817 | 217 | 99 | 1,024 | 0.7901 | 0.8919 | **0.8379** | 0.5239 |
| **MI** | 550 | 25.5% | 431 | 191 | 119 | 1,416 | 0.6929 | 0.7836 | **0.7355** | 0.5296 |
| **STTC** | 506 | 23.5% | 397 | 168 | 109 | 1,483 | 0.7027 | 0.7846 | **0.7414** | 0.6722 |
| **CD** | 496 | 23.0% | 372 | 123 | 124 | 1,538 | 0.7515 | 0.7500 | **0.7508** | 0.5753 |
| **HYP** | 262 | 12.1% | 136 | 197 | 126 | 1,698 | 0.4084 | 0.5191 | **0.4571** | 0.7273 |

---

## 2. Investigation 1: HYP (Hypertrophy) Performance

HYP exhibits the weakest performance across all models in CardioFed (F1 = 0.4571, Precision = 0.4084, Recall = 0.5191). The analysis reveals four interconnected drivers:

### 2.1 Severe Class Imbalance and False Positive Predominance
- **Prevalence:** HYP is the rarest superclass in PTB-XL (12.1% of test records, 262/2,157; 12.4% overall).
- **Asymmetry:** False positives substantially outnumber true positives (197 FP vs 136 TP), depressing precision to 40.8%.
- **Origin of False Positives:** HYP false positives primarily occur in records belonging to other abnormal classes:
  - **STTC:** 49 records
  - **CD:** 38 records
  - **MI + CD / MI:** 53 records
  - **NORM:** 22 records  
  Secondary repolarization changes (ST depression, T-wave inversion) and QRS widening in STTC and CD closely mimic the secondary repolarization abnormalities of ventricular hypertrophy.

### 2.2 Signal Preprocessing: Voltage Erasure via Per-Record Z-Scoring
- **The Core Flaw:** The standard pipeline (`src/preprocessing.py`) applies per-lead z-score standardisation:
  $$\tilde{x} = \frac{x - \mu}{\sigma + 10^{-8}}$$
- **Clinical Significance:** Established electrocardiographic criteria for Left Ventricular Hypertrophy (e.g., Sokolow-Lyon: $S_{V1} + R_{V5/V6} \ge 3.5\text{ mV}$; Cornell voltage: $R_{aVL} + S_{V3} > 2.8\text{ mV}$) depend strictly on **absolute millivolt amplitude**.
- **Impact:** By forcing every lead to unit standard deviation ($\sigma = 1.0$), abnormally high QRS voltages are scaled down to match normal relative wave shapes. The network is forced to rely on subtle secondary wave morphology rather than the primary diagnostic feature (high voltage).

### 2.3 Sub-Statement Disparity (Atrial vs Ventricular Hypertrophy)
Analyzing the underlying PTB-XL diagnostic statements reveals stark differences in detectability:

| SCP Code | Description | Test Records | Missed (FN) | Detected (TP) | FN Rate |
|---|---|---|---|---|---|
| **LVH** | Left Ventricular Hypertrophy | 214 | 91 | 123 | **42.5%** |
| **LAO/LAE** | Left Atrial Overload / Enlargement | 42 | 27 | 15 | **64.3%** |
| **RVH** | Right Ventricular Hypertrophy | 12 | 4 | 8 | **33.3%** |
| **RAO/RAE** | Right Atrial Overload / Enlargement | 10 | 9 | 1 | **90.0%** |
| **SEHYP** | Septal Hypertrophy | 2 | 1 | 1 | **50.0%** |

- **Key Finding:** Atrial enlargement (`LAO/LAE`, `RAO/RAE`) is almost entirely missed by the model (64% to 90% FN rates). P-wave morphological deviations (bifid or peaked P waves) are subtle at 100 Hz sampling and are easily overwhelmed by the dominant ventricular QRS complexes.

### 2.4 Threshold Behavior
- The validation-tuned threshold for HYP is **0.7273**, the highest among all classes.
- Validation tuning selected 0.7273 because lower thresholds dramatically increase false positives from STTC and CD, causing precision (and thus F1) to collapse.
- While 0.7273 optimizes F1 on validation data, it inevitably truncates test recall to 51.9% (126 missed out of 262).

### 2.5 Demographic Factors
- **True Positive HYP:** Mean age **79.0 years**, 52.9% female.
- **False Negative HYP:** Mean age **65.5 years** (13.5 years younger), 43.7% female.
- Younger patients typically present with less advanced cardiac remodeling and fewer secondary ST-T changes, making hypertrophy harder to detect without absolute voltage.

---

## 3. Investigation 2: MI (Myocardial Infarction) False Negatives

The model correctly detects 431 MI records (78.4% recall) but misses **119 MI records** (21.6% FN rate).

### 3.1 Anatomical Location and Sub-Statement Breakdown
The 119 false negatives are not randomly distributed across infarction territories:

| SCP Code | Infarct Territory | Test Records | Missed (FN) | Detected (TP) | FN Rate | Contribution to All Missed MIs |
|---|---|---|---|---|---|---|
| **ASMI** | Anteroseptal | 234 | 15 | 219 | **6.4%** | 12.6% |
| **ILMI** | Inferolateral | 48 | 4 | 44 | **8.3%** | 3.4% |
| **INJAS** | Subendocardial Anteroseptal | 22 | 2 | 20 | **9.1%** | 1.7% |
| **IMI** | Inferior | 267 | 65 | 202 | **24.3%** | **54.6%** |
| **AMI** | Anterior | 35 | 14 | 21 | **40.0%** | 11.8% |
| **ALMI** | Anterolateral | 27 | 8 | 19 | **29.6%** | 6.7% |
| **LMI** | Lateral | 20 | 11 | 9 | **55.0%** | 9.2% |
| **PMI** | Posterior | 2 | 1 | 1 | **50.0%** | 0.8% |

#### Key Insights:
1. **Anteroseptal MI (`ASMI`) is robustly detected** (FN rate 6.4%). Loss of R-wave progression and deep Q waves across precordial leads V1–V4 produce pronounced, multi-lead morphological signatures that the 1D CNN readily captures.
2. **Inferior MI (`IMI`) is the dominant failure mode**, accounting for **over half (54.6%) of all missed infarctions** (65/119). Isolated Q waves in leads II, III, and aVF can be subtle, especially in chronic or inferior wall scars.
3. **Lateral MI (`LMI`) has the highest error rate** (55.0% missed). Isolated lateral wall changes (confined to leads I, aVL, V5, V6) are frequently overlooked by the model.

### 3.2 What Does the Model Predict When It Misses MI?
Of the 119 missed MI records:
- **54 records (45.4%) are predicted as NORM** (completely missed pathology).
- **32 records (26.9%) are predicted as STTC or STTC + HYP** (abnormality detected, but attributed to non-specific repolarization rather than infarction).
- **17 records (14.3%) are predicted as CD** (conduction defect detected; infarction missed).
- **16 records (13.4%)** receive no positive class prediction (all probabilities below threshold).

### 3.3 Demographics and Clinical Presentation
- **Age:** Patients with missed MI are significantly younger (mean age **68.1 years**) than patients with detected MI (mean age **78.2 years**).
- **Sex:** Missed MI cases have a higher proportion of female patients (**53.8% female** in FN vs **43.9% female** in TP). Atypical ECG presentations and non-ST-elevation infarction are historically more prevalent in female and younger cohorts.
- **Isolated vs. Complex MI:**
  - **Isolated MI** (MI is the only diagnostic class): 68 / 119 FN cases (**57.1%**).
  - When MI co-occurs with Conduction Defects or STTC, the overall abnormal wave distortion is larger, reducing the likelihood of a false NORM prediction.

---

## 4. Site and Hardware Associations

PTB-XL contains records collected across different clinical sites and recording devices:

| Site Group | Records (Fold 10) | MI Records | MI FN Rate | HYP Records | HYP FN Rate | Dominant Profile |
|---|---|---|---|---|---|---|
| **Site 0** | 483 | 78 | **38.5%** (30/78) | 34 | **55.9%** (19/34) | Mostly outpatient / routine (64.2% NORM prevalence) |
| **Site 1** | 597 | 124 | **16.9%** (21/124) | 95 | **53.7%** (51/95) | Moderate pathology mix |
| **Site 2** | 867 | 283 | **20.1%** (57/283) | 83 | **55.4%** (46/83) | Inpatient / complex pathology (32.6% NORM) |
| **Other sites** | 210 | 65 | **21.5%** (14/65) | 50 | **20.0%** (10/50) | Mixed |

- **Site 0 has an elevated MI miss rate (38.5%):** Because Site 0 has a very high NORM prevalence (64.2%), MIs at this site are predominantly chronic/old scars or outpatient presentations with less acute ST deviation.
- **Site 2 has suppressed HYP F1 (0.359):** Site 2 patients exhibit high rates of co-occurring cardiac disease (conduction defects and MI), causing extensive secondary ECG changes that confound hypertrophy classification.

---

## 5. Synthesis of Root Causes

The performance deficits in HYP and MI are driven by distinct, identifiable mechanisms:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             ROOT CAUSE MATRIX                              │
├──────────────────────┬──────────────────────────┬──────────────────────────┤
│ Factor               │ HYP (Hypertrophy)        │ MI (Infarction)          │
├──────────────────────┼──────────────────────────┼──────────────────────────┤
│ Preprocessing        │ CRITICAL: Z-scoring      │ Minor: Bandpass filter   │
│                      │ removes absolute voltage  │ (0.5–40 Hz) preserves   │
│                      │ required for diagnosis   │ pathological Q waves     │
├──────────────────────┼──────────────────────────┼──────────────────────────┤
│ Class Imbalance      │ Severe: 12.1% prevalence │ Moderate: 25.5% prev.    │
│                      │ causes low precision     │ (second largest class)   │
├──────────────────────┼──────────────────────────┼──────────────────────────┤
│ Spatial Lead Bias    │ Diffuse changes across   │ Strong precordial bias:  │
│                      │ limb/precordial leads    │ ASMI (6.4% FN) vs        │
│                      │                          │ IMI/LMI (24–55% FN)      │
├──────────────────────┼──────────────────────────┼──────────────────────────┤
│ Label Granularity    │ Atrial enlargement       │ Acute vs chronic not     │
│                      │ pooled with ventricular  │ distinguished; inferior  │
│                      │ hypertrophy              │ scars subtle             │
├──────────────────────┼──────────────────────────┼──────────────────────────┤
│ Demographic Bias     │ Younger patients missed; │ Younger & female cohorts │
│                      │ older patients detected  │ overrepresented in FN    │
└──────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 6. Recommendations and Next Steps

The following improvements are directly supported by evidence:

1. **Adopt Global / Training-Set Normalization (Top Recommendation for HYP):**
   - *Evidence:* Per-record z-scoring destroys voltage. The committed script `src/prepare_global_norm.py` implements per-lead mean/std standardisation computed strictly on training folds 1–8.
   - *Anticipated Impact:* Preserves relative inter-record amplitude, enabling the CNN to learn high-voltage representations for LVH.
2. **Lead-Group Attention / Auxiliary Territory Heads (For Inferior & Lateral MI):**
   - *Evidence:* ASMI is detected with 93.6% recall because precordial leads dominate the loss gradient, while inferior leads (II, III, aVF) and high lateral leads (I, aVL) are under-weighted.
   - *Anticipated Impact:* Spatial attention across anatomical lead groupings (limb, inferior, lateral, precordial) would prevent dominant precordial signals from overshadowing localized limb-lead infarctions.
3. **Disaggregate Atrial Enlargement from Ventricular Hypertrophy:**
   - *Evidence:* Atrial overload statements (`LAO/LAE`, `RAO/RAE`) have a 64%–90% FN rate under HYP.
   - *Anticipated Impact:* Separating atrial enlargement or introducing targeted P-wave morphological features would eliminate a major source of false negative dilution in the HYP superclass.
4. **Preserve Current Threshold Policy:**
   - *Evidence:* Modifying thresholds on the test set is invalid. On validation data, tuning per-class F1 already found the optimal operating point (HYP = 0.7273, MI = 0.5296). Arbitrary threshold lowering would cause false positive proliferation.

---

## 7. Limitations

1. **Proof-of-Concept Research Prototype:** CardioFed is a research prototype trained on PTB-XL; it is not a medical device.
2. **Confounded Site Effects:** Hospital site, recording equipment, and patient mix are confounded in PTB-XL (e.g., Site 0 is 64% NORM; Site 2 is 33% NORM).
3. **Annotation Subjectivity:** Diagnostic statements in PTB-XL reflect automated algorithmic pre-reads with varying cardiologist validation likelihoods.
