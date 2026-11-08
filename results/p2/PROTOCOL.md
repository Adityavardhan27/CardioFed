# P2 Experiment Protocol (pre-specified)

Written and committed **before** any P2 training run (2026-11-07). Results
must be reported for every run listed here, whatever the outcome. No
hyper-parameter is tuned on the test set; nothing below is changed after
results are seen. Any deviation is recorded in Section 7.

## 1. Common settings

| Item | Setting |
|---|---|
| Data | `data/processed` (per-record normalisation, CardioFed label policy) for E1–E3 |
| Splits | official folds: 1–8 train, 9 validation, 10 test |
| Architecture | Multi-scale residual CNN (`src/model.py`) |
| Hardware | CPU only (as for the original runs); 2 threads per job |
| Threshold selection | per class, maximise F1 on validation (fold 9) |
| Model selection | best validation macro-AUROC (epoch or round) |
| Evaluation | `src/evaluate_checkpoint.py`: validation-tuned thresholds, test fold evaluated once, per-site metrics |
| Seeds | 42, 43, 44 |

The existing artifacts are reused, not retrained:
- centralized multi-scale seed 42 (`results/scheduler_experiment/`);
- FedAvg 3 and 8 rounds (`results/checkpoints/`).

## 2. Experiments, in priority order

**E1: Local-only baselines.** Each of the 4 federated clients trains alone
on its own FedAvg partition (`src/local_training.py`).
- Recipe: the centralized multi-scale recipe (Adam lr 1e-3, batch 16,
  ReduceLROnPlateau on validation loss, early stopping on validation
  macro-AUROC with patience 5, at most 20 epochs).
- Positive-class weights from the client's own labels.
- Seeds 42, 43, 44, giving 12 runs.
- Simulation assumption: model selection and thresholds use the central
  validation fold, as in the FedAvg simulation.

**E2: FedAvg for 20 rounds** (`src/federated_train.py --rounds 20`).
- Recipe: unchanged (lr 5e-4, 1 local epoch, batch 16, 4 clients).
- Seeds 42, 43, 44, giving 3 runs.
- Convergence: per-round validation macro-AUROC and client losses; best
  round; first round within 0.005 of the run's maximum.
- Reproduction check: seed 42 rounds 1–8 compared with the stored
  `results/metrics/federated_history.json`.

**E3: Centralized multi-scale, additional seeds** (`src/train_multiscale.py`,
default recipe). Seeds 43 and 44, giving 2 runs; seed 42 already exists.

**E4: Alternative normalisation**, only if resources permit.
- Filtered signals (same 0.5–40 Hz band-pass) standardised with per-lead
  mean and std computed on training folds 1–8 only, then applied to all
  folds.
- Stored separately in `data/p2_global_norm/`.
- Centralized multi-scale with the default recipe; seeds 42, 43, 44 if
  feasible, otherwise 42 only.

## 3. Endpoints

- **Primary:** test macro-AUROC.
- **Secondary:** test macro-F1, per-class AUROC/F1 (HYP for E4), per-site
  macro-AUROC.

## 4. Pre-specified comparisons

Seed-averaged, paired patient-level bootstrap on the test fold (2,000
resamples). In each resample, each arm's metric is averaged over its seeds,
then the arms are differenced. Two-sided p-values with Holm correction across
C1–C3 (macro-AUROC; macro-F1 reported separately with its own Holm
correction).

- **C1:** FedAvg-20 vs centralized multi-scale.
- **C2 (×4):** FedAvg-20 vs local-only client k, on the full test fold.
- **C3:** global vs per-record normalisation (centralized), only if E4 runs.

Descriptive (no hypothesis test): each local-only model on its own site's
test subset, compared with FedAvg-20 on the same subset.

Seed-to-seed variability is reported as mean ± SD across seeds.

## 5. Interpretation rules

- A difference is called significant only if Holm-adjusted p < 0.05.
- Non-significant differences are reported as "no detectable difference",
  never as "equivalent".
- Local-only vs FedAvg answers whether federation improves on isolated
  training for each client. It does not establish privacy benefits.

## 6. Resource and stopping rules

- Up to 5 concurrent CPU jobs.
- If the total wall time is expected to exceed about 6 hours, or memory or
  disk becomes insufficient, drop items in reverse priority (E4 seeds 43–44,
  then E4, then E3) and record that here.
- Existing files are never overwritten (all scripts guard outputs).
- New outputs go under `results/p2/` and `data/p2_global_norm/`.

## 7. Deviations

1. **Scope reduced by user instruction (2026-11-08)**, after the first
   session was interrupted while the job queue was running.
   - *Kept:* local-only baselines with seed 42 (one per client), and FedAvg
     for at least 10 rounds.
   - *Dropped:* additional local-only seeds (43, 44) and E4 (alternative
     normalisation).
   - *Consequence for comparisons:* the primary comparisons (C1, C2) became
     seed-matched on seed 42, with Holm correction across those five tests.
     The three-seed FedAvg vs centralized comparison is supplementary. C3 was
     not run.
   - The extra runs that had already finished (centralized seeds 43/44,
     FedAvg seeds 43/44) are reported, not discarded.
2. **Interrupted runs.**
   - Local-only clients 3 and 4 (seed 42) were interrupted at epochs 7 and 8
     by the session ending. They were rerun from scratch into
     `results/p2/local/client{3,4}_seed42_complete/`.
   - Training on CPU is deterministic only for a fixed number of threads.
     The first attempts used 2 threads; the reruns used 4 threads
     (`OMP_NUM_THREADS=4`). Their trajectories therefore differ slightly
     (e.g. client 3, epoch 1 validation AUROC 0.8488 vs 0.8464).
   - Thread counts of all analysed runs: 2 threads for FedAvg (hard-coded),
     centralized seeds 43/44 and local clients 1/2; 4 threads for local
     clients 3/4. The thread count of the original seed-42 centralized run
     was not recorded.
   - The interrupted folders (`client3_seed42`, `client4_seed42`) and the
     partial seed-43 local runs (`client1_seed43`, `client2_seed43`, plus an
     empty `local_c3_s43` log) are kept unchanged but not analysed.
3. **E4 not run.** The globally normalised dataset had been built
   (`src/prepare_global_norm.py`, git-ignored, regenerable), but no model was
   trained on it.
4. **FedAvg rounds.** 20 rounds were run (minimum 10 required). Validation
   macro-AUROC peaked at rounds 13–14 in all three seeds and did not improve
   afterwards, so no further rounds were run.
