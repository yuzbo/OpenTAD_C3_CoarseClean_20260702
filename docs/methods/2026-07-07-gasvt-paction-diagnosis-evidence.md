---
updated: 2026-07-07
status: active
scope: Evidence collected for diagnosing GAS-VT plateau and PAction learned fixed_384 advantage
out-of-scope: Claiming final causal proof, changing running experiments, or replacing matched reruns
---

# GAS-VT vs PAction Diagnosis Evidence

This note records the first evidence pass for the question:

Why did GAS-VT reach around 40 mAP early and then plateau, while PAction learned fixed_384 reached stronger mAP?

It combines parallel code explorers and read-only remote log parsing. It is not yet a matched rerun result.

## Current Verdict

I do not fully accept the explanation as proven. I accept it as the strongest working hypothesis.

The explanation is plausible:

- GAS-VT fixed_384 gives AdaTAD enough sparse context for coarse low-IoU detection early.
- Later progress is limited by ledger geometry, boundary precision, high-IoU localization, or train/apply mismatch.
- PAction learned fixed_384 likely benefits from simpler optimization, direct p_action/delta features, and an effective gap-loss policy.

But causal proof still requires:

- matched reruns from the same commit and same pretrain/data/config;
- formal ledger statistics with top-k overlap and selected-count histograms;
- ablations for budget-conditioned GAS apply, CVaR loss, hard repair, and PAction hard repair.

## mAP Curves

Source logs:

- GAS-VT: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/gas_vt_adatad/c3_gas_vt_stage01_gpu0_provok_20260706_151908_+0800/logs/gas_vt_fixed_384/train.out`
- PAction: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/paction_learned_adatad/c3_paction_learned_g30_gpu1_d413df8_20260706_214309_+0800/logs/learned_fixed_384/train.out`
- Dense teacher: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800/train.out`

### GAS-VT fixed_384

| logged epoch | Avg mAP | tIoU 0.30 | tIoU 0.40 | tIoU 0.50 | tIoU 0.60 | tIoU 0.70 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.69 | 33.65 | 25.21 | 17.06 | 8.95 | 3.57 |
| 6 | 40.96 | 56.30 | 50.59 | 42.71 | 32.93 | 22.29 |
| 11 | 43.87 | 59.85 | 53.73 | 45.29 | 36.27 | 24.19 |
| 16 | 42.74 | 57.68 | 51.80 | 43.88 | 35.34 | 25.00 |
| 21 | 43.65 | 59.62 | 52.96 | 44.28 | 35.63 | 25.74 |
| 26 | 45.87 | 61.51 | 55.85 | 46.51 | 37.98 | 27.52 |
| 31 | 46.55 | 61.97 | 56.22 | 47.52 | 38.60 | 28.45 |
| 36 | 45.33 | 60.75 | 54.77 | 46.42 | 37.80 | 26.91 |
| 41 | 44.17 | 59.58 | 53.02 | 45.34 | 36.76 | 26.15 |
| 46 | 45.05 | 59.96 | 54.20 | 46.59 | 37.35 | 27.17 |
| 51 | 45.41 | 60.22 | 54.64 | 47.35 | 37.36 | 27.48 |
| 56 | 44.96 | 60.15 | 53.93 | 46.72 | 37.26 | 26.75 |
| 59 | 44.90 | 60.09 | 53.83 | 46.39 | 37.28 | 26.92 |

Observation: GAS-VT jumps from 17.69 to 40.96 by logged epoch 6, then remains in the 43-46 mAP band.

### PAction learned fixed_384

| logged epoch | Avg mAP | tIoU 0.30 | tIoU 0.40 | tIoU 0.50 | tIoU 0.60 | tIoU 0.70 |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 48.08 | 69.50 | 60.89 | 50.26 | 37.99 | 21.74 |
| 19 | 60.41 | 76.79 | 70.27 | 62.78 | 53.15 | 39.08 |
| 29 | 61.02 | 76.03 | 71.71 | 64.07 | 53.21 | 40.05 |
| 39 | 60.49 | 75.33 | 70.76 | 63.25 | 53.17 | 39.94 |
| 49 | 59.73 | 75.07 | 69.48 | 62.49 | 52.10 | 39.49 |

Observation: PAction starts higher at first observed eval and improves high-IoU mAP much more than GAS-VT.

### Dense teacher

| logged epoch | Avg mAP | tIoU 0.30 | tIoU 0.40 | tIoU 0.50 | tIoU 0.60 | tIoU 0.70 |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 31.64 | 54.08 | 43.17 | 31.76 | 20.03 | 9.17 |

Dense teacher is still early and should not yet be treated as final teacher quality.

## Ledger Statistics

Remote validation summaries already contain boundary/action/hole/uniform fields, but they do not yet contain formal top-k overlap or selected-count histograms.

### Validation summary comparison

| split/method | action coverage | boundary r1 | max hole | p95 hole | max uniform sim | mean selected |
|---|---:|---:|---:|---:|---:|---:|
| val GAS fixed_384 | 0.5445 | 0.1426 | 96 | 96.0 | 0.5195 | 366.14 |
| test GAS fixed_384 | 0.5388 | 0.1433 | 96 | 96.0 | 0.5195 | 372.37 |
| val PAction fixed_384 | 0.5272 | 0.2361 | 117 | 2.0 | 0.5359 | 366.14 |
| test PAction fixed_384 | 0.5273 | 0.2371 | 170 | 2.0 | 0.5359 | 372.37 |

Important reading:

- PAction has lower action-positive coverage than GAS-VT by this metric.
- PAction has substantially stronger boundary support r1.
- PAction has much smaller p95 unselected hole, despite a larger worst-case max hole.
- Both methods have similar selected-count means because short-valid-ratio rows reduce the nominal 384.

### Additional sampled strategy-overlap analysis

The following was computed from `strategy_selected_positions` in remote samples.

| split/method | topk action Jaccard | delta p_action Jaccard | boundary score Jaccard | uniform Jaccard mean | uniform Jaccard max |
|---|---:|---:|---:|---:|---:|
| val GAS fixed_384 | 0.4372 | 0.3640 | 0.4120 | 0.3326 | 0.3446 |
| test GAS fixed_384 | 0.4308 | 0.3615 | 0.4084 | 0.3326 | 0.3446 |
| val PAction fixed_384 | 0.4083 | 0.3389 | 0.3853 | 0.3350 | 0.3711 |
| test PAction fixed_384 | 0.4074 | 0.3396 | 0.3877 | 0.3352 | 0.3711 |

Important reading:

- PAction is not simply closer to `topk_action_logit`; its overlap with top-k is slightly lower than GAS-VT.
- The stronger PAction result is more consistent with a learned ranking plus gap/loss geometry than with raw p_action top-k imitation.
- This still does not prove causal superiority because the runs are not a same-commit matched matrix.

## Evidence Gaps

Parallel explorers identified these missing pieces.

1. mAP curve extraction is possible from `train.out`; `result_detection.json` is not reliable as a curve source because it is not guaranteed to exist and may only contain the latest predictions.
2. Current validator lacks formal:
   - p_action top-k overlap;
   - selected-count histogram;
   - richer uniformity metrics such as phase-shift uniformity, gap CV, and CDF/KS distance.
3. Matched comparison is not proven:
   - PAction run uses remote snapshot `opentad_gasvt_d413df8_g30_20260706`;
   - current branch HEAD is later;
   - GAS-VT GPU0 run and PAction GPU1 run are not a clean same-commit matrix.

## Ablation Matrix

### Directly queueable after precheck

| ID | Purpose | Launcher/env |
|---|---|---|
| G0 | GAS-VT current pre-fix fixed_384 with hard repair | `scripts/run_c3_gas_vt_policy_adatad_full_train_gpu1.sh`, `GAS_VT_ADATAD_VARIANTS="gas_vt_fixed_384"` |
| P0 | PAction fixed_384 without hard repair | `scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh`, `PACTION_ADATAD_VARIANTS="learned_fixed_384"`, no `MAX_UNSELECTED_HOLE` |
| P1 | PAction fixed_384 with hard repair | same launcher, set `MAX_UNSELECTED_HOLE=96`, `MAX_P95_UNSELECTED_HOLE=96` |

### Requires code patch first

| ID | Purpose | Blocker |
|---|---|---|
| G1 | GAS-VT post-fix fixed_384 with hard repair | apply-time target-budget conditioning fix |
| G2 | GAS-VT post-fix no CVaR and no hard repair | loss-weight CLI/env knob plus repair-off launcher support |

## Immediate Implementation Implications

1. Fix GAS-VT apply-time budget conditioning.
2. Fix Stage2 detector-aware apply-time budget conditioning because it reuses GAS-VT feature machinery.
3. Add official top-k overlap and selected-count histogram fields to the validators.
4. Add repair statistics and richer uniformity metrics.
5. Only then run matched Stage1 ablations and Stage2/3 claims.

## Answer To The Diagnosis Question

The current evidence supports this partial diagnosis:

- GAS-VT likely reaches 40+ early because half-density sparse inputs are sufficient for coarse AdaTAD detection.
- GAS-VT plateaus because its ledger gives weaker boundary support and has large p95 holes in validation summaries; its current implementation may also suffer from train/apply budget-conditioning mismatch.
- PAction learned fixed_384 likely performs better because it provides much stronger boundary support and far smaller p95 holes while keeping similar selected-count density.
- PAction's advantage is not explained by simply copying p_action top-k, because its Jaccard overlap with top-k is slightly lower than GAS-VT's.

But this is not final causal proof. It is the evidence-backed next hypothesis to test with matched reruns and ablations.
