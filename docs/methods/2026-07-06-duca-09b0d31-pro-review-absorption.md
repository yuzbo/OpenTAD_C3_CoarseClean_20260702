---
updated: 2026-07-06
status: active
scope: Absorbed external GPT/Pro static review for DUCA-TAD commit 09b0d31
out-of-scope: Claiming issues are fixed, reporting detector mAP, or changing experiment results
---

# DUCA-TAD 09b0d31 Pro Review Absorption

Raw record: `docs/methods/reviews/2026-07-06-duca-09b0d31-pro-review-raw.txt`

Reviewed commit: `09b0d312cdb54c590fd4c25b78aeac373d92a10c`

Primary branch: `codex/gas-vt-stage23-detector-aware-20260706`

## Validity Of This Review

This is a valid static code review artifact, but not an experiment reproduction.

The reviewer confirmed that the GitHub commit was visible and inspected code at the target SHA plus the in-repo review map. The review did not run training, did not reproduce detector mAP, and did not verify remote experiment logs. Its conclusions should therefore guide code review, claim locks, and experiment planning, while all metric claims must still come from controlled runs and recorded result files.

Local spot checks support the main static observations:

- `tools/bata/detector_aware_acquisition_policy.py` defines Stage2 as an offline selector, reuses `gas_vt.GAS_VT_FEATURE_NAMES`, and records `end_to_end=False`.
- `tools/bata/train_detector_aware_acquisition_policy.py` saves `policy_family=detector_aware_offline_selector` and `end_to_end=False`.
- `tools/bata/detector_teacher_utility.py` serializes `marginal_gain_frame_utility` using absolute signed utility.
- `opentad/models/detectors/single_stage.py` has a `frame_selector` hook and selected-axis remap support.
- `configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py` is explicitly `smoke_only=True` and disables paper / metric / end-to-end claims.

## Absorbed Verdict

Overall verdict:

- Research / paper claims: `HOLD`
- Engineering scaffolding: `PASS`
- Publishable DUCA-TAD method: not yet proven

The current implementation is directionally aligned with the intended DUCA-TAD story, but it does not yet satisfy the full project objective. It contains strong scaffolding and claim locks, but still needs completed detector mAP evidence, stronger detector utility semantics, and a validated real end-to-end selector + AdaTAD training path.

## P0 Findings Absorbed

1. Stage2 "detector-aware" is mainly offline teacher-target supervision.
   - The deploy-time feature path is still GAS-VT / p_action style.
   - The detector-aware signal currently enters mostly as an offline training target.
   - Required correction: either word Stage2 as offline distillation baseline, or implement online detector-loss / detector-state coupling before claiming detector-aware deployable acquisition.

2. No publishable detector mAP evidence is complete yet.
   - Intermediate GAS-VT mAP is useful, but not a full matched table.
   - Required correction: complete dense / uniform / random / p_action / GAS-VT / detector-aware / end-to-end matrix before any performance claim.

3. Teacher utility semantics are too weak for a strong detector-utility claim.
   - Current utility can degenerate to proposal center score spreading.
   - The serialized absolute signed utility can collapse negative risk into positive magnitude if reused.
   - Required correction: export real detector train-forward utility with point assignment, cls/reg loss, quality, boundary responsibility, and signed positive-gain vs negative-risk separation.

4. Stage3 end-to-end exists as skeleton / smoke, not production evidence.
   - `SingleStageDetector` has selector integration, loss merge, and remap hooks.
   - The configured route is explicitly smoke-only with no paper or metric claim.
   - Required correction: run real AdaTAD training with selector enabled and prove detector loss produces non-zero gradients on selector parameters in the production path.

5. High-IoU localization is not protected by evidence.
   - True-time remap exists, but evidence must include mAP@0.6/0.7, boundary-error diagnostics, selected-axis roundtrip tests on real detections, and matched-budget baselines.

## P1 Findings Absorbed

1. Dynamic budget calibration remains weak.
   - Current thresholding is positive teacher-score calibration, not detector-mAP or high-IoU marginal utility calibration.
   - Required correction: train-split marginal detector-utility calibration, frozen for val/test/deploy, plus matched-average-K fixed baseline.

2. Policy architecture is still GAS-VT-like.
   - The current novelty is mostly supervision and metadata, not a clearly detector-specific observable model.
   - Required correction: add detector-specific observable inputs or online detector-feedback training while preserving deployability.

3. Bootstrap policy must stay test-only.
   - Any hand-engineered bootstrap score route is allowed for CI/smoke only.
   - Required correction: exclude bootstrap results from paper claims and fail closed if used in formal mAP runs.

4. Export validation should fail earlier.
   - Evidence validators are strong, but export constructors should also require teacher checkpoint/config hashes and provenance.
   - Required correction: fail at export time when teacher evidence is absent, not only later in validation.

## P2 Findings Absorbed

- Replace internal route names such as `DIVERGENT_INNOVATION_*_DO_NOT_MERGE_WITH_C3` before public artifact release.
- Add seed variance, wall-clock, FLOPs, memory, checkpoint hash, and data manifest accounting.
- Add no-gap-repair / no-hole-guard ablations so improvements cannot be attributed only to hand-designed coverage repair.

## Claim Support Lock After Review

| Claim | Absorbed status | Required evidence |
|---|---:|---|
| Sparse acquisition improves TAD mAP | HOLD | Completed matched detector mAP table |
| Detector-aware acquisition improves over p_action-only | HOLD / current FAIL | Stage2 matched PAction/GAS-VT/random/uniform comparison |
| Dynamic budget is calibrated | WARN / HOLD | Train-only marginal utility calibration, frozen threshold, matched-average-K baseline |
| End-to-end selector + AdaTAD is implemented | Partial skeleton only | Production AdaTAD training with selector and non-zero selector gradients from detector loss |
| High-IoU localization is protected | HOLD / current FAIL | mAP@0.6/0.7, true-time remap audit, boundary localization ablations |

## Immediate Implementation Priority

1. Fix signed utility payload semantics.
   - Keep `signed_frame_utility`.
   - Add `positive_observation_gain`.
   - Add `negative_observation_risk`.
   - Relegate `abs_signed_utility` to diagnostic-only naming.

2. Strengthen teacher utility export.
   - Prefer dense AdaTAD train-forward diagnostics over proposal center score spreading.
   - Require teacher checkpoint/config hashes at export time.
   - Record point-axis metadata, fps, feature stride, window offset, dense length, and utility coordinate semantics.

3. Promote Stage3 from smoke to real experiment.
   - Add a non-smoke config or runner for real AdaTAD + selector training.
   - Add a production-path selector gradient proof.
   - Confirm evaluation uses sparse selector only and remaps selected-axis segments before NMS/mAP.

4. Complete Stage1/2 detector mAP matrix.
   - Dense official AdaTAD.
   - Uniform/random/p_action/GAS-VT/detector-aware fixed 384 and 768.
   - Dynamic variants with matched-average-K baselines.
   - Multiple seeds before final claim.

5. Add high-IoU and boundary audit.
   - Report mAP@0.6 and mAP@0.7 as first-class metrics.
   - Add boundary error histograms and short-action failure cases.

## Experiment Roadmap Absorbed

Stage A: Freeze trustworthy dense detector and simple baselines.

Stage B: Replace score-center utility with dense detector train-forward utility.

Stage C: Split signed positive gain and negative risk.

Stage D: Validate offline detector-aware acquisition under matched budgets.

Stage E: Promote Stage3 from smoke to real end-to-end sparse AdaTAD training.

Stage F: Run high-IoU localization audit and claim support analysis.

## Paper Story Constraint

The review supports the intended paper story only if the evidence catches up:

Dense TAD wastes computation, but actionness-only sparse sampling is not enough for high-IoU localization. DUCA-TAD should learn detector-utility-calibrated temporal acquisition: selecting frames that improve proposals, boundaries, ranking, and hard-negative suppression while preserving true-time geometry. The publishable novelty is not p_action or gap-aware top-k alone; it must be detector utility, signed gain/risk, true-time sparse detector geometry, calibrated dynamic budget, and verified detector-loss-to-selector optimization.

## Local Absorption Note

No code fix is marked complete by this document. It records a verified external review and updates the project claim gate. Each P0/P1 item still needs an implementation patch, focused tests, and remote precheck/full-run evidence before being marked resolved.

