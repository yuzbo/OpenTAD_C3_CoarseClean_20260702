---
updated: 2026-07-06
status: active
scope: Absorbed Pro code review for visible DUCA-TAD commit 5f87baf
out-of-scope: Claiming issues are fixed, reporting new detector mAP, or changing experiment policy without verification
---

# DUCA-TAD 5f87baf Pro Review Absorption

Raw record: `docs/methods/reviews/2026-07-06-duca-5f87baf-pro-review-raw.txt`

Reviewed commit: `5f87bafc7c63a525ea7c91b7ef4b76b310cc86cf`

Primary branch: `codex/gas-vt-stage23-detector-aware-20260706`

## Core Verdict

This is a valid external code and experiment-route review. Unlike the earlier blocked review, the reviewer confirmed that the target commit and key GitHub blobs were visible. The review therefore counts as a substantive code-review artifact, not merely a visibility gate.

Overall absorbed verdict:

- Engineering implementation: `HOLD / WARN`
- Experiment evidence: `FAIL`
- Paper-claim readiness: `FAIL`

The target commit is best described as a Stage2/3/4 evidence-gate and smoke-prototype integration, not a complete DUCA-TAD end-to-end sparse detector.

## Stage Completeness Recorded

| Stage | Recorded completeness | Absorbed interpretation |
|---|---:|---|
| Stage1 GAS-VT / p_action strict ledger -> AdaTAD mAP | 60-65% | Offline ledger and tooling exist, but full detector mAP matrix and high-IoU evidence are not unlocked by this commit. |
| Stage2 dense teacher utility / detector-aware selector | 45-55% | Teacher utility adapter and offline selector route exist, but dynamic budget is still rank-derived and deploy leakage hardening is incomplete. |
| Stage3 TrueTime ST hard selector inside detector | 20-25% | Selector module has hard/ST mechanics, but detector classes do not call it, so detector loss is not yet proven to reach selector. |
| Stage4 curriculum / bilevel stable training | 10-15% | Evidence gate exists, but real curriculum/bilevel training loop is not implemented. |

## P0 Findings Absorbed

1. Detector forward does not integrate the TrueTime selector.
   - Files: `opentad/models/detectors/single_stage.py`, `opentad/models/detectors/two_stage.py`
   - Impact: no end-to-end claim; no proof that detector loss backpropagates to selector.
   - Required fix: build/call selector inside detector forward and merge selector losses with detector losses.

2. Selected-axis proposals are not forced back to physical/true time before post-processing.
   - Files: `single_stage.py`, `two_stage.py`
   - Impact: NMS and mAP may evaluate on the wrong temporal axis.
   - Required fix: explicit selected-axis to true-time remap before `convert_to_seconds` / NMS.

3. TrueTime metadata records selected valid length incorrectly.
   - File: `opentad/models/selectors/truetime_joint_selector.py`
   - Impact: sparse selected length may be confused with dense valid length.
   - Required fix: separate `irregular_dense_valid_len`, `irregular_selected_valid_len`, and `irregular_selected_count`.

4. Dynamic budget is not calibrated dynamic.
   - File: `tools/bata/train_detector_aware_acquisition_policy.py`
   - Impact: cannot claim calibrated dynamic budget.
   - Required fix: train-split global marginal-gain calibration; freeze threshold for val/test/deploy.

5. Deploy forbidden payload stripping and validation are incomplete.
   - Files: `apply_detector_aware_acquisition_policy.py`, `convert_detector_aware_samples_to_value_transport_ledger.py`
   - Impact: possible teacher/proposal/prediction-cache leakage in val/test/deploy artifacts.
   - Required fix: shared recursive forbidden-key validator used by apply, convert, and validate.

6. Teacher proposal to dense point coordinate assumptions are too weakly guarded.
   - File: `tools/bata/detector_teacher_utility.py`
   - Impact: teacher utility may be assigned to the wrong frame/time coordinate.
   - Required fix: require proposal axis, fps, stride, window offset, dense length, and source-axis metadata.

## P1 Findings Absorbed

1. Detector-aware selector input is still largely GAS-VT / p_action feature reuse.
   - Required interpretation: current route is closer to teacher-supervised p_action selector than a clearly detector-specific observable selector.
   - Required ablations: p_action-only, teacher-utility target, teacher-feature oracle, observable-only detector-aware.

2. Signed utility is partly collapsed by absolute-value marginal gain.
   - Risk: harmful negative utility can be treated as high-value observation.
   - Required fix: separate positive observation gain and negative observation risk.

3. Fixed budgets are requested fixed budgets with short-video effective adjustment.
   - Required wording: do not describe them as strictly fixed in every sample.
   - Required metadata: requested budget, effective budget, selected count, valid length, dense length, short-valid-ratio flag.

4. Hard gap-aware top-k can hide coverage-prior / gap-repair effects.
   - Required metadata: per-position selection reason such as utility top-k, gap repair, forced coverage.
   - Required ablation: no-repair vs gap-repaired top-k.

5. Stage4 is currently an evidence gate, not a trainable curriculum/bilevel system.
   - Required wording: Stage4 planned gate / protocol only.
   - Required future implementation: freeze/unfreeze schedule, temperature schedule, budget curriculum, sparse distillation training loop.

## Claim Gate Absorbed

Current forbidden claims:

- Sparse acquisition improves TAD mAP.
- Detector-aware acquisition is proven.
- End-to-end training is implemented.
- Dynamic budget is calibrated.
- CVPR-ready method is complete.
- High-IoU localization is protected.

Current allowed cautious wording:

- The commit integrates offline detector-aware selector tooling, TrueTime/ST selector smoke components, and Stage4 fail-closed evidence gates.
- It does not yet prove detector-aware sparse acquisition improves AdaTAD mAP.
- It does not yet implement a detector-loss-to-selector end-to-end training path.

## Required Fix Order

1. Fix recursive no-leakage validation across detector-aware apply, convert, and validate.
2. Fix TrueTime selected-valid-length metadata.
3. Add selected-axis to true-time segment remap before detector post-processing.
4. Add detector wrapper / detector forward integration for TrueTime selector.
5. Add real detector-loss gradient proof with selector parameter grad norm.
6. Replace rank-derived dynamic budget with train-split calibrated marginal-gain threshold.
7. Split signed utility into positive gain and negative risk targets.
8. Add no-repair vs gap-repair ablation metadata and tests.

## Required Experiment Roadmap

Seven-day gate:

- Fix leakage validator, TrueTime metadata, and time remap.
- Require teacher utility manifest axis and split provenance.
- Prove a real detector forward/backward path reaches selector.
- Run PRECHECK_ONLY remote smoke without unlocking mAP claims.

Four-week gate:

- Run dense, uniform, random, p_action, delta-p_action, GAS-VT, and detector-aware full AdaTAD mAP matrix.
- Include high-IoU metrics, especially mAP@0.6 and mAP@0.7.
- Include dynamic budget calibration and matched-average-K fixed baseline.

Eight-week gate:

- Implement sparse distillation and TrueTime ST end-to-end detector integration.
- Add curriculum/bilevel schedule only after Stage2 and Stage3 evidence is positive.
- Report collapse diagnostics, compute accounting, and full mAP.

## Paper Story Absorbed

The review supports the DUCA-TAD story only under strict evidence:

Temporal Action Detection sparse acquisition should learn detector-specific value of observation rather than actionness alone. The acquisition module should learn which observations improve proposals, boundaries, ranking, background suppression, and high-IoU localization. If detector mAP does not improve, the story should shrink to a negative study and diagnostic protocol for why actionness-based acquisition is insufficient.

## Local Interpretation

This review should drive implementation, but each issue still needs local verification against the current worktree before applying patches. It is not evidence that issues are already fixed. It is a prioritized external finding list and claim-lock policy for the next development cycle.
