---
updated: 2026-07-07
status: active
scope: Absorption record for external HOLD review of DUCA-TAD commit acc6960
out-of-scope: Reporting new detector mAP, claiming issues are fixed without current-state evidence, or replacing experiment results
---

# DUCA-TAD acc6960 HOLD Review Absorption

Raw record: `docs/methods/reviews/2026-07-07-duca-acc6960-hold-review-raw.txt`

Reviewed commit: `acc696039d53458cbe36d65b72a84bb55fdafe6f`

Current branch: `codex/gas-vt-stage23-detector-aware-20260706`

Current branch HEAD when absorbed: `378bbc1782407eec1a58ab404e2494a25d94f5f0`

Latest implementation target already reviewed in our project prompt: `3f0041c9847ffc50b43a55d3845ec37ec089c026`

## Validity

This is a valid static review of the public GitHub commit `acc6960`, but it is not a runtime reproduction. The reviewer inspected source files and configs, did not run pytest, did not run OpenTAD training, and did not verify remote logs.

The review should be treated as a claim-control and implementation-planning artifact. It cannot prove any metric, but it does correctly identify that paper-level claims remain blocked until detector mAP, matched baselines, dense teacher provenance, and real end-to-end training evidence exist.

## Absorbed Verdict

I accept the main verdict:

- Research and paper claims remain `HOLD`.
- Engineering scaffolding is useful, but not a complete method proof.
- The route is still aligned with DUCA-TAD, but the evidence tree is incomplete.

The review target was `acc6960`. Since then, `3f0041c` partially addressed one of the review's Stage3 gate concerns by binding Stage3 precheck/full-run artifacts to config/proof hashes. That does not change the global HOLD verdict, because most blockers are experimental or method-level rather than wrapper-level.

## Findings We Should Carry Forward

### P0: No performance claim yet

The reviewer is correct. We still cannot claim sparse acquisition superiority until a complete detector mAP matrix exists:

- dense AdaTAD official-like baseline / teacher
- uniform fixed 384 / 768
- random fixed 384 / 768
- p_action fixed 384 / 768 / dynamic
- GAS-VT fixed 384 / 768 / dynamic
- Stage2 detector-aware fixed 384 / 768 / dynamic
- Stage3 true end-to-end sparse selector

Required metrics remain: mAP@0.3/0.4/0.5/0.6/0.7, average mAP, wall-clock, memory, selected count, max gap, p95 gap, boundary bracket coverage, action coverage, and high-IoU localization diagnostics.

### P0: Stage2 utility is not yet strong detector utility

The review correctly calls out that current Stage2 utility can be read as proposal-score / proposal-center saliency rather than true detector utility. The next stronger implementation should use real AdaTAD train-forward evidence:

- point positive assignment responsibility
- per-point classification and regression loss contribution
- proposal quality / NMS-sensitive responsibility
- signed positive gain versus harmful false-positive risk
- optional mask-span / leave-span-out counterfactual proxy

Until then, Stage2 should be described as an offline teacher-supervised selector candidate, not as a proven detector-utility acquisition policy.

### P0: Stage3 is gradient-path evidence, not a THUMOS end-to-end result

The review's wording is accurate. Stage3 currently proves a candidate path: detector loss can reach selector parameters in a synthetic / precheck ActionFormer route. It does not yet prove:

- full THUMOS training stability
- sparse AdaTAD detector mAP
- high-IoU preservation
- selector collapse resistance
- production end-to-end performance

The current Stage3 precheck queue now uses the newer `3f0041c` snapshot, not the old `acc6960` snapshot, but it remains a precheck until the full run and mAP matrix are complete.

### P1: GAS-VT may be mostly p_action plus priors

We should preserve this critique in the experiment plan. GAS-VT needs ablations:

- remove gap urgency / CVaR max-hole pressure
- remove boundary bracket objective
- remove action interior objective
- remove hard max-hole constraints
- p_action-only with same budget
- random with same max-hole constraint
- uniform with same K

This is necessary to show that any gain comes from learned acquisition rather than hidden hand-designed repair.

### P1: Dynamic budget calibration is weak

The current dynamic budget path is not yet a strong deployment claim. We need a train-only frozen calibrator with:

- explicit marginal detector utility semantics
- target average K
- confidence intervals or bootstrap stability
- no val/test refit
- matched-average-K fixed baseline
- per-video selected count distribution

Dynamic budget should remain a candidate until these checks exist.

### P1: True-time geometry is necessary but not proven

True-time remap and physical-grid support are directionally right. Missing proof:

- mAP@0.6/0.7 improvements or non-regression
- selected-axis naive postprocess versus true-time postprocess ablation
- real sliding-window offset/fps/unit tests
- boundary error distribution
- selected-axis to true-time roundtrip checks on real predictions

### P2: Stage4 is still a protocol/gate, not a curriculum trainer

The review is correct. Stage4 should not be described as implemented curriculum training until code exists for:

1. dense warmup
2. selector pretraining
3. frozen selector sparse detector training
4. partial unfreeze
5. ST joint fine-tune
6. dynamic budget calibration
7. collapse diagnostics
8. final compute-matched table

### P2: Recursive no-leakage artifact scanner is still valuable

Existing metadata flags are useful, but nested artifacts can still leak. A recursive scanner should inspect JSON/JSONL/NPZ/PT metadata payloads for forbidden keys or values in val/test/deploy artifacts:

- `gt`, `ground_truth`, `oracle`
- `teacher`, `dense_teacher`
- `raw_prediction`, `prediction_cache`, `proposal_cache`
- utility payloads without explicit train-only signed semantics

This is a near-term engineering improvement because it strengthens all routes, not just Stage2.

## Current-State Delta Since acc6960

Already partially addressed after the reviewed commit:

- `3f0041c` added stronger Stage3 full-run/precheck binding around config and proof hashes.
- The remote Stage3 precheck queue was replaced with the `3f0041c` snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_stage23_3f0041c9847f_20260707_001421`.
- The project prompt now locks the implementation review target to `3f0041c` while letting later documentation commits move independently.

Still not addressed:

- complete detector mAP matrix
- dense AdaTAD official-like teacher checkpoint and train-only utility export
- stronger per-point / loss-based detector utility
- Stage3 full THUMOS end-to-end run
- Stage4 curriculum trainer
- recursive artifact leakage scanner
- GAS-VT ablations
- dynamic budget calibration evidence

## Next Implementation Bias

The next code improvement with the best risk/reward ratio is the recursive no-leakage artifact scanner plus tests, because it can harden PAction, GAS-VT, Stage2, and Stage3 without waiting for the dense teacher job.

The next experimental dependency is the dense AdaTAD teacher checkpoint. Without it, Stage2 cannot honestly graduate from offline scaffold to detector-aware utility evidence.

The next claim dependency is the running Stage0/1 and PAction detector mAP matrix. No detector mAP means no performance claim.
