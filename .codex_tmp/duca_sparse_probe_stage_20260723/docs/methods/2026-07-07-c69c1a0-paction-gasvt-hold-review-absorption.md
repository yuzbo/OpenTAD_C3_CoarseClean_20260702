---
updated: 2026-07-07
status: active
scope: Absorbed external GPT static review for PAction learned vs GAS-VT at commit c69c1a0
out-of-scope: Claiming fixes are complete, changing remote experiments, or accepting reported mAP without matched logs
---

# PAction vs GAS-VT HOLD Review Absorption

Raw record: `docs/methods/reviews/2026-07-07-c69c1a0-paction-gasvt-hold-review-raw.txt`

Reviewed branch: `codex/gas-vt-stage23-detector-aware-20260706`

Reviewed commit: `c69c1a01cbb86cdce1ceba13b3a491d0b489f76f`

## Validity

This is a valid static code and experimental-logic review artifact, not a reproduction.

The reviewer confirmed public GitHub visibility for the target branch and commit, but also correctly noted that the repository does not contain long remote training logs, checkpoints, result JSONs, or full metric evidence. Therefore, all mAP comparisons remain monitoring hypotheses until backed by controlled matched runs and archived result artifacts.

The review is especially useful because it separates engineering scaffolding from paper-level claims:

- Code correctness: `HOLD`
- Experimental logic: `HOLD`
- Paper claim readiness: `FAIL`
- GAS-VT current status: demote to engineered Stage1 baseline unless repaired
- PAction result status: unverified without matched logs and full result artifacts
- Stage2 status: blocked/incomplete without teacher utility export surface
- Stage3 status: precheck-oriented, not mAP-claim-ready

## Local Spot Check

After recording the review, local inspection of the current worktree supports several core findings.

- `tools/bata/export_dense_adatad_teacher_utility.py` is absent in the current branch surface.
- `tools/bata/detector_teacher_utility.py` exists, but it is not the requested dense AdaTAD teacher utility exporter entry point.
- `tools/bata/train_gap_aware_acquisition_policy.py` builds GAS features with `target_budget`.
- `tools/bata/apply_gap_aware_acquisition_policy.py` still has `checkpoint_policy_scores()` building GAS features without a requested budget, while a bootstrap path does use a max dynamic bucket.
- `tools/bata/train_detector_aware_acquisition_policy.py` builds detector-aware features with target-budget conditioning.
- `tools/bata/apply_detector_aware_acquisition_policy.py` still has a checkpoint scoring path that can inherit the same target-budget mismatch.

These checks mean the review should not be treated as merely theoretical. Its main implementation blockers are present in the code surface that matters for Stage1/2 claims.

## Absorbed P0 Findings

1. GAS-VT train/apply feature mismatch.
   - Training conditions feature construction on target budget.
   - Application can score with feature construction that omits target budget.
   - This is a real distribution shift and a plausible reason for GAS-VT plateau or uniform-like behavior.
   - Required action: pass the concrete requested budget into apply-time feature construction for fixed variants; for dynamic variants, predict budget first, then rebuild features with that predicted budget before final scoring.

2. GAS-VT is not a faithful sequential value-transport policy.
   - Current implementation exposes state-like features, but the train/apply path does not perform an autoregressive selection loop with updated `selected_so_far`.
   - The hard decoder delegates to constrained top-k machinery.
   - Required action: either rename/demote GAS-VT to an engineered constrained p_action baseline, or implement true iterative state update and verify it.

3. Hard gap repair can produce uniform-like lattices while uniform-fill flags stay false.
   - `uses_uniform_fill=False` and `uses_uniform_scaffold=False` are not enough to prove non-uniform acquisition.
   - Required action: record hard-repair counts, max hole before/after, selected-position uniformity metrics, and whether repair caused uniform-like output.

4. Stage2 dense teacher utility exporter is missing from the reviewed branch.
   - The requested `tools/bata/export_dense_adatad_teacher_utility.py` is not present locally either.
   - Required action: add a train-only exporter that signs and records dense AdaTAD teacher utility, checkpoint/config hashes, split provenance, and coordinate semantics.

## Absorbed P1 Findings

1. PAction vs GAS-VT validation gates are asymmetric.
   - Direct comparison is weakened if one route fails stricter geometry gates while another route is mainly judged by detector mAP.
   - Required action: use the same ledger validation schema for PAction, GAS-VT, detector-aware, and uniform/random/top-k baselines.

2. `uniform_similarity` is too narrow.
   - Exact overlap with one rounded uniform reference can both over-trigger and miss phase-shifted uniform grids.
   - Required action: add gap CV, rank-distance-to-uniform, selected CDF/KS style distance, p95/p99 hole, and per-row worst-case summaries.

3. PAction validation JSONL support must stay train-diagnostic only.
   - GT-derived action/boundary targets must not drive official val/test policy selection.
   - Required action: fail closed if non-train GT-derived payloads enter policy training or checkpoint selection.

4. Dynamic-budget variants need matched selected-count accounting.
   - Dynamic results cannot be compared directly to fixed budgets without selected-count histograms and matched-average-K baselines.

5. Stage2 can inherit the same budget-conditioning mismatch as GAS-VT.
   - Detector-aware features reuse GAS-VT feature machinery.
   - Required action: repair Stage2 apply-time feature construction at the same time as GAS-VT.

6. Stage3 TrueTime currently proves contract/shape more than detector-aware learning.
   - Required action: show real detector-loss gradients into selector parameters, selector parameter updates, and mAP/high-IoU gains under matched budgets.

## Interpretation Of PAction Better Than GAS-VT

The review's explanation is absorbed as the current working hypothesis:

- PAction learned fixed_384 may be better because it is simpler and easier to optimize.
- Fixed_384 is already a high-density selector on a 768-length window, so strong hole and boundary metrics can arise from budget density plus p_action/delta correlations.
- GAS-VT adds more constraints and losses, but without a faithful sequential decoder or detector utility it may over-regularize frame values toward coverage/uniformity.
- Early GAS-VT mAP can rise quickly because AdaTAD learns coarse action classification from half-density inputs; later plateau likely reflects boundary/high-IoU localization limits.

This does not prove PAction is the final research method. It makes PAction learned fixed_384 a strong Stage1 baseline that Stage2/3 must beat under matched conditions.

## Claim Gate

| Claim | Status after absorption | Required evidence |
|---|---:|---|
| PAction learned fixed_384 is better than GAS-VT fixed_384 | HOLD | Matched rerun from same commit, same data SHA, same pretrain, same eval schedule, archived logs/results |
| GAS-VT is a publishable main method | FAIL for now | Budget-consistent apply path, true sequential decoder or renamed baseline, matched mAP gain |
| Sparse ledger improves AdaTAD mAP | HOLD | Uniform/random/top-k/PAction/GAS-VT/detector-aware matched matrix |
| Stage2 is detector-aware | HOLD | Train-only dense teacher utility exporter and matched Stage2 mAP table |
| Stage3 is end-to-end | HOLD | Detector-loss gradient proof plus full mAP run |
| High-IoU localization is protected | HOLD | mAP@0.6/0.7, boundary error audit, true-time remap audit |

## Immediate Work Queue

1. Add or restore `tools/bata/export_dense_adatad_teacher_utility.py`.
2. Fix GAS-VT apply-time target-budget conditioning.
3. Fix Stage2 apply-time target-budget conditioning.
4. Add hard-repair and richer uniformity metrics to ledger summaries.
5. Rerun matched Stage1 matrix after fixes:
   - uniform fixed_384
   - random fixed_384
   - p_action top-k fixed_384
   - PAction learned fixed_384 with and without hard repair
   - GAS-VT fixed_384 before/after budget-conditioning repair
   - fixed_768 and dynamic variants with selected-count histograms
6. Complete dense teacher utility export from train split only.
7. Run Stage2 precheck and full only after exporter evidence exists.
8. Promote Stage3 from precheck to real full run only after proving detector-loss gradients reach selector parameters.

## Paper Story Constraint

The review reinforces the story constraint:

PAction and current GAS-VT are Stage1 baselines and safety anchors. The paper-grade method cannot be "p_action plus gap-aware constrained top-k" alone. The publishable route must be detector-utility-calibrated temporal acquisition, preserving true-time geometry and high-IoU localization while reducing pre-backbone computation. Stage2 should prove dense AdaTAD teacher utility improves acquisition over p_action-only. Stage3 should prove a selector and AdaTAD detector can be optimized in one training graph without collapse.

## Absorption Status

This document records and absorbs the review. It does not mark any code issue as fixed. The review should be used as a claim lock and implementation checklist before any paper-facing comparison between PAction learned, GAS-VT, Stage2 detector-aware selection, or Stage3 joint training.
