---
updated: 2026-07-07
status: active
scope: Absorb the external static review of commit 46cacc113042fcf0931c70774491d44665246e32 and translate it into claim locks, code gates, and the next experiment plan.
out-of-scope: Treating the review as reproduced mAP evidence, changing running jobs, or claiming the current sparse-acquisition route is paper-ready.
---

# 46cacc1 CVPR HOLD Review Absorption

Raw record:
`docs/methods/reviews/2026-07-07-46cacc1-cvpr-hold-review-raw.txt`

Reviewed target:

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Branch: `codex/gas-vt-stage23-detector-aware-20260706`
- Commit: `46cacc113042fcf0931c70774491d44665246e32`
- Patch under review: lattice provenance inference for source exports.

## Verdict Absorbed

I accept the review as a valid static code and research-route critique. It is
not an independent experiment reproduction.

The absorbed verdict is:

- PASS for the focused `46cacc1` provenance patch.
- WARN for current sparse-result interpretation.
- HOLD for paper or CVPR-level method claims.

The patch is useful because it lets the lattice pipeline recover p_action
provenance from canonical source exports when the deploy ledger can still strip
GT diagnostic payload and remain no-leak. That does not make the full sparse
selector route detector-aware or end-to-end.

## Claim Locks

The following claims remain locked:

- No claim that lattice replacement is a scaffold-free intelligent selector.
- No claim that Stage2 is detector-loss-aware until utility includes detector
  responsibility, cls/reg sensitivity, saliency, or counterfactual terms.
- No claim that Stage3 is end-to-end until a real full training run shows
  nonzero selector loss, detector-loss gradient into selector parameters, and
  mAP/tIoU evaluation.
- No claim that the `<=384` sparse route beats matched uniform_384 until a
  same-config result exceeds the approximately 65 mAP anchor.
- No claim from 768 or dynamic variants for the fixed-budget sparse paper main
  result; those are diagnostic ceilings only.

## P0 Findings To Implement

1. Lattice truthfulness:
   Lattice replacement starts from a uniform lattice and performs local
   p_action-score replacement. Its summary must say
   `uses_uniform_scaffold=True`, with
   `scaffold_type="uniform_lattice_local_replacement"`,
   protected/replaceable counts, and uniform overlap or Jaccard statistics.

2. Explicit paper-main provenance:
   `allow_inferred_paction_positive_provenance=True` is acceptable for migration
   and diagnostics, but paper-main fulltrain should require explicit
   `paction_positive_provenance`, checkpoint sha, and manifest sha.

3. Detector utility upgrade:
   Stage2 must move beyond proposal-score-derived utility. The next main route
   should export train-only dense AdaTAD responsibility targets: point
   responsibility, matched proposal responsibility, cls/reg loss sensitivity,
   high-IoU boundary utility, and optional counterfactual drop/add utility.

4. Real Stage3:
   Stage3 is currently a gradient/precheck route. It must become a real
   full-train route before any end-to-end claim: real batches, real detector
   loss, nonzero selector regularization, collapse metrics, selected-position
   movement, mAP, and high-IoU reporting.

## P1 Findings To Implement

- Add strict `valid_positions` checks: duplicate, nonmonotonic, and out-of-range
  positions must fail in paper-main mode.
- Enforce exact budget at decoder level when `valid_len >= budget`; do not rely
  only on downstream converter or validator.
- Report short-valid exceptions separately from fixed_384 selected-count
  histograms.
- Add alias conflict checks for same `(subset, video, window_start, dense_len,
  valid_len)` with different sample ids.
- Separate paper-main fixed_384 configs from 768 and dynamic variants, or add a
  manifest gate that prevents result mixing.
- Add config parity checks against official AdaTAD and matched uniform_384:
  only loader/ledger/target-length/projection differences should vary.
- Lock checkpoint selection and eval cadence so best-epoch reporting cannot be
  confounded by different validation schedules.
- Make every fulltrain summary record `PRECHECK_ONLY`, unlock flag, Slurm job,
  selected-count histogram, short-valid count, and repair/replacement counts.

## Experimental Reprioritization

The review reinforces the current north star:

> First obtain a convincing `<=384` pre-backbone sparse TAD result under matched
> AdaTAD settings. Then build the CVPR story around detector-responsibility
> acquisition.

Execution order:

1. Keep dense AdaTAD teacher running to epoch 59 and export train-only utility.
2. Treat PAction learned fixed_384 as the strongest current p_action-supervised
   baseline.
3. Treat lattice replacement as a diagnostic probe for geometry and decoder
   effects, not as the main intelligent method.
4. Implement responsibility-aware Stage2 fixed_384 as the next main experiment.
5. Use Stage3 only after Stage2 shows the utility target can beat p_action-only
   or at least closes the gap to uniform_384.

## Main Comparisons Required

Fixed-budget main comparisons:

- uniform_384;
- raw p_action top-k 384;
- PAction learned fixed_384;
- GAS-VT fixed_384;
- lattice replacement diagnostic 384;
- Stage2 responsibility-aware utility selector 384;
- Stage3 ST joint selector 384, only after real fulltrain is available.

Diagnostics only:

- fixed_768;
- dynamic budget;
- no-CVaR or no-repair variants;
- scaffold or replacement ablations.

## Success Criteria

A route can become paper-main only if it satisfies all of the following:

1. selected positions are exactly `<=384` for valid windows;
2. no val/test GT, teacher prediction cache, oracle boundary, or raw detector
   prediction is available to deploy selection;
3. mAP beats matched uniform_384 by a meaningful margin, with high-IoU mAP not
   collapsing;
4. ledger reports selected count, max/p95 holes, boundary bracket/support,
   action coverage, short-valid exceptions, utility coverage, and replacement
   or repair counts;
5. all claims are tied to commit, manifest, checkpoint sha, config diff, and
   Slurm/precheck evidence.

## Immediate Code Queue

The next engineering pass should prioritize:

- fix lattice scaffold metadata and tests;
- add formal-mode explicit provenance gate;
- add strict valid-position and exact-budget tests;
- add fixed_384 claim-manifest gate;
- implement dense teacher responsibility export;
- implement Stage2 utility precheck that rejects score-only utility when a
  detector-aware claim is requested;
- keep Stage3 claim-locked until fulltrain evidence exists.

