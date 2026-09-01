---
updated: 2026-07-07
status: active
scope: Absorption of an external static review on no-leak sparse acquisition, Stage1/2/3 readiness, and CVPR-grade method risk.
out-of-scope: Treating static review as reproduced evidence, claiming current code is CVPR-ready, or changing running experiments without precheck.
---

# CVPR Intelligent Acquisition Review Absorption

Raw record:
`docs/methods/reviews/2026-07-07-cvpr-intelligent-acquisition-no-leak-review-raw.txt`

## Verdict

I accept the review as a valid route-level code and research critique, not as an
experiment reproduction. The strongest current interpretation is:

- Engineering defenses are clearly stronger: no-leak records, provenance, strict
  ledger validation, focused tests, and remote prechecks are now real assets.
- Method and paper claims remain HOLD. The current route is still closer to a
  robust sparse-ledger engineering scaffold than to a CVPR-grade intelligent
  acquisition method.
- PAction learned is a useful proxy and the strongest Stage1 signal observed so
  far, but it does not prove detector-aware sparse acquisition by itself.
- GAS-VT should stay as a diagnostic baseline unless it is upgraded into a true
  sequential, gap-aware, detector-utility-calibrated selector.

## Accepted Findings

1. The selector currently learns actionness or proposal-score proxy signals more
   than high-IoU TAD utility. This explains why action coverage can improve while
   boundary localization and high-IoU mAP do not.
2. Soft gap, CVaR, and repair losses do not guarantee hard deployment geometry.
   Any gap control must be audited in the emitted ledger and compared against the
   actual selected positions.
3. Selected-axis remapping in AdaTAD remains a likely high-IoU bottleneck. A
   true-time or hybrid geometry experiment is needed before making localization
   claims.
4. Stage2 dense-teacher utility is currently a proposal-score utility baseline
   unless it explicitly exports point responsibility, cls/reg loss, boundary
   utility, saliency, or counterfactual utility.
5. Stage3 cannot be called end-to-end until detector loss is shown to move the
   selector, with non-zero gradients or measurable selected-position changes.
6. Main-route budgets must be fixed at 384 selected positions or fewer. 640/768
   variants may be kept only as diagnostic ceilings.

## Immediate Implementation Gates

- Add fail-closed 384-or-less budget checks for any route that will support a
  main sparse-acquisition claim.
- Separate deploy metadata for `policy_source`, `selector_decoder`,
  `train_supervision`, and `deploy_inputs`.
- Keep PAction score-only lattice replacement as a diagnostic/performance probe,
  not the final paper method.
- Record repair/replacement counts and pre/post hole statistics whenever a
  post-hoc geometry decoder is used.
- Extend dense-teacher utility provenance so each exported row states whether it
  came from proposal score, point responsibility, cls/reg loss, saliency, or
  counterfactual utility.
- Add matched 384-only baselines before claiming a win: uniform, raw PAction
  top-k, PAction no repair, PAction repair, PAction lattice, GAS-VT fixed, and
  teacher utility selector.
- Add true-time versus selected-axis remap comparison for at least the strongest
  384 selector.

## Claim Boundary

This review does not invalidate the current experiments. It narrows what they
can honestly prove:

- PAction learned fixed_384 can prove that low-cost p_action-like signals are
  useful for sparse TAD input construction.
- PAction lattice replacement can test whether learned scores plus detector
  geometry constraints can recover performance under a 384 budget.
- Dense teacher Stage2 can test whether detector-derived utility improves the
  acquisition policy over p_action-only supervision.
- Stage3 can only become an end-to-end claim after selector-detector gradient
  coupling is verified.

The route should not claim "intelligent acquisition" from no-leak ledgers alone.
The paper-level claim needs detector-utility evidence, high-IoU preservation,
and matched 384-budget baselines.

