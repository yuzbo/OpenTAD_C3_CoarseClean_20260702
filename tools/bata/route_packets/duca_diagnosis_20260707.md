# DUCA Diagnosis Packet 2026-07-07

Route:
`DIVERGENT_INNOVATION_DETECTOR_UTILITY_CALIBRATED_ACQUISITION`

External-review finding absorbed:

GAS-VT should not be treated as the final method. Current evidence suggests it
can cover action interior/action-positive regions, but it does not reliably
provide the boundary-neighborhood, boundary bracket, and local geometry that
AdaTAD needs for high-IoU temporal localization. PAction learned fixed_384 is
currently stronger because it preserves boundary support and small holes better,
not because it simply copies raw p_action top-k.

Boundary-courage correction absorbed:

The research target is not action coverage. Action coverage is only a diagnostic
counterexample: GAS-VT can report higher action coverage while producing worse
high-IoU mAP. The target should be stated as p_action-transition-based boundary
acquisition: use low-cost p_action shape, delta, uncertainty, and entropy as
scout signals for boundary-neighborhood evidence, boundary brackets, and local
geometry needed by the detector.

Current interpretation:

- GAS-VT is a Stage1 engineered sparse-selector baseline.
- PAction learned fixed_384 is the strongest current p_action-supervised Stage1
  boundary-proxy baseline, but it is still mixed with action BCE and must be
  ablated before it can support a method claim.
- The main research route is DUCA-TAD: detector-utility-calibrated pre-backbone
  acquisition.
- Stage2 should use train-only dense AdaTAD teacher utility to calibrate selector
  targets, with start/end/high-IoU boundary utility higher priority than class
  utility.
- Stage3 should test true joint selector-detector optimization with straight
  through hard selection and true-time geometry.

Design direction:

- Selector targets should separate detector utility heads instead of collapsing
  all supervision into one generic value score: classification utility, start
  boundary utility, end boundary utility, uncertainty/context utility, and false
  positive risk.
- Coverage can remain as an explicit safety role, but hidden hard repair must be
  measured and ablated. A run rescued by repair should not be described as a
  purely learned acquisition policy.
- The final detector interface should preserve original dense local time and
  selected observation geometry. Selected-rank remapping is an engineering
  baseline, not the final sparse TAD geometry.
- Pairwise/ranking utility losses are more aligned with top-k acquisition than
  only pointwise BCE.
- The selector should move from one frame-value head toward separate start,
  end, transition, uncertainty/context, and false-positive-risk utility heads.
- Any action-interior objective should be a low-weight context/safety term, not
  a main objective. Boundary hit, boundary bracket, boundary ranking, and
  boundary-local hole losses are the main objectives for the boundary-courage
  route.

Evidence gates before claims:

- matched same-commit mAP curves for GAS-VT, PAction, Stage2, and Stage3 under
  identical AdaTAD settings;
- ledger summaries with selected-count distribution, boundary support, action
  coverage, max/p95 holes, uniformity diagnostics, and p_action top-k overlap;
- ablations for PAction without boundary/gap/repair terms and GAS-VT without
  CVaR/action-interior/hard repair;
- Stage2 teacher utility export must be train-only and must not leak into
  val/test selection;
- Stage3 must show non-zero detector-loss gradient into the selector before any
  end-to-end claim.
- a valid boundary-courage claim requires the pattern: action coverage does not
  need to increase, while start/end boundary support or bracket success and
  mAP@0.6/0.7 improve.

Minimum ablation gates:

- high-action-interior top-k as a negative control;
- high-delta top-k and high-uncertainty top-k to isolate transition and
  uncertainty scout signals;
- PAction no-action-BCE, no-boundary-loss, and no-gap/hole-loss;
- GAS no-action-interior-bin and GAS boundary-only;
- teacher boundary utility selector, with start/end/high-IoU utility separated
  from class utility.

Immediate implementation gates:

1. Verify GAS-VT and Stage2 apply-time target-budget conditioning.
2. Add/verify validator statistics for p_action top-k overlap, boundary-distance
   quantiles, selected-count histograms, repair counts, gap CV, and phase-shift
   uniformity.
3. Collect diagnostic-only evidence summaries for current GAS-VT fixed_384,
   PAction fixed_384, and dense teacher curves.
4. Queue matched ablations only after their prechecks pass under the latest
   snapshot.

Claim boundary:

This packet is diagnostic only. It supports route prioritization and evidence
collection, not a paper claim about mAP improvement, deployment efficiency, or
end-to-end training.

Paper-story anchor:

We do not learn to cover actions. We learn to cover boundary evidence that a TAD
detector needs for high-IoU localization under pre-backbone sparse observation.

Performance-first 384-or-less plan absorbed:

The immediate objective is to build a deployable pre-backbone selector that can
beat the current uniform-sampling anchor around 65 Average-mAP before spending
GPU budget on a large ablation matrix. The budget constraint is fixed at 384
selected positions or fewer; larger settings such as 640/768 may only be used as
debug ceilings and cannot be the main performance claim. The preferred
short-term candidate is a Boundary-Uniform Hybrid Selector implemented as
boundary-aware lattice replacement, not budget expansion:

- keep explicit uniform anchors to preserve AdaTAD-friendly global temporal
  geometry and reduce selected-axis remap distortion;
- add signed p_action transition slots for start/end boundary brackets;
- add abs-delta, entropy, and uncertainty slots for ambiguous boundary evidence;
- keep a small action-context quota so low-IoU recall does not collapse;
- use explicit safety gap repair only as an audited role, with inserted/replaced
  counts and pre/post hole statistics.

First performance candidates:

1. `hybrid_fixed_384_lattice_replace`: start from a 384 uniform lattice and
   replace only a bounded number of low-utility anchors with boundary,
   transition, and uncertainty candidates.
2. `hybrid_fixed_384_boundary_aggressive`: lower the protected uniform quota
   and allocate more slots to signed start/end brackets, while retaining enough
   lattice anchors to avoid selected-axis geometry collapse.
3. `hybrid_dynamic_256_320_384`: use 384 only for complex windows and fewer
   positions for easy/background windows; average budget must be below 384.
4. `paction_uniform_safety_fixed_384`: reuse the strong PAction score source but
   protect a uniform scaffold and perform explicit boundary-aware replacement.
5. `hybrid_fixed_768_ceiling`: optional sanity/upper-bound only if 768 equals the
   dense local window; it is not a sparse claim.

GPU priority update:

- Do not spend new GPU time on pure GAS-VT as a main route.
- Let the current dense teacher and PAction runs finish unless they fail.
- Prepare hybrid policy, metadata, precheck, and ledger diagnostics off-GPU.
- Start `hybrid_fixed_384_lattice_replace` on the first free GPU, then
  `hybrid_dynamic_256_320_384` or `paction_uniform_safety_fixed_384` on the next
  free GPU.

Success criterion:

The first claim-worthy win is not just Average-mAP over 65. It must use at most
384 selected positions, be a no-leak deploy ledger under matched AdaTAD
settings, with no val/test GT/teacher/prediction-cache selection input, plus
high-IoU mAP not collapsing. For a boundary-courage claim, boundary/bracket
metrics should improve even if action coverage does not.

External GPT PLR/BPLR plan absorbed:

The newest review recommends not abandoning PAction learned. Instead, PAction
learned should stop being a global top-k selector and become a
detector-geometry-constrained scoring backbone. The proposed next method is
PAction learned scoring plus a geometry-aware local lattice replacement decoder:
start from a detector-friendly uniform 384 lattice, protect a uniformly
distributed scaffold, and only replace local low-utility anchors with
PAction-derived boundary, transition, uncertainty, and context candidates.

Core plan:

- keep `uniform_384` as the detector-friendly geometry baseline;
- keep `PAction learned global_topk_384` as a diagnostic baseline, because it
  can score useful frames but may distort selected-axis geometry and high-IoU
  localization;
- keep `PAction learned max-hole repair_384` as a middle baseline, but audit it
  as a post-hoc repair rather than a pure learned selector;
- make `PAction learned local lattice replacement_384` the P0 performance route;
- defer dynamic `256/320/384` until fixed 384 proves value under matched AdaTAD
  settings.

Implementation target:

- `tools/bata/paction_lattice_replacement_policy.py`: pure decoder containing
  feature construction, candidate roles, local replacement, safety repair, and
  diagnostics;
- `tools/bata/apply_paction_lattice_replacement_policy.py`: load the PAction
  checkpoint, compute learned frame value, call the decoder, and emit deployable
  samples;
- `tools/bata/run_paction_lattice_replacement_ledger_pipeline.py`: reuse the
  canonicalize -> deploy-source-strip -> apply -> convert -> validate contract;
- `tools/bata/validate_paction_lattice_replacement_ledger.py`: fail closed on
  selected count, sorted uniqueness, max budget, policy source, leakage, role
  counts, gap, replacement diagnostics, and uniform-similarity bounds;
- patch
  `tools/bata/convert_lowres_probe_samples_to_value_transport_ledger.py` with a
  new deploy-safe policy source for lattice replacement and an allowlisted
  replacement diagnostics payload;
- add AdaTAD config and GPU launcher that keep detector/head/postprocess/eval
  settings identical to the PAction learned and uniform baselines;
- add focused tests for policy behavior and end-to-end ledger pipeline.

Recommended variants:

- `paction_lattice_replace_fixed_384_conservative`: budget 384, protected
  uniform 288, replacement slots split roughly into boundary, transition,
  uncertainty, and context roles. This is the first full-run candidate.
- `paction_lattice_replace_fixed_384_boundary_aggressive`: protected uniform
  256 and more boundary/transition slots. Run only after conservative precheck
  and preferably after the conservative full run gives a clean signal.
- `paction_lattice_replace_dynamic_256_320_384`: P1 only. It should not be used
  as the first paper claim because variable count and reader masking introduce
  extra confounders.

Safety and ablation requirements:

- deploy ledgers must not contain GT boundaries, action coverage, boundary
  support, teacher predictions, oracle positions, or raw prediction cache;
- boundary support and bracket metrics are audit-only outputs joined outside
  deploy ledger generation;
- same budget, same detector, same source, and same ledger converter are needed
  for attribution;
- rule-only replacement, shuffled learned score, global top-k, max-hole repair,
  and uniform-only baselines are required to prove the learned scoring backbone
  contributes beyond a hand-written delta rule.

My accepted interpretation:

I agree with this plan as the right short-term performance route. It keeps the
strongest current signal source, PAction learned, but constrains it by the
geometry that AdaTAD appears to need. It is more defensible than pure
hand-designed boundary-uniform hybrid because the learned frame value remains
the scoring backbone, while the decoder is framed as a detector-compatibility
constraint. The immediate implementation should therefore be conservative
fixed_384 first, not 640/768 and not dynamic first.

External no-leak/CVPR-readiness review absorbed:

The newest review judges the current engineering defenses as materially
stronger but still blocks paper-level intelligent-acquisition claims. I accept
that distinction. No-leak provenance, strict ledgers, and focused tests are
necessary infrastructure; they do not by themselves show that the selector is
learning detector-relevant acquisition. The route must therefore keep a sharp
line between engineering readiness and method evidence.

Accepted constraints:

- Main sparse-acquisition claims must use 384 selected positions or fewer.
  640/768 variants remain diagnostic ceilings only.
- PAction learned is the strongest current Stage1 proxy, but it is still a
  p_action/proposal signal unless detector utility improves it under matched
  settings.
- PAction score-only lattice replacement is a diagnostic performance probe, not
  the final CVPR method. It is useful only if it is reported with replacement
  counts, pre/post gap metrics, and matched baselines.
- GAS-VT is demoted to an engineered Stage1 ablation until it becomes a true
  sequential gap-aware selector with measured hard deployment geometry.
- Stage2 dense-teacher utility must identify the utility source per row:
  proposal score, point responsibility, cls/reg loss, saliency, or
  counterfactual utility. Proposal score alone is a baseline, not the full
  detector-aware method.
- Stage3 needs direct evidence that detector loss reaches and changes the
  selector before any end-to-end claim is allowed.
- Selected-axis remapping remains a likely high-IoU confounder. A true-time or
  hybrid geometry comparison is required for localization claims.

Immediate queue update:

1. Keep recording all external reviews as raw artifacts plus absorption notes
   before changing code.
2. Finish the current PAction score-only lattice replacement path as a
   fixed_384 diagnostic probe and precheck it before any full run.
3. Add fail-closed 384-or-less budget summaries to any main-claim route.
4. Extend evidence summaries to separate engineering-pass signals from
   method-claim evidence: matched mAP curves, high-IoU deltas, boundary/hole
   metrics, replacement/repair counts, and utility-source provenance.
5. Prioritize Stage2 detector-utility export and true-time geometry comparison
   after the dense teacher checkpoint and train-only utility evidence are
   available.
