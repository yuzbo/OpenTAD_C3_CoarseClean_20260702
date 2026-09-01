---
title: BUILDER_DUCA_DYNAMIC_INNER_MECHANISM_MINIMAL_CHANGE_PLAN
version: v001
date: 2026-08-14
material_epoch: LOCAL_ONLY-v013-20260814T172500+0800
route_id: DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION-v001
stage: DESIGNED_ONLY / MINIMAL_CHANGE_PLAN
author_role: Builder (project-owned ARIS DeepSeek V4 Pro executor)
evidence_class: STATIC_PLAN_ONLY / NOT_EXECUTED
v008_status: QUARANTINED / UNKNOWN_REMOTE_STATE / never relied upon
supersedes_note: distinct new local material epoch; does NOT retransmit or rely on
  PROJECT_SOURCE_SYNC_REQUEST-v008.json, CURRENT_RESEARCH_STATE-v012.md, or
  DUCA_DYNAMIC_FINAL_PRO_ADJUDICATION-v001.md
next_owner: fresh exact-Project Pro (via a separately granted central
  Sources-to-Pro lease)
single_recovery: if a future leased source mutation becomes unknown, stop without
  retransmission and request one separately leased read-only inventory
---

# Builder minimal-change plan — dynamic inner-mechanism question (A vs B)

## 0. Decision boundary

This document is a **no-code, no-execution** minimal-change plan. It reframes the
final dynamic-budget scientific question as a two-way inner-mechanism comparison
and hands it to a fresh exact-Project Pro. It authorizes no implementation, no
PRE_RUN admission, no data access, no dataset listing/mount, no model/checkpoint
access, no GPU/CUDA, no Slurm, no remote work, no training, no inference, no
evaluation, no metric computation, no cost measurement, no test, no browser, no
Project Sources, no Pro invocation, no Git action, and no paper claim.

`BLOCKED_PRE_RESULT` is preserved. Dynamic budget is the required core. Fixed K
is only baseline/attribution/fallback. Candidate A is not recovered; Candidate C
is not an automatic fallback.

## 1. v008 quarantine and the distinct new local material epoch

The previous source batch (`PROJECT_SOURCE_SYNC_REQUEST-v008.json` referencing
`CURRENT_RESEARCH_STATE-v012.md` and `DUCA_DYNAMIC_FINAL_PRO_ADJUDICATION-v001.md`)
is `UNKNOWN_REMOTE_STATE / quarantined`. It is neither retransmitted nor relied
upon by this plan. This document is a **distinct new local material epoch**
(`LOCAL_ONLY-v013-20260814T172500+0800`) authored from scratch against the local
dirty working-tree surface and `research-wiki/` only.

## 2. The new final scientific question (presented, not decided)

Given that the previous Route-B inner decoder is a *monotone, bounded-displacement
physical exact-K transport*, and its Critic exposure (`F1`) was a canonical-uniform
identity contradiction inside that decoder, the final scientific round is
reframed as a comparison of **two non-equivalent inner mechanisms**, each under the
same outer dynamic-K policy:

- **A — arbitrary non-contiguous selection + exact per-frame physical timestamps.**
  The outer policy selects `K`. The inner selector emits *any* `K` distinct frame
  indices (full clustering freedom, no monotone/bounded-dilation constraint),
  and each selected frame carries its **exact original physical timestamp**; a
  frozen timestamp/coordinate map transports raw detector proposals back to
  physical time before filtering, top-k, IoU, NMS, or evaluation.

- **B — monotone / local physical exact-K transport.**
  The outer policy selects `K`. The inner decoder is the bounded monotone
  density transport (strictly increasing inverse-CDF positions with a bounded
  local stride and displacement cap); constant density must degenerate to the
  canonical physical uniform tuple. This is the pure-plugin path that keeps the
  detector's ordinal-grid temporal assumption approximately valid and maps raw
  proposals back pre-NMS.

### 2.1 The forbidden candidate (structural invariant, not a threshold)

Any candidate that **loses real timestamps** or that **presents selected ordinal
positions as uniform time** is forbidden. This is exactly the historical
selected-rank / selected-axis distortion. The two candidates must both satisfy:

1. selected positions are real physical frame indices with exact timestamps;
2. raw detector proposals are transported to physical time before any IoU,
   top-k, clipping, NMS, voting, or serialization;
3. no hidden dense heavy path or `Kmax` padding substitutes for the declared
   effective `K`.

The difference between A and B is therefore **where** the timestamps act:

- A pushes exact timestamps **into** the detector's time semantics (true-time
  positional conditioning or a timestamp-aware coordinate adapter). This is
  *enhanced integration*, not a pure pre-backbone plugin.
- B keeps the detector untouched and corrects only at the proposal→physical-time
  boundary, at the cost of constraining the selection geometry to a bounded
  monotone warp.

This is the scientific fork Pro must adjudicate, because the repository's pure
pre-backbone plugin contract (RTK.md) forbids mixing detector-internal time
injection into the *pure-plugin main result*. If A wins, it must be labeled and
reported as an enhanced-integration variant; if B wins, F1 (its decoder identity)
must be resolved.

## 3. Why each candidate is interesting

### 3.1 A — arbitrary non-contiguous selection + exact timestamps

- **Full allocation freedom.** It is the closest to the historical GT-oracle
  "boundary burst" behavior: it can concentrate frames around start/end
  boundaries and short actions without a monotone or max-gap constraint.
- **Honest time, not geometric constraint.** Correctness comes from carrying the
  true timestamps, separating "which frames" (free) from "how the detector knows
  time" (exact), instead of forcing near-uniform geometry.
- **Directly tests the strongest failure hypothesis.** The recurring DUCA
  failure is selected-rank time distortion. A is the maximally direct counter:
  if even exact timestamps cannot recover high-IoU, then the damage is in the
  backbone's irregular-input handling, not merely in proposal mapping.
- **Risk.** The backbone (VideoMAE tubelets, temporal conv, position embeddings,
  AdaTAD adapter) assumes regular spacing; arbitrary non-contiguous timestamps
  break those priors unless the detector is made timestamp-aware. A therefore
  risks becoming detector-internal time injection, which is out of scope for a
  pure-plugin main claim.

### 3.2 B — monotone / local physical exact-K transport

- **Pure-plugin compatibility.** Bounded monotone transport keeps the detector's
  ordinal-grid assumption approximately valid, so the main result can remain a
  pure pre-backbone plugin.
- **Formal coverage guarantee.** The bounded stride and displacement cap bound
  the maximum gap and per-position deviation, which is a *hard* constraint (not
  an empirical surrogate).
- **Reproducible uniform identity.** Constant density must equal the canonical
  physical uniform tuple, giving a clean fixed-K attribution anchor.
- **Risk.** The monotone constraint limits clustering freedom. The historical
  fixed-K learned density never beat matched uniform at high tIoU, so B's
  interestingness hinges on whether outer K heterogeneity (not inner shape) can
  deliver the gain — exactly what O1/O2 must test.

## 4. What video complexity causes different K

Candidate drivers of per-window/video budget heterogeneity (to be measured, not
assumed):

- number of action instances and boundary density;
- short-action fraction (short actions are boundary-dominated);
- action duration spread and background proportion;
- boundary temporal ambiguity (motion smear, occlusion, camera cuts);
- transition sharpness between adjacent/nearby actions.

Videos with many short, boundary-dense, ambiguous actions should warrant higher
`K`; videos with few long, well-separated actions should warrant lower `K`. The
claim under test is that matched-realized-cost dynamic allocation protects high
tIoU where uniform fixed-K cannot.

## 5. Which boundaries / action phases could benefit

- action start and end boundaries (the highest-IoU-sensitive regions);
- short actions (one boundary error destroys IoU);
- overlapping / temporally-adjacent actions;
- high-motion transition phases near boundaries;
- paired start+end events whose joint localization crosses below tIoU 0.7 when
  budget is reduced (the paired-boundary risk target).

## 6. Novelty / prior-art invalidators

Generic dynamic budget, budget scorers, inverse-CDF sampling, ILP, nested prefix
sets, cheap-global→sparse-heavy allocation, and risk calibration are all
pre-occupied: AdaFrame, MGSampler, AdapTok, AdaFocusV3, SMART, TAPS, Progressive
Block Drop, keyframe sampling, and semantic-boundary wavelet selection.

- For **A**, timestamp-aware / time-aligned detector coordinates are also
  pre-occupied (TE-TAD time-aligned coordinates; PhysTime/TrueTime). A's novelty
  risk is high unless the *combination* with outer hard-utility dynamic K and
  paired-endpoint risk is materially distinct.
- For **B**, bounded monotone inverse-CDF uniform-compatible transport is a
  near-neighbor of Hartley systematic sampling and Uni-AdaFocus inverse-CDF.

The only potentially publishable claim is the still-unverified combination:
train-only hard budget-conditional utility + paired endpoint/high-IoU risk +
physical exact-K + batch-invariant realized cost + unchanged detector internals.
Pro must reject whichever candidate cannot isolate a materially distinct
mechanism.

## 7. Six-arm FIT/CAL/PILOT contract (mechanism isolation)

All arms share the official THUMOS14/OpenTAD-AdaTAD detector architecture,
projection/FPN, head, assignment, losses, optimizer/LR schedule, augmentation,
video/window order, RNG streams, 6,000-successful-update budget, pre-NMS physical
mapping, filtering/top-k/NMS/voting, class map, unchanged official evaluator, and
terminal-EMA-only checkpoint rule. Official validation/test remain inaccessible
to selection, epoch/K/seed/teacher/utility/risk construction.

| Arm | Frozen behavior | Isolates |
|---|---|---|
| `dense` | Native 768 AdaTAD/OpenTAD, natural cost, no scout/acquisition | cost + upper-mAP anchor |
| `uniform_k384` | Canonical physical uniform K384, no learned scout | fixed-K baseline; F1 identity object; F2 permutation reference |
| `dynamic_A` | Outer `G−β·R−λ·C` dynamic K + inner A (arbitrary non-contiguous + exact timestamps) | A full mechanism |
| `dynamic_B` | Outer `G−β·R−λ·C` dynamic K + inner B (monotone/local bounded transport) | B full mechanism |
| `k_shuffle` | Full dynamic assignments (chosen inner's outer policy) permuted by the F2 rule within exact valid-length/candidate-set strata; K histogram + cost preserved, content↔budget broken | outer content-to-budget causality |
| `no_risk` | Same trunk/utility/risk heads as the chosen inner; only `β·R` masked; λ refrozen on CAL to the same cost target | paired-boundary/high-IoU risk value |

`dynamic_A` vs `dynamic_B` is a clean A-vs-B comparison: the outer policy is
identical, only the inner geometry/time-representation differs. The fixed-K A-vs-B
attribution is delivered by O2 (frozen-detector inner-family comparison at matched
K), not by an extra learned-K arm. `k_shuffle` and `no_risk` are mandatory
controls and must run; their exact placement (formal training arm vs O4 gate
control) is a Pro confirmation item, but both must exist with frozen definitions.

## 8. Exact falsifiers

### F-O1 — dynamic budget headroom (unchanged, inner-agnostic)
Frozen-detector dynamic Oracle (with the chosen inner) vs best fixed K at matched
realized mean cost. No positive video-cluster headroom kills the dynamic main
route.

### F-O2 — inner geometry and decoder identity (new)
At matched K and matched realized cost, on the frozen FIT-only detector:

- B's monotone decoder must satisfy cardinality, uniqueness, strict increase,
  physical in-range mapping, constant→canonical-uniform identity, and lose no
  more than the predeclared bound at mAP@0.7 vs its learned fixed-K reference
  (this is where F1 is decisive);
- A's timestamp transport must be exactly invertible / round-trip bounded on the
  canonical uniform subset, must never present ordinal-as-uniform, and must not
  require detector-internal time injection unless labeled enhanced integration;
- report A-vs-B at matched K on high tIoU, short-action, boundary error, and
  pure-plugin-vs-enhanced-integration classification. Whichever loses its
  mechanism isolation is dropped.

### F-O3 — hard-utility predictability (`G_rank`, unchanged)
Train-only hard detector-utility must predict budget-conditional gain above
chance. Failure removes the utility head.

### F-O4 — paired-boundary/high-IoU risk (unchanged, outer mechanism)
Paired risk must beat actionness/transition/no-risk/K-shuffle. Failure removes
the risk contribution and kills the paper route.

### F-INV — timestamp invariant (hard, structural)
Any arm losing real timestamps or presenting selected ordinal positions as
uniform time is a P0 stop, regardless of metric.

## 9. N16R4 seed / launcher / evaluator / full-stack-cost plan (form only)

- **Seed policy:** first seed = development screening, excluded from final
  statistics. Pre-registered fresh seeds only after structure freeze. No
  training-side held-out set → terminal EMA at update 6,000.
- **Split:** training-population-only, whole-video, video-disjoint FIT/CAL/HOLD
  (FIT ≈ 3/5, CAL ≈ 1/5, HOLD ≈ 1/5) via a frozen SHA-256 rank; five-fold
  FIT cross-fit for hard utility/risk labels (no same-video labeling).
- **Launcher:** one Slurm GPU (`--gpus=1 --cpus-per-task=6`), `source /etc/profile`
  before `set -u` and every `module load`, in-process `cuda:0`, never override
  `CUDA_VISIBLE_DEVICES`, no login-node training.
- **Evaluator:** unchanged official OpenTAD/AdaTAD evaluator, invoked only once
  on terminal EMA, pooled complete predictions, no per-video AP averaging,
  video-cluster paired bootstrap.
- **Full-stack cost:** source-decode → preprocess → H2D → coarse probe → outer
  scorer → inner decoder/gather → heavy backbone → adapter/proj/FPN/head →
  pre-NMS physical map → NMS → serialization; hard-label/refresh GPU-hours and
  training cost listed separately. Nominal K/FLOPs/backbone-only timing cannot
  support a cost claim; hidden dense path or padded Kmax is a terminal stop.

## 10. Stop rules

- O1 no headroom → kill dynamic main route (fall back to fixed-K attribution or
  Candidate C only with a separate decision);
- O2 F1 unresolved or B's decoder identity invalid → B is dropped; if A then
  requires detector-internal time injection, A is labeled enhanced integration,
  not pure plugin;
- O3 `G_rank` failure → remove utility head;
- O4 pair-risk not superior → remove risk contribution, kill paper route;
- development seed ≤ best fixed K at matched cost, or high-IoU/short-action
  degradation, or padded Kmax, or no net full-stack saving → stop multi-seed /
  second-detector / efficiency claim.

## 11. F1 / F2 preserved as unresolved Pro questions

- **F1 (canonical decoder identity).** For B, Pro must freeze one exact monotone
  decoder convention whose constant-density specialization, CDF/in-bin
  coordinate, integer projection, and tie rule are mutually identical, or
  REVISE/PIVOT/STOP. For A, Pro must freeze the canonical-uniform-subset +
  exact-timestamp identity and the round-trip-bounded transport contract.
- **F2 (K-shuffle reproducibility).** Pro must freeze the nonce/seed, canonical
  row order, permutation algorithm, applicable stratum, and collapsed/invalid
  stratum fallback, or reject the control as unable to isolate outer
  content-to-budget causality.

Neither is decided by this plan. Neither may be silently resolved by a future
Builder under the existing contract.

## 12. Strict no-result boundary and handoff

This plan is `STATIC_PLAN_ONLY / NOT_EXECUTED`. It produces no Route-B
implementation, no PRE_RUN, no data traversal, no pilot, no training, no metric,
no cost, and no claim. Fixed K remains baseline/control/fallback only. The sole
next action is a separately granted central Sources-to-Pro lease delivering this
distinct new local material epoch to a fresh exact-Project Pro, which must
FREEZE F1/F2 and choose A-vs-B or REVISE/PIVOT/STOP.
