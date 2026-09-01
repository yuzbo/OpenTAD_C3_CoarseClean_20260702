---
title: DUCA_DYNAMIC_INNER_MECHANISM_FINAL_PRO_ADJUDICATION
version: v001
date: 2026-08-14
stage: FINAL_SCIENTIFIC_ROUND_MATERIAL_ONLY
author_role: coordinator
route_id: DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION-v001
material_epoch: LOCAL_ONLY-v013-20260814T172500+0800
evidence_class: STATIC_PLAN_ONLY / NOT_EXECUTED
supersedes: quarantined DUCA_DYNAMIC_FINAL_PRO_ADJUDICATION-v001 transport epoch
---

# Final Pro packet — choose the honest-time inner mechanism or revise/pivot/stop

## Binding question

This is the final allowed scientific round for Dynamic DUCA. Fresh exact-Project
Pro must return one decisive `CONTINUE`, `REVISE`, `PIVOT`, or `STOP`
disposition. It alone must freeze F1 and F2 and decide which (if either) inner
mechanism yields an interesting, isolatable, publishable dynamic-budget route.
Dynamic budget is mandatory; fixed K may be used only as baseline, attribution,
or fallback, never as the final route.

Under the same outer train-only dynamic-K objective `G - beta*R - lambda*C`,
compare:

- **A — arbitrary non-contiguous selection + exact timestamp remap.** Select any
  K distinct physical frame indices. Every selected frame retains its original
  timestamp/physical coordinate; raw proposals map to physical time before
  filtering, top-k, IoU, NMS, voting, or serialization. The detector must be
  explicitly timestamp-aware, so A is enhanced integration rather than the
  repository's pure pre-backbone plugin.
- **B — monotone/local physical exact-K transport.** Select K strictly increasing
  physical positions using bounded local density transport; preserve detector
  internals and map proposals to physical time pre-NMS. Constant density must
  reduce to one frozen canonical physical-uniform tuple.

Neither candidate may turn selected ordinal rank into uniform time, discard true
timestamps, hide a dense heavy path, or pad Kmax in place of declared effective
K. These are structural stops, not metric thresholds.

## Why this fork could be interesting—and what kills it

The proposed surprise is not generic adaptive sampling: train-only utility and
paired boundary risk should allocate different heavy budgets to videos/windows
with many short actions, dense or ambiguous boundaries, transitions, overlap, or
duration heterogeneity, while preserving high-IoU localization at matched
realized cost. A tests whether full boundary-focused freedom is viable once time
is represented honestly; B tests whether a pure plugin can preserve that value
with bounded geometry and canonical uniform attribution.

Pro must reject a candidate if its novelty reduces to known adaptive frame/token
selection, inverse-CDF sampling, time-aligned coordinates, or a non-isolatable
detector modification. Relevant invalidators include AdaFrame, MGSampler,
AdapTok, AdaFocusV3, SMART, TAPS, Progressive Block Drop, keyframe and semantic
boundary sampling; A also faces TE-TAD/PhysTime/TrueTime; B faces Hartley
systematic sampling and Uni-AdaFocus. No novelty or efficacy is currently
established.

## Definitions Pro must freeze, not delegate to a Builder

**F1 — canonical physical-time contract.** For B, specify one exact CDF/bin
coordinate, integer projection objective and tie rule whose constant-density
specialization is exactly the stated canonical physical-uniform tuple. For A,
specify the canonical-uniform subset, exact timestamp/coordinate representation,
and round-trip bound. State whether A can remain scientifically comparable to B
despite enhanced integration; otherwise revise/pivot/stop it.

**F2 — K-shuffle control.** Specify nonce/seed, canonical eligible-row ordering,
stratum definition, deterministic permutation algorithm, and collapse/invalid
stratum behavior. It must preserve the K histogram and cost while breaking the
content-to-budget relationship, without breach of the FIT/CAL/HOLD firewall.

## Frozen experimental decision contract—context only

Do not authorize execution in this turn. If Pro retains a route, its later
admission must preserve official THUMOS14/OpenTAD-AdaTAD architecture, losses,
NMS, evaluator, and video-disjoint training-population FIT/CAL/HOLD. Official
validation/test must remain inaccessible to selection. The six arms are:

1. `dense`; 2. `uniform_k384`; 3. `dynamic_A`; 4. `dynamic_B`; 5. `k_shuffle`;
6. `no_risk`.

All share the detector, optimizer, augmentation, RNG partitioning, 6,000-update
budget, terminal-EMA rule, pre-NMS physical mapping, and full-stack cost ledger.
O1 dynamic headroom at matched cost, O2 inner geometry/time validity, O3
train-only hard-utility predictability, O4 paired-boundary risk, F-INV timestamp
integrity, high-IoU/short-action behavior, and no hidden dense path are hard
kill gates. A real N16R4 pilot follows only a fresh accepted definition plus
structural implementation, Critic, Evaluator PRE_RUN, data/firewall and resource
gates; formal training additionally requires pilot support. This packet grants
none of those actions.

## Evidence and negative boundary

The binding plan is
`.cvpr-pro-lab/role-returns/BUILDER_DUCA_DYNAMIC_INNER_MECHANISM_MINIMAL_CHANGE_PLAN-v001.md`.
The prior Route-B Critic found the original F1/F2 ambiguity. The old v008 Project
Source batch is `UNKNOWN_REMOTE_STATE / QUARANTINED` and cannot be used or
retransmitted. The root is dirty and `a6bdc084...` is not a DUCA revision.
Everything here is `STATIC_PLAN_ONLY / NOT_EXECUTED`: no code, test, data,
PRE_RUN, pilot, training, evaluation, metric, cost, performance result, or claim.
