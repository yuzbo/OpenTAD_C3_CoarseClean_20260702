---
type: experiment
node_id: exp:scnr-dynamic-stage1-executor-v1
title: "SCNR-TAD dynamic exact-budget ragged executor v1"
stage: designed
status: approved_implementation_in_progress
outcome: no_result
added: 2026-08-02
updated: 2026-08-02
---

# SCNR-TAD dynamic exact-budget ragged executor v1

## Purpose

Build the first executable Stage-1 form of the approved ROI + TokenSelect
Hybrid.  The experiment is an implementation and no-performance admission
gate; it cannot produce mAP, efficiency, floor-optimality, or paper evidence.

## Frozen model contract

- The decision unit is one native two-frame VideoMAE tubelet patch.  A window
  has `T=384` tubelets and the production source lattice has `N=11*20=220`
  physical candidates per tubelet.
- One global constrained projection selects exactly the configured window
  budget `B` unique physical `(t,n)` candidates.  There is no independent
  count head, fixed context quota, per-tubelet quota, padding repair, or dummy
  heavy token.
- The policy has one shared base utility and two modifiers:
  `u_hard=q_base+max(0,delta_roi,delta_res)`.  The same argmax assigns context,
  ROI, or residual role; role IDs do not alter heavy execution or pooling.
- The backward relaxation is
  `u_soft=q_base+tau*logsumexp((0,delta_roi,delta_res)/tau)`.  A global sigmoid
  threshold projection over valid physical candidates produces strict soft
  probabilities with `sum p=B`.  Hard forward membership remains exact top-B.
- The main route permits `K_t=0`.  Its carrier is an exactly zero heavy feature
  plus an explicit boolean heavy-valid mask.  Empty tubelets and empty clips
  execute no patch embedding, attention, MLP, or Adapter token.
- The ROI floor is runtime `(1/W_grid,1/H_grid)` with no full-frame, area,
  coverage, smoothness, expected-cost, fixed-context, or fixed-`K_t` loss.

## Ragged execution contract

Selected physical indices are sorted only after hard top-B.  Each selected
token carries `(batch,tubelet,clip,local_tubelet,spatial_index)` provenance.
Patch embedding runs once on the flat selected-token tensor.  Within every
VideoMAE block, non-empty clips are grouped only with clips having the same
true token count; every bucket executes attention and MLP without padding.
The coordinate-lineage Adapter operates on flat selected tokens and looks up
only identical spatial indices in adjacent global tubelets.  Missing neighbors
contribute zero.

The executor records per-window `K_t`, per-clip `b_c`, `sum_c b_c^2`, the
number and sizes of real ragged buckets, executed patch tokens, attention
pairs, MLP tokens, Adapter tokens, and the requested/unique/padded/executed
counts separately.  `padded=0` is a hard invariant, not a reporting choice.

## Compatibility and failure behavior

The new route is opt-in under a new dynamic mode.  Legacy dense, fixed-K,
structured `8/28/28`, packed Adapter, P0, checkpoint, and audit schemas retain
their existing code paths.  Dynamic execution fails closed on an invalid
budget, duplicate or unsorted physical index, out-of-grid provenance, a soft
budget residual above tolerance, a nonzero padded count, an unmasked empty
tubelet, or any mismatch between selected and executed tokens.

## Implementation milestones

| Milestone | Scope | Required evidence | State |
| --- | --- | --- | --- |
| D0 | Pure global allocator and differentiable exact-sum soft projection | known answers, uniqueness, dynamic roles, finite dense gradients | designed |
| D1 | Native ragged VideoMAE + coordinate-lineage Adapter | zero-padding, empty-clip, full-token parity, exact ledger KATs | designed |
| D2 | Wrapper integration, masked-zero aggregation, scout stop-gradient and proxy schedule | one real detector backward, successful-step schedule, no-leak audit | blocked on D0/D1 |
| D3 | Clean N16R4 Linux/CUDA no-performance P0 | exact source, clean tree, Slurm GPU, zero metric/checkpoint | blocked on D2 |
| D4 | Matched development G1/G2 floor arms | only after D3 PASS | blocked |

## Current boundary

The specification is approved through the user's prior section-by-section
decisions and the explicit instruction to execute when the route is clear.
Until D3 passes, the dynamic Stage-1 route is `designed`, not implemented,
tested, empirically supported, or eligible for performance training.
