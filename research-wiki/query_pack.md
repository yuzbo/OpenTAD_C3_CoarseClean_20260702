# DUCA-RIME Current Query Pack

Last updated: `2026-07-27`

## Current decision

The user approved direct execution of the four-stage DUCA-RIME adjudication.
The route is an **offline TAD pre-backbone acquisition plugin**. It is not
Online TAD, and it is not yet the paper's final method.

Current evidence level:

| Item | State |
|---|---|
| Scientific route | `user_approved` |
| Four-stage implementation | `implemented` |
| Focused local checks | `tested` |
| Remote authoritative code gate | `pending` |
| Four-stage Slurm DAG | `pending_submission` |
| DUCA-RIME empirical superiority | `not_yet_empirically_supported` |
| Paper-ready method | `not_yet_paper_ready` |

## Central research question

Can cheap, inference-visible action/state evidence estimate detector utility and
pair/boundary risk well enough to allocate real pre-backbone computation, while
an exact-K monotone physical-time decoder protects high-IoU temporal
localization and lowers measured full-stack cost?

## Frozen method semantics

1. The external detector grid is 768 candidate positions.
2. Candidate heavy budgets are `K=(192,256,384,512)`, quantum 16.
3. The heavy VideoMAE backbone receives exactly the selected effective K; no
   Kmax padding is allowed.
4. The detector backbone, projection, adapter, head, losses, and NMS remain the
   registered ActionFormer or TriDet backend.
5. Selection decisions may use only cheap inference-visible evidence. GT,
   teacher outputs, validation/test labels, raw-prediction caches, and
   counterfactual ledgers are forbidden at inference.
6. Predictions are mapped from the selected axis back to physical time before
   official evaluation and NMS.

## Four stages and what they produce

### Phase 1 — execution and geometry closure

Produces exact-K physical execution, dense/uniform/no-probe/probe controls,
coordinate round-trip audits, inference ledgers, and real cost instrumentation.
This is an algorithmic/evidence foundation, not a new final model.

### Phase 2 — trainable baseline and causal admission

Produces the probe-free `U-mixed-K` detector, whose per-video 60-epoch exposure
histogram is exactly `(8,12,16,24)` over `(192,256,384,512)`, hence mean K=384.
It also produces cross-fitted targets, counterfactual measurements, O1–O4
causal gates, and two frozen budget protocols. This is a new trainable baseline
and decision protocol, not the final DUCA-RIME model.

### Phase 3 — first DUCA-RIME candidate

Produces the first trainable candidate (`RIME-full`) and its causal arm matrix:
`U-fixed`, `F-bound`, `D-no-risk`, `AdapTok-TAD`, `D-shuffle`, plus evaluation-
only `U-same-K`. Every train arm has exactly 6000 successful detector updates.
Only a passing development receipt authorizes Phase 4.

### Phase 4 — frozen publication validation

Retrains and evaluates the frozen candidate over:

- detector: ActionFormer, TriDet;
- panel: K384, K192;
- fresh seed: 5801, 8123, 12011.

This produces 12 formal cells and a fail-closed matrix receipt. It does not
invent a fourth model; it determines whether the Phase-3 candidate is
empirically supportable and transferable.

## Budget-panel correction

- `K384`: `frozen_price_dynamic_budget`; content-conditioned dynamic allocation
  is allowed and must realize at least two requested K values.
- `K192`: `fixed_floor_budget_position_only`; all requested budgets are exactly
  192. Risk predictions may still supervise learned positions, but they do not
  allocate K. No dynamic-budget claim is allowed for this panel.

Reason: when 192 is the minimum candidate budget, a risk-triggered fallback to
larger K makes a mean-192 dynamic policy mathematically infeasible.

## Cost correction

Variable-K RIME is cost-matched against `U-same-K`, which replays the exact
per-video realized K sequence from the RIME inference ledger. `U-fixed` remains
the fixed-budget accuracy comparator. The profiler reads `effective_k` before
legacy `effective_budget`; otherwise RIME would be incorrectly reported as
dense K=768.

## Claim gate

A positive paper claim requires:

1. development Phase 3 passes before the official-final set is opened;
2. all 12 Phase-4 cells are present and hash-bound;
3. RIME beats both best fixed and uniform same-K under paired video-cluster
   bootstrap;
4. high-IoU, short-action, and pair-support non-degradation gates pass;
5. measured full-stack latency is below dense;
6. seed directions are positive for every detector/budget panel.

Until those artifacts exist, the correct status is `implemented/tested` or
`experiment_running`, never `empirically_supported` or `paper_ready`.

## Immediate execution

1. Finish local and remote code gates on the exact clean commit.
2. Freeze and hash all checkpoint/protocol prerequisites.
3. Submit the fail-closed four-stage Slurm DAG.
4. Record job IDs, dependencies, external run root, and exact commit in
   `experiments/duca-dynamic-k-rime-oracle.md` and `log.md`.
5. Monitor without changing the frozen method after Phase 4 authorization.
