# DUCA-RIME Four-Phase Implementation Decision

Status: `user_approved / implemented / focused_tested / deployment_pending`

## Decision

The approved route is one staged adjudication, not four independent paper
models. Phase 1 closes execution and evidence semantics; Phase 2 creates a
mixed-budget baseline and causal admission protocol; Phase 3 creates the first
trainable DUCA-RIME candidate; Phase 4 freezes and validates that candidate.

## Phase outputs

| Phase | New technical output | Publication role |
|---|---|---|
| 1 | exact-K physical-time decoder, inverse mapping, controls, ledgers, full-stack profiler | correctness and measurement |
| 2 | stateless `U-mixed-K`, cross-fitted utility/risk evidence, O1–O4 gates, frozen budget protocols | causal admission and fair baseline |
| 3 | `RIME-full` plus five causal training controls and `U-same-K` replay | first candidate method |
| 4 | 12 fresh formal cells across ActionFormer/TriDet, K384/K192, and three seeds | transfer/stability decision |

## Two budget panels

K384 is the dynamic-allocation panel. It uses a frozen price and must realize
multiple K values. K192 is the minimum candidate budget; therefore it is forced
to exact K=192 and tests learned positions only. Calling K192 dynamic would be
mathematically inconsistent with a risk fallback to larger budgets.

## Cost matching

RIME full-stack cost is paired with `U-same-K`, which replays the exact
per-video realized-K sequence. `U-fixed` remains the primary fixed-budget
accuracy comparator. The profiler treats `effective_k` as the authoritative
executed heavy-frame count and only falls back to the legacy
`effective_budget` field.

## Formal provenance

Every formal Phase-4 RIME cell binds:

```text
authorization receipt
  -> exact Phase-2 receipt
  -> K-specific protocol path + SHA-256
  -> checkpoint training audit
  -> compact terminal checkpoint
  -> terminal evaluation identity
  -> localization/cost/comparison artifacts
  -> matrix seal
```

The official-final set remains unopened until the Phase-3 development GO.

## Evidence language

Passing local tests means `tested`. Successful submission means
`experiment_running`. Only completed, sealed, statistically passing Phase-4
evidence can support the registered paper claim.
