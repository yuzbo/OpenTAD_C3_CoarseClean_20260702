# Independent static review — DUCA Dynamic B

- **Frozen revision:** `9eb328f99c10c04240770d282aad2097384a6eb8`
- **Verdict:** `DYNAMIC_ROUTE_B_STATIC_BLOCKED`
- **Classification:** `IMPLEMENTATION_CORRECTION`
- **Evidence class:** static code review; no data, held-out, GPU, remote, training, inference, evaluation, metric, or efficacy evidence.

## Blocking findings

1. `opentad/models/selectors/duca_dynamic_physical.py:59-67` falls back to `candidates[:k]` when anchor locality cannot be met. That output can violate the frozen bounded/local physical exact-K contract.
2. `opentad/models/selectors/duca_dynamic_physical.py:62` treats `local_radius == 0` as unconditional acceptance, disabling the locality bound.
3. `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:2551-2553` reaches this helper for `dynamic_B`.
4. `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:2562` advertises F2 nonce/Fisher--Yates metadata without invoking the F2 transform on that path. The execution and metadata must be reconciled.

The existing focused tests do not exercise locality failure or dynamic-B F2 execution. The six-arm declarations and fail-closed launcher otherwise remain static-only and do not constitute PRE_RUN or performance evidence.

## Bounded handoff

- **next_owner:** Builder
- **next_action:** exactly one focused correction: preserve the locality contract by deterministic contract-preserving behavior or fail-closed rejection, and reconcile the F2 metadata with the actual control path; add only targeted static tests.
- **dependency:** none beyond the clean frozen worktree.
- **expected_return_at:** one clean focused Builder commit, then the same Critic's single focused recheck.
- **single_recovery:** consumed by that correction and recheck; an equivalent second deterministic defect terminates the loop.
