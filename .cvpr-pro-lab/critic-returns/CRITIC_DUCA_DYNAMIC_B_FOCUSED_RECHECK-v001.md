# Focused static recheck — DUCA Dynamic B

- **Frozen correction commit:** `3e551595f9ca151fa2625181f19b8447feec15bc`
- **Verdict:** `DYNAMIC_ROUTE_B_FOCUSED_STATIC_BLOCKED`
- **Disposition:** correction loop exhausted; `PRE_RUN_NOT_READY`.
- **Evidence class:** deterministic static helper execution and focused static review. This is not data, held-out, runtime, efficacy, or result evidence.

## Second equivalent locality defect

`opentad/models/selectors/duca_dynamic_physical.py:58-63` seeds selected anchor indices before any nonzero locality check. For the direct static input `bounded_monotone_local_exact_k([0, 1, 0, 1], 2, local_radius=1)`, it returns `[0, 3]`, whose separation is `3 > 1`. Thus the supposedly bounded/local physical exact-K mechanism still accepts a locality-violating anchor pair. `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:2547-2553` calls that helper on the dynamic-B path. The focused test suite passes but `tests/test_duca_dynamic_physical_contract.py:37-45` covers only the zero-radius branch, not this nonzero-anchor path.

The earlier correction did remove the prefix fallback, made zero radius fail closed, and honestly marked F2 as not executed on the dynamic-B claim path. Those changes do not close the second locality breach.

## Terminal handoff

- **next_owner:** Coordinator terminal hold
- **next_action:** do not make a third Builder correction or another Critic recheck; preserve the route as `BLOCKED_PRE_RUN` with no efficacy claim.
- **dependency:** a new explicit authority would be required before any scientifically distinct replacement could be considered.
- **expected_return_at:** none for this exhausted correction loop.
- **single_recovery:** none.
