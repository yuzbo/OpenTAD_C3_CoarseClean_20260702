---
doc_id: CRITIC_DUCA_P0_C_PROJ_001_FOCUSED_STATIC_REREVIEW
version: v001
stage: DRAFT_P0_STATIC_CORRECTION
author_role: critic
parent_message_id: msg-20260812T133904Z-a53c61748824
parent_decision: PRO_P0_PROJECTION_POLICY-v001
parent_critic_return: CRITIC_DUCA_P0_PROJECTION_POLICY_STATIC_REVIEW-v001
parent_builder_return: BUILDER_DUCA_P0_C_PROJ_001_CORRECTION-v001
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
builder_snapshot: C:\Users\skywalker\.codex\worktrees\07c1\OpenTAD_C3_CoarseClean_20260702
verdict: C_PROJ_001_CLOSED
remaining_findings: NONE
scientific_ambiguity: NONE
evidence_class: BLOCKED_PRE_RESULT
execution: NOT_RUN
---

# Focused Critic re-review — C-PROJ-001

## Receipt and boundary

- Consumed only durable queue `msg-20260812T133904Z-a53c61748824` and its
  bounded cited decision, original Critic finding, Builder correction receipt,
  corrected decoder, and corrected static coverage.
- Frozen comparison revision:
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df`.
- Review surface was restricted to
  `decode_duca_density_positions_v001` in
  `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py` and
  `test_nonconstant_path_has_no_repair_fallback_or_second_decoder` in
  `tests/test_duca_p0_projection_policy.py`.
- This is static authored-code evidence only. No execution evidence is admitted.

## Focused attacks and evidence

1. **Explicit endpoints — conformant.** The decoder allocates the complete
   target vector and writes `targets[0] = 0.0` and
   `targets[-1] = float(valid_len - 1)` before constructing any quantile lookup
   (`pc_ot_mras_prebackbone_frame_selector.py:454-460`).

2. **Inverse-CDF lookup restricted to internal ranks — conformant.**
   `internal_ranks` is exactly `1..effective_k-2`; only its corresponding
   `internal_quantile_mass` is passed to `torch.searchsorted`, and interpolation
   is assigned only to `targets[1:-1]`
   (`pc_ot_mras_prebackbone_frame_selector.py:461-477`). Neither endpoint enters
   the inverse-CDF interval lookup.

3. **No clipping or replacement repair in the nonconstant decoder —
   conformant.** The inspected decoder contains no `.clamp(` and no endpoint
   overwrite after inverse-CDF lookup. It builds endpoints explicitly, fills
   only the internal slice, and invokes the frozen projector once
   (`pc_ot_mras_prebackbone_frame_selector.py:454-500`). The later fixed-point
   tuple construction preserves the already-explicit endpoint contract; it is
   not a lookup repair (`pc_ot_mras_prebackbone_frame_selector.py:481-490`).

4. **Out-of-range internal lookup is typed fail-closed — conformant.** Any
   internal interval index outside `0..valid_len-2` raises
   `DUCAProjectionError`; that typed error is re-raised, while unexpected errors
   are wrapped as `DUCAProjectionError`
   (`pc_ot_mras_prebackbone_frame_selector.py:473-500`).

5. **Static no-clipping coverage now inspects the decoder — conformant.** The
   test obtains source for both `project_duca_fixed_targets_v001` and
   `decode_duca_density_positions_v001`, then applies the same forbidden-token
   loop, including `.clamp(`, to both sources
   (`tests/test_duca_p0_projection_policy.py:397-414`).

## Verdict

`C-PROJ-001 IS CLOSED` within the authorized static-correction boundary.

The original finding remains correctly classified as an
`IMPLEMENTATION_CORRECTION`; the requested deterministic correction is present.
There are no remaining findings to classify and no `SCIENTIFIC_AMBIGUITY`.
Nothing in this focused correction changes the frozen objective, feasible set,
identity protocol, route, claim, split, metric, threshold, evaluator, or
evidence class.

Fairness/leakage verdict for this bounded change: unchanged; the inspected
decoder and static assertion add no data, GT, teacher, cache, evaluator, metric,
or result surface. No scientific route was created, selected, revised, or
advanced.

## Gate boundary

P0 remains `BLOCKED_PRE_RESULT`. This static closure does not discharge the
separately governed cross-implementation identity/optimality gate and does not
admit any result or paper claim.

`NO_EXECUTION_ATTESTATION`: text-only read inspection. No repository edit,
shell command, test, Python/import/model process, data/checkpoint/metric access,
CPU/GPU workload, Slurm, browser, network, Git, experiment, Pro, Sources, route,
claim, or evidence-promotion action was performed. The only write is this
explicitly requested durable Critic return.

`EVIDENCE_CLASS: BLOCKED_PRE_RESULT`
