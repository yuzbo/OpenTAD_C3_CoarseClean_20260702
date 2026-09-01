# EVALUATOR_PJST_D1_CYCLE3_N16R4_PRERUN-v001

status: PRE_RUN_BLOCKED
evaluated_commit: cbefa51563adce5c512403695259f2fcb3da16fa
evaluated_branch: codex/duca-pjst-cycle3-builder-20260826
remote_checkout: /data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle3_cbefa515_20260826
tree_status: clean; HEAD exact commit
formal_jobs_submitted: none

## Evidence

Canonical environment was loaded on N16R4 (`cuda/11.8`, `miniforge3/24.11`,
`opentad` conda environment, `PYTHONNOUSERSITE=1`). The exact checkout was
fetched/checked out detached at the required commit and had empty
`git status --porcelain`.

Command:

```text
python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q
```

Result: `20 passed, 10 failed` in `101.57s`, exit status 1.

Blocking failures (remote paths/lines from traceback):

- `opentad/models/utils/temporal_grid.py:197`, `apply_pjst_derivative_only`: `torch.where` receives incompatible dimensions (`The size of tensor a (8) must match the size of tensor b (3) at non-singleton dimension 2`). This fails `test_uniform_mixed_batch_byte_identity`, `test_invalid_partial_pair_byte_identity`, `test_explicit_formula`, `test_constant_pair_invariance`, `test_gap_scaling_halves_derivative`, `test_dtype_restoration`, and `test_finite_nonzero_input_gradient`.
- `tests/test_duca_pjst_d1_derivative_only.py:294` and `:305`: `SingleStageDetector.__new__` cannot assign a module before `Module.__init__`, failing `test_single_stage_metas_reach_backbone` and `test_single_stage_forward_train_passes_metas`.
- `tests/test_duca_pjst_d1_derivative_only.py:371`: post-processing source contains two occurrences of `_remap_selector_segments_for_post_processing(` where the exactly-once assertion requires one, failing `test_remap_occurs_exactly_once_before_filtering`.

The command also reached the required Linux Torch test execution; therefore
these are admission failures, not Windows skip artifacts. No validation/test
data, held-out data, mAP, or efficacy evaluation was opened.

Static checks were not used to override the failed focused gate. No formal
STAGE2_OFF/STAGE2_ON jobs were submitted.

## Disposition

current_scientific_question: Does the Critic-passed PJST-D1 Cycle-3 snapshot satisfy the Linux/N16R4 structural contract before matched Stage-2 training?
next_owner: Builder/implementer, then independent Critic recheck, then Evaluator.
next_action: Fix only the three evidenced structural blockers, produce a new clean exact commit, and rerun the focused Linux suite plus required PRE_RUN gates; do not submit formal training from this snapshot.
dependency: A new clean commit with zero focused-test failures and an independent Critic-passed checkpoint-binding review.
expected_return_at: After the corrected exact commit is available and its independent Critic recheck is recorded.
single_recovery: Preserve this receipt and the immutable checkout/logs; repair the PJST transform broadcasting, detector test construction compatibility, and exactly-once remap source contract in a successor commit, then re-evaluate from that successor only.

No scientific efficacy claim is made. No model/config code was edited by the Evaluator.
