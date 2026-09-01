# EVALUATOR_PJST_D1_CYCLE3_N16R4_PRERUN_RECHECK-v001

status: PRE_RUN_BLOCKED
evaluated_commit: a16a67c4f74ce19de640704c357850c0e7b85ba3
evaluated_branch: codex/duca-pjst-cycle3-builder-20260826
remote_checkout: /data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle3_cbefa515_20260826
tree_status: clean; HEAD exact commit
formal_jobs_submitted: none

## Linux focused gate

Environment: N16R4, `/etc/profile`, `cuda/11.8`, `miniforge3/24.11`,
`/data/run01/sczc063/yuzibo/conda_envs/opentad`, `PYTHONNOUSERSITE=1`.

Exact commands:

```text
git fetch origin codex/duca-pjst-cycle3-builder-20260826
git checkout --detach a16a67c4f74ce19de640704c357850c0e7b85ba3
git rev-parse HEAD
git status --porcelain
python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q -rs
```

Identity output: `HEAD=a16a67c4f74ce19de640704c357850c0e7b85ba3`; status was empty.

Focused result: `3 failed, 27 passed in 75.31s (0:01:15)`, exit 1. Therefore the
required zero-failure/zero-skip Linux gate fails and all remaining PRE_RUN gates
were intentionally not run; no formal OFF/ON jobs were submitted.

Raw failures:

1. `tests/test_duca_pjst_d1_derivative_only.py:160`,
`test_constant_pair_invariance`: `assert torch.equal(y, x)` failed. The
constant pair (`x[:,:,1] = x[:,:,0]`) is modified by
`apply_pjst_derivative_only` despite zero pair derivative.

2. `tests/test_duca_pjst_d1_derivative_only.py:300`,
`test_single_stage_metas_reach_backbone`: production
`opentad/models/detectors/single_stage.py:356`, `_call_backbone`, calls
`self.backbone(inputs, masks)`; `_RecordingBackbone.forward()` accepts only
2 positional arguments including `self`, causing
`TypeError: _RecordingBackbone.forward() takes 2 positional arguments but 3 were given`.

3. `tests/test_duca_pjst_d1_derivative_only.py:319`,
`test_single_stage_forward_train_passes_metas`: same production call and same
`TypeError` at `opentad/models/detectors/single_stage.py:356`.

The prior failed `cbefa515` receipt/evidence remains preserved at
`.cvpr-pro-lab/role-returns/EVALUATOR_PJST_D1_CYCLE3_N16R4_PRERUN-v001.md`.
This recheck consumed the third focused correction and closes Cycle-3 for an
equivalent deterministic implementation failure. No data, validation/test
evaluation, mAP, efficacy claim, or browser/Pro/Sources action was performed.

current_scientific_question: Does PJST-D1 Cycle-3 satisfy the Linux/N16R4 structural contract before matched Stage-2 training?
next_owner: Builder/implementer and project coordinator; no automatic patch request from this recheck.
next_action: Preserve this receipt and raw remote log; decide project-level closure or a separately authorized successor route. Do not submit Stage-2 from this commit.
dependency: A separately authorized clean successor must first eliminate the three exact Linux failures and pass the complete zero-failure/zero-skip focused gate, followed by all original PRE_RUN gates.
expected_return_at: Not applicable until a separately authorized successor is presented.
single_recovery: Keep the immutable `a16a67c4` checkout/log and prior `cbefa515` failed evidence; do not mutate either or reuse this commit for formal training.
