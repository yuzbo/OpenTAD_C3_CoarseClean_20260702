# BUILDER_PJST_D1_CYCLE4_RUNTIME_CORRECTION-v001

- role: Builder / Executor
- process_id: aris-duca-pjst-d1-cycle4-runtime-20260826
- project_root: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`
- writable_workspace: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-builder-20260826`
- frozen_parent: `a16a67c4f74ce19de640704c357850c0e7b85ba3`
- accepted_science: `PJST-D1 derivative-only physical Jacobian at the first VideoMAE temporal mixing; frozen/replayed H65 selector; K=384; matched OFF/ON representation attribution`
- evidence_boundary: implementation correction only; no data, metric, GPU, Slurm, experiment, or efficacy claim

## Current scientific question

Can derivative-only physical time scaling at the first VideoMAE tubelet mixing preserve the exact H65 selected RGB set while improving representation of irregular physical spacing? This task must not change that frozen question.

## Required Builder action

Return a `MINIMAL_CHANGE_PLAN` first, then complete the smallest runnable correction in the same invocation.

Only the following two deterministic test-contract defects are in scope:

1. `tests/test_duca_pjst_d1_derivative_only.py::test_constant_pair_invariance` makes only pair 0 constant but asserts invariance over all eight pairs. Correct the assertion/fixture so it tests exactly the constant pair without weakening the production formula.
2. `_RecordingBackbone` in the same file does not implement the production detector call shape. Make the test double accept the real `(x, masks=None, metas=None, **kwargs)` contract and record `metas`, so both metadata-forwarding tests exercise production behavior.

## Hard scope

- Allowed implementation file: `tests/test_duca_pjst_d1_derivative_only.py` only.
- Do not edit production model, config, launcher, schedule, selector, split, metric, threshold, claim, or evidence code.
- Do not add a framework, contract generator, new entry point, or documentation chain.
- If a focused test exposes a genuine production defect rather than the known fixture defects, stop with `NEEDS_ATTENTION`; do not widen scope.

## Required checks

From the clean writable workspace:

```bash
python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q -rs
git diff --check
git status --porcelain=v1
```

Commit the minimal change on `codex/duca-pjst-cycle4-builder-20260826`, leave the worktree clean, and return:

- parent and candidate commit;
- exact changed paths and diff summary;
- exact commands and outcomes;
- confirmation that production/model/config/launcher files are unchanged;
- evidence class `STATIC_NO_DATA_IMPLEMENTATION_CORRECTION`;
- `next_owner=independent Critic`;
- `dependency=clean frozen candidate`.

