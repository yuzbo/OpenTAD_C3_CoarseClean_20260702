# ZoomToken GridFuse32-L6 G0 construction-witness replacement minimal-change plan

## Authority and immutable identity

- Fresh exact-Project Pro decision: `REVISE / CONTINUE_ONCE_WITH_EXACT-CONSTRUCTION-WITNESSED_G0_REPLACEMENT`.
- Conversation: `6a9494ad-dab4-83ea-83f6-e9cc2fabc722`.
- Unique task: `ZOOMTOKEN-GRIDFUSE32-L6-G0-CONSTRUCTION-WITNESS-AND-RPL1-v001`.
- Execution base: `0b734ab839973b2c945b012f066db8222d235bb9`.
- Candidate branch: `codex/zoomtoken-gridfuse32-l6-v001`.
- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>.
- Reviewed base commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0b734ab839973b2c945b012f066db8222d235bb9>.

The repair candidate will be a minimal clean, pushed descendant on the same branch. Its exact commit URL replaces the reviewed-base URL in every subsequent Critic, Evaluator, formal-run, and fresh-Pro handoff.

## Observed blocker and repair boundary

Formal job `1262090` stopped before checkpoint load, warmup, timing, memory measurement, prediction, metric, gate evaluation, or parameter update because `Rearrange` was absent from the mmengine transform registry during detector construction. This is a result-blind construction failure and supplies no GridFuse performance evidence.

The production training and test entrypoints initialize the existing OpenTAD transform registry by importing `opentad.datasets` before `build_detector`. The segment profiler must use that canonical package initialization in the same Python process. It must not bypass or delete the configured preprocessing pipeline, copy transform classes, modify dataset or transform source, or manually or forcibly register entries.

## Allowed implementation surface and symbols

Only these scientific-code files may change:

1. `tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py`
   - add one shared production preparation function;
   - initialize the canonical OpenTAD dataset/transform package before detector construction;
   - perform `Config.fromfile`, frozen route assertions, `build_detector`, exact epoch-59 `state_dict_ema` strict load, 12-block/final-six-Adapter binding, and one untimed, unmetered, no-prediction/no-metric dense/candidate dry ledger assertion;
   - return the prepared model, blocks, real-shape inputs, lineage, execution closure, and witnessed ledgers for the unchanged formal profiler.
2. `tests/test_zoomtoken_gridfuse32_l6.py`
   - prove in an isolated subprocess that the old construction sequence without canonical transform initialization reproduces the registry failure;
   - prove the production preparation function resolves `Rearrange`, `Reduce`, and `Interpolate`, constructs the detector, strictly loads the exact epoch-59 EMA with no missing or unexpected keys, and produces the frozen N512/N256 six-block ledger without timing or performance output.
3. `scripts/run_zoomtoken_gridfuse32_l6_gated_n16r4.sh`
   - make `PRECHECK_ONLY=1` invoke the same production construction-witness function and stop after the witnessed dry ledger;
   - preserve the G0 command, one-GPU/four-CPU resource semantics, checkpoint, shapes, dtype, warmup, iterations, alternating order, and gates.

Forbidden changes include `opentad/models/backbones/vit_adapter.py`, the GridFuse config, dataset/transform source, the full-stack profiler, `tools/train.py`, model/checkpoint/data/shape/dtype/resource/warmup/iteration/order/gate semantics, manual registry mutation, G1, and G2.

## Checks that change the next action

The exact final candidate must pass:

1. Python compilation and launcher `bash -n`;
2. the focused GridFuse suite, R1 regression suite, and strict-rectangle regression suite;
3. the isolated old-path negative subprocess;
4. the exact N16R4 production construction witness with canonical registry initialization, detector construction, strict epoch-59 EMA load, and the dry ledger:
   - dense final six blocks: Q/K/V/MLP `6 x 512`, Adapter `6 x 512`;
   - candidate final six blocks: Q/K/V/MLP `6 x 256`, Adapter `6 x 512`;
5. exact clean local and remote Git identity, including the pushed branch and full commit;
6. one fresh change-surface Critic and one fresh result-blind Evaluator.

Only the exact Evaluator verdict `PRE_RUN_READY_G0_REPLACEMENT` opens formal execution.

## Submission and return discipline

- Job `1262090` remains scheduler ordinal 1 and is never retried or reinterpreted.
- The new task permits exactly one scheduler ordinal 2 / G0 measurement-attempt ordinal 1 replacement with one GPU, four CPUs, and a two-hour walltime.
- No third scheduler submission exists.
- A construction-witness failure, replacement engineering blocker, or valid G0 gate failure returns to a fresh exact-Project Pro conversation without G1, G2, rescue, tuning, or retest.
- A valid G0 pass only opens fresh Pro adjudication; it does not automatically authorize G1.
- Every fresh Pro prompt must include the then-latest repository, branch, and exact implementation commit URLs.

## Beijing deadlines inherited from Pro

- Builder plan: `2026-08-31T05:45:00+08:00`.
- Candidate: `2026-08-31T08:30:00+08:00`.
- Critic: `2026-08-31T09:30:00+08:00`.
- Evaluator construction witness: `2026-08-31T10:45:00+08:00`.
- Formal action: `2026-08-31T11:30:00+08:00`.
- Queue check/blocker: `2026-09-01T08:00:00+08:00` / `08:15:00+08:00`.
- Scientific return: `2026-09-01T12:00:00+08:00`.
- Every terminal outcome or objective blocker returns to fresh Pro within `PT1H`.
