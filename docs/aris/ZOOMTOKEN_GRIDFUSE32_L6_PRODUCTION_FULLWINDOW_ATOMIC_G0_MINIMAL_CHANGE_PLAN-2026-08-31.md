# ZoomToken GridFuse32-L6 production-full-window atomic G0 minimal-change plan

## Authority and immutable identity

- Fresh exact-Project Pro decision: `REVISE / STOP_CURRENT_8-TUBELET_SEGMENT_G0_PROTOCOL / CONTINUE_GRIDFUSE32_L6_ONCE_WITH_PRODUCTION-FULL-WINDOW_ATOMIC_G0`.
- Conversation: `6a949bec-1334-83ea-b410-a47ecdd451f7`.
- Unique task: `ZOOMTOKEN-GRIDFUSE32-L6-PRODUCTION-FULLWINDOW-ATOMIC-G0-v001`.
- Execution base: `b5993faaaa59be318557ca314697e38c4b39b6a1`.
- Candidate branch: `codex/zoomtoken-gridfuse32-l6-v001`.
- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>.
- Execution-base commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b5993faaaa59be318557ca314697e38c4b39b6a1>.

The exact final candidate will be a minimal clean, pushed descendant on the same branch. Its new exact commit URL replaces the execution-base URL in Critic, Evaluator, formal-run and post-terminal Pro materials.

## Scientific and protocol boundary

The candidate mechanism, model/config/checkpoint, pairing/fusion/broadcast, block split, dtype, resources, warmup/iterations and G0 gates do not change. The only change is to remove the disproven protocol assumption that one 8-tubelet VideoMAE attention bucket can stand in for the production 384-tubelet Adapter window.

The final atomic G0 constructs one real full window with `B=1`, `T=384`, `K=64`, `N=24,576`, 48 attention buckets, dense bucket size 512 and candidate bucket size 256. All 384 tubelets have valid strictly increasing native lineage; no fake metadata, padding, dummy, zero fill or omitted tubelets are allowed. Dense and candidate arms reuse the identical full-window input and lineage.

## Allowed implementation surface

Only these files may change:

1. `tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py`
   - generalize lineage/input preparation from one bucket to 384 tubelets and 48 deterministic buckets;
   - execute each final-six VideoMAE block across all 48 buckets, restore the N24,576 carrier, then execute the unchanged temporal-384 dense Adapter once per block;
   - preserve shared production preparation, strict epoch-59 EMA load, fp16 autocast and dense/candidate equivalence;
   - fail closed on the Pro-frozen full-window ledgers before any timing or memory measurement;
   - in the same invocation, run 100 alternating warmups, 500 alternating timed samples per arm, peak allocated/reserved memory and the unchanged G0 gates;
   - emit exclusive profile/terminal receipt for pass, valid gate failure or blocker.
2. `tests/test_zoomtoken_gridfuse32_l6.py`
   - assert production Adapter temporal size 384 and stable rejection of the old 8-tubelet production call;
   - add a small-embed full-window 384-tubelet Adapter+GridFuse forward;
   - prove 48 buckets cover all 24,576 native tokens exactly once without padding;
   - assert the exact dense/candidate full-window ledger and atomic launcher contract;
   - retain all existing GridFuse/R1/strict-rectangle regressions.
3. `scripts/run_zoomtoken_gridfuse32_l6_gated_n16r4.sh`
   - remove the obsolete standalone GPU construction-witness action from the final path;
   - keep one `G0` action that performs construction and measurement atomically in the same process/job;
   - preserve one GPU, four CPUs, two-hour walltime, exact source/config/checkpoint and exclusive result root.

Forbidden files and changes include `vit_adapter.py`, every config, Adapter temporal size, model/checkpoint/data/evaluator/NMS, GridFuse mechanism, G1/G2/training/prediction/metric/energy/full-stack code, compile/CUDA graph/candidate-only optimization, K/layer/pairing/threshold sweeps, and any replacement or third submission.

## Frozen full-window ledgers

Dense final six must report: 288 attention bucket calls; 147,456 attention/KV/MLP tokens; 75,497,472 attention pairs; 147,456 Adapter tokens.

GridFuse final six must report: 288 GridFuse bucket calls; 73,728 attention/KV/MLP tokens; 18,874,368 attention pairs; 147,456 Adapter tokens; 24,576 restored native tokens per block.

Any mismatch stops before timing.

## Verification and one final action

1. Local diff/compile/launcher checks and focused tests where the Windows environment permits.
2. Exact clean N16R4 GridFuse, R1-regression and strict-rectangle suites.
3. One fresh change-surface Critic; only `PASS` proceeds.
4. One fresh result-blind Evaluator; only `PRE_RUN_READY_ATOMIC_FULLWINDOW_G0` proceeds.
5. Exactly one final scheduler-ordinal-2 / scientific-measurement-attempt-1 Slurm job, one GPU, four CPUs, two-hour walltime. No separate GPU witness and no replacement.

Success is `GRIDFUSE32_L6_PRODUCTION_FULLWINDOW_G0_PASS_PENDING_FRESH_PRO`; a complete gate failure is `STOP_GRIDFUSE32_L6_EXACT_ROUTE_VALID_G0_NEGATIVE`; any construction/execution/artifact blocker is `STOP_GRIDFUSE32_L6_EXACT_ROUTE_FINAL_EXECUTION_BLOCKER`. Every terminal returns to a fresh exact-Project Pro within PT1H, and every Pro handoff includes the then-latest repository, branch and exact commit links.

## Beijing deadlines

- Role rules sync: `2026-08-31T06:00:00+08:00`.
- Builder plan: `2026-08-31T06:15:00+08:00`.
- Candidate: `2026-08-31T09:00:00+08:00`.
- Critic: `2026-08-31T10:00:00+08:00`.
- Evaluator: `2026-08-31T10:45:00+08:00`.
- Formal action: `2026-08-31T11:30:00+08:00`.
- Queue start/blocker: `2026-09-01T08:00:00+08:00` / `08:30:00+08:00`.
- Scientific return: within `PT1H` of terminal and no later than `2026-09-01T12:00:00+08:00`.
