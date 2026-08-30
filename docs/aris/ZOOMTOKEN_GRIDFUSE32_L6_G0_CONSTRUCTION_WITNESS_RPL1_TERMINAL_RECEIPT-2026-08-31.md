# ZoomToken GridFuse32-L6 G0 construction-witness RPL1 terminal receipt

## Frozen identity

- Unique task: `ZOOMTOKEN-GRIDFUSE32-L6-G0-CONSTRUCTION-WITNESS-AND-RPL1-v001`
- Execution base: `0b734ab839973b2c945b012f066db8222d235bb9`
- Exact clean/pushed candidate: `b5993faaaa59be318557ca314697e38c4b39b6a1`
- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>
- Exact commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b5993faaaa59be318557ca314697e38c4b39b6a1>
- Remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_src_b5993faa`

## Minimal implementation and checks

Only the three Pro-authorized files changed: the G0 segment profiler, its launcher, and the focused GridFuse test. The candidate initializes the canonical `opentad.datasets` transform registry, shares one production preparation function between witness and formal G0, strictly loads the exact epoch-59 `state_dict_ema`, binds the 12 blocks and final-six Adapters, and runs one untimed/unmetered/no-prediction/no-metric real-shape dry ledger.

Local `git diff --check`, Python compilation and launcher `bash -n` passed. Local pytest could not collect because the known Windows Torch `c10.dll` loader failed with WinError 1114; this was not counted as a pass. On the exact clean N16R4 checkout, GridFuse, R1-regression and strict-rectangle suites passed as `12 passed in 80.51s`, `12 passed in 41.73s`, and `8 passed in 0.76s`.

## Construction-witness terminal

- Slurm job: `1262099`
- Job name: `zt-gf32-l6-cw-rpl1`
- Resources: `gpu`, one GPU, four CPUs, 30-minute walltime
- Node: `g0063`
- Start/end: `2026-08-31T05:01:38+08:00` / `2026-08-31T05:03:30+08:00`
- State/exit: `FAILED 2:0`
- Elapsed: `00:01:52`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_logs_b5993faa_20260831`
- Intended unused result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_g0_cw_b5993faa_20260831`

The isolated old-path test reproduced the missing-registry failure, while the new canonical path constructed the real detector. In the Slurm witness, registry initialization, real config parsing, detector construction and strict checkpoint load completed. The first dense real-shape dry ledger then entered the final-six execution path and stopped in `vit_adapter.py` with:

```text
ValueError: ragged Adapter temporal axis differs from pretrained Adapter
```

The traceback is `construction-witness main -> prepare_gridfuse32_l6_g0 -> execute("dense") -> block.forward_native_ragged -> adapter.forward_native_ragged`. This occurred before candidate execution, timing, memory measurement, prediction, evaluator metric, gate evaluation, training or parameter update. No G0 performance evidence exists.

## Classification and frozen response

This is a second, independent construction/shape-contract blocker. The registry defect is fixed, but the production construction witness did not pass. Under the exact Pro task, the failure does not authorize a model/config/shape repair, fresh Critic/Evaluator, formal replacement, third scheduler submission, G1 or G2. The only authorized action is a fresh exact-Project Pro adjudication with the complete blocker and latest GitHub implementation identity.
