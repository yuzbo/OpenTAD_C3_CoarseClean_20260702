# ZoomToken GridFuse32-L6 G0 terminal receipt

## Frozen identity

- Task: `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`
- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>
- Exact candidate: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0b734ab839973b2c945b012f066db8222d235bb9>
- Remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_src_0b734ab8`
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_gated_0b734ab8_seed42_20260901`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_logs_0b734ab8_20260901`

## Slurm terminal

- Job: `1262090`, `zt-gf32-l6-g0`
- State / exit: `FAILED / 2:0`
- Submit / start / end: `2026-08-31T04:27:53+08:00` / `04:28:05+08:00` / `04:28:20+08:00`
- Elapsed / node: `00:00:15` / `g0030`
- Allocation: one GPU, four CPUs, `62200M`, walltime `02:00:00`
- Formal G0 submission count: `1/1`

## Exact blocker

The profiler loaded the model config and reached `build_detector`, then failed while `BackboneWrapper` constructed `custom.pre_processing_pipeline`. `mmengine.dataset.Compose` attempted to build `Rearrange` and raised:

```text
KeyError: 'Rearrange is not in the mmengine::transform registry.'
```

The remote terminal receipt records:

- schema: `zoomtoken_gridfuse32_l6_g0_terminal_v001`;
- status: `GRIDFUSE32_L6_G0_ENGINEERING_OR_PROTOCOL_BLOCKER`;
- error type: `KeyError`;
- terminal JSON: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_gated_0b734ab8_seed42_20260901/g0/terminal_receipt.json`.

No warmup, alternating timed iteration, memory measurement or gate evaluation started. `profile.json` is absent. The two stdout lines about parameter counts and random initialization occurred during construction and are not performance evidence; the exact epoch-59 EMA checkpoint had already passed precheck and was not evaluated here.

## Scientific boundary

This terminal state is an engineering/protocol blocker, not positive or negative GridFuse efficiency evidence. The p50, p95, allocated-memory and reserved-memory gates are all unevaluated. G1 and G2 remain closed. The candidate is not repaired or rerun before one fresh exact-Project Pro independently adjudicates this terminal outcome and issues exactly one next task with Beijing deadlines.
