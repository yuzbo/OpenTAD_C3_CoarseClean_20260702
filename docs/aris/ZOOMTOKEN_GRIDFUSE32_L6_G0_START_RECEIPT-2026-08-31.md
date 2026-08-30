# ZoomToken GridFuse32-L6 G0 start receipt

## Identity and GitHub provenance

- Task: `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`
- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>
- Exact commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0b734ab839973b2c945b012f066db8222d235bb9>
- Remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_src_0b734ab8`
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_gated_0b734ab8_seed42_20260901`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_logs_0b734ab8_20260901`

Immediately before submission, the login node freshly fetched
`refs/heads/codex/zoomtoken-gridfuse32-l6-v001` into the persistent
`refs/remotes/origin/codex/zoomtoken-gridfuse32-l6-v001`; the remote-tracking
ref, clean detached HEAD and expected SHA all matched the exact commit above.

## Precheck closure

- `1262078`: pre-execution GitHub DNS failure on the compute node; no tests or scientific action.
- `1262079`: same pre-execution GitHub DNS failure; no tests or scientific action.
- The launcher-only identity correction was pushed and independently reviewed.
- Final precheck `1262089`: `COMPLETED 0:0`, focused `9 passed`, marker `PRECHECK_READY`.
- Fresh exact-candidate Critic/Evaluator: `PASS / PRE_RUN_READY`.

## Sole formal G0 action

- Job ID: `1262090`
- Job name: `zt-gf32-l6-g0`
- Submission time: `2026-08-31T04:27:53+08:00`
- State at receipt: `PENDING`, reason `Priority`
- Slurm resources: partition `gpu`, one GPU, four CPUs, walltime `02:00:00`
- Frozen launcher: `scripts/run_zoomtoken_gridfuse32_l6_gated_n16r4.sh`
- Phase: `G0`; formal submission count `1/1`
- Terminal-only watcher: FastCtx job `j-lmy3cj`, 300-second real waits

G0 uses R1 epoch-59 EMA blocks 6--11, `B=1`, eight tubelets, dense N512 versus GridFuse N256, dimension 384, six heads and FP16. It performs 100 warmups and 500 alternating synchronized iterations per arm. Pass requires p50 speedup `>=1.35x` and allocated/reserved memory ratios each `<=1.05`; p95 is report-only. Running or partial values are not read. G1/G2 remain closed until complete terminal evidence is ingested.
