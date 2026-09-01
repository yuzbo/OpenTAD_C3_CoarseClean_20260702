# Evaluator — PJST-D1 c73e8418 real initialization smoke and launch

Use exact clean commit `c73e8418` from local worktree `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-builder-20260826`. Create a fresh remote checkout `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle4_c73e8418_20260826` via exact bundle/SCP if needed; prove HEAD and empty porcelain. Do not reuse prior checkouts or output roots.

Use canonical N16R4 environment and the frozen Stage-1 checkpoint:

- `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- SHA256 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- epoch `29`, state `state_dict_ema`

Run the full existing PRE_RUN: py_compile, entire PJST focused test file with no fail/skip, validator, bash syntax, canonical dataset resolution, OFF and ON `PRECHECK_ONLY=1`.

Additionally perform an actual CPU model-initialization smoke for both resolved configs using the same production model builder and `initialize_model_from_checkpoint` path used by `tools/train.py`:

1. strict-load the frozen EMA state into each model; no missing/unexpected parameter is allowed;
2. assert `backbone.model.backbone.blocks.0.relative_physical_time_scale` exists and equals exact scalar zero after load in both;
3. assert `single_clock_admission is False` for both;
4. run the smallest representative forward/path smoke that proves no actual/canonical/relative physical-time tensor is admitted, while OFF keeps PJST pair metadata absent and ON admits only PJST pair metadata. Do not use data or claim efficacy.

If and only if all gates pass, immediately submit matched formal jobs using:

- `sbatch --account=sczc063 --partition=gpu --qos=normal`
- launcher `scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`
- seed 3407; identical Stage-1 path/SHA/epoch; distinct master ports
- fresh roots `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/{off,on}`

Return exact checks, state-load receipt, job IDs/states/log paths. Stop on any deterministic failure. No code/config/data/science edits and no metric interpretation.
