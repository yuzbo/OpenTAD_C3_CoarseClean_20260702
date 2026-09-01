# Evaluator — PJST-D1 Cycle 4 dc260fad PRE_RUN and launch

Use exact local candidate `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-builder-20260826` at clean commit `dc260fad`.

Create a fresh exact remote checkout (do not modify or reuse the failed c195 checkout/output) under `/data/run01/sczc063/yuzibo/projects/`, preferably `duca_pjst_d1_cycle4_dc260fad_20260826`. GitHub TLS is unreliable, so a local exact Git bundle plus SCP is permitted. Verify remote HEAD=`dc260fad` and empty porcelain.

Canonical environment:

```bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1
```

Stage-1 identity:

- checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- SHA256: `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- epoch: `29`

PRE_RUN must include clean identity, py_compile, all of `tests/test_duca_pjst_d1_derivative_only.py` with zero failures/skips, validator PASS, `bash -n`, and both OFF/ON `PRECHECK_ONLY=1`. Also prove the resolved config has canonical annotation/class-map/video paths, not the former `$HOME/run/yuzibo` path.

If and only if PRE_RUN passes, immediately submit both formal jobs from the same clean checkout with:

- launcher: `scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`
- `sbatch --account=sczc063 --partition=gpu --qos=normal`
- OFF config `configs/adatad/thumos/duca_pjst_d1_stage2_off.py`
- ON config `configs/adatad/thumos/duca_pjst_d1_stage2_on.py`
- seed 3407, same Stage-1 path/SHA/epoch
- distinct master ports and output roots under `/data/run01/sczc063/yuzibo/duca_pjst_d1_dc260fad_20260826/{off,on}`

Return exact PRE_RUN results, resolved bindings, job IDs/states, revision, commands, outputs, and evidence class. Stop on any deterministic PRE_RUN failure. Do not change code, configs, data, science, or metrics; do not infer efficacy before terminal evaluation.
