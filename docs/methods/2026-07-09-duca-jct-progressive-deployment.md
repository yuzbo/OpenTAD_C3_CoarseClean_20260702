---
updated: 2026-07-09
status: active
scope: Record the DUCA-JCT progressive collaborative training implementation and remote experiment deployment state for commit 308088c.
out-of-scope: Final mAP numbers; result interpretation; paper claims before full runs complete.
---

# DUCA-JCT Progressive Deployment

## Code State

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`
- Branch: `codex/gas-vt-stage23-detector-aware-20260706`
- Commit: `308088c`
- Commit title: `Add DUCA joint training loss schedule`
- Remote worktree: `/data/run01/sczc063/yuzibo/projects/opentad_stage23_308088c_20260709_jct`

This commit is the current implementation checkpoint for the single-run DUCA-JCT route:

```text
online coarse actionness probe
-> DUCA indirect temporal acquisition
-> selected-axis official ActionFormer/AdaTAD head
-> progressive joint loss schedule
```

## Training Design Locked By Validators

The main fixed-384 and DUCA-MUST configs both validate:

- `loss_schedule_policy = progressive_joint`
- `loss_schedule_shape = cosine`
- `loss_schedule_warmup_steps = 500`
- `loss_schedule_transition_steps = 4000`
- `loss_schedule_detector_start = 0.0`
- `loss_schedule_detector_end = 1.0`
- `loss_schedule_actionness_start = 1.0`
- `loss_schedule_actionness_end = 0.25`
- `loss_schedule_hole_start = 0.0`
- `loss_schedule_hole_end = 0.05`

For DUCA-MUST, the dynamic-budget loss is also scheduled:

- `loss_schedule_lagrangian_budget_start = 0.0`
- `loss_schedule_lagrangian_budget_end = 1.0`

This means the intended training sequence is continuous within one run:

1. Coarse actionness supervision dominates early training.
2. Detector loss and selection distribution loss are gradually enabled.
3. Dynamic budget regularization is gradually enabled for DUCA-MUST.

This is not a multi-stage p_action export pipeline.

## Verified Gates

Remote focused validation completed before deployment:

- `validate_duca_official_adatad_backend.py`: passed for fixed DUCA-384.
- `validate_duca_must_dynamic_official_adatad_backend.py`: passed for DUCA-MUST.
- `validate_duca_x3d_official_adatad_backend.py`: passed as train-free X3D downstream config without requiring JSONL existence.
- `validate_duca_must_dynamic_x3d_official_adatad_backend.py`: passed as train-free X3D downstream config without requiring JSONL existence.
- Focused pytest on remote: `25 passed in 67.36s`.

The remote pytest set covered:

- `tests/test_duca_joint_training_contract.py`
- `tests/test_duca_online_coarse_probe_actionness.py`
- `tests/test_duca_online_precheck_config.py`

## Remote Deployment

Deployment root:

`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_jct_progressive_308088c_20260709`

Slurm jobs currently queued:

- DUCA-JCT fixed-384 official full train: `1151134`
- DUCA-MUST dynamic official full train: `1151135`

These two are the current main paper-method jobs for commit `308088c`.

## X3D Train-Free Baseline

The X3D train-free downstream detector jobs are not yet in Slurm because the account hit `AssocMaxSubmitJobLimit`.

Current dependency:

- X3D grid job: `1151093`
- Intended downstream dependency: `afterok:1151093`
- Waiting submitter PID: `2125160`
- Waiting submitter state file:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_jct_progressive_308088c_20260709/x3d_wait_submitter_state.json`

The downstream configs remain baselines:

- `external_actionness_source = train_free_x3d_jsonl`
- `requires_external_actionness = True`
- `x3d_downstream_detector_full_train = True`

They must not be described as the main joint-trainable coarse-probe method.

## Known Queue Constraints

Slurm rejected initial submissions until the wrapper used:

- `#SBATCH --cpus-per-task=4`
- no explicit `#SBATCH --mem=...`

Further submission is currently blocked by:

`AssocMaxSubmitJobLimit`

No existing user jobs were cancelled.

## Current Completion Status

Achieved:

- Single-run progressive DUCA-JCT implementation.
- Fixed and dynamic main-method configs with strict schedule validators.
- Train-free X3D downstream configs kept separate from the main method.
- Main fixed-384 and DUCA-MUST full runs queued for commit `308088c`.

Still pending:

- Main full-run completion and mAP collection.
- X3D grid completion and production of the formal X3D actionness JSONL.
- X3D downstream detector jobs entering Slurm after submit limit clears.
- Full result analysis and paper-claim audit.
