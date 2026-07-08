---
updated: 2026-07-09
status: active
scope: Record the DUCA-JCT progressive collaborative training implementation, detector-gradient bridge refinement, and remote experiment deployment state.
out-of-scope: Final mAP numbers; result interpretation; paper claims before full runs complete.
---

# DUCA-JCT Progressive Deployment

## Code State

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`
- Branch: `codex/gas-vt-stage23-detector-aware-20260706`
- Base deployment commit: `308088c`
- Base deployment title: `Add DUCA joint training loss schedule`
- Current local refinement: detector loss remains active for the backend while
  the detector-to-selector ST gradient bridge is scheduled separately.
- Current dynamic-budget refinement: DUCA-MUST now updates the budget
  controller dual variable after each optimizer step from the controller's
  expected selected-cost mean, closing the primal-dual budget constraint loop.
- Current experiment-suite entrypoint:
  `scripts/submit_duca_jct_experiment_suite.sh`
- Current experiment-suite monitor:
  `scripts/monitor_duca_jct_experiment_suite.sh` and
  `tools/bata/monitor_duca_jct_experiment_suite.py`
- Current paper-evidence collector:
  `scripts/collect_duca_jct_paper_evidence.sh` and
  `tools/bata/collect_duca_jct_paper_evidence.py`
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
- `loss_schedule_detector_loss_always_on = true`
- `loss_schedule_detector_gradient_start = 0.0`
- `loss_schedule_detector_gradient_end = 1.0`
- `loss_schedule_actionness_start = 1.0`
- `loss_schedule_actionness_end = 0.25`
- `loss_schedule_hole_start = 0.0`
- `loss_schedule_hole_end = 0.05`

For DUCA-MUST, the dynamic-budget loss is also scheduled:

- `loss_schedule_lagrangian_budget_start = 0.0`
- `loss_schedule_lagrangian_budget_end = 1.0`
- `dynamic_budget_dual_update_after_optimizer_step = true`
- `dynamic_budget_dual_update_source = dynamic_must_expected_cost`

This means the intended training sequence is continuous within one run:

1. Coarse actionness supervision dominates early training.
2. The detector backend trains from the start, while the detector-to-selector
   ST/surrogate gradient bridge and selection distribution losses are gradually
   enabled.
3. Dynamic budget regularization is gradually enabled for DUCA-MUST; after
   each optimizer update, the budget controller updates its dual variable from
   the observed expected cost, so the learned `K(x)` is governed by a budget
   constraint rather than a forced budget curve.

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

The newer suite runner writes a fresh run root and submits:

- focused DUCA-JCT tests and validators;
- DUCA-JCT fixed-384 official full train;
- DUCA-MUST dynamic official full train;
- train-free X3D interval grid with pre-registered formal JSONL materialization;
- X3D -> DUCA fixed-384 official downstream full train;
- X3D -> DUCA-MUST dynamic official downstream full train.

The X3D downstream jobs depend on the X3D grid job via `afterok`, and read the
formal `best_x3d_actionness.jsonl` materialized by
`tools/bata/materialize_trainfree_x3d_actionness.py`.

After submission, the monitor should be run against the suite
`deployment_summary.json`:

```bash
bash scripts/monitor_duca_jct_experiment_suite.sh /path/to/deployment_summary.json
```

The monitor records Slurm state, log failure markers, materialized X3D
readiness, result artifacts, and missing-result gates in
`duca_jct_suite_monitor.summary.json`. It does not turn a finished job into a
paper metric claim unless detector result artifacts or logged mAP values are
actually present.

After the monitor summary exists, the paper-evidence collector can build the
method table and claim gate:

```bash
bash scripts/collect_duca_jct_paper_evidence.sh \
  /path/to/duca_jct_suite_monitor.summary.json \
  /path/to/matched_baselines.json
```

The collector writes `duca_jct_paper_evidence.summary.json` and
`duca_jct_paper_evidence.table.tsv`. It keeps `paper_claim_allowed=false` if
matched baselines are missing, detector result artifacts are missing, Avg-mAP
does not clear the configured delta, DUCA/X3D train-free downstream runs are
incomplete, or mAP@0.6/0.7 is absent or worse than the matched baseline.

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
- DUCA-MUST primal-dual budget update hook is implemented through the generic
  training-engine `after_optimizer_step` callback.
- DUCA-JCT suite monitoring is implemented for Slurm state, failure logs,
  formal X3D actionness readiness, and missing detector-result artifacts.
- DUCA-JCT paper-evidence collection is implemented for method rows, X3D
  train-free baselines, high-IoU mAP gates, and matched-baseline claim locks.
- Train-free X3D downstream configs kept separate from the main method.
- Main fixed-384 and DUCA-MUST full runs queued for commit `308088c`.

Still pending:

- Main full-run completion and mAP collection.
- X3D grid completion and production of the formal X3D actionness JSONL.
- X3D downstream detector jobs entering Slurm after submit limit clears.
- Full result analysis and paper-claim audit.
