---
type: experiment
node_id: exp:scnr-residual-centering-paired-cost-v1
title: "SCNR residual-centering same-GPU paired full-stack cost v1"
stage: implemented
status: local_contract_tests_pass_remote_precheck_pending
outcome: pending
added: 2026-08-09
updated: 2026-08-09
---

# SCNR residual-centering same-GPU paired full-stack cost v1

## Purpose

Measure whether the seed-3407 residual-centering accuracy repair preserves
complete system cost relative to its fresh matched control. This experiment is
the authorized successor of
`exp:scnr-residual-centering-matched-training-v1`; it does not retrain either
checkpoint and does not alter the model.

## Frozen protocol

- source checkpoints: audited `none_control` and `residual_window_center`
  epoch-59 EMA artifacts from runtime `16137484`;
- execution: one Slurm Job, one physical GPU/logical `cuda:0`, world size one,
  batch one, four detector/input CPUs plus one sidecar CPU;
- order: `none, center, center, none, center, none, none, center`;
- pair map: `(0,1)`, `(3,2)`, `(5,4)`, `(6,7)` as control/center;
- 50 warmup samples per pass, no post-hoc pass/sample removal;
- one continuous independently scheduled 20-ms NVML sampler across all eight
  passes;
- complete Gate decode, preprocessing, H2D, scout/route, patch embedding,
  native ragged heavy backbone, adapter, projection/neck/head, postprocessing,
  gather/NMS, memory, energy, `K_t`, roles, and attention-pair ledger;
- diagnostic telemetry disabled inside timed forward; no GT, teacher, oracle,
  raw-prediction cache, official test, training, or resume.

## Frozen decision rule

Primary centered/control ratios are order-balanced end-to-end p50 latency and
mean energy per measured sample. The estimate is the geometric mean of four
paired-pass ratios. A deterministic 10,000-replicate 95% interval resamples
complete video clusters and counterbalanced pass pairs.

Cost is non-inferior only when both upper bounds are `<=1.05`. Together with
the already sealed positive mAP@0.6/mAP@0.7 and nonnegative Avg-mAP signs, this
opens only a separately frozen seeds-3408/3409 matched confirmation. Any
complete violation is HOLD; any missing stage/timer/power/hash/population/GPU
receipt is FAIL with no inference. A strict Pareto observation is recorded
separately but is not a paper claim.

## Implementation

- `tools/bata/georoute_residual_centering_cost_contract.py`: immutable source,
  config, deployment, raw-profile, clustered-bootstrap, and finalization gates;
- `tools/bata/profile_georoute_residual_centering_cost.py`: eight-pass same-GPU
  full-stack profiler reusing the repaired M2 timing primitives;
- `tools/bata/deploy_georoute_residual_centering_cost.py`: one-job held atomic
  deployment with explicit model-runtime/cost-execution separation and a
  sensitive-source Git diff gate;
- `tools/bata/finalize_georoute_residual_centering_cost.py`: same-job
  artifact-driven finalization;
- `scripts/run_georoute_residual_centering_cost_slurm.sh`: Slurm allocation,
  CPU isolation, torchrun rendezvous, profiler and same-job finalizer;
- `tests/test_georoute_residual_centering_cost.py`: order, config/no-retrain,
  bootstrap gate, deployment, finalizer, and launcher coverage.

Local compile and focused study tests pass `8/8`. The combined study,
matched-training, inherited M2, and required C3 regression matrix passes
`65/65`. Remote Linux/Torch checks remain pending before deployment.

## Claim boundary

No cost result exists yet. The implementation is not evidence. A future
single-job pass is a development promotion gate, not paper-grade efficiency;
independent repeated cost Jobs, multi-seed accuracy, matched baselines,
short-action/boundary analysis, second detector/dataset, and sealed official
test remain required.

Full preregistration:
`docs/superpowers/specs/2026-08-09-scnr-residual-centering-paired-cost-v1-design.md`.
