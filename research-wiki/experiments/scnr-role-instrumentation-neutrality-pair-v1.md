---
type: experiment
node_id: exp:scnr-role-instrumentation-neutrality-pair-v1
title: "SCNR role-instrumentation same-GPU neutrality pair v1"
stage: implemented
status: same_gpu_serial_pair_pending_remote_validation
outcome: pending
added: 2026-08-06
updated: 2026-08-06
---

# SCNR role-instrumentation same-GPU neutrality pair v1

## Purpose

Separate role-calibration instrumentation effects from ordinary frozen-checkpoint
rerun drift after Jobs `1223640/1223641` failed source raw prediction-SHA parity.
This is an integrity experiment, not a model comparison and not a repair of the
Hybrid mechanism.

## Frozen pair

Run exactly one M2 arm per Slurm job and exactly one visible GPU. Inside that job,
run two checkpoint-inference evaluations serially:

1. `role_off`: retain the source formal diagnostic telemetry, disable only the
   nested role-calibration extension;
2. `role_on`: enable that extension.

Both runs share the frozen M2 config/checkpoint, seed, 136-window development
population, evaluator/NMS path, exact `B=24576`, formal telemetry, `profile=false`,
environment, node and physical GPU. Training, resume, cost profiling, official
test, GT/teacher routing and result-based selection are absent.

## Gates

- Validate source config/checkpoint/prediction hashes, source/runtime commits,
  source population SHA-256 and dataset count before either run.
- Require both runs to emit the formal no-leak telemetry schema over the exact
  population; OFF must omit `policy_calibration`, ON must contain it.
- Require no profiler artifact in either arm.
- Compare source/OFF, source/ON and OFF/ON predictions by raw SHA, semantic JSON
  equality and exact candidate identity without reading performance metrics.
- Hard-fail unless OFF and ON raw prediction SHA-256 are identical.
- Never summarize or expose ON role statistics inside this runner.

## Interpretation

| Pair result | Meaning | Authorized action |
| --- | --- | --- |
| OFF != ON | instrumentation is not neutral | debug instrumentation; role evidence remains closed |
| OFF == ON, source mismatch | same-job causal neutrality passes, but historical source replay drift remains | record integrity diagnosis only; original frozen role analysis remains closed pending an explicit contract decision |
| OFF == ON == source | both neutrality and original source parity pass | analyze the already registered role-calibration statistics, then select one minimal separately trained repair |

No outcome is mAP, cost, floor-causality, operational-Hybrid, M3, official-test or
paper evidence.

## Implementation

- `tools/bata/run_georoute_role_instrumentation_pair.py`
- `scripts/run_georoute_role_instrumentation_pair_slurm.sh`
- focused tests in `tests/test_georoute_dynamic_role_calibration.py`

Local `py_compile`, Bash syntax and whitespace checks pass. Windows Torch test
collection remains blocked by the known `c10.dll` load error. Clean N16R4 tests,
preflight and Slurm execution are pending.
