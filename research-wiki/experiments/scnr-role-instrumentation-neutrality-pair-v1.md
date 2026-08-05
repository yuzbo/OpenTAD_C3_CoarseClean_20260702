---
type: experiment
node_id: exp:scnr-role-instrumentation-neutrality-pair-v1
title: "SCNR role-instrumentation same-GPU neutrality pair v1"
stage: tested
status: neutrality_supported_source_replay_nondeterminism_localized
outcome: strict_triplet_exact_categorical_bridge_open
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

Clean N16R4 validation passed. Same-GPU R3 Jobs `1223686/1223687` proved OFF/ON
role instrumentation causal neutrality at the route surface, while both replay
predictions drifted from the historical source. Legacy OFF-A/OFF-B/ON Jobs
`1223707/1223708` showed OFF-A versus OFF-B drift as well, despite exact route
hash equality over `136/136` windows. The first downstream warning points to
memory-efficient CUDA SDPA in `vit_adapter.py` after routing.

Strict math-SDPA Jobs `1223727/1223728` emitted no nondeterminism warning and
made OFF-A, OFF-B and ON prediction files byte-identical: G1 SHA-256
`578860552cf02544253f88776bcc25b33d0ee3ecf7ab24a2de1079d9fa8e331e`, G2
`f0ce98cec8abafa243bde39e0cdaeb8e73b0043f16af51355d1b7414e1d4d834`.
Because strict math SDPA is not historical source parity, this closes observer
neutrality but does not reopen continuous telemetry. Commit `ede8af53` adds a
field-minimized categorical bridge and validates hard roles only. Remote focused
dynamic tests pass `45/45`; required C3 regressions pass `20/20`.
