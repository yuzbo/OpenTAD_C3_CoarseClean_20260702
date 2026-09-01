# ActionFormer SparseHead DCSR G0/G1 Implementation and Execution

Date: 2026-07-30

Status: `tested`; G1 completed negative and the current DCSR route is terminated

Implementation status: `implemented` and `tested`

Paper status: `paper_ready=false`

## Scientific question

The falsified hard-K384 method removed most native proposals and positive
training support. DCSR tests the narrower repair hypothesis: keep a prediction
and supervision floor at every valid native query, and sparsify only an
expensive residual refinement branch.

This execution is the validation-only G0/G1 architecture gate. It is not an
official test-set result and cannot be a paper performance row.

## Frozen implementation

Candidate branch:
`codex/actionformer-dcsr-g0-g1-20260730`.

Exact implementation commit/tree:
`bf0df83d7400c89fc61f38d169d68085420a2263` /
`2f9346fcfd2bfb7fc5a76a86ef65545030a67469`.

Core DCSR implementation began at
`4107d257fa6f354d46ad11d468566655fb51dcba`. The final two descendants only
repair Slurm launch and Python module-invocation contracts; they do not change
the model, data, evaluator, seeds, schedule or gates.

Two modes are deliberately separated:

- G0 `official_identity` uses the complete official dense head as the
  scaffold, disables residual refinement, and tests routing/geometry/decoder
  identity. It is not a cheap-head performance or efficiency result.
- G1 `cheap_dense_scaffold` applies a one-layer dense scaffold to all valid
  native queries and a signed three-layer residual head only at deterministic
  uniform K384 queries. Unselected queries retain scaffold logits and offsets.
  The original full FPN masks, full-grid targets and official positive
  normalizer remain in force.

The residual branch is zero-initialized at its final projection so G1 begins
from the scaffold rather than from arbitrary selected-query perturbations.

## Internal holdout and comparability boundary

The deterministic manifest is
`DCSR_INTERNAL_HOLDOUT.json`, SHA-256
`ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`.
It partitions the official THUMOS `validation` training split into 160
development-training and 40 holdout videos, with all 20 classes present on
both sides. It selects no `test` record and uses no test GT, AP, prediction or
teacher signal.

Frozen development seeds are
`2026073001`, `2026073002`, `2026073003`; they are disjoint from the five
preregistered final official seeds. Both dense and DCSR arms use the same
seed, data, 5-warmup + 30-optimizer-epoch schedule, terminal epoch-35 EMA,
official Soft-NMS and holdout evaluator.

Therefore:

- G1 may accept or reject the architecture;
- G1 numbers must never be copied into a paper main table;
- a paper-comparable result still requires G0--G4 freeze, five paired
  full-`validation`-to-official-`test` seeds, paired uncertainty and complete
  feature-to-final-detection cost.

## Verification

The clean N16R4 runtime is
`/data/run01/sczc063/yuzibo/projects/actionformer_dcsr_g0_g1_20260730_v5`.
The exact Linux suite passed `31 passed`, including the existing native-grid
SparseHead contracts, DCSR head behavior, holdout/no-leak checks, launcher
contracts and module-entry smoke tests.

Real-CUDA G0 Job `1206168` completed `0:0` on an RTX 4090. Receipt:

`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_g0_gate_20260730_v5/seed_2026073001/DCSR_G0_EQUIVALENCE.json`

Receipt SHA-256:
`b87fc59ec6529e83e99f7bf5fbfb7f3bff5ec637060c62057da07a669a8c1ff4`.

All exact checks pass:

- state-dict keys;
- native points;
- complete FPN masks;
- pre-decode class logits;
- pre-decode offsets;
- decoded official Soft-NMS labels, scores and physical timestamps.

The receipt explicitly records
`test_gt_used=false`, `test_predictions_used=false`,
`model_selection_performed=false`, `metric_claim_allowed=false` and
`efficiency_claim_allowed=false`.

## Preserved engineering failures

- Job `1206160`, v3 root:
  `slurm_wrap_shell_and_unexported_variable_scope_v1`. An ad-hoc `--wrap`
  used the wrong job shell and did not export submission-local paths. It
  stopped before model execution.
- Job `1206166`, v4 root:
  `python_script_repository_import_scope_v1`. The checked-in Bash launcher and
  focused tests ran, but direct `python tools/...` invocation excluded the
  repository root from `sys.path`. It stopped before G0 tensor comparison.

Both roots and logs are immutable engineering evidence. Neither produced a
model metric. The final launcher uses a checked-in `g0_only` mode and invokes
repository tools with `python -m`.

## G1 execution and result

Formal Slurm array: `1206273_[0-2]`.

Run root:
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_g0_g1_internal_20260730_v5`.

`1206266` is only an `sbatch --test-only` number and is not a job.

Each task runs, fail-closed:

1. environment, source, data and manifest identity;
2. focused tests;
3. real-CUDA G0;
4. fresh dense development training and holdout evaluation;
5. fresh DCSR G1 training and matched holdout evaluation;
6. independent raw-prediction recomputation;
7. a per-seed paired completion receipt.

After all three tasks complete, the frozen aggregate gate is:
mean Avg-mAP delta `>= -0.50 pp` and mean mAP@0.6/mAP@0.7 deltas each
`>= -1.00 pp`.

Negative deltas are model evidence and must not be converted into engineering
failures. A failed G1 gate triggers evidence-grounded Pro analysis before any
redesign; it does not authorize silent tuning or retraining.

All three array tasks completed `0:0`. Aggregate receipt SHA-256 is
`b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939`.
The paired completion SHA-256 values for seeds
`2026073001/2026073002/2026073003` are:

- `c924ed997a438f14e3d4660906635e2cca90b34b8ac2d2dc7c4170df2a4a5867`;
- `a2ecb27e8485c10fe97a5319b930c1e7d49d5f918f02c64269ec6281d87f88da`;
- `9b85c1f38f5ecb4ba7fbb9e60c39c9e6005dc8201d828aa79cce0a365cdfcd40`.

Per-seed dense/DCSR Avg-mAP is:

| seed | dense | DCSR G1 | delta |
|---|---:|---:|---:|
| 2026073001 | 0.5769068582 | 0.5032135335 | -7.369333 pp |
| 2026073002 | 0.5619997727 | 0.4881934096 | -7.380636 pp |
| 2026073003 | 0.5653126303 | 0.4861262566 | -7.918637 pp |

Mean DCSR-minus-dense is `-7.556202 pp`; mAP@0.3--0.7 deltas are
`-5.719942/-3.507626/-6.490487/-11.043134/-11.019821 pp`. The frozen G1
gate is false. This is a model/method negative, not an engineering failure.

## No-training negative diagnostics

Diagnostic source commit/tree:
`8d6f6e5e7fcf8c27b6aa46870bc4c0b242f6314b` /
`1ac5a68c6b8d0b1c9028ea3154765ae20e87622a`.

Clean N16R4 runtime and run root:

- `/data/run01/sczc063/yuzibo/projects/actionformer_dcsr_negative_diagnostics_20260730_v3`;
- `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_negative_diagnostics_20260730_v3`.

The Linux exact suite passed `38 passed`. The three counterfactual tasks and
dependent aggregate Job `1207441` all completed `0:0`. Completion,
prediction-diagnostic and checkpoint-dynamics SHA-256 values are:

- `954d7944428fcf0d26dd917ff9562a9c3e7a53de71c09e9a382aaf49f5bd4a53`;
- `47dcca7e179544e348966bf92cf92cddeff19a1fdc8cfea100150dc1bc580a36`;
- `c596bc942d2617e3824d21c96d0289316be4ee1ad465f23dc507b2d90466e006`.

The completion records `validation_pass=true`, `training_performed=false`,
`test_subset_used=false`, `diagnostic_only=true` and
`paper_performance_row_allowed=false`.

Three pre-Slurm deployment failures are preserved as engineering-only roots:

1. `diagnostic_deployment_inline_ssh_quoting_v1`;
2. `diagnostic_deployment_nonlogin_module_function_v1`;
3. `diagnostic_deployment_profile_under_nounset_v1`.

None created a model result or changed the frozen scientific conditions.
Detailed attribution and the paper claim boundary are in
`actionformer-sparsehead-dcsr-g1-negative-analysis-20260730.md` and
`actionformer-sparsehead-dcsr-g1-integrity-audit-20260730.md`.
