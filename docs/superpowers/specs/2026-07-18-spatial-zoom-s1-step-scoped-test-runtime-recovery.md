# Spatial Zoom S1 Step-Scoped Test Runtime Recovery

## Failure

Formal matrix Job `1170468` completed the frozen ordinal-0
`dense256/seed3408` cell, then failed before opening ordinal-1
`dense224/seed3409`. The official test process ran
`tools/test.py` from training commit `18139b9`. That historical entrypoint
accepted only one value in `SLURM_JOB_GPUS`, while the preregistered
high-memory execution contract uses a two-GPU outer reservation and an exact
one-GPU Slurm step. The active step was auditable through
`SLURM_STEP_GPUS=2`, but the historical guard rejected the unchanged outer
`SLURM_JOB_GPUS=2,4`.

This is an infrastructure failure. It is not accuracy, model, checkpoint,
dataset, evaluator, profile-cadence, or cost evidence.

## Invariants

The recovery must preserve:

- training/model commit `18139b9`, bound configs, selected checkpoints, and
  pretrained weights;
- the sealed-test certificate and every existing test artifact;
- the official OpenTAD test/evaluator path;
- the 20 ms target and 100 ms maximum power-sampling gap;
- the exact one-GPU/five-CPU/96,000 MiB inner step and 4+1 CPU partition;
- the frozen 3x3 order, statistical analysis, and GO/KILL thresholds.

The failed campaign `20fe22c380fd38bd`, its matrix lock, and Job `1170468`
remain immutable. They cannot be resumed or reused.

## Rejected Repairs

1. Rewriting `SLURM_JOB_GPUS` around `tools/test.py` would falsify scheduler
   provenance.
2. Editing the clean `18139b9` snapshot in place would break checkpoint and
   source identity.
3. Reusing the failed campaign lock or skipping ordinal-1 test would violate
   the frozen matrix.
4. Opening the remaining tests in separate jobs would break matrix-start and
   same-hardware bindings.

## Selected Design

Create a new recursive recovery schema and campaign. It validates the parent
v4 certificate, failed matrix start/submission receipts, stdout/stderr,
ordinal-0 descriptor and all descriptor artifacts, the absence of a completion
receipt, and the exact ordinal-1 failure signature. The parent-to-runtime diff
uses a dedicated nine-path allowlist rather than the broader historical S1
infrastructure allowlist. Entry into the failed cell is accepted only when one
structured stdout record jointly identifies ordinal 1, resolution 224, and
seed 3409.

Formal tests with no existing evidence run `tools/test.py` from the new
profile/runtime commit. The recovery certificate proves that the diff from
`18139b9` is limited to S1 profile, evidence, test-entrypoint, launcher,
tests, and research records; changes under `opentad/`, S1 configs, model code,
or evaluator code remain forbidden. The test entrypoint validates the
step-scoped GPU identity, the clean runtime commit, and the recovery
certificate before opening sealed test data.

The test-open marker and test evidence bind:

- training commit and runtime commit;
- recovery certificate path, file hash, internal hash, and campaign ID;
- the versioned step-scoped runtime mode.

New non-legacy descriptors must carry the same fields. The existing
`dense256/seed3408` evidence remains the sole explicitly certified legacy
unbound test result.

## Verification And Deployment

Before any new Slurm submission:

1. focused local tests and syntax checks pass;
2. an independent read-only review reports no P0/P1;
3. the branch is committed, pushed, and replayed in a clean Linux snapshot;
4. a new immutable recovery certificate recursively validates Job `1170468`;
5. one no-new-test-open full-path Gate passes all 792 loader exposures and
   791 physical windows.

Only then may exactly one replacement frozen-order serial matrix be submitted.
No analyzer, S1 GO/KILL, or Pro review is allowed before all nine descriptors
and the matrix completion receipt validate.
