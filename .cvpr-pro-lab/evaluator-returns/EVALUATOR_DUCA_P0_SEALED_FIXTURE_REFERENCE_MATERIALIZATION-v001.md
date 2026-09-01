---
doc_id: EVALUATOR_DUCA_P0_SEALED_FIXTURE_REFERENCE_MATERIALIZATION
version: v001
status: SEALED_MATERIALIZED_NOT_GATE_EXECUTED
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T200640Z-304f13e14329
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_evaluator_artifact: EVALUATOR_DUCA_P0_IDENTITY_AUTHORING-v001
parent_builder_artifact: BUILDER_DUCA_P0_IDENTITY_CLEAN_SNAPSHOT_PREPARATION-v001
evidence_class: remote_fixture_reference_materialization_preparation
---

# Evaluator DUCA P0 sealed fixture/reference materialization

The exact remote-only preparation queue
`msg-20260812T200640Z-304f13e14329` completed without executing the P0
identity gate. The remote evaluator root is:

`/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001`

The root was absent at preflight, created exclusively, populated, checked, and
made recursively non-writable. No existing root was reused or cleaned.

## Exact materializer command

Executed once on registered N16R4 only:

```text
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -B /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_FIXTURE_MATERIALIZER-v001.py --matrix /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_IDENTITY_FIXTURE_MATRIX-v001.json --out-root /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized
```

The command returned:

```text
MATERIALIZATION_COMPLETE
FIXTURE_COUNT=27
MUTATION_COUNT=6
```

The standalone materializer imports only Python standard-library modules. It
imports and invokes neither the production projector nor the independent
reference projector. It validates the frozen matrix authority, exact closed
membership, and order before creating a temporary directory, and promotes that
directory only after every artifact is written successfully. An existing output,
partial root, unknown recipe, authority mismatch, membership/order mismatch, or
write failure exits nonzero without publishing the requested output root.

## Sealed remote artifacts

Source copies in the evaluator root:

1. `source/DUCA_P0_IDENTITY_FIXTURE_MATRIX-v001.json`
2. `source/DUCA_P0_REFERENCE_PROJECTOR-v001.py`
3. `source/DUCA_P0_FIXTURE_MATERIALIZER-v001.py`

Materialized artifacts:

1. `materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl`
2. `materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json`
3. `materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json`

The reference source was transferred unchanged and was not imported, parsed,
compiled, or executed. Each of the 27 JSONL records is canonical UTF-8 with LF
termination and field order `T,K,Q,u,a`; every record contains full integer
arrays. The index binds line number to fixture ID, category, and frozen expected
typed status. No reference winner or production output is present.

## Frozen ordering and counts

Positive count `18`, in order:

`G16-U`, `G17-E2`, `G17-EINF`, `G17-E1`, `G17-U1`, `G17-PLEX`,
`G31-U`, `G32-U`, `G383-U`, `G384-U`, `G385-X`, `G767-U`,
`F768-U`, `F768-PERIODIC`, `F768-DISP16`, `F768-CONVEX`,
`F768-CONCAVE`, `F768-ALT`.

Negative count `9`, in order:

`N-T15`, `N-K`, `N-U-LEN`, `N-A-LEN`, `N-U-CANON`, `N-A-END`,
`N-A-ORDER`, `N-INFEASIBLE`, `N-ARITH`.

Certificate-mutation definition count `6`, in order:

`M-DUPLICATE`, `M-STRIDE5`, `M-DISP17`, `M-OBJECTIVE`,
`M-SCALAR-TIE-LOSER`, `M-CANDIDATE-ORDER`.

The static closeout confirmed exactly three materialized files, 27 JSONL lines,
declared counts `18/9/6`, no `.partial` directory, and read-only file/directory
permissions. An initial closeout wrapper exited before mutation because nested
shell quoting removed literal JSON quotes from its grep pattern. A read-only
diagnostic established that the package itself was complete; a quote-free static
closeout then passed. Materialization was not rerun and no artifact was repaired
or rewritten.

## Zero-forbidden-access attestation

- Production projector/import/output: `NOT_ACCESSED_OR_INVOKED`
- Independent reference import/compile/execution/output: `NOT_INVOKED`
- Production/reference comparison: `NOT_EXECUTED`
- Tests/pytest/validator/gate: `NOT_EXECUTED`
- Reference expectations/winners/objectives: `NOT_CALCULATED`
- Dataset/media/annotation/model/checkpoint/metric: `NOT_ACCESSED`
- GPU/CUDA initialization/Slurm: `NOT_USED`
- Browser/Project/Pro operation: `NOT_USED`
- Local Python/materializer/reference execution: `NOT_EXECUTED`
- Remote activity: `AUTHORIZED_SSH_TRANSPORT_STATIC_SHELL_AND_ONE_MATERIALIZER_ONLY`
- Subagents/probes/extra role processes: `NONE`
- Scope deviation: `none`

This preparation is not the P0 identity gate and creates no conformance result.
`P0=BLOCKED_PRE_RESULT`, `P1=BLOCKED`, and `PRE_RUN=BLOCKED`. Production,
reference, comparison, and Critic closure require later exact durable queues.

`EVALUATOR_DECISION: SEALED_FIXTURE_REFERENCE_MATERIALIZED_NOT_GATE_EXECUTED`.
