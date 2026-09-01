---
doc_id: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE
version: v001
status: CANDIDATE_NOT_EXECUTED_NOT_PRE_RUN_READY
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T210234Z-8c74f8097f7c
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_plan: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN-v002
critic_closure: CRITIC_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_PLAN_V002_FOCUSED_RECHECK-v001
phase: E1_INDEPENDENT_REFERENCE_FREEZE
evidence_class: p0_identity_e1_admission_candidate
execution_state: NOT_EXECUTED
pre_run_state: NOT_READY
---

# E1 independent-reference admission-manifest candidate

This candidate consumes exactly durable queue
`msg-20260812T210234Z-8c74f8097f7c`. It names the future E1 envelope but does
not authorize it, does not create an execution queue, and does not establish
`PRE_RUN_READY`. No command was issued and no runtime output was accessed.

The candidate preserves accepted `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001` and
role-separated plan v002 without scientific change: `Q=1048576`, ordered
positive/negative/mutation counts `18/9/6`, exact lexicographic objective and
ascending candidate order, typed failures, exact input-line bytes, exhaustive
ascending reference for every applicable positive `T<=385` fixture,
independently structured exact DAG/shortest-path reference for positive
`T=767/768` fixtures, E1-before-B1 separation, and stop-on-first-failure.

## 1. Literal E1 identities and paths

- Host class: registered N16R4 remote CPU login environment only.
- Interpreter:
  `/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python`.
- Working directory:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source`.
- Independent reference source:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py`.
- Sealed fixture stream:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl`.
- Sealed fixture index:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json`.
- Sealed mutation definitions:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json`.
- `REFERENCE_OUTPUT_ROOT`:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference`.
- Expected E1 receipt:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference/DUCA_P0_E1_REFERENCE_RECEIPT-v001.json`.
- `PRODUCTION_OUTPUT_ROOT`:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production`.
- `COMPARISON_OUTPUT_ROOT`:
  `/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison`.

No path inference, alias, environment expansion, alternate root, or relative
path is permitted.

## 2. Environment denials

The later exact E1 execution queue must set exactly these task-specific values
without loading modules or activating a training environment:

```text
CUDA_VISIBLE_DEVICES=
PYTHONNOUSERSITE=1
DUCA_P0_DENY_DATASET=1
DUCA_P0_DENY_MODEL=1
DUCA_P0_DENY_CHECKPOINT=1
DUCA_P0_DENY_METRICS=1
DUCA_P0_DENY_CUDA=1
DUCA_P0_DENY_SLURM=1
DUCA_P0_DENY_NETWORK=1
DUCA_P0_DENY_BUILDER_ROOT=1
DUCA_P0_DENY_PRODUCTION=1
DUCA_P0_DENY_ADAPTER=1
DUCA_P0_DENY_VALIDATOR=1
DUCA_P0_DENY_DETECTOR=1
DUCA_P0_DENY_TRAINING=1
```

`-B` disables bytecode writes. The E1 process may read only the four literal
Evaluator-owned source/input paths in section 1 and Python standard-library
files required by the interpreter. It may write only a newly created
`REFERENCE_OUTPUT_ROOT` and its temporary sibling during atomic publication.

## 3. One literal remote CPU argv

The candidate argv has exactly eleven elements. `argv[3]` is exactly the UTF-8
text between `E1_INLINE_PROGRAM_BEGIN` and `E1_INLINE_PROGRAM_END`, excluding
those two marker lines and the Markdown fence. This definition uses no shell
interpolation. Its deliberately non-admitted execution-queue sentinel is the
explicit blocker described after the program.

```text
argv[0]  = /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python
argv[1]  = -B
argv[2]  = -c
argv[3]  = E1_INLINE_PROGRAM-v001 defined below
argv[4]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py
argv[5]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl
argv[6]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json
argv[7]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json
argv[8]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference
argv[9]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production
argv[10] = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison
```

E1_INLINE_PROGRAM_BEGIN

```python
from datetime import datetime, timezone
from pathlib import Path
import copy
import importlib.util
import json
import os
import sys

PROJECT_ID = "g-p-6a796fef9a00819194024cf1de3bd697"
PARENT_DECISION = "PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001"
AUTHORING_QUEUE_ID = "msg-20260812T210234Z-8c74f8097f7c"
QUEUE_ID = "FUTURE_EXACT_E1_EXECUTION_QUEUE_REQUIRED"
Q = 1 << 20
EXPECTED_INTERPRETER = "/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python"
EXPECTED_CWD = "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source"
EXPECTED_PATH_ARGS = (
    "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py",
    "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl",
    "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json",
    "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json",
    "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference",
    "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production",
    "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison",
)
REQUIRED_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONNOUSERSITE": "1",
    "DUCA_P0_DENY_DATASET": "1",
    "DUCA_P0_DENY_MODEL": "1",
    "DUCA_P0_DENY_CHECKPOINT": "1",
    "DUCA_P0_DENY_METRICS": "1",
    "DUCA_P0_DENY_CUDA": "1",
    "DUCA_P0_DENY_SLURM": "1",
    "DUCA_P0_DENY_NETWORK": "1",
    "DUCA_P0_DENY_BUILDER_ROOT": "1",
    "DUCA_P0_DENY_PRODUCTION": "1",
    "DUCA_P0_DENY_ADAPTER": "1",
    "DUCA_P0_DENY_VALIDATOR": "1",
    "DUCA_P0_DENY_DETECTOR": "1",
    "DUCA_P0_DENY_TRAINING": "1",
}
POSITIVE_IDS = (
    "G16-U", "G17-E2", "G17-EINF", "G17-E1", "G17-U1", "G17-PLEX",
    "G31-U", "G32-U", "G383-U", "G384-U", "G385-X", "G767-U",
    "F768-U", "F768-PERIODIC", "F768-DISP16", "F768-CONVEX",
    "F768-CONCAVE", "F768-ALT",
)
NEGATIVE_IDS = (
    "N-T15", "N-K", "N-U-LEN", "N-A-LEN", "N-U-CANON", "N-A-END",
    "N-A-ORDER", "N-INFEASIBLE", "N-ARITH",
)
MUTATION_IDS = (
    "M-DUPLICATE", "M-STRIDE5", "M-DISP17", "M-OBJECTIVE",
    "M-SCALAR-TIE-LOSER", "M-CANDIDATE-ORDER",
)

class E1Failure(Exception):
    def __init__(self, phase, fixture_id, source_line, field, detail):
        super().__init__(detail)
        self.record = {
            "phase": phase,
            "fixture_id": fixture_id,
            "source_line": source_line,
            "field": field,
            "detail": detail,
        }

def stop(phase, fixture_id, source_line, field, detail):
    raise E1Failure(phase, fixture_id, source_line, field, detail)

def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")

def seal(root):
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)

observed_original_argv = list(getattr(sys, "orig_argv", ()))
observed_path_args = tuple(sys.argv[1:])
reference_path, fixture_path, index_path, mutation_path, output_root, production_root, comparison_root = (
    Path(value) for value in EXPECTED_PATH_ARGS
)
partial_root = output_root.with_name(output_root.name + ".partial")
receipt_name = "DUCA_P0_E1_REFERENCE_RECEIPT-v001.json"
records = []
mutation_records = []
first_failure = None
artifact_status = "E1_REFERENCE_COMPLETE"
started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

try:
    if sys.executable != EXPECTED_INTERPRETER:
        stop("E1_PRE_RUN", None, None, "interpreter", f"required {EXPECTED_INTERPRETER}, observed {sys.executable}")
    if str(Path.cwd()) != EXPECTED_CWD:
        stop("E1_PRE_RUN", None, None, "working_directory", f"required {EXPECTED_CWD}, observed {Path.cwd()}")
    if observed_path_args != EXPECTED_PATH_ARGS:
        stop("E1_PRE_RUN", None, None, "argv_paths", "literal path argv mismatch")
    if len(observed_original_argv) != 11 or observed_original_argv[0] != EXPECTED_INTERPRETER or observed_original_argv[1:3] != ["-B", "-c"] or tuple(observed_original_argv[4:]) != EXPECTED_PATH_ARGS:
        stop("E1_PRE_RUN", None, None, "argv", "literal eleven-element argv mismatch")
    for name, required_value in REQUIRED_ENV.items():
        if os.environ.get(name) != required_value:
            stop("E1_PRE_RUN", None, None, "environment", f"{name} must equal {required_value!r}")
    for root in (output_root, production_root, comparison_root, partial_root):
        if root.exists():
            stop("E1_PRE_RUN", None, None, "root_state", f"required-absent path exists: {root}")
    for path in (reference_path, fixture_path, index_path, mutation_path):
        if not path.is_file() or path.stat().st_mode & 0o222:
            stop("E1_PRE_RUN", None, None, "sealed_input", f"required sealed non-writable file unavailable: {path}")
    if Q != 1048576:
        stop("E1_PRE_RUN", None, None, "Q", "Q mismatch")

    raw_fixture_bytes = fixture_path.read_bytes()
    raw_lines = raw_fixture_bytes.splitlines(keepends=True)
    if len(raw_lines) != 27 or any(not line.endswith(b"\n") or line.endswith(b"\r\n") for line in raw_lines):
        stop("E1_INPUT", None, None, "fixture_lines", "fixture stream is not exactly 27 LF lines")
    index = json.loads(index_path.read_bytes().decode("utf-8"))
    mutations = json.loads(mutation_path.read_bytes().decode("utf-8"))
    ordered = index.get("ordered_fixtures")
    mutation_defs = mutations.get("ordered_mutations")
    if not isinstance(ordered, list) or len(ordered) != 27:
        stop("E1_INPUT", None, None, "fixture_index", "fixture index count mismatch")
    if tuple(item.get("id") for item in ordered) != POSITIVE_IDS + NEGATIVE_IDS:
        stop("E1_INPUT", None, None, "fixture_order", "fixture order mismatch")
    if tuple(item.get("id") for item in mutation_defs or ()) != MUTATION_IDS:
        stop("E1_INPUT", None, None, "mutation_order", "mutation order mismatch")

    spec = importlib.util.spec_from_file_location("duca_p0_evaluator_reference_v001", reference_path)
    if spec is None or spec.loader is None:
        stop("E1_REFERENCE", None, None, "reference_interface", "reference loader unavailable")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    record_by_id = {}

    for source_line, (raw_line, descriptor) in enumerate(zip(raw_lines, ordered), start=1):
        fixture_id = descriptor["id"]
        try:
            exact_text = raw_line[:-1].decode("utf-8")
            payload = json.loads(exact_text)
        except (UnicodeError, json.JSONDecodeError) as error:
            stop("E1_INPUT", fixture_id, source_line, "exact_input_line_text", str(error))
        if exact_text.encode("utf-8") != raw_line[:-1]:
            stop("E1_INPUT", fixture_id, source_line, "exact_input_line_bytes", "UTF-8 round-trip mismatch")
        if descriptor.get("line") != source_line:
            stop("E1_INPUT", fixture_id, source_line, "source_line", "index line mismatch")
        if payload.get("Q") != Q:
            stop("E1_INPUT", fixture_id, source_line, "Q", "fixture Q mismatch")
        category = descriptor.get("category")
        force_exhaustive = category == "positive" and payload.get("T") <= 385
        result = reference.reference_project(payload, force_exhaustive=force_exhaustive)
        required_status = "PASS" if category == "positive" else descriptor.get("expected_status")
        if result.get("typed_status") != required_status:
            stop("E1_REFERENCE", fixture_id, source_line, "typed_status",
                 f"required {required_status}, observed {result.get('typed_status')}")
        if category == "positive":
            method = result.get("global_optimality", {}).get("method")
            required_method = (
                "independent_exhaustive_ascending" if payload["T"] <= 385
                else "independent_staged_exact_dag_dp"
            )
            if method != required_method:
                stop("E1_REFERENCE", fixture_id, source_line, "reference_method",
                     f"required {required_method}, observed {method}")
        record = {
            "source_line": source_line,
            "fixture_id": fixture_id,
            "category": category,
            "exact_input_line_text": exact_text,
            "exact_input_line_utf8_attestation": True,
            "input": payload,
            "reference_result": result,
        }
        records.append(record)
        record_by_id[fixture_id] = record

    def certificate_for(fixture_id):
        result = record_by_id[fixture_id]["reference_result"]
        return {
            "p": copy.deepcopy(result["p"]),
            "objective": copy.deepcopy(result["objective"]),
            "candidate_expansions": [],
        }

    for definition in mutation_defs:
        mutation_id = definition["id"]
        if mutation_id in {"M-DUPLICATE", "M-STRIDE5", "M-OBJECTIVE", "M-CANDIDATE-ORDER"}:
            base_id = "G17-E2"
        elif mutation_id == "M-DISP17":
            base_id = "F768-DISP16"
        else:
            base_id = "G17-PLEX"
        base_record = record_by_id[base_id]
        payload = base_record["input"]
        certificate = certificate_for(base_id)
        if mutation_id == "M-DUPLICATE":
            certificate["p"][2] = certificate["p"][1]
        elif mutation_id == "M-STRIDE5":
            certificate["p"][1] = 5
        elif mutation_id == "M-DISP17":
            p, u = certificate["p"], payload["u"]
            candidates = [j for j in range(1, len(p) - 1) if abs(p[j] - u[j]) == 16]
            if not candidates:
                stop("E1_MUTATION", mutation_id, None, "mutation_precondition", "no displacement-16 position")
            j = candidates[0]
            certificate["p"][j] += 1 if p[j] > u[j] else -1
        elif mutation_id == "M-OBJECTIVE":
            certificate["objective"]["E2"] = str(int(certificate["objective"]["E2"]) + 1)
        elif mutation_id == "M-SCALAR-TIE-LOSER":
            certificate["p"] = [value for value in range(17) if value != 6]
        elif mutation_id == "M-CANDIDATE-ORDER":
            certificate["candidate_expansions"] = [{"candidates": [2, 1]}]
        observed = reference.validate_external_certificate(payload, certificate)
        required = definition["required_code"]
        if observed.get("typed_status") != required:
            stop("E1_MUTATION", mutation_id, None, "typed_status",
                 f"required {required}, observed {observed.get('typed_status')}")
        mutation_records.append({
            "mutation_id": mutation_id,
            "base_fixture_id": base_id,
            "required_code": required,
            "reference_result": observed,
        })
except E1Failure as error:
    artifact_status = "E1_REFERENCE_BLOCKED"
    first_failure = error.record
except BaseException as error:
    artifact_status = "E1_REFERENCE_BLOCKED"
    first_failure = {
        "phase": "E1_UNEXPECTED",
        "fixture_id": None,
        "source_line": None,
        "field": type(error).__name__,
        "detail": str(error),
    }

completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
exit_status = 0 if artifact_status == "E1_REFERENCE_COMPLETE" else 2
receipt = {
    "schema_version": "duca-p0-e1-reference-receipt-v001",
    "artifact_status": artifact_status,
    "project_id": PROJECT_ID,
    "parent_decision": PARENT_DECISION,
    "admission_manifest_authoring_queue_id": AUTHORING_QUEUE_ID,
    "execution_queue_id": QUEUE_ID,
    "role": "evaluator",
    "phase": "E1_INDEPENDENT_REFERENCE_FREEZE",
    "evidence_class": "P0_PROJECTOR_CONFORMANCE_ONLY",
    "interpreter": EXPECTED_INTERPRETER,
    "working_directory": EXPECTED_CWD,
    "argv": observed_original_argv,
    "environment": REQUIRED_ENV,
    "host": {"class": "registered_N16R4_remote_CPU_login_environment", "nodename": os.uname().nodename},
    "started_at": started_at,
    "completed_at": completed_at,
    "exit_status": exit_status,
    "input_paths": {
        "reference_source": str(reference_path),
        "fixture_stream": str(fixture_path),
        "fixture_index": str(index_path),
        "certificate_mutations": str(mutation_path),
    },
    "output_paths": {
        "reference_output_root": str(output_root),
        "expected_receipt": str(output_root / receipt_name),
        "production_output_root_required_absent": str(production_root),
        "comparison_output_root_required_absent": str(comparison_root),
    },
    "Q": Q,
    "counts": {"positive": 18, "negative": 9, "mutations": 6},
    "fixture_order": list(POSITIVE_IDS + NEGATIVE_IDS),
    "mutation_order": list(MUTATION_IDS),
    "reference_method_allocation": {
        "positive_T_le_385": "independent_exhaustive_ascending",
        "positive_T_767_768": "independent_staged_exact_dag_dp",
    },
    "reference_source": str(reference_path),
    "fixture_source": str(fixture_path),
    "independence_attestation": {
        "reference_owned_by_evaluator": True,
        "production_module_imported": False,
        "production_helper_objective_certificate_candidate_selector_imported": False,
        "shared_surface_limited_to_sealed_input_bytes_and_frozen_contract": True,
    },
    "records": records,
    "mutation_records": mutation_records,
    "first_failure": first_failure,
    "one_shot": True,
    "reference_batch_processes": 1,
    "production_invocations": 0,
    "builder_root_access": False,
    "data_model_checkpoint_metric_access": False,
    "gpu_cuda_slurm_browser_access": False,
    "network_program_access": False,
    "tolerance_normalization_reserialization": False,
    "scope_deviation": "none",
}

try:
    partial_root.mkdir(parents=True)
    receipt_path = partial_root / receipt_name
    with receipt_path.open("xb") as handle:
        handle.write(canonical_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    seal(partial_root)
    partial_root.rename(output_root)
except BaseException as error:
    print(f"E1_RECEIPT_PUBLICATION_BLOCKED: {error}", file=sys.stderr)
    raise SystemExit(3)
raise SystemExit(exit_status)
```

E1_INLINE_PROGRAM_END

The token `FUTURE_EXACT_E1_EXECUTION_QUEUE_REQUIRED` is a deliberate literal
admission blocker. Under plan v002 it is an unresolved authority binding and
therefore a hard failure, not a usable execution-queue ID. A future Coordinator
queue must cause a new versioned manifest to replace that constant in `argv[3]`
with its own exact durable message ID before admission. Until then the candidate
argv must not issue and `PRE_RUN` remains blocked. No other argv field may
change.

## 4. Phase-relative root state

Immediately before a future E1 command, a separately authorized read-only
PRE_RUN check must establish all of the following simultaneously:

- `REFERENCE_OUTPUT_ROOT`: absent;
- `REFERENCE_OUTPUT_ROOT.partial`: absent;
- `PRODUCTION_OUTPUT_ROOT`: absent;
- `COMPARISON_OUTPUT_ROOT`: absent;
- the four allowed Evaluator input/source files: present, regular, sealed
  non-writable, and project-local;
- no prior or active E1/B1/E2 command or result for identity-gate v001.

Any mismatch yields `PRE_RUN_BLOCKED`; no E1 command issues. There is no deletion,
cleanup, reuse, overwrite, append, repair, alternate root, resume, or retry.

## 5. One-shot and role separation

- Process limit: one remote Python process for one E1 batch.
- Reference batch limit: one ordered pass producing one sealed E1 receipt.
- Retry limit: zero under this identity after the process starts.
- Production/reference comparison: forbidden in E1.
- Builder checkout, production JSONL adapter, production output, any adapter,
  validator, detector/model/decoder path: forbidden.
- A failure or incomplete receipt blocks B1, E2, C1, P0, P1, and PRE_RUN
  progression under this execution identity.

The inline batch envelope is orchestration only. It calls the existing
Evaluator-owned `reference_project(payload, force_exhaustive=...)` and
`validate_external_certificate(payload, certificate)` interfaces. It does not
define a third projector, production path, fallback, tolerance, or alternate
objective.

## 6. Expected E1 receipt schema

The only expected receipt path is the literal path in section 1. It contains one
canonical JSON object with schema version
`duca-p0-e1-reference-receipt-v001`. Required top-level fields are exactly those
emitted by `E1_INLINE_PROGRAM-v001`:

- authority and phase: `schema_version`, `artifact_status`, `project_id`,
  `parent_decision`, `admission_manifest_authoring_queue_id`,
  `execution_queue_id`, `role`, `phase`, `evidence_class`;
- literal execution identity: `interpreter`, `working_directory`, full eleven-
  element `argv`, required `environment`, registered host class/nodename,
  `started_at`, `completed_at`, and `exit_status`;
- literal surfaces: `input_paths`, `output_paths`, `reference_source`,
  `fixture_source`, and `independence_attestation`;
- frozen contract: `Q`, `counts`, `fixture_order`, `mutation_order`,
  `reference_method_allocation`;
- evidence: `records`, `mutation_records`, `first_failure`;
- controls: `one_shot`, `reference_batch_processes`,
  `production_invocations`, all forbidden-access booleans, and
  `scope_deviation`.

Every fixture record preserves its one-based `source_line`, fixture ID, category,
exact LF-stripped `exact_input_line_text`, affirmative UTF-8 byte attestation,
parsed `T,K,Q,u,a` object, and full reference result. Positive results contain
exact `p`, feasibility, objective components, interior vector, candidate order,
and global-optimality evidence. Negative results contain the exact typed failure.
Mutation records contain the ordered mutation ID, base fixture, required code,
and observed independent-reference result.

Successful completion requires exactly 27 ordered records, six ordered mutation
records, null `first_failure`, `scope_deviation="none"`, one reference batch
process, zero production invocations, and all forbidden-access booleans false.
The output root and receipt must be sealed non-writable before return.

## 7. First-fail contract

The first failing precondition, input line, reference method/status, mutation
code, arithmetic/certificate condition, forbidden access, or write/publish
condition stops the batch. No later fixture or mutation is evaluated after a
caught contract failure. A caught contract failure produces
`artifact_status="E1_REFERENCE_BLOCKED"`, the completed prefix only, and one
`first_failure` with phase, fixture/mutation identity when applicable, source
line when applicable, field, and detail. Receipt-publication failure exits 3 and
does not promote the partial root.

There is no repair, rerun, third implementation, tolerance, fallback, fixture
change, expected-output change, output reuse, or evidence promotion. E1 failure
only establishes incomplete or failed independent-reference evidence for this
package.

## 8. Remaining admission blockers

This candidate remains blocked and must not issue because:

1. it is not a Coordinator `PRE_RUN_READY` record;
2. it is not a phase-specific E1 execution queue;
3. `QUEUE_ID` still contains the deliberate future-queue sentinel and therefore
   requires one versioned deterministic manifest substitution;
4. no immediate pre-command read-only binding/root/duplicate-run check has been
   performed;
5. remote credentials, service availability, interpreter availability, and
   bounded completion resources have not been checked in this task; and
6. the literal environment and zero-forbidden-access enforcement have not been
   admitted by Coordinator for execution.

## 9. No-execution receipt

- Candidate: `AUTHORED_NOT_EXECUTED`
- PRE_RUN: `NOT_READY`
- E1 authority: `NOT_GRANTED`
- E1/B1/E2/C1: `NOT_EXECUTED`
- P0: `BLOCKED_PRE_RESULT`
- P1: `BLOCKED`
- Reference import/compile/invocation/output access: `NONE`
- Builder checkout/production adapter/output access: `NONE`
- Fixture materialization or runtime fixture read: `NONE`
- Local/remote command or SSH: `NOT_EXECUTED`
- Data/model/checkpoint/metric: `NOT_ACCESSED`
- GPU/CUDA/Slurm/browser: `NOT_USED`
- Subagents/probes/extra processes: `NONE`
- Scope deviation: `none`

`EVALUATOR_DECISION: E1_ADMISSION_MANIFEST_CANDIDATE_AUTHORED_NOT_EXECUTED_NOT_PRE_RUN_READY`.
