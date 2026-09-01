---
doc_id: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE
version: v003
status: CANDIDATE_NOT_EXECUTED_NOT_PRE_RUN_READY
date: 2026-08-13
author_role: evaluator
project_id: g-p-6a796fef9a00819194024cf1de3bd697
queue_message_id: msg-20260812T213026Z-2c2fa1e59073
parent_decision: PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001
parent_plan: EVALUATOR_DUCA_P0_IDENTITY_ROLE_SEPARATED_PRE_RUN_EXECUTION_PLAN-v002
critic_correction: CRITIC_DUCA_P0_E1_REFERENCE_ADMISSION_MANIFEST_V002_FOCUSED_RECHECK-v001
supersedes: EVALUATOR_DUCA_P0_E1_REFERENCE_PRE_RUN_ADMISSION_MANIFEST_CANDIDATE-v002
phase: E1_INDEPENDENT_REFERENCE_FREEZE
evidence_class: p0_identity_e1_admission_candidate
execution_state: NOT_EXECUTED
pre_run_state: NOT_READY
---

# E1 independent-reference admission-manifest candidate

This v003 candidate consumes exactly durable correction queue
msg-20260812T213026Z-2c2fa1e59073 and implements only deterministic Critic
finding P0-E1-MANIFEST-001R. It does not authorize E1, create an execution
queue, or establish PRE_RUN_READY. No command was issued and no runtime output
or implementation surface was accessed.

All frozen science remains unchanged: Q=1048576; ordered positive, negative,
and mutation counts 18/9/6; the exact lexicographic objective and ascending
candidate order; exact LF-stripped input-line bytes; exhaustive reference for
every applicable positive T<=385; independently structured exact DAG/
shortest-path reference for all positive T=767/768; Evaluator reference
ownership; E1-before-B1 role separation; phase-relative roots; one shot; first
fail; no hash, tolerance, objective recomputation, second solver, fallback,
repair, fixture change, or route expansion.

## 1. Literal identities, paths, and shared-parent condition

- Host class: registered N16R4 remote CPU login environment only.
- Interpreter:
  /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python.
- Working directory:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source.
- Independent reference source:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py.
- Sealed fixture stream:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl.
- Sealed fixture index:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json.
- Sealed mutation definitions:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json.
- Existing shared identity-gate parent, separately admitted by Coordinator:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001.
- REFERENCE_OUTPUT_ROOT:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference.
- Temporary sibling:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference.partial.
- Expected E1 receipt:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference/DUCA_P0_E1_REFERENCE_RECEIPT-v001.json.
- PRODUCTION_OUTPUT_ROOT:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production.
- COMPARISON_OUTPUT_ROOT:
  /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison.

The v002 shared-parent correction is preserved unchanged. The shared parent
must already exist as the exact project-local directory and be named by the
future Coordinator admission record. E1 may not create, replace, or modify it.
No path inference, alias, alternate root, relative path, or recursive parent
creation is admitted.

## 2. Environment bindings and denials

The later exact E1 queue must set these values without loading modules or
activating a training environment:

~~~text
CUDA_VISIBLE_DEVICES=
PYTHONNOUSERSITE=1
DUCA_P0_IDENTITY_GATE_PARENT_ADMITTED=1
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
~~~

The parent-admission value binds a separately existing Coordinator record; it
does not self-authorize. Python -B disables bytecode writes. The E1 process may
read only the four literal Evaluator source/input paths and Python
standard-library files. It may write only the exclusively created temporary
sibling and atomically promote that sibling to the absent reference output.

## 3. One literal remote CPU argv

The candidate argv has exactly eleven elements. argv[3] is exactly the UTF-8
text between E1_INLINE_PROGRAM_BEGIN and E1_INLINE_PROGRAM_END, excluding the
marker lines and Markdown fence. There is no shell interpolation. The deliberate
future-queue sentinel remains an admission blocker.

~~~text
argv[0]  = /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python
argv[1]  = -B
argv[2]  = -c
argv[3]  = E1_INLINE_PROGRAM-v003 defined below
argv[4]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source/DUCA_P0_REFERENCE_PROJECTOR-v001.py
argv[5]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURES-v001.jsonl
argv[6]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_FIXTURE_INDEX-v001.json
argv[7]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/materialized/DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json
argv[8]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-reference
argv[9]  = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/builder-production
argv[10] = /data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001/evaluator-comparison
~~~

E1_INLINE_PROGRAM_BEGIN

~~~python
from datetime import datetime, timezone
from pathlib import Path
import copy
import importlib.util
import json
import os
import sys

PROJECT_ID = "g-p-6a796fef9a00819194024cf1de3bd697"
PARENT_DECISION = "PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001"
MANIFEST_VERSION = "v003"
AUTHORING_QUEUE_ID = "msg-20260812T213026Z-2c2fa1e59073"
QUEUE_ID = "FUTURE_EXACT_E1_EXECUTION_QUEUE_REQUIRED"
Q = 1 << 20
EXPECTED_INTERPRETER = "/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python"
EXPECTED_CWD = "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/evaluator/fixture-reference-v001/source"
EXPECTED_PARENT = "/data/run01/sczc063/yuzibo/cvpr-pro-lab/g-p-6a796fef9a00819194024cf1de3bd697/p0/identity-gate-v001"
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
    "DUCA_P0_IDENTITY_GATE_PARENT_ADMITTED": "1",
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
EXHAUSTIVE_IDS = POSITIVE_IDS[:11]
FULL_SCALE_IDS = POSITIVE_IDS[11:]
FULL_WITNESS_COUNTS = {
    "G17-E2": 15,
    "G17-EINF": 15,
    "G17-E1": 15,
    "G17-U1": 15,
    "G17-PLEX": 15,
    "G385-X": 383,
}
CANDIDATE_ORDER_DECLARATION = {
    "within_expansion": "strictly_ascending_physical_position",
    "exhaustive_sequence_order": "lexicographically_ascending",
    "incumbent_replacement": "strictly_smaller_complete_exact_key_only",
    "unordered_parallel_reduction": "FORBIDDEN",
}
TIE_REQUIREMENTS = {
    "G17-E2": (),
    "G17-EINF": ((("E2",), (3, 8)),),
    "G17-E1": ((("E2", "E_infinity"), (1, 7)),),
    "G17-U1": ((("E2", "E_infinity", "E1"), (6, 3)),),
    "G17-PLEX": ((("E2", "E_infinity", "E1", "U1"), (10, 6)),),
    "G385-X": ((("E2", "E_infinity", "E1", "U1"), (191, 193)),),
}
SUCCESS_ONLY_FIELDS = ("p", "feasibility", "objective", "global_optimality")
FEASIBILITY_FIELDS = (
    "length", "endpoints", "minimum_stride", "maximum_stride",
    "maximum_uniform_displacement",
)
OBJECTIVE_FIELDS = ("E2", "E_infinity", "E1", "U1", "interior_position_vector")
ROOT_CERTIFICATE_FIELDS = (
    "typed_status", "method", "independent_reference", "p", "feasibility",
    "objective", "candidate_order", "root_key",
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

def require_mapping(value, fixture_id, source_line, field):
    if not isinstance(value, dict) or not value:
        stop("E1_REFERENCE", fixture_id, source_line, field,
             "required non-empty evidence object is missing or malformed")
    return value

def require_decimal(value, fixture_id, source_line, field):
    if (not isinstance(value, str) or not value.isdecimal() or
            (len(value) > 1 and value[0] == "0")):
        stop("E1_REFERENCE", fixture_id, source_line, field,
             "required canonical non-negative exact-integer text is malformed")

def validate_objective(value, p, fixture_id, source_line, field="objective"):
    objective = require_mapping(value, fixture_id, source_line, field)
    for name in OBJECTIVE_FIELDS:
        if name not in objective:
            stop("E1_REFERENCE", fixture_id, source_line, f"{field}.{name}",
                 "mandatory objective evidence is missing")
    for name in OBJECTIVE_FIELDS[:4]:
        require_decimal(objective[name], fixture_id, source_line, f"{field}.{name}")
    interior = objective["interior_position_vector"]
    if not isinstance(interior, list) or interior != p[1:-1]:
        stop("E1_REFERENCE", fixture_id, source_line,
             f"{field}.interior_position_vector",
             "interior vector is malformed or inconsistent with exact p")
    return objective

def validate_candidate_order(value, fixture_id, source_line, field):
    if value != CANDIDATE_ORDER_DECLARATION:
        stop("E1_REFERENCE", fixture_id, source_line, field,
             "candidate-order evidence does not exactly match the frozen ascending declaration")
    return value

def validate_full_candidate_rows(witness, expected_count, K, fixture_id, source_line):
    if witness.get("candidate_count") != expected_count:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.exhaustive_witness.candidate_count",
             f"required complete candidate count {expected_count}")
    candidates = witness.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.exhaustive_witness.candidates",
             f"required exactly {expected_count} ordered candidate records")
    prior_p = None
    candidate_by_p = {}
    for candidate_index, candidate in enumerate(candidates):
        candidate = require_mapping(candidate, fixture_id, source_line,
                                    f"candidate[{candidate_index}]")
        candidate_p = candidate.get("p")
        if (not isinstance(candidate_p, list) or len(candidate_p) != K or
                any(type(item) is not int for item in candidate_p)):
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"candidate[{candidate_index}].p",
                 "candidate p is missing or malformed")
        if prior_p is not None and candidate_p <= prior_p:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"candidate[{candidate_index}].p",
                 "complete candidate witness is not strictly lexicographically ascending")
        objective = validate_objective(
            candidate.get("objective"), candidate_p, fixture_id, source_line,
            f"candidate[{candidate_index}].objective",
        )
        candidate_by_p[tuple(candidate_p)] = objective
        prior_p = candidate_p
    return candidate_by_p

def omitted_vector(T, omitted):
    return tuple(value for value in range(T) if value != omitted)

def validate_ties(witness, payload, candidate_by_p, fixture_id, source_line):
    ties = witness.get("ties")
    requirements = TIE_REQUIREMENTS[fixture_id]
    if not isinstance(ties, list) or len(ties) != len(requirements):
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.exhaustive_witness.ties",
             "frozen tie-witness count is missing or malformed")
    for tie_index, (tie, (prefix_fields, omissions)) in enumerate(
            zip(ties, requirements)):
        tie = require_mapping(tie, fixture_id, source_line,
                              f"tie[{tie_index}]")
        if tie.get("objective_prefix_fields") != list(prefix_fields):
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"tie[{tie_index}].objective_prefix_fields",
                 "tie prefix does not match the frozen objective prefix")
        tied = tie.get("candidates")
        if not isinstance(tied, list) or len(tied) != len(omissions):
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"tie[{tie_index}].candidates",
                 "tied candidate-vector count is missing or malformed")
        tied_keys = []
        prefixes = []
        for candidate_index, candidate_p in enumerate(tied):
            if (not isinstance(candidate_p, list) or
                    len(candidate_p) != payload["K"] or
                    any(type(item) is not int for item in candidate_p)):
                stop("E1_REFERENCE", fixture_id, source_line,
                     f"tie[{tie_index}].candidates[{candidate_index}]",
                     "tied candidate is not a full length-K integer vector")
            candidate_key = tuple(candidate_p)
            if candidate_key not in candidate_by_p:
                stop("E1_REFERENCE", fixture_id, source_line,
                     f"tie[{tie_index}].candidates[{candidate_index}]",
                     "tied candidate is absent from the complete witness")
            objective = candidate_by_p[candidate_key]
            tied_keys.append(candidate_key)
            prefixes.append(tuple(objective[name] for name in prefix_fields))
        required_keys = {omitted_vector(payload["T"], omitted)
                         for omitted in omissions}
        if set(tied_keys) != required_keys:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"tie[{tie_index}].candidates",
                 "tied candidates do not equal the frozen T17/G385 alternatives")
        if len(set(prefixes)) != 1:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"tie[{tie_index}].objective_prefix",
                 "tied candidates do not share the required objective prefix")
        if tie.get("objective_prefix") != list(prefixes[0]):
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"tie[{tie_index}].objective_prefix",
                 "declared objective prefix is absent or inconsistent")

def validate_root_certificate(root, result, feasibility, objective,
                              candidate_order, required_method,
                              fixture_id, source_line):
    for name in ("method", "p", "feasibility", "objective",
                 "candidate_order", "certificate"):
        if name not in root:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"global_optimality.root_optimum.{name}",
                 "mandatory root-optimum field is missing")
    expected_root_key = [
        objective["E2"], objective["E_infinity"], objective["E1"],
        objective["U1"], *result["p"][1:-1],
    ]
    if root["method"] != required_method:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.root_optimum.method",
             "root method is inconsistent with the frozen allocation")
    if root["p"] != result["p"]:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.root_optimum.p",
             "root p is inconsistent with result p")
    if root["feasibility"] != feasibility:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.root_optimum.feasibility",
             "root feasibility is inconsistent with result feasibility")
    if root["objective"] != objective:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.root_optimum.objective",
             "root objective is inconsistent with result objective")
    if root["candidate_order"] != candidate_order:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.root_optimum.candidate_order",
             "root candidate order is inconsistent with the frozen declaration")
    certificate = require_mapping(
        root["certificate"], fixture_id, source_line,
        "global_optimality.root_optimum.certificate",
    )
    for name in ROOT_CERTIFICATE_FIELDS:
        if name not in certificate:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"global_optimality.root_optimum.certificate.{name}",
                 "mandatory root-certificate field is missing")
    required_values = {
        "typed_status": "PASS",
        "method": required_method,
        "independent_reference": True,
        "p": result["p"],
        "feasibility": feasibility,
        "objective": objective,
        "candidate_order": candidate_order,
        "root_key": expected_root_key,
    }
    for name in ROOT_CERTIFICATE_FIELDS:
        if certificate[name] != required_values[name]:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"global_optimality.root_optimum.certificate.{name}",
                 "root-certificate field is inconsistent with root/result")

def validate_positive(result, payload, fixture_id, source_line, required_method):
    p = result.get("p")
    if (not isinstance(p, list) or len(p) != payload["K"] or
            any(type(item) is not int for item in p)):
        stop("E1_REFERENCE", fixture_id, source_line, "p",
             "exact p is missing or malformed")
    u = payload.get("u")
    if (not isinstance(u, list) or len(u) != payload["K"] or
            any(type(item) is not int for item in u)):
        stop("E1_REFERENCE", fixture_id, source_line, "input.u",
             "sealed uniform vector is missing or malformed")
    feasibility = require_mapping(
        result.get("feasibility"), fixture_id, source_line, "feasibility",
    )
    for name in FEASIBILITY_FIELDS:
        if name not in feasibility or feasibility[name] is None:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"feasibility.{name}",
                 "mandatory feasibility evidence is missing")
    if feasibility["length"] != payload["K"]:
        stop("E1_REFERENCE", fixture_id, source_line,
             "feasibility.length", "length mismatch")
    if feasibility["endpoints"] != [0, payload["T"] - 1]:
        stop("E1_REFERENCE", fixture_id, source_line,
             "feasibility.endpoints", "endpoint mismatch")
    strides = [right - left for left, right in zip(p, p[1:])]
    computed_minimum_stride = min(strides)
    computed_maximum_stride = max(strides)
    computed_maximum_displacement = max(
        abs(position - uniform) for position, uniform in zip(p, u)
    )
    computed_feasibility = {
        "minimum_stride": computed_minimum_stride,
        "maximum_stride": computed_maximum_stride,
        "maximum_uniform_displacement": computed_maximum_displacement,
    }
    for name, computed in computed_feasibility.items():
        if type(feasibility[name]) is not int or feasibility[name] != computed:
            stop("E1_REFERENCE", fixture_id, source_line,
                 f"feasibility.{name}",
                 "reported feasibility value is malformed or inconsistent with p/u")
    if computed_minimum_stride < 1:
        stop("E1_REFERENCE", fixture_id, source_line,
             "feasibility.minimum_stride", "frozen minimum-stride bound failed")
    if computed_maximum_stride > 4:
        stop("E1_REFERENCE", fixture_id, source_line,
             "feasibility.maximum_stride", "frozen maximum-stride bound failed")
    if computed_maximum_displacement > 16:
        stop("E1_REFERENCE", fixture_id, source_line,
             "feasibility.maximum_uniform_displacement",
             "frozen maximum-displacement bound failed")
    objective = validate_objective(
        result.get("objective"), p, fixture_id, source_line,
    )
    global_evidence = require_mapping(
        result.get("global_optimality"), fixture_id, source_line,
        "global_optimality",
    )
    if global_evidence.get("method") != required_method:
        stop("E1_REFERENCE", fixture_id, source_line,
             "global_optimality.method", f"required {required_method}")
    candidate_order = validate_candidate_order(
        global_evidence.get("candidate_order"), fixture_id, source_line,
        "global_optimality.candidate_order",
    )
    if fixture_id in EXHAUSTIVE_IDS:
        if global_evidence.get("exhaustive_complete") is not True:
            stop("E1_REFERENCE", fixture_id, source_line,
                 "global_optimality.exhaustive_complete",
                 "complete exhaustive evidence is not affirmed")
        summary = require_mapping(
            global_evidence.get("exhaustive_summary"), fixture_id, source_line,
            "global_optimality.exhaustive_summary",
        )
        if (type(summary.get("candidate_count")) is not int or
                summary["candidate_count"] < 1 or
                type(summary.get("minimizer_count")) is not int or
                summary["minimizer_count"] < 1):
            stop("E1_REFERENCE", fixture_id, source_line,
                 "global_optimality.exhaustive_summary",
                 "complete candidate/minimizer counts are malformed")
        if summary.get("winner_key") in (None, [], {}):
            stop("E1_REFERENCE", fixture_id, source_line,
                 "global_optimality.exhaustive_summary.winner_key",
                 "winner key is missing or empty")
        if fixture_id in FULL_WITNESS_COUNTS:
            witness = require_mapping(
                global_evidence.get("exhaustive_witness"),
                fixture_id, source_line,
                "global_optimality.exhaustive_witness",
            )
            candidate_by_p = validate_full_candidate_rows(
                witness, FULL_WITNESS_COUNTS[fixture_id], payload["K"],
                fixture_id, source_line,
            )
            validate_ties(
                witness, payload, candidate_by_p, fixture_id, source_line,
            )
    elif fixture_id in FULL_SCALE_IDS:
        root = require_mapping(
            global_evidence.get("root_optimum"), fixture_id, source_line,
            "global_optimality.root_optimum",
        )
        validate_root_certificate(
            root, result, feasibility, objective, candidate_order,
            required_method, fixture_id, source_line,
        )
    else:
        stop("E1_REFERENCE", fixture_id, source_line,
             "reference_method_allocation",
             "positive fixture is outside the frozen method allocation")

def validate_negative(result, fixture_id, source_line):
    for name in SUCCESS_ONLY_FIELDS:
        if name not in result or result[name] is not None:
            stop("E1_REFERENCE", fixture_id, source_line, name,
                 "negative result must contain an explicit null success-only field")

observed_original_argv = list(getattr(sys, "orig_argv", ()))
observed_path_args = tuple(sys.argv[1:])
reference_path, fixture_path, index_path, mutation_path, output_root, production_root, comparison_root = (
    Path(value) for value in EXPECTED_PATH_ARGS
)
shared_parent = output_root.parent
partial_root = output_root.with_name(output_root.name + ".partial")
receipt_name = "DUCA_P0_E1_REFERENCE_RECEIPT-v001.json"
records = []
mutation_records = []
first_failure = None
artifact_status = "E1_REFERENCE_COMPLETE"
started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

parent_failure = None
if (shared_parent != Path(EXPECTED_PARENT) or
        production_root.parent != shared_parent or
        comparison_root.parent != shared_parent):
    parent_failure = "literal output roots do not share the exact admitted parent"
elif os.environ.get("DUCA_P0_IDENTITY_GATE_PARENT_ADMITTED") != "1":
    parent_failure = "shared-parent admission binding is absent"
elif not shared_parent.is_dir():
    parent_failure = "exact admitted shared parent does not already exist as a directory"
elif not shared_parent.stat().st_mode & 0o222:
    parent_failure = "exact admitted shared parent cannot exclusively create the E1 sibling"
if parent_failure is not None:
    print("E1_PRE_RUN_BLOCKED: " + parent_failure, file=sys.stderr)
    raise SystemExit(2)

try:
    if sys.executable != EXPECTED_INTERPRETER:
        stop("E1_PRE_RUN", None, None, "interpreter",
             "literal interpreter mismatch")
    if str(Path.cwd()) != EXPECTED_CWD:
        stop("E1_PRE_RUN", None, None, "working_directory",
             "literal working-directory mismatch")
    if observed_path_args != EXPECTED_PATH_ARGS:
        stop("E1_PRE_RUN", None, None, "argv_paths",
             "literal path argv mismatch")
    if (len(observed_original_argv) != 11 or
            observed_original_argv[0] != EXPECTED_INTERPRETER or
            observed_original_argv[1:3] != ["-B", "-c"] or
            tuple(observed_original_argv[4:]) != EXPECTED_PATH_ARGS):
        stop("E1_PRE_RUN", None, None, "argv",
             "literal eleven-element argv mismatch")
    for name, required_value in REQUIRED_ENV.items():
        if os.environ.get(name) != required_value:
            stop("E1_PRE_RUN", None, None, "environment",
                 f"{name} mismatch")
    for root in (output_root, production_root, comparison_root, partial_root):
        if root.exists():
            stop("E1_PRE_RUN", None, None, "root_state",
                 f"required-absent path exists: {root}")
    for path in (reference_path, fixture_path, index_path, mutation_path):
        if not path.is_file() or path.stat().st_mode & 0o222:
            stop("E1_PRE_RUN", None, None, "sealed_input",
                 f"sealed input unavailable: {path}")
    if Q != 1048576:
        stop("E1_PRE_RUN", None, None, "Q", "Q mismatch")

    raw_lines = fixture_path.read_bytes().splitlines(keepends=True)
    if (len(raw_lines) != 27 or
            any(not line.endswith(b"\n") or line.endswith(b"\r\n")
                for line in raw_lines)):
        stop("E1_INPUT", None, None, "fixture_lines",
             "fixture stream is not exactly 27 LF lines")
    index = json.loads(index_path.read_bytes().decode("utf-8"))
    mutations = json.loads(mutation_path.read_bytes().decode("utf-8"))
    ordered = index.get("ordered_fixtures")
    mutation_defs = mutations.get("ordered_mutations")
    if not isinstance(ordered, list) or len(ordered) != 27:
        stop("E1_INPUT", None, None, "fixture_index",
             "fixture index count mismatch")
    if tuple(item.get("id") for item in ordered) != POSITIVE_IDS + NEGATIVE_IDS:
        stop("E1_INPUT", None, None, "fixture_order",
             "fixture order mismatch")
    if tuple(item.get("id") for item in mutation_defs or ()) != MUTATION_IDS:
        stop("E1_INPUT", None, None, "mutation_order",
             "mutation order mismatch")

    spec = importlib.util.spec_from_file_location(
        "duca_p0_evaluator_reference_v001", reference_path,
    )
    if spec is None or spec.loader is None:
        stop("E1_REFERENCE", None, None, "reference_interface",
             "reference loader unavailable")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    record_by_id = {}

    for source_line, (raw_line, descriptor) in enumerate(
            zip(raw_lines, ordered), start=1):
        fixture_id = descriptor["id"]
        try:
            exact_text = raw_line[:-1].decode("utf-8")
            payload = json.loads(exact_text)
        except (UnicodeError, json.JSONDecodeError) as error:
            stop("E1_INPUT", fixture_id, source_line,
                 "exact_input_line_text", str(error))
        if exact_text.encode("utf-8") != raw_line[:-1]:
            stop("E1_INPUT", fixture_id, source_line,
                 "exact_input_line_bytes", "UTF-8 round-trip mismatch")
        if descriptor.get("line") != source_line or payload.get("Q") != Q:
            stop("E1_INPUT", fixture_id, source_line,
                 "line_or_Q", "source-line or Q mismatch")
        category = descriptor.get("category")
        force_exhaustive = category == "positive" and payload.get("T") <= 385
        result = reference.reference_project(
            payload, force_exhaustive=force_exhaustive,
        )
        if not isinstance(result, dict):
            stop("E1_REFERENCE", fixture_id, source_line,
                 "result", "reference result is not an object")
        required_status = (
            "PASS" if category == "positive"
            else descriptor.get("expected_status")
        )
        if result.get("typed_status") != required_status:
            stop("E1_REFERENCE", fixture_id, source_line, "typed_status",
                 f"required {required_status}, observed {result.get('typed_status')}")
        if category == "positive":
            required_method = (
                "independent_exhaustive_ascending"
                if payload["T"] <= 385
                else "independent_staged_exact_dag_dp"
            )
            validate_positive(
                result, payload, fixture_id, source_line, required_method,
            )
        else:
            validate_negative(result, fixture_id, source_line)
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
        if mutation_id in {
                "M-DUPLICATE", "M-STRIDE5", "M-OBJECTIVE",
                "M-CANDIDATE-ORDER"}:
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
            candidates = [
                j for j in range(1, len(p) - 1)
                if abs(p[j] - u[j]) == 16
            ]
            if not candidates:
                stop("E1_MUTATION", mutation_id, None,
                     "mutation_precondition", "no displacement-16 position")
            j = candidates[0]
            certificate["p"][j] += 1 if p[j] > u[j] else -1
        elif mutation_id == "M-OBJECTIVE":
            certificate["objective"]["E2"] = str(
                int(certificate["objective"]["E2"]) + 1
            )
        elif mutation_id == "M-SCALAR-TIE-LOSER":
            certificate["p"] = [
                value for value in range(17) if value != 6
            ]
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

    if (tuple(record["fixture_id"] for record in records) !=
            POSITIVE_IDS + NEGATIVE_IDS or
            tuple(record["mutation_id"] for record in mutation_records) !=
            MUTATION_IDS):
        stop("E1_REFERENCE", None, None, "completion_order",
             "complete ordered 18/9/6 evidence is missing")
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
    "manifest_version": MANIFEST_VERSION,
    "admission_manifest_authoring_queue_id": AUTHORING_QUEUE_ID,
    "execution_queue_id": QUEUE_ID,
    "role": "evaluator",
    "phase": "E1_INDEPENDENT_REFERENCE_FREEZE",
    "evidence_class": "P0_PROJECTOR_CONFORMANCE_ONLY",
    "interpreter": EXPECTED_INTERPRETER,
    "working_directory": EXPECTED_CWD,
    "argv": observed_original_argv,
    "environment": REQUIRED_ENV,
    "host": {
        "class": "registered_N16R4_remote_CPU_login_environment",
        "nodename": os.uname().nodename,
    },
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
        "admitted_existing_parent": str(shared_parent),
        "reference_output_root": str(output_root),
        "temporary_sibling": str(partial_root),
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
    partial_root.mkdir(mode=0o700, parents=False, exist_ok=False)
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
~~~

E1_INLINE_PROGRAM_END

The token FUTURE_EXACT_E1_EXECUTION_QUEUE_REQUIRED is an unresolved authority
binding and therefore a hard admission failure under plan v002. A future
versioned manifest must replace only that constant with its exact durable E1
execution queue ID. Until then this candidate argv must not issue.

## 4. Correction P0-E1-MANIFEST-001R

This revision adds only direct schema and consistency checks over the returned
result, the sealed u vector, and existing witness objects:

1. It derives adjacent strides from returned p and displacement from returned
   p versus sealed u, requires exact equality with the reported feasibility
   fields, and enforces minimum_stride>=1, maximum_stride<=4, and
   maximum_uniform_displacement<=16.
2. It requires candidate-order evidence to equal the four-field frozen
   ascending declaration from DUCA_P0_PROJECTOR_NORMATIVE_SPEC-v001; a merely
   nonempty value cannot pass.
3. Each exhaustive tie contains full length-K integer vectors, every vector is
   a member of the complete ordered witness, all tied vectors have the declared
   equal objective prefix, and the vectors equal the frozen alternatives:
   G17-EINF omits 3/8; G17-E1 omits 1/7; G17-U1 omits 6/3; G17-PLEX omits
   10/6; and G385-X omits 191/193. The T17 witnesses remain exactly 15
   candidates and G385-X remains exactly 383.
4. Every full-scale root optimum contains method, p, feasibility, objective,
   candidate-order declaration, and a nonempty certificate. The certificate
   must contain typed status, method, independent-reference flag, p,
   feasibility, objective, candidate order, and root key, each exactly
   consistent with the root/result. The root key is assembled only from the
   returned exact objective fields and returned interior p; no objective is
   recomputed.

The first missing, malformed, infeasible, out-of-order, nonmember, unequal, or
inconsistent field becomes the sole E1_REFERENCE_BLOCKED first failure. These
checks do not solve, optimize, regenerate a witness, recompute an objective,
consult another implementation, change a fixture, introduce tolerance, or
create a fallback.

## 5. Preserved shared-parent publication and phase boundary

P0-E1-MANIFEST-002 remains closed without modification. Before E1, Coordinator
must admit the literal shared parent and a read-only PRE_RUN must establish that
it already exists as the exact project-local directory. The inline envelope
repeats that condition before any write. It exclusively creates only the absent
temporary sibling using:

    partial_root.mkdir(mode=0o700, parents=False, exist_ok=False)

After one receipt is written and sealed, the sibling is atomically renamed to
the absent reference output. No recursive creation, cleanup, overwrite, reuse,
append, repair, alternate root, or retry is permitted.

Immediately before a future E1 command, a separately authorized read-only
check must establish simultaneously:

- the literal shared parent exists and is admitted for this identity;
- the reference output and temporary sibling are absent;
- production and comparison output roots are absent;
- the four Evaluator input/source files are present, regular, sealed,
  non-writable, and project-local;
- no prior or active E1/B1/E2 command or result exists for this identity.

Any mismatch yields PRE_RUN_BLOCKED; no E1 command issues. E1 may invoke one
Evaluator reference batch only. It may not traverse or read the Builder
checkout, production adapter/output, any adapter, validator, detector, model,
decoder, dataset, checkpoint, metric, Torch/CUDA, training stack, Slurm,
browser, or network program.

## 6. Expected receipt, first fail, and current state

The only expected receipt remains the literal
DUCA_P0_E1_REFERENCE_RECEIPT-v001.json path in section 1. The canonical JSON
object retains schema duca-p0-e1-reference-receipt-v001 and includes manifest
version v003, exact authority/queue/phase, interpreter, cwd, full argv,
environment, host, timestamps, exit status, admitted parent, literal inputs and
outputs, Q, ordered 18/9/6 IDs, method allocation, independent-reference
attestation, complete fixture and mutation records, first failure, one-shot
controls, zero-forbidden-access attestations, and scope_deviation="none".

The first failing precondition, input byte/line, feasibility consistency or
bound, candidate-order declaration, exhaustive witness/tie condition,
root-certificate field, typed status, mutation code, write, seal, or promotion
condition stops the batch. A caught contract failure retains only the completed
prefix plus one first-failure record and cannot make B1 eligible. Shared-parent
failure occurs before any write. Publication failure exits 3 without promoting
the temporary sibling.

This candidate remains blocked because it is neither Coordinator PRE_RUN_READY
nor a phase-specific execution queue; its execution-queue ID is still the
deliberate sentinel; no immediate admission/binding/root/duplicate-run,
credential, service, interpreter, or resource check occurred; and no execution
environment has been admitted.

- Correction: P0-E1-MANIFEST-001R_APPLIED
- Prior correction: P0-E1-MANIFEST-002_PRESERVED_CLOSED
- Candidate: AUTHORED_NOT_EXECUTED
- PRE_RUN: NOT_READY
- E1 authority: NOT_GRANTED
- E1/B1/E2/C1: NOT_EXECUTED
- P0: BLOCKED_PRE_RESULT
- P1: BLOCKED
- Reference import/compile/invocation/output access: NONE
- Builder checkout/production adapter/output access: NONE
- Fixture materialization or runtime fixture read: NONE
- Local/remote command or SSH: NOT_EXECUTED
- Data/model/checkpoint/metric: NOT_ACCESSED
- GPU/CUDA/Slurm/browser: NOT_USED
- Subagents/probes/extra processes: NONE
- Scope deviation: none

EVALUATOR_DECISION: E1_ADMISSION_MANIFEST_V003_001R_AUTHORED_NOT_EXECUTED_NOT_PRE_RUN_READY.
