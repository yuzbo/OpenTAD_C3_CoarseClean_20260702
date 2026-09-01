"""Materialize the frozen DUCA P0 fixture recipes — do not import either projector.

This script is preparation-only.  It expands the already frozen Evaluator matrix
into ordered canonical UTF-8 input bytes.  It does not invoke production or the
independent reference, calculate reference winners, compare outputs, or access
data, models, checkpoints, metrics, accelerators, schedulers, or the network.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


Q = 1 << 20
QUEUE_MESSAGE_ID = "msg-20260812T200640Z-304f13e14329"
PARENT_DECISION = "PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001"

POSITIVE_IDS = (
    "G16-U",
    "G17-E2",
    "G17-EINF",
    "G17-E1",
    "G17-U1",
    "G17-PLEX",
    "G31-U",
    "G32-U",
    "G383-U",
    "G384-U",
    "G385-X",
    "G767-U",
    "F768-U",
    "F768-PERIODIC",
    "F768-DISP16",
    "F768-CONVEX",
    "F768-CONCAVE",
    "F768-ALT",
)

NEGATIVE_IDS = (
    "N-T15",
    "N-K",
    "N-U-LEN",
    "N-A-LEN",
    "N-U-CANON",
    "N-A-END",
    "N-A-ORDER",
    "N-INFEASIBLE",
    "N-ARITH",
)

MUTATION_IDS = (
    "M-DUPLICATE",
    "M-STRIDE5",
    "M-DISP17",
    "M-OBJECTIVE",
    "M-SCALAR-TIE-LOSER",
    "M-CANDIDATE-ORDER",
)


class MaterializationError(Exception):
    pass


def fail(message: str) -> None:
    raise MaterializationError(message)


def half_up_nonnegative(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        fail("invalid half-up domain")
    return (2 * numerator + denominator) // (2 * denominator)


def canonical_uniform(T: int, K: int) -> list[int]:
    if K < 2:
        return []
    return [half_up_nonnegative(j * (T - 1), K - 1) for j in range(K)]


def q_times(values: list[int]) -> list[int]:
    return [Q * value for value in values]


def canonical_input(T: int, K: int, u: list[int], a: list[int]) -> dict:
    # Insertion order is normative: T,K,Q,u,a.
    return {"T": T, "K": K, "Q": Q, "u": u, "a": a}


def positive_input(item: dict) -> dict:
    fixture_id = item["id"]
    T = item["T"]
    K = item["K"]
    u = canonical_uniform(T, K)

    if fixture_id in {
        "G16-U",
        "G17-E2",
        "G31-U",
        "G32-U",
        "G383-U",
        "G384-U",
        "G767-U",
        "F768-U",
    }:
        a = q_times(u)
    elif fixture_id in {"G17-EINF", "G17-E1", "G17-U1", "G17-PLEX"}:
        v = item["a"].get("v")
        if not isinstance(v, list) or len(v) != K or any(not isinstance(x, int) for x in v):
            fail(f"{fixture_id}: invalid frozen quarter-grid vector")
        a = [(Q // 4) * value for value in v]
    elif fixture_id == "G385-X":
        a = []
        for j in range(K):
            if j < 191:
                value = Q * j
            elif j == 191:
                value = Q * 191 + 3 * Q // 4
            elif j == 192:
                value = Q * 192 + Q // 4
            else:
                value = Q * (j + 1)
            a.append(value)
    elif fixture_id == "F768-PERIODIC":
        gaps = [1 if i % 2 == 0 else 3 for i in range(382)] + [3]
        p_star = [0]
        for gap in gaps:
            p_star.append(p_star[-1] + gap)
        a = q_times(p_star)
    elif fixture_id == "F768-DISP16":
        gaps = [u[j + 1] - u[j] for j in range(K - 1)]
        for gap_index in range(32, 48):
            gaps[gap_index] += 1
        for gap_index in range(300, 316):
            gaps[gap_index] -= 1
        p_star = [0]
        for gap in gaps:
            p_star.append(p_star[-1] + gap)
        a = q_times(p_star)
    elif fixture_id == "F768-CONVEX":
        denominator = 2 * 383 * 383
        a = [(2 * Q * 767 * j * j + 383 * 383) // denominator for j in range(K)]
    elif fixture_id == "F768-CONCAVE":
        denominator = 2 * 383 * 383
        convex = [(2 * Q * 767 * j * j + 383 * 383) // denominator for j in range(K)]
        a = [Q * 767 - convex[383 - j] for j in range(K)]
    elif fixture_id == "F768-ALT":
        b = [(2 * Q * 767 * j + 383) // (2 * 383) for j in range(K)]
        a = list(b)
        for j in range(1, K - 1):
            a[j] = b[j] + Q // 2 if j % 2 == 0 else b[j] - Q // 2
    else:
        fail(f"unrecognized positive fixture {fixture_id}")

    result = canonical_input(T, K, u, a)
    validate_positive(fixture_id, result)
    return result


def validate_positive(fixture_id: str, value: dict) -> None:
    T, K, u, a = value["T"], value["K"], value["u"], value["a"]
    expected_k = min(384, 16 * (T // 16))
    if K != expected_k:
        fail(f"{fixture_id}: K does not satisfy the frozen rule")
    if u != canonical_uniform(T, K):
        fail(f"{fixture_id}: u is not canonical")
    if len(u) != K or len(a) != K:
        fail(f"{fixture_id}: array length mismatch")
    if a[0] != 0 or a[-1] != Q * (T - 1):
        fail(f"{fixture_id}: endpoint mismatch")
    if any(a[j] > a[j + 1] for j in range(K - 1)):
        fail(f"{fixture_id}: a is not nondecreasing")


def negative_input(fixture_id: str, positives: dict[str, dict]) -> dict:
    base = positives["G17-E2"]
    if fixture_id == "N-T15":
        return canonical_input(15, 0, [], [])
    if fixture_id == "N-K":
        u = canonical_uniform(17, 17)
        return canonical_input(17, 17, u, q_times(u))

    T, K = base["T"], base["K"]
    u, a = list(base["u"]), list(base["a"])
    if fixture_id == "N-U-LEN":
        u.pop()
    elif fixture_id == "N-A-LEN":
        a.pop()
    elif fixture_id == "N-U-CANON":
        u[7] = 8
    elif fixture_id == "N-A-END":
        a[0] = 1
    elif fixture_id == "N-A-ORDER":
        a[7], a[8] = a[8], a[7]
    elif fixture_id == "N-INFEASIBLE":
        T, K = 1534, 384
        u = canonical_uniform(T, K)
        a = q_times(u)
    elif fixture_id == "N-ARITH":
        source = positives["F768-U"]
        T, K = source["T"], source["K"]
        u, a = list(source["u"]), list(source["a"])
        a[1] = 1 << 127
    else:
        fail(f"unrecognized negative fixture {fixture_id}")
    return canonical_input(T, K, u, a)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def require_matrix(matrix: object) -> tuple[list[dict], list[dict], list[dict]]:
    if not isinstance(matrix, dict):
        fail("matrix root is not an object")
    if matrix.get("schema_version") != "duca-p0-identity-fixture-matrix-v001":
        fail("matrix schema mismatch")
    if matrix.get("status") != "FROZEN_NOT_EXECUTED" or matrix.get("matrix_closed") is not True:
        fail("matrix is not frozen and closed")
    if matrix.get("queue_message_id") != "msg-20260812T185607Z-f6cc921b7d40":
        fail("matrix authoring queue mismatch")
    if matrix.get("parent_decision") != PARENT_DECISION or matrix.get("Q") != Q:
        fail("matrix authority or Q mismatch")
    positive = matrix.get("positive")
    negative = matrix.get("negative")
    mutations = matrix.get("certificate_mutations")
    if not all(isinstance(group, list) for group in (positive, negative, mutations)):
        fail("matrix groups are not arrays")
    if tuple(item.get("id") for item in positive) != POSITIVE_IDS:
        fail("positive fixture order or membership mismatch")
    if tuple(item.get("id") for item in negative) != NEGATIVE_IDS:
        fail("negative fixture order or membership mismatch")
    if tuple(item.get("id") for item in mutations) != MUTATION_IDS:
        fail("certificate-mutation order or membership mismatch")
    return positive, negative, mutations


def write_once(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def materialize(matrix_path: Path, output_root: Path) -> None:
    if output_root.exists():
        fail(f"output root already exists: {output_root}")
    matrix = json.loads(matrix_path.read_bytes().decode("utf-8"))
    positive_defs, negative_defs, mutation_defs = require_matrix(matrix)

    temporary = output_root.with_name(output_root.name + ".partial")
    if temporary.exists():
        fail(f"partial root already exists: {temporary}")
    temporary.mkdir(parents=False)
    try:
        positive_inputs: dict[str, dict] = {}
        ordered: list[tuple[str, str, str, dict]] = []
        for definition in positive_defs:
            fixture_id = definition["id"]
            value = positive_input(definition)
            positive_inputs[fixture_id] = value
            ordered.append((fixture_id, "positive", definition["expected"]["status"], value))
        for definition in negative_defs:
            fixture_id = definition["id"]
            value = negative_input(fixture_id, positive_inputs)
            ordered.append((fixture_id, "negative", definition["required_code"], value))

        jsonl = b"".join(canonical_bytes(value) for _, _, _, value in ordered)
        write_once(temporary / "DUCA_P0_SEALED_FIXTURES-v001.jsonl", jsonl)

        index = {
            "schema_version": "duca-p0-sealed-fixture-index-v001",
            "status": "MATERIALIZED_NOT_EXECUTED",
            "queue_message_id": QUEUE_MESSAGE_ID,
            "parent_decision": PARENT_DECISION,
            "input_field_order": ["T", "K", "Q", "u", "a"],
            "encoding": "UTF-8",
            "line_ending": "LF",
            "fixture_count": len(ordered),
            "positive_count": len(positive_defs),
            "negative_count": len(negative_defs),
            "ordered_fixtures": [
                {
                    "line": line,
                    "id": fixture_id,
                    "category": category,
                    "expected_status": expected_status,
                }
                for line, (fixture_id, category, expected_status, _) in enumerate(ordered, start=1)
            ],
            "production_or_reference_invoked": False,
            "reference_expectations_calculated": False,
            "comparison_or_test_executed": False,
            "data_model_checkpoint_metric_access": False,
            "gpu_cuda_slurm_browser_access": False,
            "scope_deviation": "none",
        }
        write_once(temporary / "DUCA_P0_SEALED_FIXTURE_INDEX-v001.json", canonical_bytes(index))

        mutation_manifest = {
            "schema_version": "duca-p0-sealed-certificate-mutations-v001",
            "status": "DEFINITIONS_FROZEN_NOT_EXECUTED",
            "queue_message_id": QUEUE_MESSAGE_ID,
            "parent_decision": PARENT_DECISION,
            "mutation_count": len(mutation_defs),
            "ordered_mutations": mutation_defs,
            "certificate_or_projector_invoked": False,
            "scope_deviation": "none",
        }
        write_once(
            temporary / "DUCA_P0_SEALED_CERTIFICATE_MUTATIONS-v001.json",
            canonical_bytes(mutation_manifest),
        )
        temporary.rename(output_root)
    except BaseException:
        # Fail closed: a partial directory is never promoted to the output root.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize frozen DUCA P0 fixture recipes")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        materialize(arguments.matrix, arguments.out_root)
    except (MaterializationError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"MATERIALIZATION_BLOCKED: {error}")
        return 2
    print("MATERIALIZATION_COMPLETE")
    print(f"FIXTURE_COUNT={len(POSITIVE_IDS) + len(NEGATIVE_IDS)}")
    print(f"MUTATION_COUNT={len(MUTATION_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
