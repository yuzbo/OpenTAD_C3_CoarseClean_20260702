"""Independent exact DUCA P0 projector reference — AUTHORED, NOT EXECUTED.

This source is owned by Evaluator.  It imports no OpenTAD, production projector,
selector, objective, candidate generator, certificate, Torch, CUDA, dataset,
model, evaluator, metric, or training module.  The only shared computational
inputs are canonical JSON fixtures and the frozen mathematical specification.

Do not execute this version without a new durable Evaluator queue that cites a
separate Pro execution authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


Q_FROZEN = 1 << 20
SIGNED_128_MAX = (1 << 127) - 1


class ProjectionError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def half_up_nonnegative(numerator: int, denominator: int) -> int:
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise ProjectionError("INTEGER_RANGE_OR_OVERFLOW", "half-up operands must be integers")
    if numerator < 0 or denominator <= 0:
        raise ProjectionError("INTEGER_RANGE_OR_OVERFLOW", "half-up domain violation")
    return (2 * numerator + denominator) // (2 * denominator)


def effective_k(T: int) -> int:
    if not isinstance(T, int) or isinstance(T, bool):
        raise ProjectionError("INTEGER_RANGE_OR_OVERFLOW", "T must be an integer")
    if T < 16:
        raise ProjectionError("INVALID_T_LT_16", "T is below 16")
    return min(384, 16 * (T // 16))


def canonical_uniform(T: int, K: int) -> tuple[int, ...]:
    return tuple(half_up_nonnegative(j * (T - 1), K - 1) for j in range(K))


def _checked_input_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProjectionError("INTEGER_RANGE_OR_OVERFLOW", f"{field} is not an integer")
    if value < 0 or value > SIGNED_128_MAX:
        raise ProjectionError("INTEGER_RANGE_OR_OVERFLOW", f"{field} is outside signed-128 input range")
    return value


def parse_and_validate(payload: object) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...]]:
    if not isinstance(payload, dict):
        raise ProjectionError("INTEGER_RANGE_OR_OVERFLOW", "fixture root must be an object")
    T = _checked_input_integer(payload.get("T"), "T")
    K_raw = _checked_input_integer(payload.get("K"), "K")
    Q = _checked_input_integer(payload.get("Q"), "Q")
    if Q != Q_FROZEN:
        raise ProjectionError("INTEGER_RANGE_OR_OVERFLOW", "Q differs from 2^20")
    K_expected = effective_k(T)
    if K_raw != K_expected:
        raise ProjectionError("K_EFF_MISMATCH", f"expected {K_expected}, observed {K_raw}")
    u_raw = payload.get("u")
    a_raw = payload.get("a")
    if not isinstance(u_raw, list):
        raise ProjectionError("U_LENGTH_MISMATCH", "u must be an array")
    if not isinstance(a_raw, list):
        raise ProjectionError("A_LENGTH_MISMATCH", "a must be an array")
    if len(u_raw) != K_raw:
        raise ProjectionError("U_LENGTH_MISMATCH", f"expected {K_raw}, observed {len(u_raw)}")
    if len(a_raw) != K_raw:
        raise ProjectionError("A_LENGTH_MISMATCH", f"expected {K_raw}, observed {len(a_raw)}")
    u = tuple(_checked_input_integer(x, f"u[{j}]") for j, x in enumerate(u_raw))
    a = tuple(_checked_input_integer(x, f"a[{j}]") for j, x in enumerate(a_raw))
    expected_u = canonical_uniform(T, K_raw)
    if u != expected_u:
        raise ProjectionError("U_CANONICAL_MISMATCH", "u differs from canonical integer-half-up uniform")
    if a[0] != 0 or a[-1] != Q * (T - 1):
        raise ProjectionError("A_ENDPOINT_MISMATCH", "a endpoints are not 0 and Q*(T-1)")
    if any(a[j] > a[j + 1] for j in range(K_raw - 1)):
        raise ProjectionError("A_ORDER_MISMATCH", "a is not nondecreasing")
    if (T - 1) > 4 * (K_raw - 1):
        raise ProjectionError("INFEASIBLE", "endpoint span exceeds maximum total stride")
    return T, K_raw, Q, u, a


def objective_key(p: tuple[int, ...], u: tuple[int, ...], a: tuple[int, ...], Q: int) -> tuple:
    errors = tuple(Q * p[j] - a[j] for j in range(1, len(p) - 1))
    E2 = sum(e * e for e in errors)
    E_infinity = max((abs(e) for e in errors), default=0)
    E1 = sum(abs(e) for e in errors)
    U1 = sum(abs(p[j] - u[j]) for j in range(1, len(p) - 1))
    return (E2, E_infinity, E1, U1, *p[1:-1])


def feasibility(p: tuple[int, ...], T: int, K: int, u: tuple[int, ...]) -> dict:
    strides = tuple(p[j + 1] - p[j] for j in range(K - 1)) if len(p) == K else ()
    displacement = tuple(abs(p[j] - u[j]) for j in range(K)) if len(p) == K else ()
    passed = (
        len(p) == K
        and p[0] == 0
        and p[-1] == T - 1
        and all(1 <= gap <= 4 for gap in strides)
        and all(value <= 16 for value in displacement)
    )
    return {
        "count": len(p),
        "first": p[0] if p else None,
        "last": p[-1] if p else None,
        "minimum_stride": min(strides) if strides else None,
        "maximum_stride": max(strides) if strides else None,
        "maximum_uniform_displacement": max(displacement) if displacement else None,
        "passed": passed,
    }


def _candidate_bounds(T: int, K: int, u: tuple[int, ...], j: int, predecessor: int) -> tuple[int, int]:
    remaining = K - 1 - j
    low = max(0, predecessor + 1, u[j] - 16, (T - 1) - 4 * remaining)
    high = min(T - 1, predecessor + 4, u[j] + 16, (T - 1) - remaining)
    return low, high


def enumerate_feasible(T: int, K: int, u: tuple[int, ...]):
    prefix = [0]

    def visit(j: int):
        if j == K - 1:
            if 1 <= (T - 1) - prefix[-1] <= 4:
                yield tuple(prefix + [T - 1])
            return
        low, high = _candidate_bounds(T, K, u, j, prefix[-1])
        for position in range(low, high + 1):
            prefix.append(position)
            yield from visit(j + 1)
            prefix.pop()

    yield from visit(1)


def exhaustive_reference(T: int, K: int, Q: int, u: tuple[int, ...], a: tuple[int, ...]) -> dict:
    winner = None
    winner_key = None
    scalar_tie_set = []
    count = 0
    previous = None
    for candidate in enumerate_feasible(T, K, u):
        if previous is not None and candidate <= previous:
            raise ProjectionError("CANDIDATE_ORDER_VIOLATION", "exhaustive sequences are not ascending")
        previous = candidate
        count += 1
        key = objective_key(candidate, u, a, Q)
        if winner_key is None or key < winner_key:
            winner = candidate
            winner_key = key
    if winner is None or winner_key is None:
        raise ProjectionError("INFEASIBLE", "no feasible sequence")
    scalar_prefix = winner_key[:4]
    for candidate in enumerate_feasible(T, K, u):
        if objective_key(candidate, u, a, Q)[:4] == scalar_prefix:
            scalar_tie_set.append(candidate)
    return {
        "p": winner,
        "key": winner_key,
        "feasible_count": count,
        "scalar_tie_set": tuple(scalar_tie_set),
        "candidate_order_ascending": True,
        "method": "independent_exhaustive_ascending",
    }


def _build_transitions(T: int, K: int, u: tuple[int, ...]) -> list[dict[int, tuple[int, ...]]]:
    transitions: list[dict[int, tuple[int, ...]]] = [dict() for _ in range(K)]
    reachable = {0}
    for j in range(1, K):
        next_reachable = set()
        for predecessor in sorted(reachable):
            if j == K - 1:
                candidates = (T - 1,) if 1 <= (T - 1) - predecessor <= 4 else ()
            else:
                low, high = _candidate_bounds(T, K, u, j, predecessor)
                candidates = tuple(range(low, high + 1)) if low <= high else ()
            if tuple(sorted(candidates)) != candidates or len(set(candidates)) != len(candidates):
                raise ProjectionError("CANDIDATE_ORDER_VIOLATION", f"layer {j}, predecessor {predecessor}")
            transitions[j][predecessor] = candidates
            next_reachable.update(candidates)
        reachable = next_reachable
    if (T - 1) not in reachable:
        raise ProjectionError("INFEASIBLE", "endpoint is unreachable")
    return transitions


def _node_cost(kind: str, j: int, p: int, K: int, Q: int, u: tuple[int, ...], a: tuple[int, ...]) -> int:
    if j == 0 or j == K - 1:
        return 0
    error = Q * p - a[j]
    if kind == "E2":
        return error * error
    if kind == "E1":
        return abs(error)
    if kind == "U1":
        return abs(p - u[j])
    raise ValueError(kind)


def _shortest_forward(transitions, K, Q, u, a, kind, edge_allowed):
    distances = [dict() for _ in range(K)]
    distances[0][0] = 0
    for j in range(1, K):
        for predecessor in sorted(distances[j - 1]):
            base = distances[j - 1][predecessor]
            for position in transitions[j].get(predecessor, ()):
                if not edge_allowed(j, predecessor, position):
                    continue
                value = base + _node_cost(kind, j, position, K, Q, u, a)
                old = distances[j].get(position)
                if old is None or value < old:
                    distances[j][position] = value
    return distances


def _shortest_backward(transitions, T, K, Q, u, a, kind, edge_allowed):
    distances = [dict() for _ in range(K)]
    distances[K - 1][T - 1] = 0
    for j in range(K - 1, 0, -1):
        for predecessor in sorted(transitions[j]):
            values = []
            for position in transitions[j][predecessor]:
                if position not in distances[j] or not edge_allowed(j, predecessor, position):
                    continue
                values.append(_node_cost(kind, j, position, K, Q, u, a) + distances[j][position])
            if values:
                distances[j - 1][predecessor] = min(values)
    return distances


def _optimal_edge_filter(forward, backward, optimum, K, Q, u, a, kind, base_filter):
    def allowed(j, predecessor, position):
        if not base_filter(j, predecessor, position):
            return False
        if predecessor not in forward[j - 1] or position not in backward[j]:
            return False
        return (
            forward[j - 1][predecessor]
            + _node_cost(kind, j, position, K, Q, u, a)
            + backward[j][position]
            == optimum
        )
    return allowed


def exact_dp_reference(T: int, K: int, Q: int, u: tuple[int, ...], a: tuple[int, ...]) -> dict:
    transitions = _build_transitions(T, K, u)
    allow_all = lambda j, predecessor, position: True
    forward_e2 = _shortest_forward(transitions, K, Q, u, a, "E2", allow_all)
    if (T - 1) not in forward_e2[K - 1]:
        raise ProjectionError("INFEASIBLE", "no feasible sequence")
    optimum_e2 = forward_e2[K - 1][T - 1]

    absolute_errors = sorted({
        abs(Q * position - a[j])
        for j in range(1, K - 1)
        for position in range(max(0, u[j] - 16), min(T - 1, u[j] + 16) + 1)
    }) or [0]
    low_index, high_index = 0, len(absolute_errors) - 1
    optimum_einf = absolute_errors[-1]
    while low_index <= high_index:
        middle = (low_index + high_index) // 2
        threshold = absolute_errors[middle]
        bounded = lambda j, predecessor, position, B=threshold: (
            j == K - 1 or abs(Q * position - a[j]) <= B
        )
        trial = _shortest_forward(transitions, K, Q, u, a, "E2", bounded)
        if trial[K - 1].get(T - 1) == optimum_e2:
            optimum_einf = threshold
            high_index = middle - 1
        else:
            low_index = middle + 1

    bounded = lambda j, predecessor, position: (
        j == K - 1 or abs(Q * position - a[j]) <= optimum_einf
    )
    forward_e2 = _shortest_forward(transitions, K, Q, u, a, "E2", bounded)
    backward_e2 = _shortest_backward(transitions, T, K, Q, u, a, "E2", bounded)
    e2_edges = _optimal_edge_filter(
        forward_e2, backward_e2, optimum_e2, K, Q, u, a, "E2", bounded
    )

    forward_e1 = _shortest_forward(transitions, K, Q, u, a, "E1", e2_edges)
    optimum_e1 = forward_e1[K - 1][T - 1]
    backward_e1 = _shortest_backward(transitions, T, K, Q, u, a, "E1", e2_edges)
    e1_edges = _optimal_edge_filter(
        forward_e1, backward_e1, optimum_e1, K, Q, u, a, "E1", e2_edges
    )

    forward_u1 = _shortest_forward(transitions, K, Q, u, a, "U1", e1_edges)
    optimum_u1 = forward_u1[K - 1][T - 1]
    backward_u1 = _shortest_backward(transitions, T, K, Q, u, a, "U1", e1_edges)
    final_edges = _optimal_edge_filter(
        forward_u1, backward_u1, optimum_u1, K, Q, u, a, "U1", e1_edges
    )

    p = [0]
    for j in range(1, K):
        candidates = transitions[j].get(p[-1], ())
        selected = next((value for value in candidates if final_edges(j, p[-1], value)), None)
        if selected is None:
            raise ProjectionError("INFEASIBLE", f"no optimal continuation at layer {j}")
        p.append(selected)
    output = tuple(p)
    key = objective_key(output, u, a, Q)
    if key[:4] != (optimum_e2, optimum_einf, optimum_e1, optimum_u1):
        raise ProjectionError("CERTIFICATE_REJECTED", "independent staged optimum does not recompute")
    return {
        "p": output,
        "key": key,
        "candidate_order_ascending": True,
        "root_optimum": key,
        "method": "independent_staged_exact_dag_dp",
    }


def reference_project(payload: object, force_exhaustive: bool = False) -> dict:
    try:
        T, K, Q, u, a = parse_and_validate(payload)
        if force_exhaustive or T in (17, 385):
            solved = exhaustive_reference(T, K, Q, u, a)
        else:
            solved = exact_dp_reference(T, K, Q, u, a)
        certificate = feasibility(solved["p"], T, K, u)
        if not certificate["passed"]:
            raise ProjectionError("CERTIFICATE_REJECTED", "reference output is infeasible")
        key = solved["key"]
        return {
            "typed_status": "PASS",
            "p": list(solved["p"]),
            "feasibility": certificate,
            "objective": {
                "E2": str(key[0]),
                "E_infinity": str(key[1]),
                "E1": str(key[2]),
                "U1": str(key[3]),
                "interior_position_vector": list(key[4:]),
            },
            "global_optimality": {
                "method": solved["method"],
                "feasible_count": solved.get("feasible_count"),
                "scalar_tie_set": [list(x) for x in solved.get("scalar_tie_set", ())],
                "root_optimum": [str(x) if i < 4 else x for i, x in enumerate(solved.get("root_optimum", key))],
            },
            "candidate_order_ascending": solved["candidate_order_ascending"],
            "reference_independence": "EVALUATOR_OWNED_NON_IMPORTING",
            "scope_deviation": "none",
        }
    except ProjectionError as error:
        return {
            "typed_status": error.code,
            "detail": error.detail,
            "reference_independence": "EVALUATOR_OWNED_NON_IMPORTING",
            "scope_deviation": "none",
        }


def validate_external_certificate(payload: object, certificate: object) -> dict:
    if not isinstance(certificate, dict):
        return {"typed_status": "CERTIFICATE_REJECTED", "detail": "certificate root is not an object"}
    traces = certificate.get("candidate_expansions", [])
    if not isinstance(traces, list):
        return {"typed_status": "CERTIFICATE_REJECTED", "detail": "candidate expansions are not an array"}
    for expansion in traces:
        candidates = expansion.get("candidates") if isinstance(expansion, dict) else None
        if not isinstance(candidates, list) or candidates != sorted(candidates) or len(candidates) != len(set(candidates)):
            return {"typed_status": "CANDIDATE_ORDER_VIOLATION", "detail": "candidate expansion is not strictly ascending"}
    try:
        T, K, Q, u, a = parse_and_validate(payload)
        p_raw = certificate.get("p")
        if not isinstance(p_raw, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in p_raw):
            raise ProjectionError("CERTIFICATE_REJECTED", "p is not an integer array")
        p = tuple(p_raw)
        feasible = feasibility(p, T, K, u)
        if not feasible["passed"]:
            raise ProjectionError("CERTIFICATE_REJECTED", "p violates frozen feasibility")
        key = objective_key(p, u, a, Q)
        objective = certificate.get("objective")
        expected = {
            "E2": str(key[0]),
            "E_infinity": str(key[1]),
            "E1": str(key[2]),
            "U1": str(key[3]),
            "interior_position_vector": list(key[4:]),
        }
        if objective != expected:
            raise ProjectionError("CERTIFICATE_REJECTED", "objective does not recompute exactly")
        independent = reference_project(payload, force_exhaustive=T in (17, 385))
        if independent.get("typed_status") != "PASS" or independent.get("p") != list(p):
            raise ProjectionError("CERTIFICATE_REJECTED", "certificate is not the independent optimum")
        return {"typed_status": "PASS", "scope_deviation": "none"}
    except ProjectionError as error:
        return {"typed_status": error.code, "detail": error.detail, "scope_deviation": "none"}


def _load_canonical_json(path: Path) -> object:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent DUCA P0 projector reference")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force-exhaustive", action="store_true")
    parser.add_argument("--certificate", type=Path)
    args = parser.parse_args()
    payload = _load_canonical_json(args.input)
    if args.certificate is None:
        result = reference_project(payload, force_exhaustive=args.force_exhaustive)
    else:
        result = validate_external_certificate(payload, _load_canonical_json(args.certificate))
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0 if result.get("typed_status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
