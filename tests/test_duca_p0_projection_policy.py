"""Authored-not-run conformance tests for PRO_P0_PROJECTION_POLICY-v001."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import struct
from pathlib import Path

import pytest


if os.name == "nt":
    pytest.skip("DUCA Torch tests run only in the authorized Linux OpenTAD environment", allow_module_level=True)

torch = pytest.importorskip("torch")

from opentad.models.selectors import pc_ot_mras_prebackbone_frame_selector as production
from opentad.utils.temporal_positions import canonical_endpoint_uniform_positions


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "tests" / "duca_projection"
SPEC_PATH = ARTIFACT_DIR / "DUCA_P0_NONCONSTANT_PROJECTION_SPEC-v001.json"
FIXTURE_PATH = ARTIFACT_DIR / "DUCA_P0_PROJECTION_FIXTURES-v001.json"
REFERENCE_PATH = ARTIFACT_DIR / "DUCA_P0_PROJECTION_REFERENCE-v001.py"


def _load_reference():
    module_spec = importlib.util.spec_from_file_location("duca_projection_reference_v001", REFERENCE_PATH)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


REFERENCE = _load_reference()
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
FIXTURE_ARTIFACT = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
FIXTURES = {fixture["fixture_id"]: fixture for fixture in FIXTURE_ARTIFACT["fixtures"]}
Q = 1_048_576


def _half_up(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _piecewise(j: int, anchors: tuple[tuple[int, int], ...]) -> int:
    for (j0, a0), (j1, a1) in zip(anchors, anchors[1:]):
        if j0 <= j <= j1:
            return a0 + _half_up((a1 - a0) * (j - j0), j1 - j0)
    raise AssertionError("fixture anchor coverage is incomplete")


def _fixture_certificate(certificate) -> tuple[tuple[int, ...], int, int, int, int]:
    return (
        tuple(certificate.positions),
        certificate.e2,
        certificate.e_infinity,
        certificate.e1,
        certificate.u1,
    )


def _reference_certificate(certificate: dict[str, object]) -> tuple[tuple[int, ...], int, int, int, int]:
    return (
        tuple(certificate["p"]),
        int(certificate["E2"]),
        int(certificate["E_infinity"]),
        int(certificate["E1"]),
        int(certificate["U1"]),
    )


def _assert_fixture_feasible(fixture: dict[str, object], positions: tuple[int, ...]) -> None:
    T = int(fixture["T"])
    K = int(fixture["K"])
    uniform = tuple(fixture["u"])
    assert len(positions) == K
    assert positions[0] == 0 and positions[-1] == T - 1
    assert len(set(positions)) == K
    strides = tuple(right - left for left, right in zip(positions, positions[1:]))
    assert all(stride in SPEC["feasible_set"]["adjacent_strides"] for stride in strides)
    assert max(abs(position - anchor) for position, anchor in zip(positions, uniform)) <= 16


def test_projection_domain_and_typed_fail_closed():
    with pytest.raises(production.DUCAProjectionError, match="at least 16"):
        production.decode_duca_density_positions_v001(torch.zeros(15))
    with pytest.raises(production.DUCAProjectionError, match="finite"):
        production.decode_duca_density_positions_v001(torch.tensor([0.0] * 767 + [float("nan")]))
    with pytest.raises(production.DUCAProjectionError, match="requested_k=384"):
        production.decode_duca_density_positions_v001(torch.zeros(768), requested_k=383)

    uniform = canonical_endpoint_uniform_positions(385, 384)
    targets = tuple(Q * position for position in uniform)
    with pytest.raises(production.DUCAProjectionError, match="effective K"):
        production.project_duca_fixed_targets_v001(385, 383, uniform[:-1], targets[:-1])
    with pytest.raises(production.DUCAProjectionError, match="not canonical"):
        production.project_duca_fixed_targets_v001(385, 384, (0,) * 384, targets)
    malformed_targets = list(targets)
    malformed_targets[10] = malformed_targets[9] - 1
    with pytest.raises(production.DUCAProjectionError, match="declared domain"):
        production.project_duca_fixed_targets_v001(385, 384, uniform, malformed_targets)

    large_uniform = canonical_endpoint_uniform_positions(2000, 384)
    large_targets = tuple(Q * position for position in large_uniform)
    with pytest.raises(production.DUCAProjectionError, match="no feasible state"):
        production.project_duca_fixed_targets_v001(2000, 384, large_uniform, large_targets)

    certificate = production.DUCAProjectionCertificate(
        positions=tuple(uniform),
        e2=1,
        e_infinity=0,
        e1=0,
        u1=0,
    )
    with pytest.raises(production.DUCAProjectionError, match="recomputed exactly"):
        production._validate_duca_projection_certificate_v001(
            certificate,
            385,
            384,
            uniform,
            targets,
        )

    class Incomparable:
        def __eq__(self, _other):
            return False

        def __lt__(self, _other):
            return False

    with pytest.raises(production.DUCAProjectionError, match="comparison inconsistency"):
        production._duca_compare_keys_v001((Incomparable(),), (Incomparable(),))


def test_projector_frozen_negative_failure_codes():
    T, K = 17, 16
    uniform = canonical_endpoint_uniform_positions(T, K)
    targets = tuple(Q * position for position in uniform)

    invalid_uniform = list(uniform)
    invalid_uniform[7] = 8
    invalid_endpoint_targets = list(targets)
    invalid_endpoint_targets[0] = 1
    invalid_order_targets = list(targets)
    invalid_order_targets[7], invalid_order_targets[8] = (
        invalid_order_targets[8],
        invalid_order_targets[7],
    )
    infeasible_T, infeasible_K = 1534, 384
    infeasible_uniform = canonical_endpoint_uniform_positions(infeasible_T, infeasible_K)
    infeasible_targets = tuple(Q * position for position in infeasible_uniform)
    arithmetic_T, arithmetic_K = 768, 384
    arithmetic_uniform = canonical_endpoint_uniform_positions(arithmetic_T, arithmetic_K)
    arithmetic_targets = [Q * position for position in arithmetic_uniform]
    arithmetic_targets[1] = 1 << 127

    cases = (
        ("INVALID_T_LT_16", (15, 0, (), ())),
        (
            "K_EFF_MISMATCH",
            (
                T,
                17,
                canonical_endpoint_uniform_positions(T, 17),
                tuple(Q * position for position in canonical_endpoint_uniform_positions(T, 17)),
            ),
        ),
        ("U_LENGTH_MISMATCH", (T, K, uniform[:-1], targets)),
        ("A_LENGTH_MISMATCH", (T, K, uniform, targets[:-1])),
        ("U_CANONICAL_MISMATCH", (T, K, invalid_uniform, targets)),
        ("A_ENDPOINT_MISMATCH", (T, K, uniform, invalid_endpoint_targets)),
        ("A_ORDER_MISMATCH", (T, K, uniform, invalid_order_targets)),
        ("INFEASIBLE", (infeasible_T, infeasible_K, infeasible_uniform, infeasible_targets)),
        (
            "INTEGER_RANGE_OR_OVERFLOW",
            (arithmetic_T, arithmetic_K, arithmetic_uniform, arithmetic_targets),
        ),
    )
    for expected_code, arguments in cases:
        with pytest.raises(production.DUCAProjectionError) as raised:
            production.project_duca_fixed_targets_v001(*arguments)
        assert raised.value.code == expected_code


def test_exact_constant_bypasses_nonconstant_projector(monkeypatch):
    def forbidden_projector(*_args, **_kwargs):
        raise AssertionError("the nonconstant projector was entered")

    monkeypatch.setattr(production, "project_duca_fixed_targets_v001", forbidden_projector)
    decoded = production.decode_duca_density_positions_v001(torch.zeros(768))
    assert tuple(decoded.tolist()) == canonical_endpoint_uniform_positions(768, 384)


def test_spec_uniform_formula_and_witness():
    assert SPEC["Q"] == Q
    assert SPEC["objective"]["lexicographic_order"] == [
        "E2",
        "E_infinity",
        "E1",
        "U1",
        "p_1_through_p_K_minus_2",
    ]
    assert SPEC["candidate_order"]["L_j_r"] == "max(0,r+1,u_j-16,T-1-4*(K-1-j))"
    assert SPEC["candidate_order"]["R_j_r"] == "min(T-1,r+4,u_j+16,T-1-(K-1-j))"
    assert tuple(SPEC["required_fixture_ids"]) == tuple(FIXTURES)

    for fixture in FIXTURES.values():
        T, K = fixture["T"], fixture["K"]
        assert fixture["status"] == "AUTHORED_NOT_RUN"
        assert fixture["Q"] == Q
        assert len(fixture["u"]) == K and len(fixture["a"]) == K
        assert tuple(fixture["u"]) == canonical_endpoint_uniform_positions(T, K)
        _assert_fixture_feasible(fixture, tuple(fixture["u"]))


def test_fixture_arrays_are_literal_recipe_materializations():
    exact = FIXTURES["t385_exact_s37"]
    assert exact["a"] == [Q * (j if j <= 37 else j + 1) for j in range(384)]

    tie = FIXTURES["t385_full_key_tie"]
    expected_tie = [Q * position for position in tie["u"]]
    expected_tie[191] = Q * 191 + 3 * Q // 4
    expected_tie[192] = Q * 192 + Q // 4
    assert tie["a"] == expected_tie

    smooth = FIXTURES["t768_smooth_monotone"]
    assert smooth["a"] == [
        0 if j == 0 else Q * 767 if j == 383 else _half_up(Q * 767 * j * j, 383 * 383)
        for j in range(384)
    ]

    alternating = FIXTURES["t768_alternating"]
    expected_alternating = []
    for j in range(384):
        if j == 0:
            expected_alternating.append(0)
        elif j == 383:
            expected_alternating.append(Q * 767)
        else:
            base = _half_up(Q * 767 * j, 383)
            expected_alternating.append(base + (Q // 4 if j % 2 == 0 else -Q // 4))
    assert alternating["a"] == expected_alternating

    single = FIXTURES["t768_single_boundary"]
    single_anchors = ((0, 0), (128, 320 * Q), (255, 448 * Q), (383, 767 * Q))
    assert single["a"] == [_piecewise(j, single_anchors) for j in range(384)]

    dual = FIXTURES["t768_dual_boundary"]
    dual_anchors = (
        (0, 0),
        (80, 200 * Q),
        (150, 270 * Q),
        (233, 497 * Q),
        (303, 567 * Q),
        (383, 767 * Q),
    )
    assert dual["a"] == [_piecewise(j, dual_anchors) for j in range(384)]

    saturating = FIXTURES["t768_constraint_saturating"]
    expected_sat = []
    for j, anchor in enumerate(saturating["u"]):
        if j >= 192:
            displacement = 0
        else:
            remainder = j % 24
            displacement = 2 * remainder if remainder <= 8 else 24 - remainder
        expected_sat.append(Q * (anchor + displacement))
    assert saturating["a"] == expected_sat

    half_up = FIXTURES["t768_fixed_point_half_up"]
    expected_half_up = [Q * position for position in half_up["u"]]
    expected_half_up[100] += 1
    assert half_up["a"] == expected_half_up


def test_inverse_cdf_serialization_boundary(monkeypatch):
    captured = {}
    original = production.project_duca_fixed_targets_v001

    def capture(T, K, u, a):
        captured["tuple"] = (T, K, tuple(u), tuple(a))
        return original(T, K, u, a)

    monkeypatch.setattr(production, "project_duca_fixed_targets_v001", capture)
    logits = torch.linspace(-0.25, 0.75, 385, dtype=torch.float64)
    production.decode_duca_density_positions_v001(logits)
    T, K, uniform, targets = captured["tuple"]
    assert (T, K) == (385, 384)
    assert uniform == canonical_endpoint_uniform_positions(T, K)
    assert targets[0] == 0 and targets[-1] == Q * (T - 1)
    assert all(left <= right for left, right in zip(targets, targets[1:]))
    assert all(type(value) is int for value in targets)


def test_binary64_fixed_point_half_up_bits():
    fixture = FIXTURES["t768_fixed_point_half_up"]
    payload = fixture["source_binary64_hex_by_index"]["100"]
    value = struct.unpack(">d", int(payload, 16).to_bytes(8, "big"))[0]
    expected = fixture["a"][100]
    assert production._duca_binary64_to_fixed_half_up_v001(value) == expected
    assert round(value * Q) == expected - 1
    with pytest.raises(production.DUCAProjectionError):
        production._duca_binary64_to_fixed_half_up_v001(float("nan"))
    with pytest.raises(production.DUCAProjectionError):
        production._duca_binary64_to_fixed_half_up_v001(-0.5)


@pytest.mark.parametrize("fixture_id", ["t385_exact_s37", "t385_full_key_tie"])
def test_exhaustive_t385_feasible_set_and_optimum(fixture_id):
    fixture = FIXTURES[fixture_id]
    exhaustive = REFERENCE.exhaustive_reference_t385_v001(
        fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    reference = REFERENCE.reference_project_duca_fixed_targets_v001(
        fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    production_result = production.project_duca_fixed_targets_v001(
        fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    expected = fixture["expected"]
    assert exhaustive["feasible_sequence_count"] == 383
    assert exhaustive["minimizer_count"] == 1
    assert exhaustive["selected_long_stride_index"] == fixture["expected_selected_long_stride_index"]
    assert len(exhaustive["every_loser_first_difference"]) == 382
    assert tuple(expected["p"]) == tuple(exhaustive["winner"]["p"])
    assert _reference_certificate(reference) == _fixture_certificate(production_result)
    assert _fixture_certificate(production_result) == (
        tuple(expected["p"]),
        int(expected["E2_decimal"]),
        int(expected["E_infinity_decimal"]),
        int(expected["E1_decimal"]),
        int(expected["U1_decimal"]),
    )


def test_exact_tie_resolves_by_frozen_phi():
    fixture = FIXTURES["t385_full_key_tie"]
    winner = tuple(fixture["expected"]["p"])
    competitor = tuple(j if j <= 190 else j + 1 for j in range(384))
    winner_key = production._duca_projection_objective_v001(winner, fixture["u"], fixture["a"])
    competitor_key = production._duca_projection_objective_v001(competitor, fixture["u"], fixture["a"])
    assert winner_key[:4] == competitor_key[:4]
    assert winner_key[4] < competitor_key[4]
    assert winner_key[:4] == (687194767360, 786432, 1048576, 1)


@pytest.mark.parametrize("fixture_id", tuple(FIXTURES))
def test_full_scale_feasibility_certificates(fixture_id):
    fixture = FIXTURES[fixture_id]
    production_result = production.project_duca_fixed_targets_v001(
        fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    _assert_fixture_feasible(fixture, tuple(production_result.positions))
    recomputed = production._duca_projection_objective_v001(
        production_result.positions,
        fixture["u"],
        fixture["a"],
    )
    assert recomputed == (
        production_result.e2,
        production_result.e_infinity,
        production_result.e1,
        production_result.u1,
        production_result.positions[1:-1],
    )


@pytest.mark.parametrize("fixture_id", tuple(FIXTURES))
def test_full_scale_production_reference_identity(fixture_id):
    fixture = FIXTURES[fixture_id]
    production_result = production.project_duca_fixed_targets_v001(
        fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    reference_result = REFERENCE.reference_project_duca_fixed_targets_v001(
        fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    assert _fixture_certificate(production_result) == _reference_certificate(reference_result)


@pytest.mark.parametrize(
    "fixture_id",
    [
        "t385_exact_s37",
        "t385_full_key_tie",
        "t768_constraint_saturating",
        "t768_fixed_point_half_up",
    ],
)
def test_production_reference_objective_and_certificate_identity(fixture_id):
    fixture = FIXTURES[fixture_id]
    expected = fixture["expected"]
    production_result = production.project_duca_fixed_targets_v001(
        fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    reference_result = REFERENCE.reference_objective_certificate_v001(
        expected["p"], fixture["T"], fixture["K"], fixture["u"], fixture["a"]
    )
    assert _fixture_certificate(production_result) == _reference_certificate(reference_result)
    assert _fixture_certificate(production_result)[1:] == (
        int(expected["E2_decimal"]),
        int(expected["E_infinity_decimal"]),
        int(expected["E1_decimal"]),
        int(expected["U1_decimal"]),
    )


def test_spec_candidate_order_and_reference_identity():
    fixture = FIXTURES["t768_fixed_point_half_up"]
    T, K, uniform = fixture["T"], fixture["K"], tuple(fixture["u"])
    for rank, predecessor in ((1, 0), (100, uniform[99]), (192, uniform[191]), (383, uniform[382])):
        production_candidates = tuple(
            production._duca_projection_candidates_v001(
                rank, predecessor, T, K, uniform[rank]
            )
        )
        reference_candidates = REFERENCE._next_positions(rank, predecessor, T, K, uniform)
        assert production_candidates == tuple(sorted(production_candidates))
        assert production_candidates == reference_candidates


def test_reference_has_no_production_import_or_shared_solver_symbols():
    source = REFERENCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported_roots.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "opentad" not in imported_roots
    assert "torch" not in imported_roots
    assert "_duca_compare_keys_v001" not in source
    assert "_duca_projection_candidates_v001" not in source
    assert "_duca_projection_certificate_v001" not in source


def test_nonconstant_path_has_no_repair_fallback_or_second_decoder():
    projector_source = inspect.getsource(production.project_duca_fixed_targets_v001)
    decoder_source = inspect.getsource(production.decode_duca_density_positions_v001)
    for source in (projector_source, decoder_source):
        for forbidden in (
            "round(",
            ".clamp(",
            "dedup",
            "greedy",
            "legacy",
            "fallback",
            "retry",
        ):
            assert forbidden not in source.lower()
    assert decoder_source.count("project_duca_fixed_targets_v001(") == 1
    assert "canonical_endpoint_uniform_positions" in decoder_source
    assert "torch.equal" in decoder_source
    assert "DUCAProjectionError" in decoder_source


def test_authored_not_run_receipt_boundary():
    assert FIXTURE_ARTIFACT["status"] == "AUTHORED_NOT_RUN"
    assert FIXTURE_ARTIFACT["execution"] == "NOT_AUTHORIZED"
    assert SPEC["status"] == "AUTHORED_NOT_RUN"
    assert SPEC["execution"] == "NOT_AUTHORIZED"
    unknown_expected = {
        fixture_id: FIXTURES[fixture_id]["expected"]
        for fixture_id in (
            "t768_smooth_monotone",
            "t768_alternating",
            "t768_single_boundary",
            "t768_dual_boundary",
        )
    }
    for expected in unknown_expected.values():
        assert set(expected.values()) == {None}
