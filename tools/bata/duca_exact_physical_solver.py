from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import math
import time
from typing import Any, Iterable, Mapping, Sequence

from tools.bata.duca_allocation_families import (
    AllocationContractError,
    FamilySelection,
    PhysicalAxis,
    ResolvedPhysicalCap,
    build_family_selection,
    effective_budget,
    exact_uniform_positions,
    physical_gap_report,
    validate_physical_selection,
)


@dataclass(frozen=True)
class QuantizedScores:
    values: tuple[int, ...]
    scale: int
    rounding: str = "decimal_round_half_even"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdditiveSolveResult:
    positions: tuple[int, ...]
    quantized_objective: int
    raw_score_sum: float
    quantized_scores: QuantizedScores
    solver_status: str = "OPTIMAL"
    solver_identity: str = "duca_exact_k_physical_dag_dp_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruthObjectiveSpec:
    boundary_radii: tuple[int, ...] = (0, 1, 2, 4)
    short_action_max_length: float = 16.0
    distance_scale: int = 1000
    lex_block_size: int = 20
    position_tie_break: bool = True

    def __post_init__(self) -> None:
        if not self.boundary_radii or any(int(value) < 0 for value in self.boundary_radii):
            raise AllocationContractError("boundary radii must be non-empty and non-negative")
        if tuple(sorted(set(int(value) for value in self.boundary_radii))) != self.boundary_radii:
            raise AllocationContractError("boundary radii must be unique and increasing")
        if not math.isfinite(float(self.short_action_max_length)) or self.short_action_max_length < 0:
            raise AllocationContractError("short_action_max_length must be finite and non-negative")
        if int(self.distance_scale) < 1:
            raise AllocationContractError("distance_scale must be positive")
        if int(self.lex_block_size) < 1 or int(self.lex_block_size) > 30:
            raise AllocationContractError("lex_block_size must lie in [1,30]")
        if not isinstance(self.position_tie_break, bool):
            raise AllocationContractError("position_tie_break must be boolean")


@dataclass(frozen=True)
class GroundTruthSolveResult:
    positions: tuple[int, ...]
    objective_vector: Mapping[str, int]
    metric_upper_envelopes: Mapping[str, int]
    solver_status: str
    solver_identity: str
    solver_message: str
    mip_gap: float | None
    exact: bool
    privileged: bool = True
    deployable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryBurstSolveResult:
    positions: tuple[int, ...]
    required_positions: tuple[int, ...]
    radius: int
    quota: int
    endpoint_contracts: tuple[Mapping[str, Any], ...]
    invalid_endpoint_count: int
    residual_fill_count: int
    background_selected_count: int
    background_component_count: int
    uniform_overlap: int
    max_unselected_hole: int
    solver_status: str = "OPTIMAL"
    solver_identity: str = "duca_r0_exact_quota_physical_milp_v1"
    solver_message: str = ""
    mip_gap: float | None = 0.0
    exact: bool = True
    privileged: bool = True
    deployable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _BoundaryBurstMilpResult:
    positions: tuple[int, ...]
    solver_message: str
    mip_gap: float


def _solve_boundary_burst_exact_quota_milp(
    axis: PhysicalAxis,
    endpoint_specs: Sequence[Mapping[str, Any]],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
    enforce_global_coverage: bool,
) -> _BoundaryBurstMilpResult:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except (ImportError, AttributeError) as exc:
        raise AllocationContractError(
            "SciPy HiGHS milp is required for the exact boundary-burst Oracle"
        ) from exc

    valid_len = axis.valid_len
    budget = effective_budget(valid_len, requested_budget)
    path_variables: dict[tuple[int, int], int] = {}
    next_variable = valid_len
    if enforce_global_coverage:
        source = -1
        sink = valid_len
        for right in range(valid_len):
            if cap.allows(axis, 0, right):
                path_variables[(source, right)] = next_variable
                next_variable += 1
            for left in range(right):
                if cap.allows(axis, left, right):
                    path_variables[(left, right)] = next_variable
                    next_variable += 1
        for left in range(valid_len):
            if cap.allows(axis, left, valid_len - 1):
                path_variables[(left, sink)] = next_variable
                next_variable += 1

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float],
        minimum: float,
        maximum: float,
    ) -> None:
        row = len(lower)
        for column, value in coefficients.items():
            if value != 0:
                rows.append(row)
                columns.append(int(column))
                data.append(float(value))
        lower.append(float(minimum))
        upper.append(float(maximum))

    add_constraint({position: 1 for position in range(valid_len)}, budget, budget)
    if enforce_global_coverage:
        add_constraint({0: 1}, 1, 1)
        add_constraint({valid_len - 1: 1}, 1, 1)
        add_constraint(
            {
                variable: 1
                for (left, _), variable in path_variables.items()
                if left == -1
            },
            1,
            1,
        )
        add_constraint(
            {
                variable: 1
                for (_, right), variable in path_variables.items()
                if right == valid_len
            },
            1,
            1,
        )
        for position in range(valid_len):
            incoming = {
                variable: -1
                for (_, right), variable in path_variables.items()
                if right == position
            }
            incoming[position] = 1
            add_constraint(incoming, 0, 0)
            outgoing = {
                variable: -1
                for (left, _), variable in path_variables.items()
                if left == position
            }
            outgoing[position] = 1
            add_constraint(outgoing, 0, 0)

    for spec in endpoint_specs:
        center = int(spec["center"])
        neighborhood = tuple(int(value) for value in spec["neighborhood"])
        add_constraint({center: 1}, 1, 1)
        add_constraint(
            {position: 1 for position in neighborhood},
            int(spec["quota"]),
            math.inf,
        )
        if bool(spec["bilateral_applicable"]):
            add_constraint(
                {int(position): 1 for position in spec["left_candidates"]},
                1,
                math.inf,
            )
            add_constraint(
                {int(position): 1 for position in spec["right_candidates"]},
                1,
                math.inf,
            )

    matrix = csr_matrix(
        (data, (rows, columns)),
        shape=(len(lower), next_variable),
        dtype=float,
    )
    constraints = LinearConstraint(
        matrix,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    uniform = set(exact_uniform_positions(valid_len, requested_budget))
    primary_scale = valid_len * valid_len + 1
    objective = np.zeros(next_variable, dtype=float)
    for position in range(valid_len):
        objective[position] = -(
            (primary_scale if position in uniform else 0)
            + valid_len
            - position
        )
    result = milp(
        c=objective,
        integrality=np.ones(next_variable, dtype=np.uint8),
        bounds=Bounds(
            np.zeros(next_variable, dtype=float),
            np.ones(next_variable, dtype=float),
        ),
        constraints=constraints,
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if result.status != 0 or result.x is None:
        raise AllocationContractError(
            "boundary-burst exact quota MILP was not OPTIMAL: "
            f"{getattr(result, 'message', 'unknown HiGHS failure')}"
        )
    mip_gap = float(getattr(result, "mip_gap", math.inf))
    if not math.isfinite(mip_gap) or mip_gap != 0.0:
        raise AllocationContractError(
            "boundary-burst exact quota MILP did not prove a zero MIP gap"
        )
    values = np.asarray(result.x, dtype=float)
    rounded = np.rint(values)
    if values.shape != (next_variable,) or not np.all(np.isfinite(values)):
        raise AllocationContractError(
            "boundary-burst exact quota MILP returned invalid variables"
        )
    if float(np.max(np.abs(values - rounded), initial=0.0)) > 1.0e-6:
        raise AllocationContractError(
            "boundary-burst exact quota MILP violates integrality"
        )
    positions = tuple(
        position for position in range(valid_len) if int(rounded[position]) == 1
    )
    if len(positions) != budget:
        raise AllocationContractError(
            "boundary-burst exact quota MILP terminal solution violates exact-K"
        )
    return _BoundaryBurstMilpResult(
        positions=positions,
        solver_message=str(result.message),
        mip_gap=mip_gap,
    )


def quantize_scores(
    scores: Sequence[float | int],
    *,
    scale: int = 1_000_000,
) -> QuantizedScores:
    scale = int(scale)
    if scale < 1:
        raise AllocationContractError("score quantization scale must be positive")
    quantized: list[int] = []
    multiplier = Decimal(scale)
    for index, value in enumerate(scores):
        number = float(value)
        if not math.isfinite(number):
            raise AllocationContractError(f"score[{index}] must be finite")
        converted = (Decimal(str(number)) * multiplier).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
        quantized.append(int(converted))
    return QuantizedScores(values=tuple(quantized), scale=scale)


def solve_additive_exact_k_physical(
    axis: PhysicalAxis,
    scores: Sequence[float | int],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
    quantization_scale: int = 1_000_000,
) -> AdditiveSolveResult:
    """Solve exact-K additive allocation under the physical source-to-sink cap."""

    raw_scores = _finite_score_vector(scores, axis.valid_len)
    quantized = quantize_scores(raw_scores, scale=quantization_scale)
    budget = effective_budget(axis.valid_len, requested_budget)

    predecessors: list[tuple[int, ...]] = []
    for right in range(axis.valid_len):
        predecessors.append(
            tuple(left for left in range(right) if cap.allows(axis, left, right))
        )

    previous_scores: dict[int, int] = {}
    previous_ranks: dict[int, int] = {}
    parents: list[dict[int, int | None]] = [{}, {}]
    for position in range(axis.valid_len):
        if cap.allows(axis, 0, position):
            previous_scores[position] = quantized.values[position]
            parents[1][position] = None
    for rank, position in enumerate(sorted(previous_scores)):
        previous_ranks[position] = rank

    for count in range(2, budget + 1):
        current_scores: dict[int, int] = {}
        current_parents: dict[int, int | None] = {}
        for position in range(axis.valid_len):
            candidates = [value for value in predecessors[position] if value in previous_scores]
            if not candidates:
                continue
            predecessor = min(
                candidates,
                key=lambda value: (-previous_scores[value], previous_ranks[value]),
            )
            current_scores[position] = previous_scores[predecessor] + quantized.values[position]
            current_parents[position] = predecessor
        if not current_scores:
            raise AllocationContractError(
                f"no exact physical path reaches selection count {count}/{budget}"
            )
        ordered = sorted(
            current_scores,
            key=lambda position: (previous_ranks[int(current_parents[position])], position),
        )
        current_ranks = {position: rank for rank, position in enumerate(ordered)}
        previous_scores = current_scores
        previous_ranks = current_ranks
        parents.append(current_parents)

    terminals = [
        position
        for position in previous_scores
        if cap.allows(axis, position, axis.valid_len - 1)
    ]
    if not terminals:
        raise AllocationContractError("no exact-K physical path reaches the sink")
    terminal = min(
        terminals,
        key=lambda position: (-previous_scores[position], previous_ranks[position]),
    )

    reverse_positions = [terminal]
    current = terminal
    for count in range(budget, 1, -1):
        predecessor = parents[count][current]
        if predecessor is None:
            raise RuntimeError("physical DP parent chain ended before recovering exact-K")
        current = predecessor
        reverse_positions.append(current)
    positions = tuple(reversed(reverse_positions))
    if len(positions) != budget:
        raise RuntimeError("physical DP did not recover exactly K positions")
    return AdditiveSolveResult(
        positions=positions,
        quantized_objective=previous_scores[terminal],
        raw_score_sum=sum(raw_scores[index] for index in positions),
        quantized_scores=quantized,
    )


def solve_boundary_burst_oracle(
    axis: PhysicalAxis,
    gt_segments: Sequence[Sequence[float | int]],
    gt_boundary_validity: Sequence[Sequence[bool | int]],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
    radius: int,
    quota: int,
    max_unselected_hole: int,
    enforce_global_coverage: bool = True,
) -> BoundaryBurstSolveResult:
    """Construct the constrained R0 Oracle: local bursts plus global fill.

    This diagnostic never participates in training or inference. The exact
    MILP jointly chooses quota positions so nearby endpoints can share samples
    while the same exact-K and physical coverage contract remains enforced.
    """

    radius = int(radius)
    quota = int(quota)
    max_unselected_hole = int(max_unselected_hole)
    if radius < 1 or quota < 3:
        raise AllocationContractError("boundary-burst radius/quota are too small")
    if max_unselected_hole < 0:
        raise AllocationContractError("max_unselected_hole must be non-negative")
    segments = _canonical_gt_segments(gt_segments, axis.valid_len)
    validity = tuple(tuple(bool(value) for value in row) for row in gt_boundary_validity)
    if len(validity) != len(segments) or any(len(row) != 2 for row in validity):
        raise AllocationContractError("boundary validity must be [num_segments,2]")

    required: set[int] = set()
    endpoint_specs: list[dict[str, Any]] = []
    invalid_endpoint_count = 0
    for segment_index, ((start, end), valid_row) in enumerate(zip(segments, validity)):
        centers = (
            int(math.floor(start)),
            int(math.ceil(end) - 1),
        )
        for side, (center, is_valid) in enumerate(zip(centers, valid_row)):
            if not is_valid:
                invalid_endpoint_count += 1
                continue
            center = min(max(center, 0), axis.valid_len - 1)
            left = tuple(range(max(0, center - radius), center))
            right = tuple(range(center + 1, min(axis.valid_len, center + radius + 1)))
            neighborhood = tuple(range(max(0, center - radius), min(axis.valid_len, center + radius + 1)))
            target_quota = min(quota, len(neighborhood))
            endpoint_required = (center,)
            required.add(center)
            endpoint_specs.append(
                {
                    "segment_index": int(segment_index),
                    "endpoint": "start" if side == 0 else "end",
                    "center": int(center),
                    "radius": radius,
                    "quota": int(target_quota),
                    "neighborhood": neighborhood,
                    "left_candidates": left,
                    "right_candidates": right,
                    "required_positions": endpoint_required,
                    "bilateral_applicable": bool(left and right),
                }
            )

    # Projected families pin both dense-axis endpoints so the physical DP and
    # the declared edge-hole contract agree.  The unrestricted R0 upper bound
    # deliberately omits this scaffold; it keeps exact-K and the same burst
    # objective, but allows the remaining budget to move anywhere.
    if enforce_global_coverage:
        required.update({0, axis.valid_len - 1})
    budget = effective_budget(axis.valid_len, requested_budget)
    if len(required) > budget:
        raise AllocationContractError(
            f"boundary-burst endpoint centers exceed K_eff: {len(required)} > {budget}"
        )
    uniform = set(exact_uniform_positions(axis.valid_len, requested_budget))
    solved = _solve_boundary_burst_exact_quota_milp(
        axis,
        endpoint_specs,
        requested_budget=requested_budget,
        cap=cap,
        enforce_global_coverage=enforce_global_coverage,
    )
    selected = set(solved.positions)
    missing = sorted(required - selected)
    if missing:
        raise AllocationContractError(
            f"boundary-burst required positions are infeasible under K/G: {missing[:16]}"
        )
    if enforce_global_coverage:
        validate_physical_selection(
            axis,
            solved.positions,
            requested_budget=requested_budget,
            cap=cap,
        )
    gap_report = physical_gap_report(axis, solved.positions)
    if (
        enforce_global_coverage
        and gap_report.dense_max_unselected_hole > max_unselected_hole
    ):
        raise AllocationContractError("boundary-burst violates candidate-axis max-hole")

    validated_contracts: list[Mapping[str, Any]] = []
    burst_union: set[int] = set()
    for spec in endpoint_specs:
        neighborhood = set(spec["neighborhood"])
        left = set(spec["left_candidates"])
        right = set(spec["right_candidates"])
        local_count = len(selected & neighborhood)
        left_count = len(selected & left)
        right_count = len(selected & right)
        quota_pass = local_count >= int(spec["quota"])
        bilateral_pass = (
            not bool(spec["bilateral_applicable"])
            or (left_count >= 1 and right_count >= 1)
        )
        if not quota_pass or not bilateral_pass:
            raise AllocationContractError("boundary-burst endpoint contract was not satisfied")
        burst_union.update(neighborhood)
        validated_contracts.append(
            {
                **spec,
                "selected_positions_in_radius": tuple(sorted(selected & neighborhood)),
                "selected_in_radius": local_count,
                "selected_left": left_count,
                "selected_right": right_count,
                "quota_pass": quota_pass,
                "bilateral_pass": bilateral_pass,
            }
        )

    background = sorted(set(range(axis.valid_len)) - burst_union)
    components: list[tuple[int, ...]] = []
    for position in background:
        if not components or position != components[-1][-1] + 1:
            components.append((position,))
        else:
            components[-1] = components[-1] + (position,)
    if enforce_global_coverage:
        for component in components:
            if len(component) > max_unselected_hole and not (selected & set(component)):
                raise AllocationContractError(
                    "boundary-burst global background fill is incomplete"
                )

    return BoundaryBurstSolveResult(
        positions=solved.positions,
        required_positions=tuple(sorted(required)),
        radius=radius,
        quota=quota,
        endpoint_contracts=tuple(validated_contracts),
        invalid_endpoint_count=invalid_endpoint_count,
        residual_fill_count=len(selected - required),
        background_selected_count=len(selected & set(background)),
        background_component_count=len(components),
        uniform_overlap=len(selected & uniform),
        max_unselected_hole=gap_report.dense_max_unselected_hole,
        solver_message=solved.solver_message,
        mip_gap=solved.mip_gap,
    )


def solve_additive_one_per_cell_physical(
    axis: PhysicalAxis,
    scores: Sequence[float | int],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
    one_per_cell_bounds: Sequence[Sequence[int]],
    quantization_scale: int = 1_000_000,
) -> AdditiveSolveResult:
    """Solve additive one-per-cell selection under the same physical cap."""

    raw_scores = _finite_score_vector(scores, axis.valid_len)
    quantized = quantize_scores(raw_scores, scale=quantization_scale)
    budget = effective_budget(axis.valid_len, requested_budget)
    cell_bounds = _canonical_one_per_cell_bounds(
        one_per_cell_bounds,
        valid_len=axis.valid_len,
        budget=budget,
    )
    if cell_bounds is None:
        raise AllocationContractError("one-per-cell bounds are required")

    previous: dict[int, tuple[int, tuple[int, ...]]] = {}
    for cell_index, (start, end) in enumerate(cell_bounds):
        current: dict[int, tuple[int, tuple[int, ...]]] = {}
        for position in range(start, end):
            candidates: list[tuple[int, tuple[int, ...]]] = []
            if cell_index == 0:
                if cap.allows(axis, 0, position):
                    candidates.append((quantized.values[position], (position,)))
            else:
                for predecessor, (prefix_score, prefix) in previous.items():
                    if cap.allows(axis, predecessor, position):
                        candidates.append(
                            (
                                prefix_score + quantized.values[position],
                                prefix + (position,),
                            )
                        )
            if candidates:
                current[position] = min(
                    candidates,
                    key=lambda item: (-item[0], item[1]),
                )
        if not current:
            raise AllocationContractError(
                f"one-per-cell physical path is infeasible at cell {cell_index}"
            )
        previous = current

    terminals = [
        item
        for position, item in previous.items()
        if cap.allows(axis, position, axis.valid_len - 1)
    ]
    if not terminals:
        raise AllocationContractError("one-per-cell physical path cannot reach the sink")
    objective, positions = min(terminals, key=lambda item: (-item[0], item[1]))
    return AdditiveSolveResult(
        positions=positions,
        quantized_objective=objective,
        raw_score_sum=sum(raw_scores[index] for index in positions),
        quantized_scores=quantized,
        solver_identity="duca_exact_one_per_cell_physical_dp_v1",
    )


def solve_additive_unrestricted(
    axis: PhysicalAxis,
    scores: Sequence[float | int],
    *,
    requested_budget: int,
    quantization_scale: int = 1_000_000,
) -> AdditiveSolveResult:
    raw_scores = _finite_score_vector(scores, axis.valid_len)
    quantized = quantize_scores(raw_scores, scale=quantization_scale)
    budget = effective_budget(axis.valid_len, requested_budget)
    positions = tuple(
        sorted(
            sorted(
                range(axis.valid_len),
                key=lambda index: (-quantized.values[index], index),
            )[:budget]
        )
    )
    return AdditiveSolveResult(
        positions=positions,
        quantized_objective=sum(quantized.values[index] for index in positions),
        raw_score_sum=sum(raw_scores[index] for index in positions),
        quantized_scores=quantized,
        solver_identity="duca_exact_k_unrestricted_topk_v1",
    )


def select_family_d(
    axis: PhysicalAxis,
    scores: Sequence[float | int],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
    quantization_scale: int = 1_000_000,
) -> tuple[FamilySelection, AdditiveSolveResult]:
    solved = solve_additive_exact_k_physical(
        axis,
        scores,
        requested_budget=requested_budget,
        cap=cap,
        quantization_scale=quantization_scale,
    )
    family = build_family_selection(
        family="D_global_exact_k_physical_gap",
        axis=axis,
        positions=solved.positions,
        requested_budget=requested_budget,
        score_sum=solved.raw_score_sum,
        exact=True,
        deployable=True,
        privileged=False,
        solver_status=solved.solver_status,
        cap=cap,
    )
    if not family.physical_cap_compliant:
        raise RuntimeError("exact physical solver emitted a cap-violating selection")
    return family, solved


def solve_ground_truth_lexicographic(
    axis: PhysicalAxis,
    gt_segments: Sequence[Sequence[float | int]],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap | None,
    objective_spec: GroundTruthObjectiveSpec = GroundTruthObjectiveSpec(),
    one_per_cell_bounds: Sequence[Sequence[int]] | None = None,
    compute_upper_envelopes: bool = False,
    time_limit_seconds: float | None = None,
) -> GroundTruthSolveResult:
    """Solve the privileged GT allocation with sequentially pinned objectives."""

    try:
        import numpy as np
        from scipy import __version__ as scipy_version
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except (ImportError, AttributeError) as exc:
        raise AllocationContractError("SciPy HiGHS milp is required for exact GT ceilings") from exc

    segments = _canonical_gt_segments(gt_segments, axis.valid_len)
    budget = effective_budget(axis.valid_len, requested_budget)
    cell_bounds = _canonical_one_per_cell_bounds(
        one_per_cell_bounds,
        valid_len=axis.valid_len,
        budget=budget,
    )
    model = _GroundTruthMilpModel(
        axis=axis,
        segments=segments,
        budget=budget,
        cap=cap,
        spec=objective_spec,
        one_per_cell_bounds=cell_bounds,
    )
    matrix = csr_matrix(
        (model.constraint_data, (model.constraint_rows, model.constraint_cols)),
        shape=(len(model.constraint_lower), model.variable_count),
        dtype=float,
    )
    constraints: list[Any] = [
        LinearConstraint(
            matrix,
            np.asarray(model.constraint_lower, dtype=float),
            np.asarray(model.constraint_upper, dtype=float),
        )
    ]
    bounds = Bounds(
        np.zeros(model.variable_count, dtype=float),
        np.ones(model.variable_count, dtype=float),
    )
    integrality = np.asarray(model.integrality, dtype=np.uint8)
    options: dict[str, Any] = {"presolve": True, "mip_rel_gap": 0.0}
    deadline: float | None = None
    if time_limit_seconds is not None:
        if not math.isfinite(float(time_limit_seconds)) or float(time_limit_seconds) <= 0.0:
            raise AllocationContractError("time_limit_seconds must be finite and positive")
        deadline = time.monotonic() + float(time_limit_seconds)

    def current_options() -> dict[str, Any]:
        current = dict(options)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise AllocationContractError("GT MILP total solver deadline was exceeded")
            current["time_limit"] = remaining
        return current

    pinned_values: dict[str, int] = {}
    last_result: Any = None
    last_values: Any = None
    last_integer_values: Any = None
    last_mip_gap: float | None = None
    pinned_coefficients: dict[str, Mapping[int, int]] = {}

    def numeric_result_field(name: str, result: Any, field: str) -> float:
        value = getattr(result, field, None)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
        ):
            raise AllocationContractError(
                f"GT MILP objective {name} did not report numeric {field}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise AllocationContractError(
                f"GT MILP objective {name} reported non-finite {field}"
            )
        return number

    def validate_optimal_result(
        name: str,
        result: Any,
    ) -> tuple[Any, Any, float, float, float]:
        if result.status != 0 or result.x is None:
            message = str(getattr(result, "message", "unknown HiGHS failure"))
            raise AllocationContractError(f"GT MILP objective {name} was not OPTIMAL: {message}")

        mip_gap = numeric_result_field(name, result, "mip_gap")
        if mip_gap != 0.0:
            raise AllocationContractError(
                f"GT MILP objective {name} did not prove a zero MIP gap"
            )
        objective_value = numeric_result_field(name, result, "fun")
        objective_bound = numeric_result_field(name, result, "mip_dual_bound")

        values = np.asarray(result.x, dtype=float)
        if values.ndim != 1 or values.shape[0] != model.variable_count:
            raise AllocationContractError(
                f"GT MILP objective {name} returned an invalid solution shape"
            )
        if not np.all(np.isfinite(values)):
            raise AllocationContractError(
                f"GT MILP objective {name} returned non-finite variables"
            )
        integer_indices = np.flatnonzero(integrality != 0)
        integer_values = np.rint(values)
        if integer_indices.size:
            residual = float(
                np.max(np.abs(values[integer_indices] - integer_values[integer_indices]))
            )
            if residual > 1.0e-6:
                raise AllocationContractError(
                    f"GT MILP objective {name} violates integrality"
                )
        if np.any(values < -1.0e-6) or np.any(values > 1.0 + 1.0e-6):
            raise AllocationContractError(
                f"GT MILP objective {name} violates variable bounds: "
                f"min={float(values.min()):.12g}, max={float(values.max()):.12g}"
            )
        return values, integer_values, mip_gap, objective_value, objective_bound

    def encoded_integer_objective_value(
        name: str,
        coefficient: Mapping[int, int],
        integer_values: Any,
    ) -> int:
        variables = tuple(int(variable) for variable in coefficient)
        if any(integrality[variable] == 0 for variable in variables):
            raise RuntimeError(f"GT MILP objective {name} is not integer encoded")
        return sum(
            int(coefficient[variable]) * int(integer_values[variable])
            for variable in variables
        )

    def positions_from_integer_values(name: str, integer_values: Any) -> tuple[int, ...]:
        positions = tuple(
            index
            for index in range(axis.valid_len)
            if int(integer_values[model.x_index(index)]) == 1
        )
        if len(positions) != budget:
            raise AllocationContractError(f"GT MILP objective {name} violates exact-K")
        return positions

    def exact_objective_from_positions(
        name: str,
        coefficient: Mapping[int, int],
        integer_values: Any,
    ) -> int:
        positions = positions_from_integer_values(name, integer_values)
        semantic_values = model.objective_values_from_positions(positions)
        if name in semantic_values:
            semantic_value = semantic_values[name]
            if all(integrality[int(variable)] != 0 for variable in coefficient):
                encoded_value = encoded_integer_objective_value(
                    name,
                    coefficient,
                    integer_values,
                )
                if encoded_value != semantic_value:
                    raise AllocationContractError(
                        f"GT MILP objective {name} is inconsistent with selected positions"
                    )
            return semantic_value
        return encoded_integer_objective_value(name, coefficient, integer_values)

    def validate_objective_certificate(
        name: str,
        integer_value: int,
        *,
        maximize: bool,
        objective_value: float,
        objective_bound: float,
    ) -> None:
        signed_value = -int(integer_value) if maximize else int(integer_value)
        for label, reported in (
            ("fun", objective_value),
            ("mip_dual_bound", objective_bound),
        ):
            nearest_integer = int(round(reported))
            if abs(reported - nearest_integer) >= 0.25:
                raise AllocationContractError(
                    f"GT MILP objective {name} has ambiguous {label}"
                )
            if nearest_integer != signed_value:
                raise AllocationContractError(
                    f"GT MILP objective {name} contradicts solver {label}"
                )

    def solve_and_pin(name: str, coefficient: Mapping[int, int], *, maximize: bool) -> int:
        nonlocal last_integer_values, last_mip_gap, last_result, last_values
        objective = np.zeros(model.variable_count, dtype=float)
        for variable, value in coefficient.items():
            objective[int(variable)] = -int(value) if maximize else int(value)
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=current_options(),
        )
        (
            values,
            integer_values,
            mip_gap,
            objective_value,
            objective_bound,
        ) = validate_optimal_result(name, result)
        integer_value = exact_objective_from_positions(
            name,
            coefficient,
            integer_values,
        )
        validate_objective_certificate(
            name,
            integer_value,
            maximize=maximize,
            objective_value=objective_value,
            objective_bound=objective_bound,
        )
        columns = np.fromiter(coefficient.keys(), dtype=int)
        pin_values = np.fromiter(
            (int(coefficient[index]) for index in coefficient),
            dtype=float,
        )
        pin_row = csr_matrix(
            (pin_values, (np.zeros(len(columns), dtype=int), columns)),
            shape=(1, model.variable_count),
        )
        constraints.append(LinearConstraint(pin_row, [integer_value], [integer_value]))
        pinned_values[name] = integer_value
        pinned_coefficients[name] = coefficient
        last_result = result
        last_values = values
        last_integer_values = integer_values
        last_mip_gap = mip_gap
        return integer_value

    objective_sequence = model.objective_sequence()
    upper_envelopes: dict[str, int] = {}
    if compute_upper_envelopes:
        for name, coefficient, maximize in objective_sequence:
            standalone = milp(
                c=model.numpy_objective(coefficient, maximize=maximize),
                integrality=integrality,
                bounds=bounds,
                constraints=constraints[:1],
                options=current_options(),
            )
            (
                values,
                integer_values,
                _,
                objective_value,
                objective_bound,
            ) = validate_optimal_result(
                f"upper envelope {name}",
                standalone,
            )
            upper_envelope = exact_objective_from_positions(
                name,
                coefficient,
                integer_values,
            )
            validate_objective_certificate(
                f"upper envelope {name}",
                upper_envelope,
                maximize=maximize,
                objective_value=objective_value,
                objective_bound=objective_bound,
            )
            upper_envelopes[name] = upper_envelope

    for name, coefficient, maximize in objective_sequence:
        solve_and_pin(name, coefficient, maximize=maximize)

    if objective_spec.position_tie_break:
        for block_start in range(0, axis.valid_len, objective_spec.lex_block_size):
            block_end = min(axis.valid_len, block_start + objective_spec.lex_block_size)
            coefficient = {
                model.x_index(position): 1 << (block_end - position - 1)
                for position in range(block_start, block_end)
            }
            solve_and_pin(
                f"lex_block_{block_start:04d}_{block_end:04d}",
                coefficient,
                maximize=True,
            )

    if (
        last_result is None
        or last_values is None
        or last_integer_values is None
        or last_mip_gap is None
    ):
        raise RuntimeError("GT MILP completed without a terminal result")
    positions = tuple(
        index
        for index in range(axis.valid_len)
        if int(last_integer_values[model.x_index(index)]) == 1
    )
    if len(positions) != budget:
        raise AllocationContractError("GT MILP terminal solution violates exact-K")
    if cell_bounds is not None and any(
        sum(start <= position < end for position in positions) != 1
        for start, end in cell_bounds
    ):
        raise AllocationContractError(
            "GT MILP terminal solution violates one-per-cell constraints"
        )
    if cap is not None:
        from tools.bata.duca_allocation_families import validate_physical_selection

        validate_physical_selection(
            axis,
            positions,
            requested_budget=requested_budget,
            cap=cap,
        )
    terminal_semantic_values = model.objective_values_from_positions(positions)
    for name, expected_value in pinned_values.items():
        if name in terminal_semantic_values:
            actual_value = terminal_semantic_values[name]
        else:
            actual_value = encoded_integer_objective_value(
                name,
                pinned_coefficients[name],
                last_integer_values,
            )
        if actual_value != expected_value:
            raise AllocationContractError(
                f"GT MILP terminal solution violates pinned objective {name}"
            )
    return GroundTruthSolveResult(
        positions=positions,
        objective_vector=pinned_values,
        metric_upper_envelopes=upper_envelopes,
        solver_status="OPTIMAL",
        solver_identity=(
            f"scipy_highs_milp_{scipy_version}"
            + ("_one_per_cell" if cell_bounds is not None else "")
            + (
                "_position_lexicographic"
                if objective_spec.position_tie_break
                else "_semantic_optimum"
            )
        ),
        solver_message=str(last_result.message),
        mip_gap=last_mip_gap,
        exact=True,
    )


class _GroundTruthMilpModel:
    def __init__(
        self,
        *,
        axis: PhysicalAxis,
        segments: tuple[tuple[float, float], ...],
        budget: int,
        cap: ResolvedPhysicalCap | None,
        spec: GroundTruthObjectiveSpec,
        one_per_cell_bounds: tuple[tuple[int, int], ...] | None,
    ) -> None:
        self.axis = axis
        self.segments = segments
        self.budget = budget
        self.cap = cap
        self.spec = spec
        self.one_per_cell_bounds = one_per_cell_bounds
        self._next_variable = axis.valid_len
        self.integrality: list[int] = [1] * axis.valid_len
        self.constraint_rows: list[int] = []
        self.constraint_cols: list[int] = []
        self.constraint_data: list[float] = []
        self.constraint_lower: list[float] = []
        self.constraint_upper: list[float] = []
        self.path_variables: dict[tuple[int, int], int] = {}
        self.hit_variables: dict[tuple[int, int], int] = {}
        self.both_variables: dict[tuple[int, int], int] = {}
        self.assignment_variables: dict[tuple[int, int], int] = {}
        self.support_variables: dict[int, int] = {}
        self.endpoints = tuple(value for segment in segments for value in segment)
        self._build()

    @property
    def variable_count(self) -> int:
        return self._next_variable

    def x_index(self, position: int) -> int:
        return int(position)

    def _new_variable(self, *, integer: bool) -> int:
        index = self._next_variable
        self._next_variable += 1
        self.integrality.append(1 if integer else 0)
        return index

    def _add_constraint(
        self,
        coefficients: Mapping[int, float],
        lower: float,
        upper: float,
    ) -> None:
        row = len(self.constraint_lower)
        for column, value in coefficients.items():
            if value != 0:
                self.constraint_rows.append(row)
                self.constraint_cols.append(int(column))
                self.constraint_data.append(float(value))
        self.constraint_lower.append(float(lower))
        self.constraint_upper.append(float(upper))

    def _build(self) -> None:
        self._add_constraint(
            {self.x_index(position): 1 for position in range(self.axis.valid_len)},
            self.budget,
            self.budget,
        )
        if self.one_per_cell_bounds is not None:
            for start, end in self.one_per_cell_bounds:
                self._add_constraint(
                    {
                        self.x_index(position): 1
                        for position in range(start, end)
                    },
                    1,
                    1,
                )
        if self.cap is not None:
            self._build_path_contract()
        self._build_boundary_variables()
        self._build_distance_assignments()
        self._build_short_action_support()

    def _build_path_contract(self) -> None:
        source = -1
        sink = self.axis.valid_len
        edges: list[tuple[int, int]] = []
        for right in range(self.axis.valid_len):
            if self.cap is not None and self.cap.allows(self.axis, 0, right):
                edges.append((source, right))
            for left in range(right):
                if self.cap is not None and self.cap.allows(self.axis, left, right):
                    edges.append((left, right))
        for left in range(self.axis.valid_len):
            if self.cap is not None and self.cap.allows(self.axis, left, self.axis.valid_len - 1):
                edges.append((left, sink))
        for edge in edges:
            self.path_variables[edge] = self._new_variable(integer=True)

        self._add_constraint(
            {
                variable: 1
                for (left, _), variable in self.path_variables.items()
                if left == source
            },
            1,
            1,
        )
        self._add_constraint(
            {
                variable: 1
                for (_, right), variable in self.path_variables.items()
                if right == sink
            },
            1,
            1,
        )
        for position in range(self.axis.valid_len):
            incoming = {
                variable: -1
                for (_, right), variable in self.path_variables.items()
                if right == position
            }
            incoming[self.x_index(position)] = 1
            self._add_constraint(incoming, 0, 0)

            outgoing = {
                variable: -1
                for (left, _), variable in self.path_variables.items()
                if left == position
            }
            outgoing[self.x_index(position)] = 1
            self._add_constraint(outgoing, 0, 0)

    def _build_boundary_variables(self) -> None:
        for endpoint_index, endpoint in enumerate(self.endpoints):
            for radius_index, radius in enumerate(self.spec.boundary_radii):
                variable = self._new_variable(integer=True)
                self.hit_variables[(endpoint_index, radius_index)] = variable
                neighborhood = [
                    position
                    for position in range(self.axis.valid_len)
                    if abs(float(position) - endpoint) <= radius + 1.0e-9
                ]
                coefficients = {variable: 1}
                coefficients.update({self.x_index(position): -1 for position in neighborhood})
                self._add_constraint(coefficients, -math.inf, 0)
                for position in neighborhood:
                    self._add_constraint(
                        {self.x_index(position): 1, variable: -1},
                        -math.inf,
                        0,
                    )

        for segment_index in range(len(self.segments)):
            for radius_index, _ in enumerate(self.spec.boundary_radii):
                start_hit = self.hit_variables[(2 * segment_index, radius_index)]
                end_hit = self.hit_variables[(2 * segment_index + 1, radius_index)]
                both = self._new_variable(integer=True)
                self.both_variables[(segment_index, radius_index)] = both
                self._add_constraint({both: 1, start_hit: -1}, -math.inf, 0)
                self._add_constraint({both: 1, end_hit: -1}, -math.inf, 0)
                self._add_constraint(
                    {both: 1, start_hit: -1, end_hit: -1},
                    -1,
                    math.inf,
                )

    def _build_distance_assignments(self) -> None:
        for endpoint_index, _ in enumerate(self.endpoints):
            row: dict[int, float] = {}
            for position in range(self.axis.valid_len):
                variable = self._new_variable(integer=False)
                self.assignment_variables[(endpoint_index, position)] = variable
                row[variable] = 1
                self._add_constraint(
                    {variable: 1, self.x_index(position): -1},
                    -math.inf,
                    0,
                )
            self._add_constraint(row, 1, 1)

    def _build_short_action_support(self) -> None:
        for segment_index, (start, end) in enumerate(self.segments):
            if end - start > self.spec.short_action_max_length + 1.0e-9:
                continue
            inside = [
                position
                for position in range(self.axis.valid_len)
                if start - 1.0e-9 <= position <= end + 1.0e-9
            ]
            support = self._new_variable(integer=True)
            self.support_variables[segment_index] = support
            coefficients = {support: 1}
            coefficients.update({self.x_index(position): -1 for position in inside})
            self._add_constraint(coefficients, -math.inf, 0)
            for position in inside:
                self._add_constraint(
                    {self.x_index(position): 1, support: -1},
                    -math.inf,
                    0,
                )

    def objective_sequence(self) -> list[tuple[str, dict[int, int], bool]]:
        objectives: list[tuple[str, dict[int, int], bool]] = []
        for radius_index, radius in enumerate(self.spec.boundary_radii):
            objectives.append(
                (
                    f"both_endpoints_r{radius}",
                    {
                        variable: 1
                        for (segment, index), variable in self.both_variables.items()
                        if index == radius_index
                    },
                    True,
                )
            )
        for radius_index, radius in enumerate(self.spec.boundary_radii):
            objectives.append(
                (
                    f"distinct_endpoint_hits_r{radius}",
                    {
                        variable: 1
                        for (endpoint, index), variable in self.hit_variables.items()
                        if index == radius_index
                    },
                    True,
                )
            )
        distance_objective = {
            variable: int(
                round(
                    abs(float(position) - self.endpoints[endpoint_index])
                    * self.spec.distance_scale
                )
            )
            for (endpoint_index, position), variable in self.assignment_variables.items()
        }
        objectives.append(("total_endpoint_distance_q", distance_objective, False))
        objectives.append(
            (
                "short_action_support",
                {variable: 1 for variable in self.support_variables.values()},
                True,
            )
        )
        background = {
            self.x_index(position): 1
            for position in range(self.axis.valid_len)
            if not any(start - 1.0e-9 <= position <= end + 1.0e-9 for start, end in self.segments)
        }
        objectives.append(("selected_background", background, False))
        uniform = set(exact_uniform_positions(self.axis.valid_len, self.budget))
        objectives.append(
            (
                "exact_uniform_overlap",
                {self.x_index(position): 1 for position in uniform},
                True,
            )
        )
        return objectives

    def objective_values_from_positions(
        self,
        positions: Sequence[int],
    ) -> dict[str, int]:
        selected = frozenset(int(position) for position in positions)
        if len(selected) != self.budget:
            raise AllocationContractError("GT objective replay requires an exact-K selection")
        if any(position < 0 or position >= self.axis.valid_len for position in selected):
            raise AllocationContractError("GT objective replay position is out of range")

        values: dict[str, int] = {}
        for radius in self.spec.boundary_radii:
            values[f"both_endpoints_r{radius}"] = sum(
                any(abs(float(position) - start) <= radius + 1.0e-9 for position in selected)
                and any(abs(float(position) - end) <= radius + 1.0e-9 for position in selected)
                for start, end in self.segments
            )
            values[f"distinct_endpoint_hits_r{radius}"] = sum(
                any(
                    abs(float(position) - endpoint) <= radius + 1.0e-9
                    for position in selected
                )
                for endpoint in self.endpoints
            )
        values["total_endpoint_distance_q"] = sum(
            min(
                int(
                    round(
                        abs(float(position) - endpoint)
                        * self.spec.distance_scale
                    )
                )
                for position in selected
            )
            for endpoint in self.endpoints
        )
        values["short_action_support"] = sum(
            end - start <= self.spec.short_action_max_length + 1.0e-9
            and any(
                start - 1.0e-9 <= float(position) <= end + 1.0e-9
                for position in selected
            )
            for start, end in self.segments
        )
        values["selected_background"] = sum(
            not any(
                start - 1.0e-9 <= float(position) <= end + 1.0e-9
                for start, end in self.segments
            )
            for position in selected
        )
        uniform = set(exact_uniform_positions(self.axis.valid_len, self.budget))
        values["exact_uniform_overlap"] = len(selected & uniform)
        return values

    def numpy_objective(self, coefficient: Mapping[int, int], *, maximize: bool):
        import numpy as np

        objective = np.zeros(self.variable_count, dtype=float)
        for variable, value in coefficient.items():
            objective[int(variable)] = -int(value) if maximize else int(value)
        return objective


def _finite_score_vector(scores: Sequence[float | int], valid_len: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in scores)
    if len(values) != valid_len:
        raise AllocationContractError("score vector length must equal physical-axis length")
    if any(not math.isfinite(value) for value in values):
        raise AllocationContractError("score vector must contain only finite values")
    return values


def _canonical_gt_segments(
    gt_segments: Sequence[Sequence[float | int]],
    valid_len: int,
) -> tuple[tuple[float, float], ...]:
    canonical: list[tuple[float, float]] = []
    upper = float(valid_len - 1)
    for index, segment in enumerate(gt_segments):
        if len(segment) != 2:
            raise AllocationContractError(f"gt segment {index} must contain exactly two endpoints")
        start, end = float(segment[0]), float(segment[1])
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end > upper
            or end < start
        ):
            raise AllocationContractError(
                f"gt segment {index} is outside dense valid prefix [0,{upper}]"
            )
        canonical.append((start, end))
    return tuple(canonical)


def _canonical_one_per_cell_bounds(
    bounds: Sequence[Sequence[int]] | None,
    *,
    valid_len: int,
    budget: int,
) -> tuple[tuple[int, int], ...] | None:
    if bounds is None:
        return None
    canonical: list[tuple[int, int]] = []
    for index, cell in enumerate(bounds):
        if len(cell) != 2:
            raise AllocationContractError(
                f"one-per-cell bound {index} must contain [start,end)"
            )
        canonical.append((int(cell[0]), int(cell[1])))
    result = tuple(canonical)
    if len(result) != int(budget):
        raise AllocationContractError("one-per-cell count must equal the exact budget")
    cursor = 0
    for index, (start, end) in enumerate(result):
        if start != cursor or end <= start or end > int(valid_len):
            raise AllocationContractError(
                f"one-per-cell bounds must be a contiguous partition; invalid cell {index}"
            )
        cursor = end
    if cursor != int(valid_len):
        raise AllocationContractError("one-per-cell bounds must cover the valid prefix")
    return result
