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
    model = _GroundTruthMilpModel(
        axis=axis,
        segments=segments,
        budget=budget,
        cap=cap,
        spec=objective_spec,
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
                f"GT MILP objective {name} violates variable bounds"
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
        solver_identity=f"scipy_highs_milp_{scipy_version}",
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
    ) -> None:
        self.axis = axis
        self.segments = segments
        self.budget = budget
        self.cap = cap
        self.spec = spec
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
