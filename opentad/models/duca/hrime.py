from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any, Iterable, Mapping, Sequence


HRIME_SOLVER_VERSION = "hrime_exact_equality_mckp_v1"
HRIME_REPLAY_SCHEMA = "duca_rime_budget_replay_v1"
HRIME_REPLAY_ROLE = "hrime_joint_video_exact_mckp"
HRIME_EXECUTION_QUANTUM = 16
HRIME_SCORE_DTYPE = "int64"
HRIME_SCORE_SCALE = 1_000_000
HRIME_SCORE_ROUNDING = "ROUND_HALF_EVEN"
HRIME_SCORE_QUANTIZATION_TOLERANCE = Decimal("0.00000051")
HRIME_MAX_REACHABLE_STATES = 1_000_000
HRIME_DENSITY_PANEL_MILLI = (0, 250, 500, 750, 1000)
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    _require(
        _SHA256.fullmatch(normalized) is not None,
        f"{label} must be an exact SHA-256",
    )
    return normalized


def _as_decimal(value: Any, label: str) -> Decimal:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite decimal") from exc
    _require(result.is_finite(), f"{label} must be a finite decimal")
    return result


def _as_exact_int(value: Any, label: str) -> int:
    decimal = _as_decimal(value, label)
    _require(
        decimal == decimal.to_integral_value(),
        f"{label} must be an exact integer",
    )
    return int(decimal)


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _quantize_score(value: Decimal, label: str) -> tuple[int, Decimal]:
    scaled = value * HRIME_SCORE_SCALE
    integer = int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
    _require_int64(integer, label)
    reconstructed = Decimal(integer) / HRIME_SCORE_SCALE
    error = abs(value - reconstructed)
    _require(
        error < HRIME_SCORE_QUANTIZATION_TOLERANCE,
        f"{label} score quantization exceeded the frozen tolerance",
    )
    return integer, error


def _require_int64(value: int, label: str) -> int:
    _require(
        _INT64_MIN <= int(value) <= _INT64_MAX,
        f"{label} exceeds the frozen int64 range",
    )
    return int(value)


@dataclass(frozen=True)
class EffectiveKChoice:
    effective_k: int
    nominal_budgets: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.effective_k > 0, "effective K must be positive")
        _require(
            self.nominal_budgets
            and tuple(sorted(set(self.nominal_budgets))) == self.nominal_budgets,
            "nominal budget aliases must be positive, unique, and increasing",
        )
        _require(
            all(
                value > 0
                and value % HRIME_EXECUTION_QUANTUM == 0
                and value >= self.effective_k
                for value in self.nominal_budgets
            ),
            "nominal budget aliases must be aligned and no smaller than effective K",
        )

    @property
    def canonical_nominal_budget(self) -> int:
        return self.nominal_budgets[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_k": self.effective_k,
            "nominal_budgets": list(self.nominal_budgets),
            "canonical_nominal_budget": self.canonical_nominal_budget,
        }


@dataclass(frozen=True)
class WindowFeasibleKSet:
    valid_length: int
    execution_quantum: int
    choices: tuple[EffectiveKChoice, ...]

    def __post_init__(self) -> None:
        _require(self.valid_length > 0, "window valid length must be positive")
        _require(self.execution_quantum > 0, "execution quantum must be positive")
        effective = tuple(choice.effective_k for choice in self.choices)
        _require(
            effective and tuple(sorted(set(effective))) == effective,
            "effective-K choices must be unique and increasing",
        )
        _require(
            all(value % self.execution_quantum == 0 for value in effective),
            "effective-K choices must align with the execution quantum",
        )

    @property
    def effective_ks(self) -> tuple[int, ...]:
        return tuple(choice.effective_k for choice in self.choices)

    @property
    def nominal_to_effective(self) -> dict[int, int]:
        return {
            nominal: choice.effective_k
            for choice in self.choices
            for nominal in choice.nominal_budgets
        }

    def choice_for_effective_k(self, effective_k: int) -> EffectiveKChoice:
        for choice in self.choices:
            if choice.effective_k == int(effective_k):
                return choice
        raise ValueError(f"effective K {effective_k} is not feasible for this window")

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_length": self.valid_length,
            "execution_quantum": self.execution_quantum,
            "choices": [choice.to_dict() for choice in self.choices],
        }


def canonicalize_effective_k_options(
    valid_length: int,
    nominal_budgets: Sequence[int],
    *,
    execution_quantum: int = HRIME_EXECUTION_QUANTUM,
) -> WindowFeasibleKSet:
    valid = _as_exact_int(valid_length, "window valid length")
    quantum = _as_exact_int(execution_quantum, "execution quantum")
    budgets = tuple(
        _as_exact_int(value, "nominal budget") for value in nominal_budgets
    )
    _require(valid > 0, "window valid length must be positive")
    _require(quantum > 0, "execution quantum must be positive")
    _require(
        budgets
        and tuple(sorted(set(budgets))) == budgets
        and all(value > 0 and value % quantum == 0 for value in budgets),
        "nominal budgets must be positive, unique, increasing, and quantum aligned",
    )
    available = valid - valid % quantum
    _require(
        available > 0,
        "window is shorter than one heavy-backbone execution quantum",
    )
    aliases: dict[int, list[int]] = {}
    for nominal in budgets:
        effective = min(nominal, available)
        effective -= effective % quantum
        _require(
            effective > 0,
            "a nominal budget cannot realize one execution quantum",
        )
        aliases.setdefault(effective, []).append(nominal)
    choices = tuple(
        EffectiveKChoice(
            effective_k=effective,
            nominal_budgets=tuple(aliases[effective]),
        )
        for effective in sorted(aliases)
    )
    return WindowFeasibleKSet(
        valid_length=valid,
        execution_quantum=quantum,
        choices=choices,
    )


def _normalize_cost_rows(
    cost_options_by_window: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    rows = []
    for window_index, values in enumerate(cost_options_by_window):
        row = tuple(
            sorted(
                set(
                    _as_exact_int(
                        value,
                        f"window {window_index} effective-K option",
                    )
                    for value in values
                )
            )
        )
        _require(row, f"window {window_index} has no feasible effective-K option")
        _require(
            all(
                value > 0 and value % HRIME_EXECUTION_QUANTUM == 0
                for value in row
            ),
            f"window {window_index} effective-K options must be positive "
            "and execution-quantum aligned",
        )
        rows.append(row)
    _require(rows, "at least one window is required")
    return tuple(rows)


def enumerate_reachable_totals(
    cost_options_by_window: Sequence[Sequence[int]],
    *,
    max_states: int = HRIME_MAX_REACHABLE_STATES,
) -> tuple[int, ...]:
    rows = _normalize_cost_rows(cost_options_by_window)
    limit = int(max_states)
    _require(limit > 0, "max_states must be positive")
    states = {0}
    for window_index, row in enumerate(rows):
        states = {used + value for used in states for value in row}
        _require(
            len(states) <= limit,
            f"reachable-total state limit exceeded at window {window_index}",
        )
    return tuple(sorted(states))


@dataclass(frozen=True)
class ReachableBudgetProjection:
    raw_budget: int
    reachable_budget: int
    minimum_reachable_budget: int
    maximum_reachable_budget: int
    projection_unused_budget: int
    solver_unused_budget: int
    reachable_total_count: int
    reachable_totals_sha256: str

    def __post_init__(self) -> None:
        _require(self.raw_budget >= 0, "raw budget must be non-negative")
        _require(
            self.minimum_reachable_budget
            <= self.reachable_budget
            <= self.maximum_reachable_budget,
            "projected budget lies outside the reachable interval",
        )
        _require(
            self.reachable_budget <= self.raw_budget,
            "reachable budget must not exceed the raw cap",
        )
        _require(
            self.projection_unused_budget == self.raw_budget - self.reachable_budget,
            "projection-unused budget is inconsistent",
        )
        _require(
            self.solver_unused_budget == 0,
            "H-RIME v1 requires exact budget conservation after projection",
        )
        _require_sha256(self.reachable_totals_sha256, "reachable totals hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_budget": self.raw_budget,
            "reachable_budget": self.reachable_budget,
            "minimum_reachable_budget": self.minimum_reachable_budget,
            "maximum_reachable_budget": self.maximum_reachable_budget,
            "projection_unused_budget": self.projection_unused_budget,
            "solver_unused_budget": self.solver_unused_budget,
            "reachable_total_count": self.reachable_total_count,
            "reachable_totals_sha256": self.reachable_totals_sha256,
        }


def project_budget_to_reachable(
    raw_budget: int,
    cost_options_by_window: Sequence[Sequence[int]],
    *,
    max_states: int = HRIME_MAX_REACHABLE_STATES,
) -> ReachableBudgetProjection:
    raw = _as_exact_int(raw_budget, "raw budget")
    _require(raw >= 0, "raw budget must be non-negative")
    totals = enumerate_reachable_totals(
        cost_options_by_window,
        max_states=max_states,
    )
    candidates = tuple(value for value in totals if value <= raw)
    _require(
        candidates,
        "raw budget is below the minimum feasible whole-video execution cost",
    )
    reachable = candidates[-1]
    return ReachableBudgetProjection(
        raw_budget=raw,
        reachable_budget=reachable,
        minimum_reachable_budget=totals[0],
        maximum_reachable_budget=totals[-1],
        projection_unused_budget=raw - reachable,
        solver_unused_budget=0,
        reachable_total_count=len(totals),
        reachable_totals_sha256=canonical_sha256(list(totals)),
    )


@dataclass(frozen=True)
class MCKPOption:
    effective_k: int
    utility: Any
    risk: Any
    nominal_budgets: tuple[int, ...] = ()


@dataclass(frozen=True)
class MCKPWindow:
    window_key: str
    options: tuple[MCKPOption, ...]
    option_source_sha256: str


@dataclass(frozen=True)
class _QuantizedOption:
    effective_k: int
    utility_decimal: Decimal
    risk_decimal: Decimal
    objective_decimal: Decimal
    utility_int: int
    risk_int: int
    objective_int: int
    nominal_budgets: tuple[int, ...]
    max_quantization_error: Decimal

    def to_input_dict(self) -> dict[str, Any]:
        return {
            "effective_k": self.effective_k,
            "utility": _canonical_decimal(self.utility_decimal),
            "risk": _canonical_decimal(self.risk_decimal),
            "objective": _canonical_decimal(self.objective_decimal),
            "utility_int": self.utility_int,
            "risk_int": self.risk_int,
            "objective_int": self.objective_int,
            "nominal_budgets": list(self.nominal_budgets),
        }


@dataclass(frozen=True)
class _DPState:
    objective_int: int
    risk_int: int
    utility_int: int
    assignment: tuple[int, ...]
    option_indices: tuple[int, ...]


def _state_is_better(candidate: _DPState, previous: _DPState) -> bool:
    if candidate.objective_int != previous.objective_int:
        return candidate.objective_int > previous.objective_int
    if candidate.risk_int != previous.risk_int:
        return candidate.risk_int < previous.risk_int
    return candidate.assignment < previous.assignment


@dataclass(frozen=True)
class ExactMCKPResult:
    solver_version: str
    score_dtype: str
    score_scale: int
    score_rounding: str
    beta: str
    allocation_context_sha256: str
    target_total_cost: int
    realized_total_cost: int
    solver_unused_budget: int
    window_keys: tuple[str, ...]
    assignment: tuple[int, ...]
    selected_option_indices: tuple[int, ...]
    total_utility_int: int
    total_risk_int: int
    total_objective_int: int
    maximum_component_quantization_error: str
    solver_input_sha256: str
    assignment_sha256: str

    def __post_init__(self) -> None:
        _require(
            self.solver_version == HRIME_SOLVER_VERSION,
            "H-RIME solver version drift",
        )
        _require(
            self.score_dtype == HRIME_SCORE_DTYPE
            and self.score_scale == HRIME_SCORE_SCALE
            and self.score_rounding == HRIME_SCORE_ROUNDING,
            "H-RIME solver numeric protocol drift",
        )
        _require(
            self.target_total_cost == self.realized_total_cost,
            "exact MCKP did not conserve its reachable budget",
        )
        _require(
            self.solver_unused_budget == 0,
            "exact MCKP cannot leave reachable budget unused",
        )
        _require(
            len(self.window_keys)
            == len(self.assignment)
            == len(self.selected_option_indices),
            "MCKP result vectors do not align",
        )
        _require_sha256(self.solver_input_sha256, "solver input hash")
        _require_sha256(self.assignment_sha256, "assignment hash")
        _require_sha256(
            self.allocation_context_sha256,
            "allocation context hash",
        )

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "hrime_exact_mckp_receipt_v1",
            "status": "solved_exact",
            "solver_version": self.solver_version,
            "score_dtype": self.score_dtype,
            "score_scale": self.score_scale,
            "score_rounding": self.score_rounding,
            "beta": self.beta,
            "allocation_context_sha256": self.allocation_context_sha256,
            "target_total_cost": self.target_total_cost,
            "realized_total_cost": self.realized_total_cost,
            "solver_unused_budget": self.solver_unused_budget,
            "window_keys": list(self.window_keys),
            "assignment": list(self.assignment),
            "selected_option_indices": list(self.selected_option_indices),
            "total_utility_int": self.total_utility_int,
            "total_risk_int": self.total_risk_int,
            "total_objective_int": self.total_objective_int,
            "maximum_component_quantization_error": (
                self.maximum_component_quantization_error
            ),
            "solver_input_sha256": self.solver_input_sha256,
            "assignment_sha256": self.assignment_sha256,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
        }


def _quantize_windows(
    windows: Sequence[MCKPWindow],
    *,
    beta: Decimal,
) -> tuple[tuple[_QuantizedOption, ...], ...]:
    _require(windows, "exact MCKP requires at least one window")
    keys = tuple(str(window.window_key) for window in windows)
    _require(
        all(keys) and len(keys) == len(set(keys)),
        "MCKP window keys must be non-empty and unique",
    )
    rows = []
    for window_index, window in enumerate(windows):
        _require_sha256(
            window.option_source_sha256,
            f"window {window_index} option source hash",
        )
        _require(window.options, f"window {window_index} has no MCKP options")
        effective = tuple(
            _as_exact_int(
                option.effective_k,
                f"window {window_index} effective K",
            )
            for option in window.options
        )
        _require(
            effective == tuple(sorted(set(effective)))
            and effective[0] > 0
            and all(value % HRIME_EXECUTION_QUANTUM == 0 for value in effective),
            f"window {window_index} effective-K options must be positive, "
            "unique, increasing, and execution-quantum aligned",
        )
        quantized = []
        for option_index, option in enumerate(window.options):
            utility = _as_decimal(
                option.utility,
                f"window {window_index} option {option_index} utility",
            )
            risk = _as_decimal(
                option.risk,
                f"window {window_index} option {option_index} risk",
            )
            _require(
                risk >= 0,
                f"window {window_index} option {option_index} risk must be non-negative",
            )
            aliases = tuple(
                _as_exact_int(
                    value,
                    f"window {window_index} nominal alias",
                )
                for value in option.nominal_budgets
            )
            _require(
                not aliases
                or (
                    aliases == tuple(sorted(set(aliases)))
                    and all(
                        value > 0
                        and value % HRIME_EXECUTION_QUANTUM == 0
                        and value >= effective[option_index]
                        for value in aliases
                    )
                ),
                f"window {window_index} nominal aliases are invalid",
            )
            objective = utility - beta * risk
            utility_int, utility_error = _quantize_score(
                utility,
                f"window {window_index} option {option_index} utility",
            )
            risk_int, risk_error = _quantize_score(
                risk,
                f"window {window_index} option {option_index} risk",
            )
            objective_int, objective_error = _quantize_score(
                objective,
                f"window {window_index} option {option_index} objective",
            )
            quantized.append(
                _QuantizedOption(
                    effective_k=effective[option_index],
                    utility_decimal=utility,
                    risk_decimal=risk,
                    objective_decimal=objective,
                    utility_int=utility_int,
                    risk_int=risk_int,
                    objective_int=objective_int,
                    nominal_budgets=aliases,
                    max_quantization_error=max(
                        utility_error,
                        risk_error,
                        objective_error,
                    ),
                )
            )
        rows.append(tuple(quantized))
    return tuple(rows)


def solve_exact_mckp(
    windows: Sequence[MCKPWindow],
    *,
    target_total_cost: int,
    beta: Any,
    allocation_context_sha256: str,
) -> ExactMCKPResult:
    target = _as_exact_int(target_total_cost, "exact MCKP target")
    _require(target > 0, "exact MCKP target must be positive")
    context_sha = _require_sha256(
        allocation_context_sha256,
        "allocation context hash",
    )
    beta_decimal = _as_decimal(beta, "risk weight beta")
    _require(beta_decimal >= 0, "risk weight beta must be non-negative")
    rows = _quantize_windows(windows, beta=beta_decimal)
    reachable = enumerate_reachable_totals(
        tuple(tuple(option.effective_k for option in row) for row in rows)
    )
    _require(
        target in set(reachable),
        "exact MCKP target is not reachable by the canonical effective-K options",
    )
    input_payload = {
        "solver_version": HRIME_SOLVER_VERSION,
        "score_dtype": HRIME_SCORE_DTYPE,
        "score_scale": HRIME_SCORE_SCALE,
        "score_rounding": HRIME_SCORE_ROUNDING,
        "beta": _canonical_decimal(beta_decimal),
        "allocation_context_sha256": context_sha,
        "target_total_cost": target,
        "windows": [
            {
                "window_key": str(window.window_key),
                "option_source_sha256": window.option_source_sha256,
                "options": [option.to_input_dict() for option in row],
            }
            for window, row in zip(windows, rows)
        ],
    }
    solver_input_sha = canonical_sha256(input_payload)
    states: dict[int, _DPState] = {
        0: _DPState(
            objective_int=0,
            risk_int=0,
            utility_int=0,
            assignment=(),
            option_indices=(),
        )
    }
    for row in rows:
        next_states: dict[int, _DPState] = {}
        for used, state in states.items():
            for option_index, option in enumerate(row):
                total = used + option.effective_k
                if total > target:
                    continue
                candidate = _DPState(
                    objective_int=_require_int64(
                        state.objective_int + option.objective_int,
                        "cumulative objective",
                    ),
                    risk_int=_require_int64(
                        state.risk_int + option.risk_int,
                        "cumulative risk",
                    ),
                    utility_int=_require_int64(
                        state.utility_int + option.utility_int,
                        "cumulative utility",
                    ),
                    assignment=state.assignment + (option.effective_k,),
                    option_indices=state.option_indices + (option_index,),
                )
                previous = next_states.get(total)
                if previous is None or _state_is_better(candidate, previous):
                    next_states[total] = candidate
        _require(next_states, "exact MCKP exhausted all feasible partial states")
        states = next_states
    _require(target in states, "exact MCKP failed to fill the reachable target")
    selected = states[target]
    window_keys = tuple(str(window.window_key) for window in windows)
    assignment_payload = {
        "schema_version": "hrime_exact_mckp_assignment_v1",
        "solver_version": HRIME_SOLVER_VERSION,
        "solver_input_sha256": solver_input_sha,
        "allocation_context_sha256": context_sha,
        "target_total_cost": target,
        "windows": [
            {
                "window_key": key,
                "effective_k": effective_k,
                "selected_option_index": option_index,
            }
            for key, effective_k, option_index in zip(
                window_keys,
                selected.assignment,
                selected.option_indices,
            )
        ],
    }
    maximum_error = max(
        option.max_quantization_error for row in rows for option in row
    )
    return ExactMCKPResult(
        solver_version=HRIME_SOLVER_VERSION,
        score_dtype=HRIME_SCORE_DTYPE,
        score_scale=HRIME_SCORE_SCALE,
        score_rounding=HRIME_SCORE_ROUNDING,
        beta=_canonical_decimal(beta_decimal),
        allocation_context_sha256=context_sha,
        target_total_cost=target,
        realized_total_cost=sum(selected.assignment),
        solver_unused_budget=target - sum(selected.assignment),
        window_keys=window_keys,
        assignment=selected.assignment,
        selected_option_indices=selected.option_indices,
        total_utility_int=selected.utility_int,
        total_risk_int=selected.risk_int,
        total_objective_int=selected.objective_int,
        maximum_component_quantization_error=_canonical_decimal(maximum_error),
        solver_input_sha256=solver_input_sha,
        assignment_sha256=canonical_sha256(assignment_payload),
    )


@dataclass(frozen=True)
class VideoWindowRef:
    video_id: str
    window_start_frame: int
    valid_length: int
    source_index: int
    cheap_feature_index: int

    def __post_init__(self) -> None:
        _require(bool(self.video_id), "video ID must be non-empty")
        _require(self.window_start_frame >= 0, "window start must be non-negative")
        _require(self.valid_length > 0, "window valid length must be positive")
        _require(self.source_index >= 0, "source index must be non-negative")
        _require(
            self.cheap_feature_index >= 0,
            "cheap feature index must be non-negative",
        )

    @property
    def window_key(self) -> str:
        return hrime_window_key(self.video_id, self.window_start_frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "window_start_frame": self.window_start_frame,
            "valid_length": self.valid_length,
            "source_index": self.source_index,
            "cheap_feature_index": self.cheap_feature_index,
            "window_key": self.window_key,
        }


def hrime_window_key(video_id: str, window_start_frame: int) -> str:
    video = str(video_id)
    start = int(window_start_frame)
    _require(video and start >= 0, "window identity is invalid")
    return canonical_sha256(
        {
            "schema_version": "hrime_window_identity_v1",
            "video_id": video,
            "window_start_frame": start,
        }
    )


@dataclass(frozen=True)
class VideoWindowGroup:
    video_id: str
    windows: tuple[VideoWindowRef, ...]
    group_order_sha256: str

    def __post_init__(self) -> None:
        _require(self.windows, "video window group must be non-empty")
        _require(
            all(window.video_id == self.video_id for window in self.windows),
            "video window group mixes video identities",
        )
        ordering = tuple(
            (window.window_start_frame, window.source_index) for window in self.windows
        )
        _require(
            ordering == tuple(sorted(ordering)),
            "video windows must use stable physical ordering",
        )
        _require(
            len({window.window_start_frame for window in self.windows})
            == len(self.windows),
            "video window starts must be unique",
        )
        _require_sha256(self.group_order_sha256, "group order hash")


def group_video_windows(
    windows: Iterable[VideoWindowRef],
) -> tuple[VideoWindowGroup, ...]:
    grouped: dict[str, list[VideoWindowRef]] = {}
    source_indices = set()
    for window in windows:
        _require(
            window.source_index not in source_indices,
            "source indices must be globally unique",
        )
        source_indices.add(window.source_index)
        grouped.setdefault(window.video_id, []).append(window)
    _require(grouped, "video grouping input is empty")
    output = []
    for video_id in sorted(grouped):
        ordered = tuple(
            sorted(
                grouped[video_id],
                key=lambda item: (item.window_start_frame, item.source_index),
            )
        )
        _require(
            len({window.window_start_frame for window in ordered}) == len(ordered),
            f"video {video_id} contains duplicate window starts",
        )
        payload = {
            "schema_version": "hrime_video_window_group_v1",
            "video_id": video_id,
            "windows": [window.to_dict() for window in ordered],
        }
        output.append(
            VideoWindowGroup(
                video_id=video_id,
                windows=ordered,
                group_order_sha256=canonical_sha256(payload),
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class SharedVideoScanReceipt:
    video_id: str
    group_order_sha256: str
    scan_version: str
    scan_input_sha256: str
    video_summary_sha256: str
    window_summary_sha256: tuple[str, ...]
    shared_scan_execution_count: int
    per_window_fallback_execution_count: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.video_id), "scan video ID must be non-empty")
        _require(bool(self.scan_version), "scan version must be non-empty")
        _require_sha256(self.group_order_sha256, "scan group hash")
        _require_sha256(self.scan_input_sha256, "scan input hash")
        _require_sha256(self.video_summary_sha256, "video summary hash")
        _require(
            self.window_summary_sha256
            and all(_SHA256.fullmatch(value) for value in self.window_summary_sha256),
            "window summary hashes are invalid",
        )
        _require(
            self.shared_scan_execution_count == 1,
            "H-RIME shared scan must execute exactly once per video",
        )
        _require(
            self.per_window_fallback_execution_count == 0,
            "fallback scans must be charged separately and cannot claim shared reuse",
        )
        _require_sha256(self.receipt_sha256, "scan receipt hash")


def build_shared_scan_receipt(
    group: VideoWindowGroup,
    *,
    scan_version: str,
    scan_input_sha256: str,
    video_summary_sha256: str,
    window_summary_sha256: Sequence[str],
) -> SharedVideoScanReceipt:
    window_hashes = tuple(
        _require_sha256(value, "window summary hash")
        for value in window_summary_sha256
    )
    _require(
        len(window_hashes) == len(group.windows),
        "shared scan must emit one local summary per grouped window",
    )
    payload = {
        "schema_version": "hrime_shared_video_scan_receipt_v1",
        "video_id": group.video_id,
        "group_order_sha256": group.group_order_sha256,
        "scan_version": str(scan_version),
        "scan_input_sha256": _require_sha256(
            scan_input_sha256,
            "scan input hash",
        ),
        "video_summary_sha256": _require_sha256(
            video_summary_sha256,
            "video summary hash",
        ),
        "window_summary_sha256": list(window_hashes),
        "shared_scan_execution_count": 1,
        "per_window_fallback_execution_count": 0,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
    }
    return SharedVideoScanReceipt(
        video_id=group.video_id,
        group_order_sha256=group.group_order_sha256,
        scan_version=str(scan_version),
        scan_input_sha256=payload["scan_input_sha256"],
        video_summary_sha256=payload["video_summary_sha256"],
        window_summary_sha256=window_hashes,
        shared_scan_execution_count=1,
        per_window_fallback_execution_count=0,
        receipt_sha256=canonical_sha256(payload),
    )


@dataclass(frozen=True)
class VideoBudgetPlan:
    video_id: str
    planner_version: str
    group_order_sha256: str
    scan_receipt_sha256: str
    feasible_sets_sha256: str
    projection: ReachableBudgetProjection
    decision_input_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.video_id), "budget-plan video ID must be non-empty")
        _require(bool(self.planner_version), "planner version must be non-empty")
        _require_sha256(self.group_order_sha256, "budget-plan group hash")
        _require_sha256(self.scan_receipt_sha256, "budget-plan scan hash")
        _require_sha256(
            self.feasible_sets_sha256,
            "budget-plan feasible-set hash",
        )
        _require_sha256(
            self.decision_input_sha256,
            "budget-plan decision input hash",
        )

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "hrime_video_budget_plan_v1",
            "status": "projected_to_reachable_total",
            "video_id": self.video_id,
            "planner_version": self.planner_version,
            "group_order_sha256": self.group_order_sha256,
            "scan_receipt_sha256": self.scan_receipt_sha256,
            "feasible_sets_sha256": self.feasible_sets_sha256,
            "projection": self.projection.to_dict(),
            "decision_input_sha256": self.decision_input_sha256,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
        }


def build_video_budget_plan(
    group: VideoWindowGroup,
    scan: SharedVideoScanReceipt,
    *,
    planner_version: str,
    raw_budget: int,
    feasible_sets: Sequence[WindowFeasibleKSet],
) -> VideoBudgetPlan:
    _require(scan.video_id == group.video_id, "scan/group video identity drift")
    _require(
        scan.group_order_sha256 == group.group_order_sha256,
        "scan/group order hash drift",
    )
    _require(
        len(feasible_sets) == len(group.windows),
        "feasible sets must align with grouped windows",
    )
    for window, feasible in zip(group.windows, feasible_sets):
        _require(
            window.valid_length == feasible.valid_length,
            "feasible set valid length differs from its grouped window",
        )
    projection = project_budget_to_reachable(
        raw_budget,
        tuple(feasible.effective_ks for feasible in feasible_sets),
    )
    feasible_payload = [feasible.to_dict() for feasible in feasible_sets]
    feasible_sha = canonical_sha256(feasible_payload)
    payload = {
        "schema_version": "hrime_video_budget_plan_input_v1",
        "video_id": group.video_id,
        "planner_version": str(planner_version),
        "group_order_sha256": group.group_order_sha256,
        "scan_receipt_sha256": scan.receipt_sha256,
        "raw_budget": int(raw_budget),
        "feasible_sets": feasible_payload,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
    }
    return VideoBudgetPlan(
        video_id=group.video_id,
        planner_version=str(planner_version),
        group_order_sha256=group.group_order_sha256,
        scan_receipt_sha256=scan.receipt_sha256,
        feasible_sets_sha256=feasible_sha,
        projection=projection,
        decision_input_sha256=canonical_sha256(payload),
    )


@dataclass(frozen=True)
class DispatchBucket:
    effective_k: int
    group_window_indices: tuple[int, ...]
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class HomogeneousKDispatchPlan:
    video_id: str
    group_order_sha256: str
    assignment_sha256: str
    buckets: tuple[DispatchBucket, ...]
    heavy_execution_group_order: tuple[int, ...]
    restore_group_order: tuple[int, ...]
    heavy_frame_total: int
    heavy_bucket_count: int
    tail_padding_mode: str
    dispatch_sha256: str

    def __post_init__(self) -> None:
        _require(
            self.tail_padding_mode == "none_exact_k_bucket",
            "H-RIME dispatch cannot execute an inactive tail",
        )
        _require(
            self.heavy_bucket_count == len(self.buckets),
            "heavy bucket count is inconsistent",
        )
        _require_sha256(self.group_order_sha256, "dispatch group hash")
        _require_sha256(self.assignment_sha256, "dispatch assignment hash")
        _require_sha256(self.dispatch_sha256, "dispatch hash")


def build_homogeneous_k_dispatch(
    group: VideoWindowGroup,
    result: ExactMCKPResult,
) -> HomogeneousKDispatchPlan:
    expected_keys = tuple(window.window_key for window in group.windows)
    _require(
        result.window_keys == expected_keys,
        "MCKP assignment does not align with the grouped window order",
    )
    by_k: dict[int, list[int]] = {}
    for index, effective_k in enumerate(result.assignment):
        by_k.setdefault(effective_k, []).append(index)
    buckets = tuple(
        DispatchBucket(
            effective_k=effective_k,
            group_window_indices=tuple(by_k[effective_k]),
            source_indices=tuple(
                group.windows[index].source_index for index in by_k[effective_k]
            ),
        )
        for effective_k in sorted(by_k)
    )
    execution_order = tuple(
        index for bucket in buckets for index in bucket.group_window_indices
    )
    restore = tuple(execution_order.index(index) for index in range(len(group.windows)))
    payload = {
        "schema_version": "hrime_homogeneous_k_dispatch_v1",
        "video_id": group.video_id,
        "group_order_sha256": group.group_order_sha256,
        "assignment_sha256": result.assignment_sha256,
        "buckets": [
            {
                "effective_k": bucket.effective_k,
                "group_window_indices": list(bucket.group_window_indices),
                "source_indices": list(bucket.source_indices),
            }
            for bucket in buckets
        ],
        "heavy_execution_group_order": list(execution_order),
        "restore_group_order": list(restore),
        "heavy_frame_total": sum(result.assignment),
        "tail_padding_mode": "none_exact_k_bucket",
    }
    return HomogeneousKDispatchPlan(
        video_id=group.video_id,
        group_order_sha256=group.group_order_sha256,
        assignment_sha256=result.assignment_sha256,
        buckets=buckets,
        heavy_execution_group_order=execution_order,
        restore_group_order=restore,
        heavy_frame_total=sum(result.assignment),
        heavy_bucket_count=len(buckets),
        tail_padding_mode="none_exact_k_bucket",
        dispatch_sha256=canonical_sha256(payload),
    )


def build_hrime_replay_rows(
    group: VideoWindowGroup,
    feasible_sets: Sequence[WindowFeasibleKSet],
    plan: VideoBudgetPlan,
    result: ExactMCKPResult,
    *,
    budget_protocol_sha256: str,
) -> tuple[dict[str, Any], ...]:
    protocol_sha = _require_sha256(
        budget_protocol_sha256,
        "budget protocol hash",
    )
    _require(plan.video_id == group.video_id, "budget plan/group video drift")
    _require(
        plan.group_order_sha256 == group.group_order_sha256,
        "budget plan/group order drift",
    )
    _require(
        result.target_total_cost == plan.projection.reachable_budget,
        "solver target differs from the projected reachable budget",
    )
    _require(
        len(feasible_sets) == len(group.windows) == len(result.assignment),
        "replay inputs do not align",
    )
    feasible_payload = [feasible.to_dict() for feasible in feasible_sets]
    feasible_sha = canonical_sha256(feasible_payload)
    _require(
        feasible_sha == plan.feasible_sets_sha256,
        "replay feasible sets differ from the frozen budget plan",
    )
    plan_input_payload = {
        "schema_version": "hrime_video_budget_plan_input_v1",
        "video_id": group.video_id,
        "planner_version": plan.planner_version,
        "group_order_sha256": group.group_order_sha256,
        "scan_receipt_sha256": plan.scan_receipt_sha256,
        "raw_budget": plan.projection.raw_budget,
        "feasible_sets": feasible_payload,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
    }
    _require(
        canonical_sha256(plan_input_payload) == plan.decision_input_sha256,
        "replay inputs do not reproduce the frozen budget-plan hash",
    )
    _require(
        result.allocation_context_sha256 == plan.decision_input_sha256,
        "MCKP result is not bound to this video budget plan",
    )
    rows = []
    for window, feasible, effective_k in zip(
        group.windows,
        feasible_sets,
        result.assignment,
    ):
        _require(
            window.window_key
            == result.window_keys[len(rows)]
            and window.valid_length == feasible.valid_length,
            "replay window identity or valid length drift",
        )
        choice = feasible.choice_for_effective_k(effective_k)
        rows.append(
            {
                "schema_version": HRIME_REPLAY_SCHEMA,
                "video_id": group.video_id,
                "window_start_frame": window.window_start_frame,
                "requested_k": choice.canonical_nominal_budget,
                "effective_k": effective_k,
                "provenance": {
                    "role": HRIME_REPLAY_ROLE,
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_prediction_cache": False,
                    "uses_test_batch_composition": False,
                    "uses_whole_video_context": True,
                    "deployment_candidate": False,
                    "group_order_sha256": group.group_order_sha256,
                    "scan_receipt_sha256": plan.scan_receipt_sha256,
                    "budget_decision_input_sha256": plan.decision_input_sha256,
                    "budget_protocol_sha256": protocol_sha,
                    "solver_version": result.solver_version,
                    "solver_input_sha256": result.solver_input_sha256,
                    "allocation_context_sha256": (
                        result.allocation_context_sha256
                    ),
                    "assignment_sha256": result.assignment_sha256,
                    "raw_video_budget": plan.projection.raw_budget,
                    "reachable_video_budget": (
                        plan.projection.reachable_budget
                    ),
                    "realized_video_budget": result.realized_total_cost,
                    "projection_unused_budget": (
                        plan.projection.projection_unused_budget
                    ),
                    "solver_unused_budget": result.solver_unused_budget,
                    "canonical_nominal_aliases": list(choice.nominal_budgets),
                },
            }
        )
    _require(
        sum(int(row["effective_k"]) for row in rows)
        == result.realized_total_cost
        == plan.projection.reachable_budget,
        "replay rows do not conserve the reachable video budget",
    )
    return tuple(rows)


__all__ = [
    "DispatchBucket",
    "EffectiveKChoice",
    "ExactMCKPResult",
    "HRIME_DENSITY_PANEL_MILLI",
    "HRIME_EXECUTION_QUANTUM",
    "HRIME_REPLAY_ROLE",
    "HRIME_SCORE_DTYPE",
    "HRIME_SCORE_ROUNDING",
    "HRIME_SCORE_SCALE",
    "HRIME_SCORE_QUANTIZATION_TOLERANCE",
    "HRIME_SOLVER_VERSION",
    "HomogeneousKDispatchPlan",
    "MCKPOption",
    "MCKPWindow",
    "ReachableBudgetProjection",
    "SharedVideoScanReceipt",
    "VideoBudgetPlan",
    "VideoWindowGroup",
    "VideoWindowRef",
    "WindowFeasibleKSet",
    "build_homogeneous_k_dispatch",
    "build_hrime_replay_rows",
    "build_shared_scan_receipt",
    "build_video_budget_plan",
    "canonical_sha256",
    "canonicalize_effective_k_options",
    "enumerate_reachable_totals",
    "group_video_windows",
    "hrime_window_key",
    "project_budget_to_reachable",
    "solve_exact_mckp",
]
