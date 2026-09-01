from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence


_EPSILON = 1.0e-9


class AllocationContractError(ValueError):
    """Raised when an allocation artifact violates a fail-closed contract."""


@dataclass(frozen=True)
class PhysicalAxis:
    dense_ordinals: tuple[int, ...]
    source_frames: tuple[float, ...]
    seconds: tuple[float, ...]
    decoder_fps: float
    annotation_fps: float

    @classmethod
    def from_source_frames(
        cls,
        source_frames: Sequence[float | int],
        *,
        decoder_fps: float,
        annotation_fps: float,
    ) -> "PhysicalAxis":
        frames = tuple(float(value) for value in source_frames)
        decoder_fps = _finite_positive(decoder_fps, "decoder_fps")
        annotation_fps = _finite_positive(annotation_fps, "annotation_fps")
        if not frames:
            raise AllocationContractError("physical axis must contain at least one candidate")
        if any(not math.isfinite(value) for value in frames):
            raise AllocationContractError("source-frame coordinates must be finite")
        if any(right <= left for left, right in zip(frames, frames[1:])):
            raise AllocationContractError("source-frame coordinates must be strictly increasing")
        seconds = tuple(value / decoder_fps for value in frames)
        return cls(
            dense_ordinals=tuple(range(len(frames))),
            source_frames=frames,
            seconds=seconds,
            decoder_fps=decoder_fps,
            annotation_fps=annotation_fps,
        )

    @property
    def valid_len(self) -> int:
        return len(self.dense_ordinals)

    def subset(self, positions: Sequence[int]) -> "PhysicalAxis":
        selected = _canonical_positions(positions, self.valid_len)
        return PhysicalAxis.from_source_frames(
            [self.source_frames[index] for index in selected],
            decoder_fps=self.decoder_fps,
            annotation_fps=self.annotation_fps,
        )


@dataclass(frozen=True)
class ResolvedPhysicalCap:
    policy: str
    max_source_frame_interval: float | None
    max_seconds_interval: float | None
    reference_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.policy not in {"uniform_reference", "explicit_frames", "explicit_seconds"}:
            raise AllocationContractError(f"unknown physical cap policy: {self.policy}")
        if self.max_source_frame_interval is None and self.max_seconds_interval is None:
            raise AllocationContractError("physical cap must constrain frames or seconds")
        if self.max_source_frame_interval is not None:
            _finite_nonnegative(self.max_source_frame_interval, "max_source_frame_interval")
        if self.max_seconds_interval is not None:
            _finite_nonnegative(self.max_seconds_interval, "max_seconds_interval")

    def allows(self, axis: PhysicalAxis, left: int, right: int) -> bool:
        if left < 0 or right >= axis.valid_len or left > right:
            raise AllocationContractError("physical interval endpoints are invalid")
        if self.max_source_frame_interval is not None:
            delta_frames = axis.source_frames[right] - axis.source_frames[left]
            if delta_frames > self.max_source_frame_interval + _EPSILON:
                return False
        if self.max_seconds_interval is not None:
            delta_seconds = axis.seconds[right] - axis.seconds[left]
            if delta_seconds > self.max_seconds_interval + _EPSILON:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GapReport:
    dense_max_unselected_hole: int
    dense_max_selected_interval: int
    source_frame_max_interval: float
    seconds_max_interval: float
    source_frame_edge_intervals: tuple[float, float]
    seconds_edge_intervals: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilySelection:
    family: str
    positions: tuple[int, ...]
    budget: int
    score_sum: float | None
    exact: bool
    deployable: bool
    privileged: bool
    solver_status: str
    physical_cap_compliant: bool
    gap_report: GapReport
    scaffold_positions: tuple[int, ...] = ()
    residual_positions: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_positive(value: float | int, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise AllocationContractError(f"{name} must be finite and positive")
    return number


def _finite_nonnegative(value: float | int, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise AllocationContractError(f"{name} must be finite and non-negative")
    return number


def effective_budget(valid_len: int, requested_budget: int) -> int:
    valid_len = int(valid_len)
    requested_budget = int(requested_budget)
    if valid_len < 1:
        raise AllocationContractError("valid_len must be positive")
    if requested_budget < 1:
        raise AllocationContractError("requested_budget must be positive")
    return min(valid_len, requested_budget)


def exact_uniform_positions(valid_len: int, requested_budget: int) -> tuple[int, ...]:
    """Match the committed round-half-to-even endpoint-inclusive reference."""

    valid_len = int(valid_len)
    budget = effective_budget(valid_len, requested_budget)
    if budget == 1:
        return (0,)
    denominator = budget - 1
    positions: list[int] = []
    for index in range(budget):
        numerator = index * (valid_len - 1)
        quotient, remainder = divmod(numerator, denominator)
        if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1):
            quotient += 1
        positions.append(quotient)
    result = tuple(positions)
    if len(set(result)) != budget:
        raise RuntimeError("exact-uniform construction produced duplicate positions")
    return result


def uniform_cell_bounds(
    valid_len: int,
    requested_budget: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    anchors = exact_uniform_positions(valid_len, requested_budget)
    budget = len(anchors)
    starts = [0] * budget
    ends = [valid_len] * budget
    for index in range(1, budget):
        starts[index] = (anchors[index - 1] + anchors[index]) // 2 + 1
        ends[index - 1] = starts[index]
    if any(start >= end for start, end in zip(starts, ends)):
        raise RuntimeError("uniform cells must be non-empty")
    if any(not (start <= anchor < end) for anchor, start, end in zip(anchors, starts, ends)):
        raise RuntimeError("uniform anchor must lie inside its canonical cell")
    return anchors, tuple(starts), tuple(ends)


def resolve_physical_cap(
    axis: PhysicalAxis,
    *,
    requested_budget: int,
    policy: str = "uniform_reference",
    value: float | int | None = None,
) -> ResolvedPhysicalCap:
    reference = exact_uniform_positions(axis.valid_len, requested_budget)
    if policy == "uniform_reference":
        report = physical_gap_report(axis, reference)
        return ResolvedPhysicalCap(
            policy=policy,
            max_source_frame_interval=report.source_frame_max_interval,
            max_seconds_interval=report.seconds_max_interval,
            reference_positions=reference,
        )
    if value is None:
        raise AllocationContractError(f"{policy} requires an explicit cap value")
    cap_value = _finite_nonnegative(value, "physical cap value")
    if policy == "explicit_frames":
        return ResolvedPhysicalCap(
            policy=policy,
            max_source_frame_interval=cap_value,
            max_seconds_interval=cap_value / axis.decoder_fps,
            reference_positions=reference,
        )
    if policy == "explicit_seconds":
        return ResolvedPhysicalCap(
            policy=policy,
            max_source_frame_interval=None,
            max_seconds_interval=cap_value,
            reference_positions=reference,
        )
    raise AllocationContractError(f"unknown physical cap policy: {policy}")


def physical_gap_report(axis: PhysicalAxis, positions: Sequence[int]) -> GapReport:
    selected = _canonical_positions(positions, axis.valid_len)
    dense_intervals = [selected[0]]
    dense_intervals.extend(right - left for left, right in zip(selected, selected[1:]))
    dense_intervals.append(axis.valid_len - 1 - selected[-1])

    frame_intervals = [axis.source_frames[selected[0]] - axis.source_frames[0]]
    frame_intervals.extend(
        axis.source_frames[right] - axis.source_frames[left]
        for left, right in zip(selected, selected[1:])
    )
    frame_intervals.append(axis.source_frames[-1] - axis.source_frames[selected[-1]])

    second_intervals = [axis.seconds[selected[0]] - axis.seconds[0]]
    second_intervals.extend(
        axis.seconds[right] - axis.seconds[left]
        for left, right in zip(selected, selected[1:])
    )
    second_intervals.append(axis.seconds[-1] - axis.seconds[selected[-1]])

    dense_holes = [selected[0], axis.valid_len - 1 - selected[-1]]
    dense_holes.extend(right - left - 1 for left, right in zip(selected, selected[1:]))
    return GapReport(
        dense_max_unselected_hole=max(dense_holes),
        dense_max_selected_interval=max(dense_intervals),
        source_frame_max_interval=max(frame_intervals),
        seconds_max_interval=max(second_intervals),
        source_frame_edge_intervals=(frame_intervals[0], frame_intervals[-1]),
        seconds_edge_intervals=(second_intervals[0], second_intervals[-1]),
    )


def validate_physical_selection(
    axis: PhysicalAxis,
    positions: Sequence[int],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
) -> tuple[int, ...]:
    selected = _canonical_positions(positions, axis.valid_len)
    budget = effective_budget(axis.valid_len, requested_budget)
    if len(selected) != budget:
        raise AllocationContractError(f"selection must contain exactly {budget} positions")
    report = physical_gap_report(axis, selected)
    if (
        cap.max_source_frame_interval is not None
        and report.source_frame_max_interval > cap.max_source_frame_interval + _EPSILON
    ):
        raise AllocationContractError("selection violates source-frame interval cap")
    if (
        cap.max_seconds_interval is not None
        and report.seconds_max_interval > cap.max_seconds_interval + _EPSILON
    ):
        raise AllocationContractError("selection violates seconds interval cap")
    return selected


def minimum_physical_scaffold(
    axis: PhysicalAxis,
    cap: ResolvedPhysicalCap,
    *,
    candidate_positions: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return the shortest cap-feasible path, then the lexicographically first."""

    candidates = (
        tuple(range(axis.valid_len))
        if candidate_positions is None
        else _canonical_positions(candidate_positions, axis.valid_len)
    )
    best_paths: dict[int, tuple[int, ...]] = {}
    for position in candidates:
        candidate_paths: list[tuple[int, ...]] = []
        if cap.allows(axis, 0, position):
            candidate_paths.append((position,))
        for predecessor in candidates:
            if predecessor >= position:
                break
            prefix = best_paths.get(predecessor)
            if prefix is not None and cap.allows(axis, predecessor, position):
                candidate_paths.append(prefix + (position,))
        if candidate_paths:
            best_paths[position] = min(candidate_paths, key=lambda path: (len(path), path))

    terminal = [
        path
        for position, path in best_paths.items()
        if cap.allows(axis, position, axis.valid_len - 1)
    ]
    if not terminal:
        raise AllocationContractError("physical cap has no source-to-sink scaffold")
    return min(terminal, key=lambda path: (len(path), path))


def select_family_a(
    axis: PhysicalAxis,
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
) -> FamilySelection:
    positions = exact_uniform_positions(axis.valid_len, requested_budget)
    validate_physical_selection(axis, positions, requested_budget=requested_budget, cap=cap)
    return build_family_selection(
        family="A_exact_uniform",
        axis=axis,
        positions=positions,
        requested_budget=requested_budget,
        score_sum=None,
        exact=True,
        deployable=True,
        privileged=False,
        solver_status="OPTIMAL",
        cap=cap,
    )


def select_family_b(
    axis: PhysicalAxis,
    scores: Sequence[float | int],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
) -> FamilySelection:
    values = _finite_scores(scores, axis.valid_len)
    anchors, starts, ends = uniform_cell_bounds(axis.valid_len, requested_budget)
    positions: list[int] = []
    for anchor, start, end in zip(anchors, starts, ends):
        best = min(
            range(start, end),
            key=lambda position: (-values[position], abs(position - anchor), position),
        )
        positions.append(best)
    selected = _canonical_positions(positions, axis.valid_len)
    return build_family_selection(
        family="B_one_per_uniform_cell",
        axis=axis,
        positions=selected,
        requested_budget=requested_budget,
        score_sum=sum(values[index] for index in selected),
        exact=True,
        deployable=True,
        privileged=False,
        solver_status="OPTIMAL",
        cap=cap,
    )


def select_family_c(
    axis: PhysicalAxis,
    scores: Sequence[float | int],
    *,
    requested_budget: int,
    cap: ResolvedPhysicalCap,
) -> FamilySelection:
    values = _finite_scores(scores, axis.valid_len)
    anchors = exact_uniform_positions(axis.valid_len, requested_budget)
    scaffold = minimum_physical_scaffold(axis, cap, candidate_positions=anchors)
    budget = effective_budget(axis.valid_len, requested_budget)
    if len(scaffold) > budget:
        raise AllocationContractError("uniform-subset scaffold exceeds the exact budget")
    scaffold_set = set(scaffold)
    ordered_residual = sorted(
        (index for index in range(axis.valid_len) if index not in scaffold_set),
        key=lambda index: (-values[index], index),
    )
    residual = tuple(sorted(ordered_residual[: budget - len(scaffold)]))
    selected = tuple(sorted(scaffold + residual))
    validate_physical_selection(axis, selected, requested_budget=requested_budget, cap=cap)
    return build_family_selection(
        family="C_uniform_scaffold_residual",
        axis=axis,
        positions=selected,
        requested_budget=requested_budget,
        score_sum=sum(values[index] for index in selected),
        exact=True,
        deployable=True,
        privileged=False,
        solver_status="OPTIMAL",
        cap=cap,
        scaffold_positions=scaffold,
        residual_positions=residual,
    )


def build_family_selection(
    *,
    family: str,
    axis: PhysicalAxis,
    positions: Sequence[int],
    requested_budget: int,
    score_sum: float | None,
    exact: bool,
    deployable: bool,
    privileged: bool,
    solver_status: str,
    cap: ResolvedPhysicalCap,
    scaffold_positions: Sequence[int] = (),
    residual_positions: Sequence[int] = (),
) -> FamilySelection:
    selected = _canonical_positions(positions, axis.valid_len)
    if len(selected) != effective_budget(axis.valid_len, requested_budget):
        raise AllocationContractError("family selection violates exact-K")
    if score_sum is not None and not math.isfinite(float(score_sum)):
        raise AllocationContractError("family score sum must be finite")
    if exact and solver_status != "OPTIMAL":
        raise AllocationContractError("exact family result must have OPTIMAL status")
    if deployable and privileged:
        raise AllocationContractError("privileged selection cannot be deployable")
    report = physical_gap_report(axis, selected)
    cap_compliant = _report_satisfies_cap(report, cap)
    return FamilySelection(
        family=str(family),
        positions=selected,
        budget=len(selected),
        score_sum=None if score_sum is None else float(score_sum),
        exact=bool(exact),
        deployable=bool(deployable),
        privileged=bool(privileged),
        solver_status=str(solver_status),
        physical_cap_compliant=cap_compliant,
        gap_report=report,
        scaffold_positions=tuple(int(value) for value in scaffold_positions),
        residual_positions=tuple(int(value) for value in residual_positions),
    )


def _canonical_positions(positions: Iterable[int], valid_len: int) -> tuple[int, ...]:
    raw = tuple(int(value) for value in positions)
    if not raw:
        raise AllocationContractError("selection must not be empty")
    if tuple(sorted(raw)) != raw:
        raise AllocationContractError("selection positions must be strictly ordered")
    if len(set(raw)) != len(raw):
        raise AllocationContractError("selection positions must be unique")
    if raw[0] < 0 or raw[-1] >= int(valid_len):
        raise AllocationContractError("selection positions lie outside the valid prefix")
    return raw


def _finite_scores(scores: Sequence[float | int], valid_len: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in scores)
    if len(values) != int(valid_len):
        raise AllocationContractError("score vector length must equal valid_len")
    if any(not math.isfinite(value) for value in values):
        raise AllocationContractError("score vector must contain only finite values")
    return values


def _report_satisfies_cap(report: GapReport, cap: ResolvedPhysicalCap) -> bool:
    if (
        cap.max_source_frame_interval is not None
        and report.source_frame_max_interval > cap.max_source_frame_interval + _EPSILON
    ):
        return False
    if (
        cap.max_seconds_interval is not None
        and report.seconds_max_interval > cap.max_seconds_interval + _EPSILON
    ):
        return False
    return True
