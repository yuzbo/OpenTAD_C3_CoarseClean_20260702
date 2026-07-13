from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .actions import (
    ChronoAction,
    ChronoSchedule,
    LayerGroup,
    dense_action_tensor,
)
from .risk import ScheduleQuantileRiskPredictor
from .cost_lookup import CostLookupKey, ScheduleCostLookup
from .protocol import canonical_sha256


R2_NON_DENSE_NAMES = (
    "periodic2_transport",
    "periodic2_hold",
    "periodic4_transport",
    "periodic4_hold",
    "periodic8_transport",
    "periodic8_hold",
    "transport_only",
    "hold_only",
    "layer_only_early_recompute",
    "layer_only_early_recompute_hold",
    "layer_only_late_recompute",
    "layer_only_late_recompute_hold",
    "joint_progressive_transport",
    "joint_progressive_hold",
    "joint_reverse_transport",
    "joint_reverse_hold",
)


@dataclass(frozen=True)
class ScheduleCandidate:
    name: str
    actions: Tensor  # [C, G]

    def __post_init__(self) -> None:
        if self.actions.ndim != 2:
            raise ValueError("schedule candidate actions must have shape [C, G]")


class ScheduleLibrary:
    def __init__(
        self,
        candidates: Sequence[ScheduleCandidate],
        *,
        layer_groups: tuple[LayerGroup, ...],
    ) -> None:
        if not candidates:
            raise ValueError("schedule library must be non-empty")
        self.candidates = tuple(candidates)
        self.layer_groups = tuple(layer_groups)
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("schedule candidate names must be unique")
        if names[0] != "dense":
            raise ValueError("dense must be the first fail-closed candidate")

        shape = tuple(self.candidates[0].actions.shape)
        if shape[1] != len(self.layer_groups):
            raise ValueError("candidate group dimension does not match layer groups")
        for candidate in self.candidates:
            if tuple(candidate.actions.shape) != shape:
                raise ValueError("all schedule candidates must have identical shape")
            ChronoSchedule(
                actions=candidate.actions.unsqueeze(0),
                layer_groups=self.layer_groups,
                name=candidate.name,
            )

    @property
    def num_chunks(self) -> int:
        return int(self.candidates[0].actions.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.candidates[0].actions.shape[1])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        if set(self.names) == {"dense", *R2_NON_DENSE_NAMES}:
            return R2_NON_DENSE_NAMES + ("dense",)
        return self.names

    def canonical_payload(self) -> dict[str, object]:
        candidates = []
        for name in self.canonical_names:
            candidate = self.find(name)
            action_matrix = candidate.actions.detach().cpu().to(torch.long).tolist()
            candidates.append(
                {
                    "name": name,
                    "actions": action_matrix,
                    "hard_valid": True,
                    "action_sha256": canonical_sha256(action_matrix),
                }
            )
        is_frozen_r2 = self.canonical_names == R2_NON_DENSE_NAMES + ("dense",)
        body: dict[str, object] = {
            "schema": (
                "chronotransport-r2-library-v2"
                if is_frozen_r2
                else "chronotransport-r2-library-v1"
            ),
            "num_chunks": self.num_chunks,
            "num_groups": self.num_groups,
            "candidates": candidates,
        }
        if is_frozen_r2:
            body["layer_groups"] = [
                [int(group.start), int(group.end)] for group in self.layer_groups
            ]
        body["library_sha256"] = canonical_sha256(body)
        return body

    @property
    def library_sha256(self) -> str:
        return str(self.canonical_payload()["library_sha256"])

    def stacked_actions(self, *, device: torch.device | str | None = None) -> Tensor:
        return torch.stack([candidate.actions for candidate in self.candidates], dim=0).to(device=device)

    def find(self, name: str) -> ScheduleCandidate:
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown schedule candidate: {name}")

    @classmethod
    def r2(
        cls,
        *,
        num_chunks: int = 48,
        layer_groups: tuple[LayerGroup, ...],
    ) -> "ScheduleLibrary":
        if type(num_chunks) is not int or num_chunks != 48:
            raise ValueError("r2 schedule library requires exactly 48 clips")
        if len(layer_groups) != 3:
            raise ValueError("r2 schedule library requires exactly three layer groups")
        if (
            any(
                not isinstance(group, LayerGroup)
                or type(group.start) is not int
                or type(group.end) is not int
                for group in layer_groups
            )
            or tuple((group.start, group.end) for group in layer_groups) != (
            (0, 4),
            (4, 8),
            (8, 12),
            )
        ):
            raise ValueError("r2 schedule library requires exact layer groups [0:4]/[4:8]/[8:12]")

        def periodic(period: int, fallback: ChronoAction) -> Tensor:
            actions = torch.full((48, 3), int(fallback), dtype=torch.long)
            actions[:: int(period), :] = int(ChronoAction.RECOMPUTE)
            return actions

        def only(group: int, fallback: ChronoAction) -> Tensor:
            actions = torch.full((48, 3), int(fallback), dtype=torch.long)
            actions[:, int(group)] = int(ChronoAction.RECOMPUTE)
            actions[0, :] = int(ChronoAction.RECOMPUTE)
            return actions

        def joint(periods: tuple[int, int, int], fallback: ChronoAction) -> Tensor:
            actions = torch.full((48, 3), int(fallback), dtype=torch.long)
            for group, period in enumerate(periods):
                actions[::period, group] = int(ChronoAction.RECOMPUTE)
            actions[0, :] = int(ChronoAction.RECOMPUTE)
            return actions

        dense = torch.zeros((48, 3), dtype=torch.long)
        matrices = {
            "periodic2_transport": periodic(2, ChronoAction.TRANSPORT),
            "periodic2_hold": periodic(2, ChronoAction.HOLD),
            "periodic4_transport": periodic(4, ChronoAction.TRANSPORT),
            "periodic4_hold": periodic(4, ChronoAction.HOLD),
            "periodic8_transport": periodic(8, ChronoAction.TRANSPORT),
            "periodic8_hold": periodic(8, ChronoAction.HOLD),
            "transport_only": periodic(49, ChronoAction.TRANSPORT),
            "hold_only": periodic(49, ChronoAction.HOLD),
            "layer_only_early_recompute": only(0, ChronoAction.TRANSPORT),
            "layer_only_early_recompute_hold": only(0, ChronoAction.HOLD),
            "layer_only_late_recompute": only(2, ChronoAction.TRANSPORT),
            "layer_only_late_recompute_hold": only(2, ChronoAction.HOLD),
            "joint_progressive_transport": joint((8, 4, 2), ChronoAction.TRANSPORT),
            "joint_progressive_hold": joint((8, 4, 2), ChronoAction.HOLD),
            "joint_reverse_transport": joint((2, 4, 8), ChronoAction.TRANSPORT),
            "joint_reverse_hold": joint((2, 4, 8), ChronoAction.HOLD),
        }
        candidates = [ScheduleCandidate("dense", dense)]
        candidates.extend(ScheduleCandidate(name, matrices[name]) for name in R2_NON_DENSE_NAMES)
        return cls(candidates, layer_groups=layer_groups)

    @classmethod
    def default(
        cls,
        *,
        num_chunks: int,
        layer_groups: tuple[LayerGroup, ...],
    ) -> "ScheduleLibrary":
        num_chunks = int(num_chunks)
        num_groups = len(layer_groups)
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive")

        def periodic(period: int, fallback: ChronoAction) -> Tensor:
            actions = torch.full((num_chunks, num_groups), int(fallback), dtype=torch.long)
            actions[:: int(period), :] = int(ChronoAction.RECOMPUTE)
            actions[0, :] = int(ChronoAction.RECOMPUTE)
            return actions

        dense = torch.full(
            (num_chunks, num_groups),
            int(ChronoAction.RECOMPUTE),
            dtype=torch.long,
        )
        def layer_only(recompute_group: int, fallback: ChronoAction = ChronoAction.TRANSPORT) -> Tensor:
            actions = torch.full((num_chunks, num_groups), int(fallback), dtype=torch.long)
            actions[:, int(recompute_group)] = int(ChronoAction.RECOMPUTE)
            actions[0, :] = int(ChronoAction.RECOMPUTE)
            return actions

        def joint_progressive() -> Tensor:
            actions = torch.full(
                (num_chunks, num_groups),
                int(ChronoAction.TRANSPORT),
                dtype=torch.long,
            )
            for group_index in range(num_groups):
                # Earlier groups are refreshed less often; later task-specific
                # groups receive denser refreshes. For three groups this is 8/4/2.
                period = max(2, 2 ** (num_groups - group_index))
                actions[::period, group_index] = int(ChronoAction.RECOMPUTE)
            actions[0, :] = int(ChronoAction.RECOMPUTE)
            return actions

        hold = periodic(num_chunks + 1, ChronoAction.HOLD)
        transport = periodic(num_chunks + 1, ChronoAction.TRANSPORT)
        candidates = [
            ScheduleCandidate("dense", dense),
            # Time-only controls: every group receives the same temporal plan.
            ScheduleCandidate("periodic2_transport", periodic(2, ChronoAction.TRANSPORT)),
            ScheduleCandidate("periodic4_transport", periodic(4, ChronoAction.TRANSPORT)),
            ScheduleCandidate("periodic8_transport", periodic(8, ChronoAction.TRANSPORT)),
            ScheduleCandidate("periodic2_hold", periodic(2, ChronoAction.HOLD)),
            ScheduleCandidate("hold_only", hold),
            ScheduleCandidate("transport_only", transport),
        ]
        if num_groups > 1:
            candidates.extend(
                [
                    ScheduleCandidate("layer_only_early_recompute", layer_only(0)),
                    ScheduleCandidate("layer_only_late_recompute", layer_only(num_groups - 1)),
                    ScheduleCandidate("joint_progressive_transport", joint_progressive()),
                ]
            )
        return cls(candidates, layer_groups=layer_groups)


def validate_r2_library_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Deeply validate the frozen 16+1 r2 candidate-library artifact."""

    if not isinstance(payload, Mapping):
        raise TypeError("r2 library payload must be a mapping")
    expected_root_keys = {
        "schema",
        "num_chunks",
        "num_groups",
        "layer_groups",
        "candidates",
        "library_sha256",
    }
    if set(payload) != expected_root_keys:
        raise ValueError("r2 library fields mismatch")
    if payload["schema"] != "chronotransport-r2-library-v2":
        raise ValueError("r2 library schema mismatch")
    if (
        type(payload["num_chunks"]) is not int
        or type(payload["num_groups"]) is not int
        or payload["num_chunks"] != 48
        or payload["num_groups"] != 3
    ):
        raise ValueError("r2 library action shape must be exactly 48 x 3")
    layer_groups = payload["layer_groups"]
    if (
        not isinstance(layer_groups, list)
        or any(
            not isinstance(group, list)
            or len(group) != 2
            or any(type(endpoint) is not int for endpoint in group)
            for group in layer_groups
        )
        or layer_groups != [[0, 4], [4, 8], [8, 12]]
    ):
        raise ValueError("r2 library layer-group identity mismatch")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 17:
        raise ValueError("r2 library must contain exactly 16 non-dense actions plus dense")
    expected_library = ScheduleLibrary.r2(
        layer_groups=(LayerGroup(0, 4), LayerGroup(4, 8), LayerGroup(8, 12)),
    ).canonical_payload()
    expected_candidates = expected_library["candidates"]
    expected_names = R2_NON_DENSE_NAMES + ("dense",)
    actual_names = tuple(
        candidate.get("name") if isinstance(candidate, Mapping) else None
        for candidate in candidates
    )
    if actual_names != expected_names:
        raise ValueError("r2 library candidate order/name mismatch")
    for index, (candidate, expected) in enumerate(zip(candidates, expected_candidates)):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"r2 library candidate {index} must be a mapping")
        if set(candidate) != {"name", "actions", "hard_valid", "action_sha256"}:
            raise ValueError(f"r2 library candidate {index} fields mismatch")
        actions = candidate["actions"]
        if not isinstance(actions, list) or len(actions) != 48:
            raise ValueError(f"r2 candidate {candidate['name']} action matrix must have 48 rows")
        for row in actions:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"r2 candidate {candidate['name']} action rows must have 3 groups")
            if any(
                isinstance(action, bool)
                or not isinstance(action, int)
                or action not in (0, 1, 2)
                for action in row
            ):
                raise ValueError(f"r2 candidate {candidate['name']} contains an invalid action")
        if actions[0] != [int(ChronoAction.RECOMPUTE)] * 3:
            raise ValueError(f"r2 candidate {candidate['name']} action row zero must recompute")
        if actions != expected["actions"]:
            raise ValueError(f"r2 candidate {candidate['name']} action matrix mismatch")
        if candidate["hard_valid"] is not True:
            raise ValueError(f"r2 candidate {candidate['name']} must be hard-valid")
        if candidate["action_sha256"] != canonical_sha256(actions):
            raise ValueError(f"r2 candidate {candidate['name']} action hash mismatch")
    unsigned = dict(payload)
    library_sha256 = unsigned.pop("library_sha256")
    if library_sha256 != canonical_sha256(unsigned):
        raise ValueError("r2 library hash mismatch")
    if dict(payload) != expected_library:
        raise ValueError("r2 library payload differs from the frozen canonical library")
    return dict(payload)


@dataclass(frozen=True)
class MeasuredCostTable:
    """Profiled action cost by layer group, in milliseconds."""

    recompute: tuple[float, ...]
    transport: tuple[float, ...]
    hold: tuple[float, ...]
    scheduler_overhead: float = 0.0

    def __post_init__(self) -> None:
        lengths = {len(self.recompute), len(self.transport), len(self.hold)}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("cost vectors must share a positive group length")
        values = (*self.recompute, *self.transport, *self.hold, self.scheduler_overhead)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("cost values must be finite")
        if any(float(value) < 0.0 for value in values):
            raise ValueError("cost values must be non-negative")

    @property
    def num_groups(self) -> int:
        return len(self.recompute)

    def estimate(self, actions: Tensor) -> Tensor:
        if actions.ndim < 2 or int(actions.shape[-1]) != self.num_groups:
            raise ValueError("action tensor group dimension does not match cost table")
        dtype = torch.float32
        device = actions.device
        vectors = torch.tensor(
            [self.recompute, self.transport, self.hold],
            dtype=dtype,
            device=device,
        )
        group_ids = torch.arange(self.num_groups, device=device)
        cell_cost = vectors[actions.to(torch.long), group_ids]
        return cell_cost.sum(dim=(-1, -2)) + float(self.scheduler_overhead)


def motion_threshold_actions(
    motion: Tensor,
    *,
    num_groups: int,
    threshold: float,
    fallback: ChronoAction | int | str = ChronoAction.TRANSPORT,
) -> Tensor:
    """Build a deploy-visible motion-threshold baseline schedule.

    ``motion`` is either ``[B,C]`` or ``[B,C,G]``. Non-finite motion cells
    fail closed to RECOMPUTE, and the first chunk is always recomputed.
    """

    if not isinstance(motion, Tensor):
        motion = torch.as_tensor(motion)
    if motion.ndim not in (2, 3):
        raise ValueError("motion must have shape [B,C] or [B,C,G]")
    num_groups = int(num_groups)
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("motion threshold must be finite")
    fallback = ChronoAction.parse(fallback)
    if motion.ndim == 2:
        motion = motion.unsqueeze(-1).expand(-1, -1, num_groups)
    elif int(motion.shape[-1]) == 1 and num_groups != 1:
        motion = motion.expand(-1, -1, num_groups)
    elif int(motion.shape[-1]) != num_groups:
        raise ValueError("motion group dimension does not match num_groups")

    finite = torch.isfinite(motion)
    recompute = finite & (motion >= threshold)
    actions = torch.full(
        tuple(motion.shape),
        int(fallback),
        dtype=torch.long,
        device=motion.device,
    )
    actions[recompute | (~finite)] = int(ChronoAction.RECOMPUTE)
    actions[:, 0, :] = int(ChronoAction.RECOMPUTE)
    return actions


@dataclass(frozen=True)
class SchedulerSelection:
    schedule: ChronoSchedule
    selected_names: tuple[str, ...]
    upper_risk: Tensor
    estimated_cost: Tensor
    fail_closed: Tensor
    candidate_upper_risk: Tensor
    candidate_cost: Tensor


class RiskConstrainedScheduler(nn.Module):
    """Choose the cheapest calibrated candidate satisfying a risk bound."""

    def __init__(
        self,
        predictor: ScheduleQuantileRiskPredictor,
        schedule_library: ScheduleLibrary,
        cost_table: MeasuredCostTable,
        *,
        epsilon: float,
        max_cache_age: int,
        schedule_cost_lookup: ScheduleCostLookup | None = None,
        cost_hardware: str = "",
        cost_precision: str = "",
        cost_statistic: str = "p50",
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.schedule_library = schedule_library
        self.cost_table = cost_table
        self.epsilon = float(epsilon)
        self.max_cache_age = int(max_cache_age)
        self.schedule_cost_lookup = schedule_cost_lookup
        self.cost_hardware = str(cost_hardware)
        self.cost_precision = str(cost_precision)
        self.cost_statistic = str(cost_statistic)
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if self.max_cache_age <= 0:
            raise ValueError("max_cache_age must be positive")
        if self.cost_table.num_groups != self.schedule_library.num_groups:
            raise ValueError("cost table and schedule library group counts differ")
        if self.schedule_cost_lookup is not None:
            if not self.cost_hardware or not self.cost_precision:
                raise ValueError("nonlinear cost lookup requires hardware and precision")
            if self.cost_statistic not in {"p50", "p95"}:
                raise ValueError("cost statistic must be p50 or p95")

    def _candidate_cost(self, candidates: Tensor, batch_size: int) -> Tensor:
        if self.schedule_cost_lookup is None:
            return self.cost_table.estimate(candidates)
        costs = []
        for index, name in enumerate(self.schedule_library.names):
            actions = candidates[index]
            rows = tuple(
                int((actions[:, group] == int(ChronoAction.RECOMPUTE)).sum().item())
                for group in range(int(actions.shape[1]))
            )
            key = CostLookupKey(
                hardware=self.cost_hardware,
                precision=self.cost_precision,
                batch_size=int(batch_size),
                candidate_schedule=name,
                selected_rows_per_group=rows,
            )
            costs.append(self.schedule_cost_lookup.get(key, self.cost_statistic))
        return torch.tensor(costs, dtype=torch.float32, device=candidates.device)

    def _age_feasible(self, candidates: Tensor) -> Tensor:
        # candidates: [K,C,G]
        candidate_count, num_chunks, num_groups = candidates.shape
        feasible = torch.ones(candidate_count, dtype=torch.bool, device=candidates.device)
        age = torch.zeros((candidate_count, num_groups), dtype=torch.long, device=candidates.device)
        for chunk_index in range(num_chunks):
            action = candidates[:, chunk_index, :]
            recompute = action == int(ChronoAction.RECOMPUTE)
            age = torch.where(recompute, torch.zeros_like(age), age + 1)
            feasible &= torch.all(age <= self.max_cache_age, dim=-1)
        return feasible

    def select(self, signals: Tensor, ood_mask: Tensor | None = None) -> SchedulerSelection:
        if signals.ndim != 4:
            raise ValueError("signals must have shape [B,C,G,F]")
        batch_size, num_chunks, num_groups, _ = signals.shape
        if (num_chunks, num_groups) != (
            self.schedule_library.num_chunks,
            self.schedule_library.num_groups,
        ):
            raise ValueError("signals do not match schedule-library dimensions")

        candidates = self.schedule_library.stacked_actions(device=signals.device)
        candidate_risk = self.predictor(signals, candidates)
        # Dense is a safety action outside r2 head fit/calibration/ranking and
        # therefore has exact zero risk in the scheduler.
        dense_index = self.schedule_library.names.index("dense")
        candidate_risk = candidate_risk.clone()
        candidate_risk[:, dense_index] = 0.0
        candidate_cost_1d = self._candidate_cost(candidates, batch_size)
        candidate_cost = candidate_cost_1d.unsqueeze(0).expand(batch_size, -1)
        age_feasible = self._age_feasible(candidates).unsqueeze(0).expand(batch_size, -1)

        finite_signals = torch.isfinite(signals).reshape(batch_size, -1).all(dim=1)
        finite_risk = torch.isfinite(candidate_risk)
        feasible = finite_risk & age_feasible & (candidate_risk <= self.epsilon)
        feasible[:, dense_index] = False

        if ood_mask is None:
            ood_mask = torch.zeros(batch_size, dtype=torch.bool, device=signals.device)
        else:
            ood_mask = ood_mask.to(device=signals.device, dtype=torch.bool).reshape(-1)
            if int(ood_mask.numel()) != batch_size:
                raise ValueError("ood_mask must have one value per window")

        global_invalid = (~finite_signals) | ood_mask
        feasible = feasible & (~global_invalid).unsqueeze(1)

        inf = torch.full_like(candidate_cost, float("inf"))
        feasible_cost = torch.where(feasible, candidate_cost, inf)
        selected_index = feasible_cost.argmin(dim=1)
        has_feasible = feasible.any(dim=1)
        selected_index = torch.where(has_feasible, selected_index, torch.zeros_like(selected_index))
        fail_closed = ~has_feasible

        selected_actions = candidates[selected_index]
        schedule = ChronoSchedule(
            actions=selected_actions,
            layer_groups=self.schedule_library.layer_groups,
            name="per_window_library_selection",
            metadata={"epsilon": self.epsilon},
        )
        names = tuple(self.schedule_library.names[int(index)] for index in selected_index.tolist())
        batch_indices = torch.arange(batch_size, device=signals.device)
        upper_risk = candidate_risk[batch_indices, selected_index]
        estimated_cost = candidate_cost[batch_indices, selected_index]
        return SchedulerSelection(
            schedule=schedule,
            selected_names=names,
            upper_risk=upper_risk,
            estimated_cost=estimated_cost,
            fail_closed=fail_closed,
            candidate_upper_risk=candidate_risk,
            candidate_cost=candidate_cost,
        )
