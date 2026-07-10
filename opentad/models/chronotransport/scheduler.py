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

    def stacked_actions(self, *, device: torch.device | str | None = None) -> Tensor:
        return torch.stack([candidate.actions for candidate in self.candidates], dim=0).to(device=device)

    def find(self, name: str) -> ScheduleCandidate:
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown schedule candidate: {name}")

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
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.schedule_library = schedule_library
        self.cost_table = cost_table
        self.epsilon = float(epsilon)
        self.max_cache_age = int(max_cache_age)
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if self.max_cache_age <= 0:
            raise ValueError("max_cache_age must be positive")
        if self.cost_table.num_groups != self.schedule_library.num_groups:
            raise ValueError("cost table and schedule library group counts differ")

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
        candidate_cost_1d = self.cost_table.estimate(candidates)
        candidate_cost = candidate_cost_1d.unsqueeze(0).expand(batch_size, -1)
        age_feasible = self._age_feasible(candidates).unsqueeze(0).expand(batch_size, -1)

        finite_signals = torch.isfinite(signals).all(dim=(-1, -2, -3))
        finite_risk = torch.isfinite(candidate_risk)
        feasible = finite_risk & age_feasible & (candidate_risk <= self.epsilon)

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
