from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


class ChronoAction(IntEnum):
    """Runtime action for one temporal chunk and one layer group."""

    RECOMPUTE = 0
    TRANSPORT = 1
    HOLD = 2

    @classmethod
    def parse(cls, value: int | str | "ChronoAction") -> "ChronoAction":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper()
            try:
                return cls[normalized]
            except KeyError as exc:
                raise ValueError(f"invalid ChronoAction name: {value!r}") from exc
        try:
            return cls(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ChronoAction value: {value!r}") from exc


@dataclass(frozen=True, order=True)
class LayerGroup:
    """Half-open transformer block interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if int(self.start) < 0:
            raise ValueError("layer-group start must be non-negative")
        if int(self.end) <= int(self.start):
            raise ValueError("layer-group width must be positive")

    @property
    def width(self) -> int:
        return int(self.end) - int(self.start)


def normalize_layer_groups(
    depth: int,
    groups: Sequence[LayerGroup | Sequence[int]] | None,
) -> tuple[LayerGroup, ...]:
    """Validate that groups are contiguous, non-overlapping, and cover depth."""

    depth = int(depth)
    if depth <= 0:
        raise ValueError("depth must be positive")
    if groups is None:
        groups = ((0, depth),)

    normalized: list[LayerGroup] = []
    for value in groups:
        if isinstance(value, LayerGroup):
            group = value
        else:
            if len(value) != 2:
                raise ValueError("each layer group must contain exactly (start, end)")
            group = LayerGroup(int(value[0]), int(value[1]))
        normalized.append(group)

    if not normalized:
        raise ValueError("at least one layer group is required")
    if normalized[0].start != 0:
        raise ValueError("layer groups must start at block 0 and be contiguous")
    for previous, current in zip(normalized, normalized[1:]):
        if previous.end != current.start:
            raise ValueError("layer groups must be contiguous and non-overlapping")
    if normalized[-1].end != depth:
        raise ValueError(f"layer groups must cover depth={depth}")
    return tuple(normalized)


def dense_action_tensor(
    batch_size: int,
    num_chunks: int,
    num_groups: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    return torch.full(
        (int(batch_size), int(num_chunks), int(num_groups)),
        int(ChronoAction.RECOMPUTE),
        dtype=torch.long,
        device=device,
    )


@dataclass(frozen=True)
class ChronoSchedule:
    """Strict schedule schema consumed by the runtime.

    ``actions`` uses shape ``[B, C, G]`` for window batch, temporal chunks,
    and layer groups. The first chunk is always recomputed for every group.
    """

    actions: Tensor
    layer_groups: tuple[LayerGroup, ...]
    name: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = self.actions
        if not isinstance(actions, Tensor):
            raise TypeError("schedule actions must be a torch.Tensor")
        if actions.ndim != 3:
            raise ValueError("schedule actions must have shape [B, C, G]")
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("schedule actions must use an integer dtype")
        if int(actions.shape[0]) <= 0 or int(actions.shape[1]) <= 0 or int(actions.shape[2]) <= 0:
            raise ValueError("schedule dimensions must be positive")
        if int(actions.shape[2]) != len(self.layer_groups):
            raise ValueError("schedule group dimension does not match layer_groups")

        values = torch.unique(actions.detach().cpu())
        valid = {int(action) for action in ChronoAction}
        invalid = [int(value.item()) for value in values if int(value.item()) not in valid]
        if invalid:
            raise ValueError(f"invalid ChronoAction values in schedule: {invalid}")

        first_chunk = actions[:, 0, :]
        if not bool(torch.all(first_chunk == int(ChronoAction.RECOMPUTE)).item()):
            raise ValueError("first chunk must RECOMPUTE for every layer group")

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_chunks(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_groups(self) -> int:
        return int(self.actions.shape[2])

    def to(self, device: torch.device | str) -> "ChronoSchedule":
        return ChronoSchedule(
            actions=self.actions.to(device=device),
            layer_groups=self.layer_groups,
            name=self.name,
            metadata=dict(self.metadata),
        )

    def clone(self, *, name: str | None = None) -> "ChronoSchedule":
        return ChronoSchedule(
            actions=self.actions.clone(),
            layer_groups=self.layer_groups,
            name=self.name if name is None else str(name),
            metadata=dict(self.metadata),
        )

    def action_counts(self) -> dict[str, int]:
        return {
            action.name.lower(): int((self.actions == int(action)).sum().item())
            for action in ChronoAction
        }

    def is_dense(self) -> bool:
        return bool(torch.all(self.actions == int(ChronoAction.RECOMPUTE)).item())


def broadcast_schedule(
    actions: Tensor,
    *,
    batch_size: int,
    num_chunks: int,
    num_groups: int,
) -> Tensor:
    """Normalize ``[C,G]``, ``[1,C,G]``, or ``[B,C,G]`` actions."""

    if not isinstance(actions, Tensor):
        actions = torch.as_tensor(actions, dtype=torch.long)
    actions = actions.to(dtype=torch.long)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3:
        raise ValueError("forced actions must have shape [C,G] or [B,C,G]")
    if tuple(actions.shape[1:]) != (int(num_chunks), int(num_groups)):
        raise ValueError(
            "forced action shape mismatch: "
            f"expected (*,{num_chunks},{num_groups}), got {tuple(actions.shape)}"
        )
    if int(actions.shape[0]) == 1 and int(batch_size) != 1:
        actions = actions.expand(int(batch_size), -1, -1).clone()
    elif int(actions.shape[0]) != int(batch_size):
        raise ValueError("forced action batch dimension does not match runtime batch")
    return actions
