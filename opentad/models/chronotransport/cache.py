from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .actions import ChronoAction


DetachPolicy = Literal["always", "never", "eval_only"]


@dataclass
class CacheEntry:
    """Anchor/latest state for one stream and one layer group."""

    anchor: Tensor | None = None
    latest: Tensor | None = None
    recompute_age: int = 0
    anchor_time: int = -1
    latest_time: int = -1

    @property
    def valid(self) -> bool:
        return self.anchor is not None and self.latest is not None

    def clear(self) -> None:
        self.anchor = None
        self.latest = None
        self.recompute_age = 0
        self.anchor_time = -1
        self.latest_time = -1


class ChronoCacheBank:
    """Per-window, per-layer-group cache with explicit detach semantics."""

    def __init__(
        self,
        num_groups: int,
        *,
        detach_policy: DetachPolicy = "always",
        training: bool = False,
    ) -> None:
        self.num_groups = int(num_groups)
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if detach_policy not in {"always", "never", "eval_only"}:
            raise ValueError(f"unsupported cache detach policy: {detach_policy}")
        self.detach_policy: DetachPolicy = detach_policy
        self.training = bool(training)
        self._entries: list[list[CacheEntry]] = []

    def reset(self, batch_size: int) -> None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._entries = [
            [CacheEntry() for _ in range(self.num_groups)]
            for _ in range(batch_size)
        ]

    @property
    def batch_size(self) -> int:
        return len(self._entries)

    def _check_index(self, stream_index: int, group_index: int) -> None:
        if not self._entries:
            raise RuntimeError("cache bank must be reset before use")
        if not 0 <= int(stream_index) < self.batch_size:
            raise IndexError("stream_index out of range")
        if not 0 <= int(group_index) < self.num_groups:
            raise IndexError("group_index out of range")

    def entry(self, stream_index: int, group_index: int) -> CacheEntry:
        self._check_index(stream_index, group_index)
        return self._entries[int(stream_index)][int(group_index)]

    def _materialize(self, state: Tensor) -> Tensor:
        if not isinstance(state, Tensor):
            raise TypeError("cache state must be a tensor")
        should_detach = self.detach_policy == "always" or (
            self.detach_policy == "eval_only" and not self.training
        )
        return state.detach() if should_detach else state

    def read_anchor(self, stream_index: int, group_index: int) -> Tensor:
        entry = self.entry(stream_index, group_index)
        if not entry.valid or entry.anchor is None:
            raise RuntimeError("invalid cache: anchor is unavailable")
        return entry.anchor

    def read_latest(self, stream_index: int, group_index: int) -> Tensor:
        entry = self.entry(stream_index, group_index)
        if not entry.valid or entry.latest is None:
            raise RuntimeError("invalid cache: latest state is unavailable")
        return entry.latest

    def age(self, stream_index: int, group_index: int) -> int:
        return int(self.entry(stream_index, group_index).recompute_age)

    def commit(
        self,
        stream_index: int,
        group_index: int,
        action: ChronoAction | int | str,
        state: Tensor,
        *,
        chunk_index: int,
    ) -> None:
        action = ChronoAction.parse(action)
        entry = self.entry(stream_index, group_index)
        chunk_index = int(chunk_index)
        state = self._materialize(state)

        if action is ChronoAction.RECOMPUTE:
            entry.anchor = state
            entry.latest = state
            entry.recompute_age = 0
            entry.anchor_time = chunk_index
            entry.latest_time = chunk_index
            return

        if not entry.valid:
            raise RuntimeError(f"invalid cache: {action.name} requires a prior RECOMPUTE")

        if action is ChronoAction.TRANSPORT:
            entry.latest = state
            entry.latest_time = chunk_index
            entry.recompute_age += 1
            return

        if action is ChronoAction.HOLD:
            # HOLD is intentionally bitwise invariant. ``state`` must be the same
            # cached object/value and is not used to overwrite the cache.
            if entry.latest is None or not torch.equal(state, entry.latest):
                raise ValueError("HOLD state must equal the latest cached state")
            entry.recompute_age += 1
            return

        raise AssertionError(f"unhandled action: {action}")

    def snapshot(self) -> list[list[dict[str, int | bool]]]:
        return [
            [
                {
                    "valid": entry.valid,
                    "recompute_age": int(entry.recompute_age),
                    "anchor_time": int(entry.anchor_time),
                    "latest_time": int(entry.latest_time),
                }
                for entry in stream
            ]
            for stream in self._entries
        ]
