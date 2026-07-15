from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.utils.checkpoint as cp
from torch import Tensor, nn

from .actions import (
    ChronoAction,
    ChronoSchedule,
    LayerGroup,
    broadcast_schedule,
    dense_action_tensor,
    normalize_layer_groups,
)
from .profiler import ChronoProfiler
from .risk import ScheduleQuantileRiskPredictor
from .scheduler import MeasuredCostTable, RiskConstrainedScheduler, ScheduleLibrary
from .transport import TemporalTransportAdapter
from .cost_lookup import ScheduleCostLookup
from .protocol import canonical_sha256


class ChronoTransportRuntime(nn.Module):
    """Chunk × layer-group runtime for VideoMAE-like transformer blocks.

    Heavy attention/MLP is gathered and executed only for RECOMPUTE chunks.
    Existing AdaTAD adapters remain a dense innovation path: they see a dense
    surrogate tensor and write back every row in the original block order.
    RECOMPUTE and TRANSPORT rows remain live for the current row while cached
    history is detached; HOLD consumes the detached latest state.

    The all-RECOMPUTE path calls the original block forward exactly and is the
    numerical-compatibility anchor. Dynamic learned scheduling fails closed to
    that path until both measured costs and explicit risk readiness are present.
    """

    def __init__(
        self,
        *,
        embed_dims: int,
        depth: int,
        chunks_per_window: int,
        layer_groups: Sequence[LayerGroup | Sequence[int]] | None = None,
        enabled: bool = True,
        signal_dims: int = 6,
        risk_hidden_dims: int = 64,
        transport_bottleneck_dims: int = 64,
        risk_quantile: float = 0.9,
        risk_epsilon: float = 1.0,
        max_cache_age: int | None = None,
        hard_cache_validity_age: int = 47,
        transport_age_embedding_cap: int = 8,
        forced_schedule: str | None = None,
        forced_actions: Tensor | None = None,
        cache_detach: bool = True,
        profile_sync_cuda: bool = False,
        measured_cost: Mapping[str, Sequence[float] | float] | None = None,
        nonlinear_cost_entries: Mapping[str, Mapping[str, float]] | None = None,
        cost_hardware: str = "",
        cost_precision: str = "",
        cost_statistic: str = "p50",
        allow_unmeasured_cost_for_debug: bool = False,
        risk_ready: bool = False,
        require_checkpoint_for_dynamic: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.depth = int(depth)
        self.chunks_per_window = int(chunks_per_window)
        self.enabled = bool(enabled)
        self.signal_dims = int(signal_dims)
        # ``max_cache_age`` is retained only as a compatibility override for
        # legacy configs.  New r2 configs must use the two explicit fields.
        if max_cache_age is not None:
            hard_cache_validity_age = int(max_cache_age)
            transport_age_embedding_cap = int(max_cache_age)
        self.hard_cache_validity_age = int(hard_cache_validity_age)
        self.transport_age_embedding_cap = int(transport_age_embedding_cap)
        self.max_cache_age = self.hard_cache_validity_age
        self.forced_schedule = forced_schedule
        self.forced_action_name = "forced_actions"
        self.cache_detach = bool(cache_detach)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self.allow_unmeasured_cost_for_debug = bool(allow_unmeasured_cost_for_debug)
        self.risk_ready = bool(risk_ready)
        self.require_checkpoint_for_dynamic = bool(require_checkpoint_for_dynamic)
        self.checkpoint_loaded = False
        self.cost_is_measured = measured_cost is not None
        self.nonlinear_cost_ready = nonlinear_cost_entries is not None
        self.layer_groups = normalize_layer_groups(self.depth, layer_groups)
        if self.embed_dims <= 0 or self.chunks_per_window <= 0 or self.signal_dims <= 0:
            raise ValueError("runtime dimensions must be positive")
        if self.hard_cache_validity_age <= 0:
            raise ValueError("hard_cache_validity_age must be positive")
        if self.transport_age_embedding_cap <= 0:
            raise ValueError("transport_age_embedding_cap must be positive")
        if forced_schedule is not None and forced_actions is not None:
            raise ValueError("forced_schedule and forced_actions are mutually exclusive")
        self.register_buffer(
            "forced_actions",
            torch.empty(0, dtype=torch.long) if forced_actions is None else forced_actions.to(torch.long),
            persistent=False,
        )

        self.transport = nn.ModuleList(
            [
                TemporalTransportAdapter(
                    embed_dims=self.embed_dims,
                    bottleneck_dims=int(transport_bottleneck_dims),
                    max_age=self.transport_age_embedding_cap,
                )
                for _ in self.layer_groups
            ]
        )
        self.risk_predictor = ScheduleQuantileRiskPredictor(
            signal_dims=self.signal_dims,
            num_groups=len(self.layer_groups),
            hidden_dims=int(risk_hidden_dims),
            quantile=float(risk_quantile),
        )
        if self.chunks_per_window == 48 and len(self.layer_groups) == 3:
            self.schedule_library = ScheduleLibrary.r2(
                num_chunks=self.chunks_per_window,
                layer_groups=self.layer_groups,
            )
        else:
            self.schedule_library = ScheduleLibrary.default(
                num_chunks=self.chunks_per_window,
                layer_groups=self.layer_groups,
            )

        # Proxy costs only make forced baseline/debug execution possible. They
        # never unlock the learned deployment scheduler.
        if measured_cost is None:
            widths = tuple(float(group.width) for group in self.layer_groups)
            measured_cost = {
                "recompute": widths,
                "transport": tuple(0.15 * width for width in widths),
                "hold": tuple(0.01 for _ in widths),
                "scheduler_overhead": 0.0,
            }
        self.cost_table = MeasuredCostTable(
            recompute=tuple(float(value) for value in measured_cost["recompute"]),
            transport=tuple(float(value) for value in measured_cost["transport"]),
            hold=tuple(float(value) for value in measured_cost["hold"]),
            scheduler_overhead=float(measured_cost.get("scheduler_overhead", 0.0)),
        )
        self.scheduler = RiskConstrainedScheduler(
            predictor=self.risk_predictor,
            schedule_library=self.schedule_library,
            cost_table=self.cost_table,
            epsilon=float(risk_epsilon),
            max_cache_age=self.hard_cache_validity_age,
            schedule_cost_lookup=(
                None
                if nonlinear_cost_entries is None
                else ScheduleCostLookup(nonlinear_cost_entries)
            ),
            cost_hardware=cost_hardware,
            cost_precision=cost_precision,
            cost_statistic=cost_statistic,
        )
        self.latest_summary: dict[str, Any] | None = None
        self.latest_schedule: ChronoSchedule | None = None
        self.latest_signals: Tensor | None = None
        self.latest_output: Tensor | None = None
        self.capture_replay_signals = False

    def set_checkpoint_loaded(self, loaded: bool) -> None:
        """Mark whether a ChronoTransport-trained checkpoint populated the runtime."""

        self.checkpoint_loaded = bool(loaded)

    def set_registered_forced_actions(
        self,
        actions: Tensor,
        *,
        candidate_name: str,
    ) -> None:
        """Bind exact pre-registered action bytes to their candidate identity."""

        if not isinstance(candidate_name, str) or not candidate_name or "\x00" in candidate_name:
            raise ValueError("registered forced-action candidate_name must be non-empty and NUL-free")
        normalized = torch.as_tensor(actions, dtype=torch.long, device=self.forced_actions.device)
        if tuple(normalized.shape) != (
            self.chunks_per_window,
            len(self.layer_groups),
        ):
            raise ValueError("registered forced actions must have exact [chunks,groups] shape")
        # Constructing the schedule validates integer action values and the
        # mandatory first-chunk RECOMPUTE rule before any profiled inference.
        ChronoSchedule(
            actions=normalized.unsqueeze(0),
            layer_groups=self.layer_groups,
            name=candidate_name,
        )
        self.forced_actions = normalized.detach().clone()
        self.forced_schedule = None
        self.forced_action_name = candidate_name

    @staticmethod
    def _dense_forward(x: Tensor, blocks: Sequence[nn.Module], h: int, w: int) -> Tensor:
        out = x
        for block in blocks:
            out = block(out, h, w)
        return out

    def _signals(self, state: Tensor) -> Tensor:
        # state: [B,C,N,D]. All features are deploy-visible because they are the
        # current group input/patch-token state, never dense-reference features.
        pooled = state.float().mean(dim=2)
        energy = state.float().square().mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        delta = torch.zeros_like(pooled)
        delta[:, 1:] = pooled[:, 1:] - pooled[:, :-1]
        delta_l2 = delta.square().mean(dim=-1, keepdim=True).sqrt()
        pooled_l2 = pooled.square().mean(dim=-1, keepdim=True).sqrt()
        cosine = torch.zeros_like(delta_l2)
        if int(state.shape[1]) > 1:
            current = pooled[:, 1:]
            previous = pooled[:, :-1]
            numerator = (current * previous).sum(dim=-1, keepdim=True)
            denominator = current.norm(dim=-1, keepdim=True) * previous.norm(dim=-1, keepdim=True)
            cosine[:, 1:] = 1.0 - numerator / denominator.clamp_min(1e-6)
        chunk_position = torch.linspace(
            0.0,
            1.0,
            int(state.shape[1]),
            device=state.device,
            dtype=state.dtype,
        ).view(1, -1, 1).expand(int(state.shape[0]), -1, -1)
        finite = (
            torch.isfinite(state)
            .reshape(int(state.shape[0]), int(state.shape[1]), -1)
            .all(dim=-1)
            .unsqueeze(-1)
            .to(state.dtype)
        )
        raw = torch.cat(
            (
                energy.to(state.dtype),
                delta_l2.to(state.dtype),
                pooled_l2.to(state.dtype),
                cosine.to(state.dtype),
                chunk_position,
                finite,
            ),
            dim=-1,
        )
        if self.signal_dims < int(raw.shape[-1]):
            raw = raw[..., : self.signal_dims]
        elif self.signal_dims > int(raw.shape[-1]):
            raw = torch.nn.functional.pad(raw, (0, self.signal_dims - int(raw.shape[-1])))
        # Group identity and candidate cache age are supplied by the risk model.
        return raw.unsqueeze(2).expand(-1, -1, len(self.layer_groups), -1).contiguous()

    def _forced_schedule(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[ChronoSchedule, int, bool, Tensor] | None:
        if self.forced_actions.numel() > 0:
            actions = broadcast_schedule(
                self.forced_actions,
                batch_size=batch_size,
                num_chunks=self.chunks_per_window,
                num_groups=len(self.layer_groups),
            ).to(device=device)
            requested_actions = actions.clone()
            actions, repairs, first_forced = self._repair_schedule(actions)
            return (
                ChronoSchedule(
                    actions=actions,
                    layer_groups=self.layer_groups,
                    name=self.forced_action_name,
                ),
                repairs,
                first_forced,
                requested_actions,
            )
        if self.forced_schedule is None:
            return None
        candidate = self.schedule_library.find(self.forced_schedule)
        actions = candidate.actions.to(device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        requested_actions = actions.clone()
        actions, repairs, first_forced = self._repair_schedule(actions)
        return (
            ChronoSchedule(actions=actions, layer_groups=self.layer_groups, name=candidate.name),
            repairs,
            first_forced,
            requested_actions,
        )

    def _repair_schedule(self, actions: Tensor) -> tuple[Tensor, int, bool]:
        actions = actions.clone().to(dtype=torch.long)
        repairs = 0
        first_forced = False
        batch_size, num_chunks, num_groups = actions.shape
        valid_values = {int(action) for action in ChronoAction}
        for batch_index in range(batch_size):
            for group_index in range(num_groups):
                if int(actions[batch_index, 0, group_index].item()) != int(ChronoAction.RECOMPUTE):
                    actions[batch_index, 0, group_index] = int(ChronoAction.RECOMPUTE)
                    repairs += 1
                    first_forced = True
                age = 0
                for chunk_index in range(1, num_chunks):
                    value = int(actions[batch_index, chunk_index, group_index].item())
                    if value not in valid_values:
                        actions[batch_index, chunk_index, group_index] = int(ChronoAction.RECOMPUTE)
                        repairs += 1
                        age = 0
                        continue
                    if value == int(ChronoAction.RECOMPUTE):
                        age = 0
                    else:
                        age += 1
                        if age > self.hard_cache_validity_age:
                            actions[batch_index, chunk_index, group_index] = int(ChronoAction.RECOMPUTE)
                            repairs += 1
                            age = 0
        return actions, repairs, first_forced

    @staticmethod
    def _heavy_forward(block: nn.Module, selected: Tensor) -> Tensor:
        def inner(value: Tensor) -> Tensor:
            value = value + block.drop_path(block.attn(block.norm1(value)))
            value = value + block.drop_path(block.mlp(block.norm2(value)))
            return value

        if bool(getattr(block, "with_cp", False)) and selected.requires_grad:
            return cp.checkpoint(inner, selected, use_reentrant=False)
        return inner(selected)

    def _runtime_geometry(self, x: Tensor, h: int, w: int) -> dict[str, int]:
        spatial_tokens = int(h) * int(w)
        if spatial_tokens <= 0:
            raise ValueError("h and w must define a positive spatial token grid")
        if int(x.shape[1]) % spatial_tokens != 0:
            raise ValueError("per-clip token length must be divisible by h*w")
        tubelets_per_chunk = int(x.shape[1]) // spatial_tokens
        return {
            "chunks_per_window": self.chunks_per_window,
            "tubelets_per_chunk": tubelets_per_chunk,
            "internal_tubelet_points": self.chunks_per_window * tubelets_per_chunk,
            "spatial_tokens_per_tubelet": spatial_tokens,
        }

    def _dense_fail_closed(
        self,
        x: Tensor,
        blocks: Sequence[nn.Module],
        h: int,
        w: int,
        *,
        profiler: ChronoProfiler,
        reason: str,
        batch_size: int,
        geometry: Mapping[str, int],
    ) -> Tensor:
        actions = dense_action_tensor(
            batch_size,
            self.chunks_per_window,
            len(self.layer_groups),
            device=x.device,
        )
        schedule = ChronoSchedule(actions=actions, layer_groups=self.layer_groups, name="dense")
        self.latest_schedule = schedule
        with profiler.stage("recompute"):
            out = self._dense_forward(x, blocks, h, w)
        self.latest_output = out
        dense_rows = int(x.shape[0]) * self.depth
        self.latest_summary = {
            "schema_version": "chronotransport_runtime_v1",
            "enabled": True,
            "forced_dense_exact_path": True,
            "fail_closed_reason": str(reason),
            "selected_schedule_names": ["dense"] * batch_size,
            "recompute_rows": dense_rows,
            "transport_rows": 0,
            "hold_rows": 0,
            "adapter_dense_forward_count": sum(
                1 for block in blocks if bool(getattr(block, "use_adapter", False))
            ),
            "schedule_repair_count": 0,
            "first_chunk_forced_recompute": False,
            "runtime_fail_closed_repairs": 0,
            "dense_output_shape_preserved": True,
            "cost_is_measured": self.cost_is_measured,
            "nonlinear_cost_ready": self.nonlinear_cost_ready,
            "risk_ready": self.risk_ready,
            "checkpoint_loaded": self.checkpoint_loaded,
            "require_checkpoint_for_dynamic": self.require_checkpoint_for_dynamic,
            "cache_reset_per_window": True,
            "external_dense_grid_preserved_by_post_interpolation": True,
            **dict(geometry),
            "profile": profiler.summary(fill_missing=True),
        }
        return out

    def _run_group(
        self,
        state: Tensor,
        blocks: Sequence[nn.Module],
        actions: Tensor,
        *,
        group_index: int,
        h: int,
        w: int,
        profiler: ChronoProfiler,
        counters: dict[str, int],
    ) -> Tensor:
        batch_size, num_chunks, token_count, channels = state.shape
        if channels != self.embed_dims:
            raise ValueError("runtime state channel dimension does not match embed_dims")

        # Group-level actions are expanded across every block in the group. Each
        # block owns a rolling cache because the original AdaTAD adapter sits
        # after every transformer block and must remain in its original location.
        for block in blocks:
            current_flat = state.reshape(batch_size * num_chunks, token_count, channels)
            recompute_mask = actions == int(ChronoAction.RECOMPUTE)
            effective_flat_mask = recompute_mask.reshape(-1).clone()
            with profiler.stage("cache_movement"):
                provisional = torch.empty_like(current_flat)
                selected = current_flat[effective_flat_mask]

            if int(selected.shape[0]) > 0:
                with profiler.stage("recompute"):
                    selected_heavy = self._heavy_forward(block, selected)
                with profiler.stage("cache_movement"):
                    provisional[effective_flat_mask] = selected_heavy
                counters["recompute_rows"] += int(selected.shape[0])

            for stream_index in range(batch_size):
                anchor: Tensor | None = None
                latest: Tensor | None = None
                age = 0
                for chunk_index in range(num_chunks):
                    flat_index = stream_index * num_chunks + chunk_index
                    action = ChronoAction(int(actions[stream_index, chunk_index].item()))
                    if action is ChronoAction.RECOMPUTE:
                        anchor = provisional[flat_index]
                        latest = anchor
                        age = 0
                        if self.cache_detach:
                            anchor = anchor.detach()
                            latest = latest.detach()
                        continue

                    if anchor is None or latest is None or age >= self.hard_cache_validity_age:
                        # Runtime fail closed protects direct calls even after the
                        # schema-level repair pass.
                        with profiler.stage("recompute"):
                            value = self._heavy_forward(block, current_flat[flat_index : flat_index + 1])[0]
                        with profiler.stage("cache_movement"):
                            provisional[flat_index] = value
                        actions[stream_index, chunk_index] = int(ChronoAction.RECOMPUTE)
                        effective_flat_mask[flat_index] = True
                        anchor = value
                        latest = value
                        age = 0
                        counters["recompute_rows"] += 1
                        counters["runtime_fail_closed_repairs"] += 1
                        if self.cache_detach:
                            anchor = anchor.detach()
                            latest = latest.detach()
                        continue

                    age += 1
                    if action is ChronoAction.HOLD:
                        provisional[flat_index] = latest
                        counters["hold_rows"] += 1
                    elif action is ChronoAction.TRANSPORT:
                        with profiler.stage("transport"):
                            transported = self.transport[group_index](
                                latest.unsqueeze(0),
                                current_flat[flat_index : flat_index + 1],
                                torch.tensor([age], device=state.device),
                            )[0]
                        if not torch.isfinite(transported).all():
                            with profiler.stage("recompute"):
                                transported = self._heavy_forward(
                                    block,
                                    current_flat[flat_index : flat_index + 1],
                                )[0]
                            actions[stream_index, chunk_index] = int(ChronoAction.RECOMPUTE)
                            effective_flat_mask[flat_index] = True
                            anchor = transported
                            latest = transported
                            age = 0
                            counters["runtime_fail_closed_repairs"] += 1
                            counters["recompute_rows"] += 1
                        else:
                            latest = transported
                            counters["transport_rows"] += 1
                        provisional[flat_index] = transported
                    else:
                        raise AssertionError(f"unhandled action: {action}")

                    if self.cache_detach:
                        if anchor is not None:
                            anchor = anchor.detach()
                        if latest is not None:
                            latest = latest.detach()

            if bool(getattr(block, "use_adapter", False)):

                def adapter_forward(value: Tensor) -> Tensor:
                    return block.adapter(value, h, w)

                with profiler.stage("dense_adatad_adapter"):
                    if bool(getattr(block, "with_cp", False)) and provisional.requires_grad:
                        adapted = cp.checkpoint(adapter_forward, provisional, use_reentrant=False)
                    else:
                        adapted = adapter_forward(provisional)
                # AdaTAD's temporal adapter is part of every original block,
                # not part of the action-controlled heavy attention/MLP path.
                # Its output therefore replaces all rows.
                output = adapted
                counters["adapter_dense_forward_count"] += 1
            else:
                output = provisional
            state = output.reshape(batch_size, num_chunks, token_count, channels)
        return state

    def forward(self, x: Tensor, blocks: Sequence[nn.Module], h: int, w: int) -> Tensor:
        self.latest_signals = None
        self.latest_output = None
        if x.ndim != 3:
            raise ValueError("ChronoTransport runtime expects [B*chunks, N, C]")
        if int(x.shape[0]) % self.chunks_per_window != 0:
            raise ValueError("flattened clip batch must be divisible by chunks_per_window")
        if len(blocks) != self.depth:
            raise ValueError("runtime depth does not match transformer block count")
        if int(x.shape[-1]) != self.embed_dims:
            raise ValueError("runtime embed_dims does not match input channels")
        geometry = self._runtime_geometry(x, h, w)

        if not self.enabled:
            self.latest_schedule = None
            self.latest_summary = {
                "enabled": False,
                "forced_dense_exact_path": True,
                "dense_output_shape_preserved": True,
                **geometry,
            }
            out = self._dense_forward(x, blocks, h, w)
            self.latest_output = out
            return out

        batch_size = int(x.shape[0]) // self.chunks_per_window
        state = x.reshape(batch_size, self.chunks_per_window, int(x.shape[1]), int(x.shape[2]))
        profiler = ChronoProfiler(sync_cuda=self.profile_sync_cuda)
        signals = None
        if self.capture_replay_signals:
            with profiler.stage("innovation"):
                signals = self._signals(state).detach()
            self.latest_signals = signals
        counters = {
            "recompute_rows": 0,
            "transport_rows": 0,
            "hold_rows": 0,
            "adapter_dense_forward_count": 0,
            "runtime_fail_closed_repairs": 0,
        }

        forced = self._forced_schedule(batch_size, x.device)
        if forced is None:
            if not self.cost_is_measured and not self.allow_unmeasured_cost_for_debug:
                return self._dense_fail_closed(
                    x,
                    blocks,
                    h,
                    w,
                    profiler=profiler,
                    reason="unmeasured_cost_table",
                    batch_size=batch_size,
                    geometry=geometry,
                )
            if not self.risk_ready:
                return self._dense_fail_closed(
                    x,
                    blocks,
                    h,
                    w,
                    profiler=profiler,
                    reason="risk_not_ready",
                    batch_size=batch_size,
                    geometry=geometry,
                )
            if self.require_checkpoint_for_dynamic and not self.checkpoint_loaded:
                return self._dense_fail_closed(
                    x,
                    blocks,
                    h,
                    w,
                    profiler=profiler,
                    reason="risk_checkpoint_not_loaded",
                    batch_size=batch_size,
                    geometry=geometry,
                )
            if not self.nonlinear_cost_ready and not self.allow_unmeasured_cost_for_debug:
                return self._dense_fail_closed(
                    x,
                    blocks,
                    h,
                    w,
                    profiler=profiler,
                    reason="schedule_shape_cost_lookup_missing",
                    batch_size=batch_size,
                    geometry=geometry,
                )

        schedule_repair_count = 0
        first_chunk_forced = False
        requested_actions: Tensor
        if forced is not None:
            schedule, schedule_repair_count, first_chunk_forced, requested_actions = forced
            selected_names = tuple([schedule.name] * batch_size)
            fail_closed = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            upper_risk = torch.zeros(batch_size, dtype=x.dtype, device=x.device)
            estimated_cost = self.cost_table.estimate(schedule.actions)
        else:
            if signals is None:
                with profiler.stage("innovation"):
                    signals = self._signals(state)
            with profiler.stage("scheduler"):
                selection = self.scheduler.select(signals)
            requested_actions = selection.schedule.actions.clone()
            actions, repairs, first_chunk_forced = self._repair_schedule(selection.schedule.actions)
            schedule_repair_count += repairs
            schedule = ChronoSchedule(
                actions=actions,
                layer_groups=self.layer_groups,
                name=selection.schedule.name,
                metadata=selection.schedule.metadata,
            )
            selected_names = selection.selected_names
            fail_closed = selection.fail_closed
            upper_risk = selection.upper_risk
            estimated_cost = selection.estimated_cost

        self.latest_schedule = schedule
        if schedule.is_dense():
            with profiler.stage("recompute"):
                out = self._dense_forward(x, blocks, h, w)
            self.latest_output = out
            dense_rows = int(x.shape[0]) * self.depth
            self.latest_summary = {
                "schema_version": "chronotransport_runtime_v1",
                "enabled": True,
                "forced_dense_exact_path": True,
                "selected_schedule_names": list(selected_names),
                "recompute_rows": dense_rows,
                "transport_rows": 0,
                "hold_rows": 0,
                "adapter_dense_forward_count": sum(
                    1 for block in blocks if bool(getattr(block, "use_adapter", False))
                ),
                "schedule_repair_count": schedule_repair_count,
                "adapter_writeback": "all_rows",
                "first_chunk_forced_recompute": first_chunk_forced,
                "runtime_fail_closed_repairs": 0,
                "dense_output_shape_preserved": True,
                "cost_is_measured": self.cost_is_measured,
                "risk_ready": self.risk_ready,
                "checkpoint_loaded": self.checkpoint_loaded,
                "require_checkpoint_for_dynamic": self.require_checkpoint_for_dynamic,
                "cache_reset_per_window": True,
                "external_dense_grid_preserved_by_post_interpolation": True,
                **geometry,
                "upper_risk": upper_risk.detach().cpu().tolist(),
                "estimated_cost": estimated_cost.detach().cpu().tolist(),
                "fail_closed": fail_closed.detach().cpu().tolist(),
                "profile": profiler.summary(fill_missing=True),
            }
            return out

        actions = schedule.actions.clone()
        for group_index, group in enumerate(self.layer_groups):
            state = self._run_group(
                state,
                blocks[group.start : group.end],
                actions[:, :, group_index],
                group_index=group_index,
                h=h,
                w=w,
                profiler=profiler,
                counters=counters,
            )

        out = state.reshape_as(x)
        whole_window_dense_fallback = False
        if not torch.isfinite(out).all():
            with profiler.stage("recompute"):
                out = self._dense_forward(x, blocks, h, w)
            counters["runtime_fail_closed_repairs"] += 1
            fail_closed = torch.ones_like(fail_closed)
            whole_window_dense_fallback = True
            actions = dense_action_tensor(
                batch_size,
                self.chunks_per_window,
                len(self.layer_groups),
                device=x.device,
            )

        executed_schedule = ChronoSchedule(
            actions=actions,
            layer_groups=self.layer_groups,
            name="dense_fail_closed" if whole_window_dense_fallback else schedule.name,
            metadata=dict(schedule.metadata),
        )
        self.latest_schedule = executed_schedule
        action_counts = executed_schedule.action_counts()
        requested_action_counts = {
            action.name.lower(): int((requested_actions == int(action)).sum().item())
            for action in ChronoAction
        }
        requested_action_sha256 = canonical_sha256(requested_actions.detach().cpu().to(torch.long).tolist())
        executed_action_sha256 = canonical_sha256(actions.detach().cpu().to(torch.long).tolist())
        executed_estimated_cost = self.cost_table.estimate(actions)
        evidence_valid = (
            int(schedule_repair_count) == 0
            and int(counters["runtime_fail_closed_repairs"]) == 0
            and not whole_window_dense_fallback
        )
        invalid_reason = None
        if int(schedule_repair_count) > 0:
            invalid_reason = "schedule_repair"
        elif int(counters["runtime_fail_closed_repairs"]) > 0:
            invalid_reason = "runtime_repair"
        elif whole_window_dense_fallback:
            invalid_reason = "whole_window_dense_fallback"
        profiler.record_actions(**action_counts)
        self.latest_summary = {
            "schema_version": "chronotransport_runtime_v1",
            "enabled": True,
            "forced_dense_exact_path": False,
            "whole_window_dense_fallback": whole_window_dense_fallback,
            "selected_schedule_names": list(selected_names),
            "executed_schedule_name": executed_schedule.name,
            **counters,
            "schedule_repair_count": int(schedule_repair_count + counters["runtime_fail_closed_repairs"]),
            "first_chunk_forced_recompute": bool(first_chunk_forced),
            "dense_output_shape_preserved": tuple(out.shape) == tuple(x.shape),
            "adapter_path_dense": True,
            "adapter_writeback": "all_rows",
            "heavy_attention_mlp_gathered": True,
            "cache_reset_per_window": True,
            "transport_uses_latest_cache": True,
            "cost_is_measured": self.cost_is_measured,
            "risk_ready": self.risk_ready,
            "checkpoint_loaded": self.checkpoint_loaded,
            "require_checkpoint_for_dynamic": self.require_checkpoint_for_dynamic,
            "external_dense_grid_preserved_by_post_interpolation": True,
            **geometry,
            "upper_risk": upper_risk.detach().cpu().tolist(),
            "estimated_cost": estimated_cost.detach().cpu().tolist(),
            "requested_estimated_cost": estimated_cost.detach().cpu().tolist(),
            "executed_estimated_cost": executed_estimated_cost.detach().cpu().tolist(),
            "fail_closed": fail_closed.detach().cpu().tolist(),
            "action_counts": action_counts,
            "requested_action_counts": requested_action_counts,
            "requested_action_sha256": requested_action_sha256,
            "executed_action_sha256": executed_action_sha256,
            "evidence_valid": evidence_valid,
            "invalid_implementation_reason": invalid_reason,
            "profile": profiler.summary(fill_missing=True),
        }
        self.latest_output = out
        return out
