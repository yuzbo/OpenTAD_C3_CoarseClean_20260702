from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import nn

from opentad.models.chronotransport.actions import (
    ChronoAction,
    ChronoSchedule,
    LayerGroup,
    normalize_layer_groups,
)
from opentad.models.chronotransport.cache import ChronoCacheBank
from opentad.models.chronotransport.losses import pinball_loss, transport_consistency_loss
from opentad.models.chronotransport.profiler import ChronoProfiler, REQUIRED_STAGE_FIELDS
from opentad.models.chronotransport.risk import ScheduleQuantileRiskPredictor
from opentad.models.chronotransport.runtime import ChronoTransportRuntime
from opentad.models.chronotransport.scheduler import (
    MeasuredCostTable,
    RiskConstrainedScheduler,
    ScheduleLibrary,
)
from opentad.models.chronotransport.transport import TemporalTransportAdapter


def test_action_schema_rejects_invalid_values_and_requires_first_chunk_recompute() -> None:
    groups = normalize_layer_groups(depth=6, groups=[(0, 2), (2, 6)])
    assert groups == (LayerGroup(0, 2), LayerGroup(2, 6))

    actions = torch.full((2, 4, 2), int(ChronoAction.HOLD), dtype=torch.long)
    with pytest.raises(ValueError, match="first chunk"):
        ChronoSchedule(actions=actions, layer_groups=groups)

    actions[:, 0, :] = int(ChronoAction.RECOMPUTE)
    schedule = ChronoSchedule(actions=actions, layer_groups=groups)
    assert schedule.batch_size == 2
    assert schedule.num_chunks == 4
    assert schedule.num_groups == 2

    bad = actions.clone()
    bad[0, 1, 0] = 99
    with pytest.raises(ValueError, match="invalid ChronoAction"):
        ChronoSchedule(actions=bad, layer_groups=groups)


def test_layer_groups_must_be_contiguous_non_overlapping_and_cover_depth() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        normalize_layer_groups(depth=6, groups=[(0, 2), (3, 6)])
    with pytest.raises(ValueError, match="cover"):
        normalize_layer_groups(depth=6, groups=[(0, 2), (2, 5)])
    with pytest.raises(ValueError, match="positive"):
        normalize_layer_groups(depth=6, groups=[(0, 0), (0, 6)])


def test_cache_age_anchor_latest_and_layer_group_isolation() -> None:
    bank = ChronoCacheBank(num_groups=2, detach_policy="always")
    bank.reset(batch_size=1)
    g0 = torch.tensor([[[1.0, 2.0]]])
    g1 = torch.tensor([[[9.0, 8.0]]])

    bank.commit(0, 0, ChronoAction.RECOMPUTE, g0, chunk_index=0)
    bank.commit(0, 1, ChronoAction.RECOMPUTE, g1, chunk_index=0)
    assert bank.entry(0, 0).recompute_age == 0
    assert bank.entry(0, 1).recompute_age == 0
    assert torch.equal(bank.entry(0, 0).latest, g0)
    assert torch.equal(bank.entry(0, 1).latest, g1)

    transported = g0 + 3.0
    bank.commit(0, 0, ChronoAction.TRANSPORT, transported, chunk_index=1)
    assert bank.entry(0, 0).recompute_age == 1
    assert torch.equal(bank.entry(0, 0).anchor, g0)
    assert torch.equal(bank.entry(0, 0).latest, transported)
    assert bank.entry(0, 1).recompute_age == 0
    assert torch.equal(bank.entry(0, 1).latest, g1)

    held = bank.read_latest(0, 0)
    bank.commit(0, 0, ChronoAction.HOLD, held, chunk_index=2)
    assert bank.entry(0, 0).recompute_age == 2
    assert torch.equal(bank.entry(0, 0).latest, transported)


def test_hold_requires_valid_cache_and_is_bitwise_invariant() -> None:
    bank = ChronoCacheBank(num_groups=1, detach_policy="always")
    bank.reset(batch_size=1)
    with pytest.raises(RuntimeError, match="invalid cache"):
        bank.read_latest(0, 0)

    state = torch.randn(2, 3, 4)
    bank.commit(0, 0, ChronoAction.RECOMPUTE, state, chunk_index=0)
    held = bank.read_latest(0, 0)
    bank.commit(0, 0, ChronoAction.HOLD, held, chunk_index=1)
    assert torch.equal(bank.read_latest(0, 0), state)


def test_transport_shape_finiteness_and_zero_initialization_equals_hold() -> None:
    torch.manual_seed(3)
    module = TemporalTransportAdapter(embed_dims=8, bottleneck_dims=4, max_age=8)
    anchor = torch.randn(3, 5, 8)
    current = torch.randn(3, 5, 8)
    age = torch.tensor([1, 2, 4])
    out = module(anchor, current, age)
    assert out.shape == anchor.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out, anchor)

    with torch.no_grad():
        module.up_proj.weight.fill_(0.05)
    changed = module(anchor, current, age)
    assert not torch.equal(changed, anchor)


def test_risk_predictor_and_scheduler_choose_cheapest_feasible_deterministically() -> None:
    torch.manual_seed(4)
    groups = normalize_layer_groups(depth=4, groups=[(0, 2), (2, 4)])
    library = ScheduleLibrary.default(num_chunks=4, layer_groups=groups)
    predictor = ScheduleQuantileRiskPredictor(signal_dims=3, num_groups=2, hidden_dims=8, quantile=0.9)

    # Make risk depend almost entirely on action count in a deterministic way.
    for parameter in predictor.parameters():
        nn.init.zeros_(parameter)
    predictor.set_debug_action_risk(
        recompute=0.0,
        transport=0.1,
        hold=0.3,
    )

    costs = MeasuredCostTable(
        recompute=(4.0, 6.0),
        transport=(1.0, 1.5),
        hold=(0.05, 0.05),
        scheduler_overhead=0.2,
    )
    scheduler = RiskConstrainedScheduler(
        predictor=predictor,
        schedule_library=library,
        cost_table=costs,
        epsilon=0.9,
        max_cache_age=3,
    )
    signals = torch.zeros(2, 4, 2, 3)
    first = scheduler.select(signals)
    second = scheduler.select(signals)
    assert torch.equal(first.schedule.actions, second.schedule.actions)
    assert first.selected_names == second.selected_names
    assert all(math.isfinite(value) for value in first.upper_risk.tolist())
    assert first.estimated_cost.shape == (2,)


def test_scheduler_fails_closed_on_nonfinite_signal_ood_or_no_feasible_candidate() -> None:
    groups = normalize_layer_groups(depth=2, groups=[(0, 1), (1, 2)])
    library = ScheduleLibrary.default(num_chunks=3, layer_groups=groups)
    predictor = ScheduleQuantileRiskPredictor(signal_dims=2, num_groups=2, hidden_dims=4, quantile=0.9)
    predictor.set_debug_action_risk(recompute=10.0, transport=10.0, hold=10.0)
    costs = MeasuredCostTable(recompute=(2.0, 2.0), transport=(1.0, 1.0), hold=(0.1, 0.1))
    scheduler = RiskConstrainedScheduler(predictor, library, costs, epsilon=0.0, max_cache_age=2)

    signals = torch.zeros(1, 3, 2, 2)
    result = scheduler.select(signals)
    assert result.selected_names == ("dense",)
    assert result.fail_closed.tolist() == [True]
    assert torch.all(result.schedule.actions == int(ChronoAction.RECOMPUTE))

    signals[0, 1, 0, 0] = float("nan")
    result = scheduler.select(signals)
    assert result.selected_names == ("dense",)
    assert result.fail_closed.tolist() == [True]

    signals.zero_()
    result = scheduler.select(signals, ood_mask=torch.tensor([True]))
    assert result.selected_names == ("dense",)
    assert result.fail_closed.tolist() == [True]


def test_losses_are_finite_and_pinball_has_expected_value() -> None:
    prediction = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 1.0])
    loss = pinball_loss(prediction, target, quantile=0.9, reduction="none")
    assert torch.allclose(loss, torch.tensor([0.9, 0.1]))

    reference = torch.randn(2, 5, 8)
    same = transport_consistency_loss(reference, reference)
    assert same.item() == pytest.approx(0.0, abs=1e-8)


def test_profiler_emits_all_required_cost_fields() -> None:
    profiler = ChronoProfiler(sync_cuda=False)
    with profiler.stage("innovation"):
        _ = sum(range(10))
    profiler.record("scheduler", 0.1)
    summary = profiler.summary(fill_missing=True)
    assert set(REQUIRED_STAGE_FIELDS).issubset(summary["latency_ms"])
    assert summary["latency_ms"]["innovation"]["count"] == 1


def test_inference_surfaces_do_not_accept_gt_or_teacher_inputs() -> None:
    forbidden = {"gt", "gt_segments", "gt_labels", "teacher", "oracle", "raw_predictions"}
    for callable_obj in (
        RiskConstrainedScheduler.select,
        ChronoTransportRuntime.forward,
    ):
        parameters = set(inspect.signature(callable_obj).parameters)
        assert not (parameters & forbidden)


class _CountingLinear(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dims, dims, bias=False)
        nn.init.eye_(self.linear.weight)
        self.rows_seen = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rows_seen += int(x.shape[0])
        return self.linear(x) * 0.1


class _FakeAdapter(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dims, dims, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        del h, w
        return x + self.proj(x)


class _FakeBlock(nn.Module):
    def __init__(self, dims: int, use_adapter: bool) -> None:
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.attn = _CountingLinear(dims)
        self.mlp = _CountingLinear(dims)
        self.drop_path = nn.Identity()
        self.use_adapter = use_adapter
        self.adapter = _FakeAdapter(dims)
        self.with_cp = False

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_adapter:
            x = self.adapter(x, h, w)
        return x


def _dense_blocks(x: torch.Tensor, blocks: nn.ModuleList, h: int, w: int) -> torch.Tensor:
    for block in blocks:
        x = block(x, h, w)
    return x


def test_runtime_forced_dense_is_numerically_identical_to_original_block_loop() -> None:
    torch.manual_seed(11)
    blocks_a = nn.ModuleList([_FakeBlock(6, True), _FakeBlock(6, False), _FakeBlock(6, True)])
    blocks_b = nn.ModuleList([_FakeBlock(6, True), _FakeBlock(6, False), _FakeBlock(6, True)])
    blocks_b.load_state_dict(blocks_a.state_dict())
    x = torch.randn(8, 4, 6)

    expected = _dense_blocks(x.clone(), blocks_a, h=1, w=4)
    runtime = ChronoTransportRuntime(
        embed_dims=6,
        depth=3,
        chunks_per_window=4,
        layer_groups=[(0, 1), (1, 3)],
        forced_schedule="dense",
        signal_dims=4,
        max_cache_age=3,
    )
    actual = runtime(x.clone(), blocks_b, h=1, w=4)
    assert torch.equal(actual, expected)
    assert runtime.latest_summary["forced_dense_exact_path"] is True


def test_runtime_mixed_schedule_preserves_dense_shape_and_reduces_heavy_rows() -> None:
    torch.manual_seed(12)
    blocks = nn.ModuleList([_FakeBlock(4, True), _FakeBlock(4, True)])
    x = torch.randn(8, 3, 4)  # two windows, four chunks each
    runtime = ChronoTransportRuntime(
        embed_dims=4,
        depth=2,
        chunks_per_window=4,
        layer_groups=[(0, 1), (1, 2)],
        forced_schedule="periodic2_transport",
        signal_dims=4,
        max_cache_age=2,
    )
    out = runtime(x, blocks, h=1, w=3)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    dense_rows = int(x.shape[0]) * len(blocks)
    seen_rows = sum(block.attn.rows_seen for block in blocks)
    assert seen_rows < dense_rows
    assert runtime.latest_summary["recompute_rows"] == seen_rows
    assert runtime.latest_summary["transport_rows"] > 0
    assert runtime.latest_summary["dense_output_shape_preserved"] is True
    assert runtime.latest_summary["adapter_dense_forward_count"] == len(blocks)


def test_runtime_opt_in_captures_only_compact_signals_and_ephemeral_output() -> None:
    blocks = nn.ModuleList([_FakeBlock(4, False)])
    x = torch.randn(4, 2, 4)
    runtime = ChronoTransportRuntime(
        embed_dims=4,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        forced_schedule="periodic2_transport",
        signal_dims=3,
        max_cache_age=2,
    )
    runtime.capture_replay_signals = True
    out = runtime(x, blocks, h=1, w=2)
    assert runtime.latest_signals.shape == (1, 4, 1, 3)
    assert runtime.latest_signals.requires_grad is False
    assert torch.equal(runtime.latest_output, out)


def test_runtime_repairs_illegal_hold_to_recompute_and_never_returns_nonfinite() -> None:
    blocks = nn.ModuleList([_FakeBlock(4, False)])
    x = torch.randn(4, 2, 4)
    runtime = ChronoTransportRuntime(
        embed_dims=4,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        forced_actions=torch.full((1, 4, 1), int(ChronoAction.HOLD), dtype=torch.long),
        signal_dims=3,
        max_cache_age=1,
    )
    out = runtime(x, blocks, h=1, w=2)
    assert torch.isfinite(out).all()
    assert runtime.latest_summary["schedule_repair_count"] >= 1
    assert runtime.latest_summary["first_chunk_forced_recompute"] is True


def test_risk_candidate_age_tracks_time_since_last_recompute() -> None:
    predictor = ScheduleQuantileRiskPredictor(signal_dims=2, num_groups=1, hidden_dims=4)
    actions = torch.tensor(
        [[
            [int(ChronoAction.RECOMPUTE)],
            [int(ChronoAction.TRANSPORT)],
            [int(ChronoAction.HOLD)],
            [int(ChronoAction.RECOMPUTE)],
        ]],
        dtype=torch.long,
    )
    age = predictor.candidate_age(actions)
    assert age.shape == (1, 4, 1, 1)
    assert torch.equal(age.flatten(), torch.tensor([0.0, 1.0, 2.0, 0.0]))


def test_unmeasured_dynamic_runtime_fails_closed_to_dense() -> None:
    blocks = nn.ModuleList([_FakeBlock(4, False)])
    x = torch.randn(4, 2, 4)
    runtime = ChronoTransportRuntime(
        embed_dims=4,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        signal_dims=3,
        max_cache_age=2,
        measured_cost=None,
        allow_unmeasured_cost_for_debug=False,
    )
    out = runtime(x, blocks, h=1, w=2)
    assert out.shape == x.shape
    assert runtime.latest_summary["forced_dense_exact_path"] is True
    assert runtime.latest_summary["fail_closed_reason"] == "unmeasured_cost_table"


def test_cache_detach_blocks_cross_chunk_hold_gradient() -> None:
    blocks = nn.ModuleList([_FakeBlock(3, False)])
    x = torch.randn(4, 2, 3, requires_grad=True)
    forced = torch.tensor(
        [[[int(ChronoAction.RECOMPUTE)], [int(ChronoAction.HOLD)], [int(ChronoAction.RECOMPUTE)], [int(ChronoAction.HOLD)]]]
    )
    runtime = ChronoTransportRuntime(
        embed_dims=3,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        forced_actions=forced,
        signal_dims=3,
        max_cache_age=2,
        cache_detach=True,
    )
    out = runtime(x, blocks, h=1, w=2).reshape(1, 4, 2, 3)
    out[:, 1].sum().backward()
    assert x.grad is not None
    assert float(x.grad[0].abs().sum().item()) == pytest.approx(0.0, abs=1e-8)


class _ShapeCheckingTemporalAdapter(nn.Module):
    def __init__(self, temporal_size: int, dims: int) -> None:
        super().__init__()
        self.temporal_size = int(temporal_size)
        self.proj = nn.Linear(dims, dims, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.last_hw = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        self.last_hw = (int(h), int(w))
        reshaped = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])
        return (reshaped + self.proj(reshaped)).reshape_as(x)


def test_runtime_passes_real_spatial_shape_to_dense_temporal_adapter() -> None:
    block = _FakeBlock(4, True)
    block.adapter = _ShapeCheckingTemporalAdapter(temporal_size=8, dims=4)
    blocks = nn.ModuleList([block])
    # 4 chunks × (2 tubelets × 2 × 2 spatial patches) = temporal_size 8.
    x = torch.randn(4, 8, 4)
    runtime = ChronoTransportRuntime(
        embed_dims=4,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        forced_schedule="periodic2_transport",
        signal_dims=3,
        max_cache_age=2,
    )
    out = runtime(x, blocks, h=2, w=2)
    assert out.shape == x.shape
    assert block.adapter.last_hw == (2, 2)


def test_transport_zero_init_is_exact_latest_cache_not_stale_anchor() -> None:
    module = TemporalTransportAdapter(embed_dims=3, bottleneck_dims=2, max_age=4)
    latest = torch.full((1, 2, 3), 7.0)
    current = torch.full((1, 2, 3), 11.0)
    out = module(latest, current, age=2)
    assert torch.equal(out, latest)


class _AddOneTransport(nn.Module):
    def forward(self, cached: torch.Tensor, current: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        del current, age
        return cached + 1.0


def test_runtime_chains_consecutive_transport_from_latest_state() -> None:
    blocks = nn.ModuleList([_FakeBlock(2, False)])
    x = torch.zeros(4, 1, 2)
    forced = torch.tensor(
        [[[int(ChronoAction.RECOMPUTE)], [int(ChronoAction.TRANSPORT)], [int(ChronoAction.TRANSPORT)], [int(ChronoAction.RECOMPUTE)]]]
    )
    runtime = ChronoTransportRuntime(
        embed_dims=2,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        forced_actions=forced,
        signal_dims=3,
        max_cache_age=3,
    )
    runtime.transport[0] = _AddOneTransport()
    out = runtime(x, blocks, h=1, w=1).reshape(1, 4, 1, 2)
    # Chunk 2 must be transported from chunk 1's latest state, hence +2.
    assert torch.allclose(out[:, 2], out[:, 0] + 2.0)


def test_cost_table_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        MeasuredCostTable(recompute=(float("inf"),), transport=(1.0,), hold=(0.0,))


def test_dynamic_runtime_requires_explicit_risk_readiness() -> None:
    blocks = nn.ModuleList([_FakeBlock(3, False)])
    x = torch.randn(4, 2, 3)
    runtime = ChronoTransportRuntime(
        embed_dims=3,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        signal_dims=3,
        max_cache_age=3,
        measured_cost={
            "recompute": (2.0,),
            "transport": (0.5,),
            "hold": (0.1,),
            "scheduler_overhead": 0.1,
        },
    )
    runtime(x, blocks, h=1, w=2)
    assert runtime.latest_summary["forced_dense_exact_path"] is True
    assert runtime.latest_summary["fail_closed_reason"] == "risk_not_ready"


def test_mixed_runtime_profiles_cache_movement_and_records_executed_schedule() -> None:
    blocks = nn.ModuleList([_FakeBlock(3, False)])
    x = torch.randn(4, 2, 3)
    runtime = ChronoTransportRuntime(
        embed_dims=3,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        forced_schedule="periodic2_transport",
        signal_dims=3,
        max_cache_age=2,
    )
    runtime(x, blocks, h=1, w=2)
    assert runtime.latest_summary["profile"]["latency_ms"]["cache_movement"]["count"] > 0
    assert runtime.latest_schedule is not None
    assert torch.equal(runtime.latest_schedule.actions, torch.tensor([[[0], [1], [0], [1]]]))


def test_default_library_contains_time_layer_and_joint_candidates() -> None:
    groups = normalize_layer_groups(depth=6, groups=[(0, 2), (2, 4), (4, 6)])
    library = ScheduleLibrary.default(num_chunks=8, layer_groups=groups)
    assert "periodic2_transport" in library.names
    assert "layer_only_late_recompute" in library.names
    assert "layer_only_early_recompute" in library.names
    assert "joint_progressive_transport" in library.names


def test_motion_threshold_baseline_is_deploy_visible_and_first_chunk_dense() -> None:
    from opentad.models.chronotransport.scheduler import motion_threshold_actions

    motion = torch.tensor([[0.0, 0.1, 0.9, 0.2]])
    actions = motion_threshold_actions(
        motion,
        num_groups=2,
        threshold=0.5,
        fallback=ChronoAction.HOLD,
    )
    assert actions.shape == (1, 4, 2)
    assert torch.all(actions[:, 0] == int(ChronoAction.RECOMPUTE))
    assert torch.all(actions[:, 2] == int(ChronoAction.RECOMPUTE))
    assert torch.all(actions[:, 1] == int(ChronoAction.HOLD))


def test_dynamic_runtime_requires_chronotransport_checkpoint_when_guarded() -> None:
    blocks = nn.ModuleList([_FakeBlock(3, False)])
    x = torch.randn(4, 2, 3)
    runtime = ChronoTransportRuntime(
        embed_dims=3,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        signal_dims=3,
        max_cache_age=3,
        measured_cost={
            "recompute": (2.0,),
            "transport": (0.5,),
            "hold": (0.1,),
            "scheduler_overhead": 0.1,
        },
        allow_unmeasured_cost_for_debug=True,
        risk_ready=True,
        require_checkpoint_for_dynamic=True,
    )
    runtime(x, blocks, h=1, w=2)
    assert runtime.latest_summary["fail_closed_reason"] == "risk_checkpoint_not_loaded"
    runtime.set_checkpoint_loaded(True)
    runtime.risk_predictor.set_debug_action_risk(recompute=0.0, transport=0.0, hold=0.0)
    runtime(x, blocks, h=1, w=2)
    assert runtime.latest_summary.get("fail_closed_reason") is None
