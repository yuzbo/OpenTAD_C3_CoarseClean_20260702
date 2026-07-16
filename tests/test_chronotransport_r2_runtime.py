import torch
from torch import nn

from opentad.models.chronotransport import ChronoAction, ChronoTransportRuntime


class _Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _AddAdapter(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x, h, w):
        del h, w
        return x + self.bias


class _Block(nn.Module):
    def __init__(self, adapter_value: float | None):
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.attn = _Zero()
        self.mlp = _Zero()
        self.drop_path = nn.Identity()
        self.with_cp = False
        self.use_adapter = adapter_value is not None
        self.adapter = _AddAdapter(adapter_value or 0.0)

    def forward(self, x, h, w):
        if self.use_adapter:
            return self.adapter(x, h, w)
        return x


def _runtime(forced, **kwargs):
    return ChronoTransportRuntime(
        embed_dims=2,
        depth=1,
        chunks_per_window=4,
        layer_groups=[(0, 1)],
        forced_actions=forced,
        signal_dims=3,
        **kwargs,
    )


def test_temporal_adapter_writes_back_all_rows_in_original_block_order():
    forced = torch.tensor([[[0], [2], [0], [2]]], dtype=torch.long)
    x = torch.zeros(4, 1, 2)
    runtime = _runtime(forced, hard_cache_validity_age=47, transport_age_embedding_cap=8)
    out = runtime(x, nn.ModuleList([_Block(3.0)]), h=1, w=1).reshape(4, 2)
    assert torch.equal(out, torch.full_like(out, 3.0))
    assert runtime.latest_summary["adapter_writeback"] == "all_rows"


def test_current_recompute_row_keeps_gradient_while_hold_history_is_detached():
    forced = torch.tensor([[[0], [2], [0], [2]]], dtype=torch.long)
    x = torch.randn(4, 1, 2, requires_grad=True)
    block = _Block(1.0)
    runtime = _runtime(
        forced,
        hard_cache_validity_age=47,
        transport_age_embedding_cap=8,
        cache_detach=True,
    )
    out = runtime(x, nn.ModuleList([block]), h=1, w=1).reshape(4, 2)
    out[2].sum().backward(retain_graph=True)
    assert x.grad is not None and x.grad[2].abs().sum() > 0
    x.grad.zero_()
    out[1].sum().backward()
    assert x.grad[0].abs().sum() == 0
    assert block.adapter.bias.grad is not None


def test_requested_and_executed_actions_are_distinct_after_repair():
    forced = torch.tensor([[[0], [2], [2], [2]]], dtype=torch.long)
    runtime = _runtime(forced, hard_cache_validity_age=1, transport_age_embedding_cap=8)
    runtime(torch.zeros(4, 1, 2), nn.ModuleList([_Block(None)]), h=1, w=1)
    summary = runtime.latest_summary
    assert summary["requested_action_counts"] == {"recompute": 1, "transport": 0, "hold": 3}
    assert summary["action_counts"] == {"recompute": 2, "transport": 0, "hold": 2}
    assert summary["requested_action_sha256"] != summary["executed_action_sha256"]
    assert summary["evidence_valid"] is False
    assert summary["invalid_implementation_reason"] == "schedule_repair"


def test_r2_default_age_allows_hold_only_through_clip_47():
    forced = torch.full((1, 48, 1), int(ChronoAction.HOLD), dtype=torch.long)
    forced[:, 0] = int(ChronoAction.RECOMPUTE)
    runtime = ChronoTransportRuntime(
        embed_dims=2,
        depth=1,
        chunks_per_window=48,
        layer_groups=[(0, 1)],
        forced_actions=forced,
        signal_dims=3,
    )
    runtime(torch.zeros(48, 1, 2), nn.ModuleList([_Block(None)]), h=1, w=1)
    assert runtime.latest_summary["schedule_repair_count"] == 0
    assert runtime.latest_summary["action_counts"]["hold"] == 47
    assert runtime.hard_cache_validity_age == 47
    assert runtime.transport_age_embedding_cap == 8


def test_registered_forced_actions_preserve_exact_candidate_name_and_bytes():
    runtime = ChronoTransportRuntime(
        embed_dims=2,
        depth=1,
        chunks_per_window=48,
        layer_groups=[(0, 1)],
        signal_dims=3,
    )
    actions = torch.full((48, 1), int(ChronoAction.HOLD), dtype=torch.long)
    actions[0] = int(ChronoAction.RECOMPUTE)
    runtime.set_registered_forced_actions(
        actions,
        candidate_name="motion_topk_p8",
    )
    runtime(torch.zeros(48, 1, 2), nn.ModuleList([_Block(None)]), h=1, w=1)
    assert runtime.latest_schedule.name == "motion_topk_p8"
    assert torch.equal(runtime.latest_schedule.actions[0].cpu(), actions)
    assert runtime.latest_summary["schedule_repair_count"] == 0


def test_registered_direct_candidate_cost_drives_forced_runtime_summary():
    runtime = _runtime(None)
    candidate = runtime.schedule_library.candidates[1]
    runtime.set_registered_forced_actions(
        candidate.actions,
        candidate_name=candidate.name,
    )
    costs = {
        name: float(index + 1)
        for index, name in enumerate(runtime.schedule_library.names)
    }
    costs[candidate.name] = 123.25
    profile_sha256 = "b" * 64
    runtime.install_registered_candidate_costs(
        costs,
        profile_sha256=profile_sha256,
    )

    runtime(torch.zeros(4, 1, 2), nn.ModuleList([_Block(None)]), h=1, w=1)
    summary = runtime.latest_summary
    assert summary["estimated_cost"] == [123.25]
    assert summary["requested_estimated_cost"] == [123.25]
    assert summary["executed_estimated_cost"] == [123.25]
    assert summary["cost_is_measured"] is True
    assert summary["registered_cost_profile_sha256"] == profile_sha256


def test_registered_post_stage_c_calibration_matches_runtime_risk_and_budget_gate():
    runtime = _runtime(
        None,
        risk_ready=True,
        require_checkpoint_for_dynamic=True,
    )
    runtime.set_checkpoint_loaded(True)
    non_dense = [name for name in runtime.schedule_library.names if name != "dense"]
    selected = non_dense[0]
    costs = {name: 2.0 for name in runtime.schedule_library.names}
    costs[selected] = 0.5
    costs["dense"] = 3.0
    runtime.install_registered_candidate_costs(costs, profile_sha256="c" * 64)
    runtime.install_registered_gate3_calibration(
        q_conf=0.25,
        budget=0.5,
        calibration_sha256="d" * 64,
    )

    def zero_risk(signals, candidates):
        return torch.zeros(
            (int(signals.shape[0]), int(candidates.shape[0])),
            device=signals.device,
            dtype=signals.dtype,
        )

    runtime.scheduler.predictor.forward = zero_risk
    runtime(torch.zeros(4, 1, 2), nn.ModuleList([_Block(None)]), h=1, w=1)
    summary = runtime.latest_summary
    assert summary["selected_schedule_names"] == [selected]
    assert summary["upper_risk"] == [0.25]
    assert summary["estimated_cost"] == [0.5]
    assert summary["registered_q_conf"] == 0.25
    assert summary["registered_budget"] == 0.5
    assert summary["registered_gate3_calibration_sha256"] == "d" * 64

    try:
        runtime.install_registered_gate3_calibration(
            q_conf=0.5,
            budget=0.5,
            calibration_sha256="d" * 64,
        )
    except RuntimeError as error:
        assert "immutable" in str(error)
    else:
        raise AssertionError("registered Gate3 calibration mutation was accepted")
