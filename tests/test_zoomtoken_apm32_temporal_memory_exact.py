import importlib.util
from pathlib import Path

import pytest
import torch
from mmengine import Config
from torch import nn

from opentad.models.backbones.georoute_routing import (
    APM32_TEMPORAL_ALIGNMENT_SCHEMA,
    build_apm32_temporal_plan,
)
from opentad.models.backbones.vit_adapter import Block, VisionTransformerAdapter


ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "zoomtoken_apm32_train_entry",
        ROOT / "tools" / "train.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _strict_indices(top: int = 0, left: int = 0) -> torch.Tensor:
    return torch.tensor(
        [
            (top + row) * 10 + left + col
            for row in range(8)
            for col in range(8)
        ],
        dtype=torch.long,
    )


def _identity_embeddings(tubelets: int, *, channels: int = 64) -> torch.Tensor:
    base = torch.eye(64, channels, dtype=torch.float32)
    return base.unsqueeze(0).repeat(tubelets, 1, 1).unsqueeze(0)


def _plan(
    embeddings: torch.Tensor,
    indices: torch.Tensor | None = None,
):
    if indices is None:
        indices = _strict_indices().view(1, 1, 64).expand(
            embeddings.shape[0],
            embeddings.shape[1],
            64,
        )
    return build_apm32_temporal_plan(
        embeddings,
        indices,
        grid_height=10,
        grid_width=10,
    )


def test_identity_motion_is_mutual_and_stable_k32_after_clip_reset():
    plan = _plan(_identity_embeddings(2))
    assert plan["schema_version"] == APM32_TEMPORAL_ALIGNMENT_SCHEMA
    assert plan["refresh_mask"].sum(-1).tolist() == [[64, 32]]
    assert plan["fallback_mask"].tolist() == [[True, False]]
    assert plan["matched_previous_slot"][0, 1].tolist() == list(range(64))
    assert plan["retained_mask"][0, 1, :32].all()
    assert not plan["retained_mask"][0, 1, 32:].any()


@pytest.mark.parametrize("shift, expected_matches", [(1, 56), (2, 48)])
def test_one_and_two_cell_support_translation_match_only_real_overlap(
    shift: int,
    expected_matches: int,
):
    embeddings = _identity_embeddings(2, channels=100)
    previous_indices = _strict_indices(0, 0)
    current_indices = _strict_indices(0, shift)
    table = torch.eye(100)
    embeddings[0, 0] = table.index_select(0, previous_indices)
    embeddings[0, 1] = table.index_select(0, current_indices)
    indices = torch.stack((previous_indices, current_indices)).view(1, 2, 64)
    plan = _plan(embeddings, indices)
    assert int(plan["matched_mask"][0, 1].sum().item()) == expected_matches
    assert plan["refresh_mask"].sum(-1).tolist() == [[64, 32]]


def test_three_cell_content_motion_is_outside_radius_and_falls_back():
    table = torch.eye(100)
    spatial = _strict_indices(0, 0)
    previous = table.index_select(0, spatial)
    shifted_content = table.index_select(0, (spatial + 3).clamp_max(99))
    embeddings = torch.stack((previous, shifted_content)).view(1, 2, 64, 100)
    plan = _plan(embeddings)
    assert int(plan["matched_mask"][0, 1].sum().item()) < 32
    assert plan["fallback_mask"].tolist() == [[True, True]]
    assert plan["refresh_mask"].sum(-1).tolist() == [[64, 64]]


def test_mutual_nearest_prevents_duplicate_previous_reuse():
    embeddings = _identity_embeddings(2)
    embeddings[0, 1, 1] = embeddings[0, 1, 0]
    plan = _plan(embeddings)
    previous = plan["matched_previous_slot"][0, 1]
    valid_previous = previous[previous >= 0]
    assert int(valid_previous.unique().numel()) == int(valid_previous.numel())


def test_future_tubelet_never_changes_current_plan_and_clip_boundary_resets():
    embeddings = _identity_embeddings(9)
    original = _plan(embeddings)
    perturbed = embeddings.clone()
    perturbed[:, 2:] = torch.randn_like(perturbed[:, 2:])
    changed = _plan(perturbed)
    assert torch.equal(
        original["refresh_mask"][:, :2],
        changed["refresh_mask"][:, :2],
    )
    assert original["forced_first_mask"].tolist() == [
        [True, False, False, False, False, False, False, False, True]
    ]
    assert original["refresh_mask"][0, 8].all()


def test_nonfinite_embedding_fails_instead_of_becoming_a_fallback():
    embeddings = _identity_embeddings(2)
    embeddings[0, 1, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="must be finite"):
        _plan(embeddings)


def _empty_stats():
    return {
        "ragged_attention_bucket_call_count": 0,
        "ragged_mlp_bucket_call_count": 0,
        "executed_attention_tokens": 0,
        "executed_kv_tokens": 0,
        "executed_attention_pairs": 0,
        "executed_mlp_tokens": 0,
    }


def test_k64_fallback_is_numerically_and_gradient_equal_to_full_block():
    torch.manual_seed(9)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    value = torch.randn(1, 64, 8, requires_grad=True)
    positions = [torch.arange(64).view(1, 64)]
    tubelets = torch.zeros(1, 64, dtype=torch.long)
    spatial = _strict_indices().view(1, 64)
    dense = block.forward_native_ragged(
        value,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=1,
        grid_height=10,
        grid_width=10,
    )
    fallback = block.forward_native_ragged(
        value,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=1,
        grid_height=10,
        grid_width=10,
        packed_stats=_empty_stats(),
        refresh_mask=torch.ones(1, 64, dtype=torch.bool),
        refresh_mode="mod32_kv",
    )
    assert torch.allclose(fallback, dense, atol=1e-6, rtol=1e-5)
    dense_grad = torch.autograd.grad(dense.sum(), value, retain_graph=True)[0]
    fallback_grad = torch.autograd.grad(fallback.sum(), value)[0]
    assert torch.allclose(fallback_grad, dense_grad, atol=1e-6, rtol=1e-5)


def test_mixed_k32_k64_clip_query_counts_execute_without_padding():
    torch.manual_seed(11)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    value = torch.randn(1, 1024, 8)
    positions = [torch.arange(1024).view(2, 512)]
    tubelets = torch.arange(16).repeat_interleave(64).view(1, -1)
    spatial = _strict_indices().repeat(16).view(1, -1)
    refresh = torch.zeros(1, 1024, dtype=torch.bool)
    refresh.reshape(1, 16, 64)[:, :, :32] = True
    refresh.reshape(1, 16, 64)[:, 0] = True
    refresh.reshape(1, 16, 64)[:, 8:10] = True
    stats = _empty_stats()
    output = block.forward_native_ragged(
        value,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=16,
        grid_height=10,
        grid_width=10,
        packed_stats=stats,
        refresh_mask=refresh,
        refresh_mode="mod32_kv",
    )
    assert output.shape == value.shape
    assert stats["executed_attention_tokens"] == int(refresh.sum().item())
    assert stats["executed_kv_tokens"] == 1024
    assert stats["executed_attention_pairs"] == 512 * int(refresh.sum().item())


def test_different_fallback_totals_across_batch_execute_in_separate_query_buckets():
    torch.manual_seed(13)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    value = torch.randn(2, 128, 8)
    positions = [torch.arange(256).view(2, 128)]
    tubelets = torch.arange(2).repeat_interleave(64).view(1, -1).expand(2, -1)
    spatial = _strict_indices().repeat(2).view(1, -1).expand(2, -1)
    refresh = torch.zeros(2, 128, dtype=torch.bool)
    refresh[0].reshape(2, 64)[:, :32] = True
    refresh[1].reshape(2, 64)[:, :32] = True
    refresh[1].reshape(2, 64)[0] = True
    stats = _empty_stats()
    output = block.forward_native_ragged(
        value,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=2,
        grid_height=10,
        grid_width=10,
        packed_stats=stats,
        refresh_mask=refresh,
        refresh_mode="mod32_kv",
    )
    assert output.shape == value.shape
    assert refresh.sum(dim=1).tolist() == [64, 96]
    assert stats["executed_attention_tokens"] == 160
    assert stats["executed_kv_tokens"] == 256
    assert stats["executed_attention_pairs"] == 128 * 160


class _FixedPatchEmbed(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.values = values

    def forward(self, inputs: torch.Tensor):
        assert int(inputs.shape[0]) == int(self.values.shape[0])
        return self.values.unsqueeze(1), None


def _tiny_temporal_backbone(values: torch.Tensor) -> VisionTransformerAdapter:
    model = VisionTransformerAdapter(
        img_size=160,
        patch_size=16,
        embed_dims=64,
        depth=1,
        num_heads=8,
        mlp_ratio=2.0,
        num_frames=16,
        tubelet_size=2,
        total_frames=16,
        adapter_index=[],
        use_mean_pooling=False,
    ).eval()
    model.patch_embed = _FixedPatchEmbed(values)
    return model


def test_apm_and_cur_share_mask_but_only_apm_substitutes_detached_memory():
    torch.manual_seed(17)
    base = torch.randn(64, 64)
    values = torch.stack(
        [base + 0.02 * tubelet * torch.randn_like(base) for tubelet in range(8)]
    ).reshape(512, 64)
    apm_values = values.clone().detach().requires_grad_(True)
    cur_values = values.clone().detach().requires_grad_(True)
    physical = torch.cat(
        [_strict_indices() + 100 * tubelet for tubelet in range(8)]
    ).view(1, -1)
    native = torch.zeros(1, 512, 3, 2, 16, 16)
    apm = _tiny_temporal_backbone(apm_values)
    cur = _tiny_temporal_backbone(cur_values)
    apm_x, _, _, _, apm_meta = apm._prepare_native_ragged_tokens(
        native,
        physical,
        total_tubelets=8,
        source_grid_hw=(10, 10),
        use_absolute_position=False,
        refresh_mode="apm32_ctx64",
    )
    cur_x, _, _, _, cur_meta = cur._prepare_native_ragged_tokens(
        native,
        physical,
        total_tubelets=8,
        source_grid_hw=(10, 10),
        use_absolute_position=False,
        refresh_mode="cur32_ctx64",
    )
    assert torch.equal(
        apm_meta["temporal_refresh_mask"],
        cur_meta["temporal_refresh_mask"],
    )
    retained = ~apm_meta["temporal_refresh_mask"].reshape(1, 8, 64)
    assert retained[:, 1:].any()
    assert not torch.equal(
        apm_x.reshape(1, 8, 64, 64)[retained],
        cur_x.reshape(1, 8, 64, 64)[retained],
    )
    loss = apm_x.reshape(1, 8, 64, 64)[:, 1][retained[:, 1]].sum()
    gradient = torch.autograd.grad(loss, apm_values)[0].reshape(8, 64, 64)
    assert torch.equal(gradient[0], torch.zeros_like(gradient[0]))
    assert gradient[1].abs().sum().item() > 0
    assert apm_meta["temporal_alignment_ledger"]["new_trainable_parameters"] == 0


def test_configs_and_launcher_bind_only_the_two_frozen_temporal_arms():
    config_dir = ROOT / "configs" / "adatad" / "thumos"
    expected = {
        "apm32_ctx64": "georoute_official_r1_apm32_ctx64_prebackbone_seed42_v001.py",
        "cur32_ctx64": "georoute_official_r1_cur32_ctx64_prebackbone_seed42_v001.py",
    }
    for mode, filename in expected.items():
        config = Config.fromfile(config_dir / filename)
        custom = config.model.backbone.custom
        assert custom.zoomtoken_refresh_carry_mode == mode
        assert (
            custom.zoomtoken_query_tokens,
            custom.zoomtoken_kv_tokens,
            custom.zoomtoken_mlp_tokens,
        ) == (32, 64, 32)
        assert custom.georoute_official_support == "strict_rect8x8"
        assert config.zoomtoken_p1_config.new_trainable_parameters == 0
        assert config.workflow.checkpoint_interval == 5
    launcher = (
        ROOT / "scripts" / "run_zoomtoken_official_prebackbone_bc_n16r4.sh"
    ).read_text(encoding="utf-8")
    assert launcher.count("R1-APM32-CTX64)") == 1
    assert launcher.count("R1-CUR32-CTX64)") == 1
    assert "georoute_official_r1_apm32_ctx64_prebackbone_seed42_v001.py" in launcher
    assert "georoute_official_r1_cur32_ctx64_prebackbone_seed42_v001.py" in launcher


def test_apm_and_cur_enter_the_existing_full_state_recovery_contract():
    train_entry = _load_train_module()
    config_dir = ROOT / "configs" / "adatad" / "thumos"
    for filename, arm in (
        (
            "georoute_official_r1_apm32_ctx64_prebackbone_seed42_v001.py",
            "R1-APM32-CTX64",
        ),
        (
            "georoute_official_r1_cur32_ctx64_prebackbone_seed42_v001.py",
            "R1-CUR32-CTX64",
        ),
    ):
        config = Config.fromfile(config_dir / filename)
        config.zoomtoken_p1_config.source_commit = "d" * 40
        config.work_dir = f"/tmp/zoomtoken-preflight/{arm}/gpu2_id0"
        contract = train_entry._zoomtoken_recovery_contract(config)
        assert contract["arm_surface"] == arm
        assert contract["interval_epochs"] == 5
        assert contract["keep_latest"] == 3
        assert contract["full_state"] is True


def test_single_batch_loader_never_consumes_a_second_batch():
    train_entry = _load_train_module()

    class _Loader:
        sampler = object()

        def __iter__(self):
            yield "first"
            raise AssertionError("second batch must not be consumed")

    loader = train_entry._SingleBatchLoader(_Loader())
    assert len(loader) == 1
    assert list(loader) == ["first"]


@pytest.mark.parametrize(
    "arm,mode",
    [
        ("R1-APM32-CTX64", "apm32_ctx64"),
        ("R1-CUR32-CTX64", "cur32_ctx64"),
    ],
)
def test_temporal_preflight_ledger_reconciles_without_metric_values(arm, mode):
    train_entry = _load_train_module()
    summary = {
        "refresh_execution_mode": mode,
        "heavy_backbone_forward_count": 1,
        "padded_heavy_tokens_per_window": 0,
        "requested_physical_tokens_per_window": 512,
        "unique_physical_tokens_per_window": 512,
        "executed_patch_tokens_per_window": 512,
        "batch_size": 1,
        "refresh_query_tokens_per_window_by_batch": [288],
        "temporal_alignment": {
            "carrier_mode": mode,
            "memory_lifetime_tubelets": 1,
            "clip_reset_tubelets": 8,
            "similarity_threshold": 0.8,
            "search_radius": 2,
            "new_trainable_parameters": 0,
            "previous_memory_detached": True,
            "current_position_restored": True,
            "future_tubelet_access": False,
            "total_tubelets": 8,
            "refreshed_tokens": 288,
            "retained_tokens": 224,
            "fallback_tubelets": 1,
            "normal_tubelets": 7,
        },
    }
    vit = type("Vit", (), {"latest_native_packed_summary": summary})()
    route = type("Route", (), {"model": type("Core", (), {"backbone": vit})()})()
    detector = type("Detector", (), {"backbone": route})()
    ddp = type("DDP", (), {"module": detector})()
    receipt = train_entry._zoomtoken_temporal_preflight_summary(
        ddp,
        {"arm_surface": arm},
    )
    assert receipt["mode"] == mode
    assert "accuracy" not in receipt
    assert "loss" not in receipt


def test_recovery_fixture_rejects_any_serialized_live_temporal_memory():
    train_entry = _load_train_module()
    valid = {
        "state_dict": {"module.weight": torch.ones(1)},
        "state_dict_ema": {"module.weight": torch.ones(1)},
        "training_state": {"next_epoch": 1},
    }
    train_entry._assert_no_temporal_memory_in_checkpoint(valid)
    invalid = dict(valid)
    invalid["state_dict"] = {"module.apm_memory": torch.ones(1)}
    with pytest.raises(RuntimeError, match="serialized live memory"):
        train_entry._assert_no_temporal_memory_in_checkpoint(invalid)


def test_launcher_exposes_only_result_blind_temporal_preflight_mode():
    launcher = (
        ROOT / "scripts" / "run_zoomtoken_official_prebackbone_bc_n16r4.sh"
    ).read_text(encoding="utf-8")
    assert 'TEMPORAL_PREFLIGHT_ONLY="${ZOOMTOKEN_TEMPORAL_PREFLIGHT_ONLY:-0}"' in launcher
    assert "--zoomtoken-temporal-preflight-only" in launcher
    assert "temporal mechanical preflight accepts only APM32/CUR32" in launcher
    assert "temporal mechanical preflight forbids resume input" in launcher


def test_temporal_route_does_not_modify_or_combine_amod_dsr6_chronotransport():
    vit_source = (
        ROOT / "opentad" / "models" / "backbones" / "vit_adapter.py"
    ).read_text(encoding="utf-8")
    assert 'raise RuntimeError("APM32/CUR32 cannot combine with strict A-MoD")' in vit_source
    assert 'refresh_mode == "dsr6_kv"' in vit_source
    assert "native ragged execution cannot combine with ChronoTransport" in vit_source
