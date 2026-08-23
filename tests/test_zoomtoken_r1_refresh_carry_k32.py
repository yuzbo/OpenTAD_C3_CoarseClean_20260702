import importlib.util
from pathlib import Path

import pytest
import torch
from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load focused module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_routing = _load_module(
    "zoomtoken_georoute_routing_focus",
    ROOT / "opentad" / "models" / "backbones" / "georoute_routing.py",
)
_vit = _load_module(
    "zoomtoken_vit_adapter_focus",
    ROOT / "opentad" / "models" / "backbones" / "vit_adapter.py",
)
_train = _load_module(
    "zoomtoken_train_entry_focus",
    ROOT / "tools" / "train.py",
)
build_refresh_mask = _routing.build_refresh_mask
Attention = _vit.Attention
Block = _vit.Block


def test_refresh_mask_exact_k32_and_age_priority():
    motion = torch.zeros(1, 2, 64)
    valid = torch.ones_like(motion, dtype=torch.bool)
    age = torch.zeros_like(motion)
    age[..., :2] = 2
    mask = build_refresh_mask(motion, valid, age)
    assert mask.shape == (1, 2, 64)
    assert mask.sum(-1).tolist() == [[32, 32]]
    assert mask[0, 0, :2].all()


def test_refresh_mask_uses_frozen_lexicographic_priority_and_index_tie():
    motion = torch.zeros(1, 1, 64)
    motion[..., 40:48] = 10.0
    valid = torch.ones_like(motion, dtype=torch.bool)
    valid[..., :32] = False
    age = torch.zeros_like(motion)
    mask = build_refresh_mask(motion, valid, age)
    assert mask[..., 40:48].all()
    assert mask[..., :24].all()
    assert not mask[..., 24:32].any()

    tied = build_refresh_mask(
        torch.zeros_like(motion),
        torch.ones_like(valid),
        torch.zeros_like(age),
    )
    assert tied[..., :32].all()
    assert not tied[..., 32:].any()


def test_base_attention_query_context_is_parameter_shared_and_full_length_equal():
    torch.manual_seed(42)
    attention = Attention(embed_dims=8, num_heads=2).eval()
    inputs = torch.randn(2, 5, 8)
    expected = attention(inputs)
    actual = attention.forward_query_context(inputs, inputs)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert {name for name, _ in attention.named_parameters()} == {
        "q_bias",
        "v_bias",
        "qkv.weight",
        "proj.weight",
        "proj.bias",
    }


def test_block_previous_tubelet_carry_is_same_spatial_and_detached():
    torch.manual_seed(7)
    current = torch.randn(1, 128, 4, requires_grad=True)
    tubelets = torch.arange(2).repeat_interleave(64).view(1, -1)
    spatial = torch.arange(64).repeat(2).view(1, -1)
    carry = Block._previous_spatial_block_input(
        current,
        tubelets,
        spatial,
        total_tubelets=2,
        spatial_tokens=100,
    )
    assert torch.equal(carry[:, :64], current[:, :64])
    assert torch.equal(carry[:, 64:], current[:, :64].detach())
    gradient = torch.autograd.grad(carry[:, 64:].sum(), current)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))


def _empty_packed_stats():
    return {
        "ragged_attention_bucket_call_count": 0,
        "ragged_mlp_bucket_call_count": 0,
        "executed_attention_tokens": 0,
        "executed_kv_tokens": 0,
        "executed_attention_pairs": 0,
        "executed_mlp_tokens": 0,
    }


def test_mod32_block_updates_only_k32_queries_against_k64_context():
    torch.manual_seed(9)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    inputs = torch.randn(1, 128, 8)
    tubelets = torch.arange(2).repeat_interleave(64).view(1, -1)
    spatial = torch.arange(64).repeat(2).view(1, -1)
    refresh = torch.zeros(1, 128, dtype=torch.bool)
    refresh[:, :32] = True
    refresh[:, 64:96] = True
    stats = _empty_packed_stats()
    output = block.forward_native_ragged(
        inputs,
        bucket_positions=[torch.arange(128).view(1, 128)],
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=2,
        grid_height=10,
        grid_width=10,
        packed_stats=stats,
        refresh_mask=refresh,
        refresh_mode="mod32_kv",
    )
    assert torch.equal(output[~refresh], inputs[~refresh])
    assert not torch.equal(output[refresh], inputs[refresh])
    assert stats["executed_attention_tokens"] == 64
    assert stats["executed_kv_tokens"] == 128
    assert stats["executed_attention_pairs"] == 64 * 128
    assert stats["executed_mlp_tokens"] == 64


def test_rc32_nonrefresh_tokens_use_detached_previous_block_input_mix():
    torch.manual_seed(11)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    inputs = torch.randn(1, 128, 8, requires_grad=True)
    tubelets = torch.arange(2).repeat_interleave(64).view(1, -1)
    spatial = torch.arange(64).repeat(2).view(1, -1)
    refresh = torch.zeros(1, 128, dtype=torch.bool)
    refresh[:, :32] = True
    refresh[:, 64:96] = True
    output = block.forward_native_ragged(
        inputs,
        bucket_positions=[torch.arange(128).view(1, 128)],
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=2,
        grid_height=10,
        grid_width=10,
        packed_stats=_empty_packed_stats(),
        refresh_mask=refresh,
        refresh_mode="rc32_kv",
        refresh_alpha=torch.zeros((), requires_grad=True),
    )
    expected = 0.5 * (inputs[:, 96:128] + inputs[:, 32:64].detach())
    assert torch.allclose(output[:, 96:128], expected)
    gradient = torch.autograd.grad(output[:, 96:128].sum(), inputs)[0]
    assert torch.equal(gradient[:, 32:64], torch.zeros_like(gradient[:, 32:64]))
    assert torch.equal(
        gradient[:, 96:128],
        torch.full_like(gradient[:, 96:128], 0.5),
    )


def test_four_arm_configs_bind_token_counts_and_rc_optimizer_group():
    config_dir = ROOT / "configs" / "adatad" / "thumos"
    names = {
        "full64": "georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py",
        "drop32": "georoute_official_r1_drop32_prebackbone_seed42_v001.py",
        "mod32_kv": "georoute_official_r1_mod32_kv_prebackbone_seed42_v001.py",
        "rc32_kv": "georoute_official_r1_rc32_kv_prebackbone_seed42_v001.py",
    }
    expected = {
        "full64": (64, 64, 64),
        "drop32": (32, 32, 32),
        "mod32_kv": (32, 64, 32),
        "rc32_kv": (32, 64, 32),
    }
    configs = {
        arm: Config.fromfile(config_dir / filename)
        for arm, filename in names.items()
    }
    for arm, config in configs.items():
        custom = config.model.backbone.custom
        assert custom.zoomtoken_refresh_carry_mode == arm
        assert (
            custom.zoomtoken_query_tokens,
            custom.zoomtoken_kv_tokens,
            custom.zoomtoken_mlp_tokens,
        ) == expected[arm]
        assert custom.georoute_official_support == "strict_rect8x8"
    rc_custom = configs["rc32_kv"].model.backbone.custom
    assert rc_custom.zoomtoken_temporal_carry is True
    assert rc_custom.zoomtoken_carry_detach is True
    assert rc_custom.zoomtoken_carry_mix_per_block is True
    optimizer_names = [
        item["name"] for item in configs["rc32_kv"].optimizer.backbone.custom
    ]
    assert optimizer_names.count("zoomtoken_refresh_carry_alpha") == 1
    alpha_group = next(
        item
        for item in configs["rc32_kv"].optimizer.backbone.custom
        if item["name"] == "zoomtoken_refresh_carry_alpha"
    )
    assert alpha_group["lr"] == 2e-4
    assert alpha_group["weight_decay"] == 0.0
    for config in configs.values():
        assert config.zoomtoken_p1_config.runner_binding_required is True
        assert config.zoomtoken_p1_config.seed == 42
        assert config.zoomtoken_p1_config.gt_for_route_allowed is False
    for arm in ("drop32", "mod32_kv", "rc32_kv"):
        assert configs[arm].official_bc_contract.support_is_only_scientific_difference is False
        assert configs[arm].official_bc_contract.temporal_refresh_arm == arm


def test_existing_official_runner_selects_all_refresh_arms():
    source = (
        ROOT / "scripts" / "run_zoomtoken_official_prebackbone_bc_n16r4.sh"
    ).read_text(encoding="utf-8")
    expected = {
        "R1-DROP32": "georoute_official_r1_drop32_prebackbone_seed42_v001.py",
        "R1-MOD32-KV": "georoute_official_r1_mod32_kv_prebackbone_seed42_v001.py",
        "R1-RC32-KV": "georoute_official_r1_rc32_kv_prebackbone_seed42_v001.py",
    }
    for arm, filename in expected.items():
        assert f"{arm})" in source
        assert f'CONFIG_NAME="{filename}"' in source


def test_refresh_configs_pass_actual_training_recovery_contract(tmp_path):
    config_dir = ROOT / "configs" / "adatad" / "thumos"
    expected = {
        "R1-DROP32": "georoute_official_r1_drop32_prebackbone_seed42_v001.py",
        "R1-MOD32-KV": "georoute_official_r1_mod32_kv_prebackbone_seed42_v001.py",
        "R1-RC32-KV": "georoute_official_r1_rc32_kv_prebackbone_seed42_v001.py",
    }
    for arm_surface, filename in expected.items():
        config = Config.fromfile(config_dir / filename)
        config.work_dir = str(tmp_path / arm_surface)
        config.zoomtoken_p1_config.source_commit = (
            "836f2ce4beafa8cbab513604dfa74be01a977a3c"
        )
        contract = _train._zoomtoken_recovery_contract(config)
        assert contract["arm_surface"] == arm_surface
        assert contract["seed"] == 42
        assert contract["interval_epochs"] == 5
        assert contract["keep_latest"] == 3
        assert arm_surface in _train.ZOOMTOKEN_UPDATE_INDEX_ARMS
        assert arm_surface in _train.ZOOMTOKEN_CANONICAL_SOURCE_ARMS


def test_training_recovery_contract_still_rejects_unknown_route(tmp_path):
    config = Config.fromfile(
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_official_r1_drop32_prebackbone_seed42_v001.py"
    )
    config.work_dir = str(tmp_path / "unknown")
    config.zoomtoken_p1_config.arm_surface = "R1-UNKNOWN"
    config.zoomtoken_p1_config.source_commit = (
        "836f2ce4beafa8cbab513604dfa74be01a977a3c"
    )
    with pytest.raises(
        ValueError,
        match="ZoomToken recovery is restricted to the frozen route surfaces",
    ):
        _train._zoomtoken_recovery_contract(config)
