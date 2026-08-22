import ast
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ROUTING_PATH = ROOT / "opentad/models/backbones/georoute_routing.py"
WRAPPER_PATH = ROOT / "opentad/models/backbones/georoute_wrapper.py"
ENGINE_PATH = ROOT / "opentad/cores/train_engine.py"
TRAIN_PATH = ROOT / "tools/train.py"
COMMON_PATH = ROOT / "configs/adatad/thumos/georoute_official_prebackbone_bc_common_seed42_v001.py"
LAUNCHER_PATH = ROOT / "scripts/run_zoomtoken_official_prebackbone_bc_n16r4.sh"
CONFIG_DIR = ROOT / "configs/adatad/thumos"
CONFIG_IDENTITIES = {
    "georoute_official_r2_strict_rect8x8_q48_prebackbone_seed42_v001.py": ("R2", "strict_rect8x8_q48"),
    "georoute_official_r2_shuf48_prebackbone_seed42_v001.py": ("R2-SHUF48", "strict_rect8x8_shuf48"),
    "georoute_official_q48_global_prebackbone_seed42_v001.py": ("Q48-GLOBAL", "q48_global"),
    "georoute_official_r3_continuous_rect_prebackbone_seed42_v001.py": ("R3", "continuous_rect_dynamic"),
    "georoute_official_r3_area_shift97_prebackbone_seed42_v001.py": ("R3-AREA-SHIFT", "continuous_rect_dynamic_area_shift97"),
    "georoute_official_r4_core49_q15_prebackbone_seed42_v001.py": ("R4", "strict_rect7x7_core49_q15"),
    "georoute_official_r4_shuf15_prebackbone_seed42_v001.py": ("R4-SHUF15", "strict_rect7x7_core49_shuf15"),
    "georoute_official_q64_global_prebackbone_seed42_v001.py": ("Q64-GLOBAL", "q64_global"),
}


def _function_source(path, name):
    source = path.read_text()
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, node)


def _class_method_source(path, class_name, method_name):
    source = path.read_text()
    class_node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def _assert_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected fail-closed ValueError")


def test_r4_implementation_builds_only_sixteen_complete_7x7_cores():
    namespace = {}
    for name in (
        "_validate_strict_rectangle_7x7_blocks",
        "strict_rectangle_7x7_blocks",
    ):
        exec(compile(_function_source(ROUTING_PATH, name), str(ROUTING_PATH), "exec"), namespace)
    validate = namespace["_validate_strict_rectangle_7x7_blocks"]
    blocks = namespace["strict_rectangle_7x7_blocks"]()
    assert len(blocks) == 16
    assert all(len(block) == len(set(block)) == 49 for block in blocks)
    assert blocks[0][:7] == tuple(range(7))
    assert blocks[-1][-7:] == tuple(range(93, 100))
    assert validate(blocks) == blocks
    _assert_value_error(lambda: validate(blocks[:-1]))
    _assert_value_error(lambda: validate(blocks, grid_width=9))
    duplicate_token = list(blocks[0])
    duplicate_token[-1] = duplicate_token[0]
    _assert_value_error(lambda: validate((tuple(duplicate_token),) + blocks[1:]))
    _assert_value_error(lambda: validate(blocks[:-1] + (blocks[0],)))
    routing = ROUTING_PATH.read_text()
    assert "6x8" not in routing and "8x6" not in routing
    assert "torch" not in sys.modules


def test_r234_source_contract_is_prepatch_one_ragged_zero_padding_and_no_leak():
    route = _class_method_source(
        WRAPPER_PATH, "GeoRouteBackboneWrapper", "_official_fixed_support_route"
    )
    forward = _class_method_source(
        WRAPPER_PATH, "GeoRouteBackboneWrapper", "_forward_official_fixed_support"
    )
    ordered = (
        "extract_native_tubelets(",
        "self._official_fixed_support_route(",
        "self._gather_selected_native_physical(",
        "self.model.backbone.forward_native_ragged(",
        "self.sparse_adapter.forward_ragged(",
        "deterministic_linear_2x(",
    )
    positions = [forward.index(binding) for binding in ordered]
    assert positions == sorted(positions)
    assert forward.count("self.model.backbone.forward_native_ragged(") == 1
    assert "forward_native_packed" not in forward
    for binding in (
        "select_rectangle_constrained_qbase_8x8(",
        "select_rectangle_core_outside_qbase_7x7(",
        "select_continuous_strict_rectangle(",
        "select_qbase_global_exact_k(",
    ):
        assert binding in route
    for binding in (
        '"padded_heavy_tokens_per_window": 0',
        '"dummy_tokens_used": False',
        '"uses_gt_for_route": False',
        '"uses_teacher": False',
        '"uses_oracle": False',
        '"uses_raw_prediction": False',
        '"q_base_roi_modifier_enabled": False',
        '"q_base_residual_modifier_enabled": False',
        '"q_base_geometry_side_channel_enabled": False',
        '"extra_anti_collapse_loss_enabled": False',
    ):
        assert binding in forward
    assert "torch" not in sys.modules


def test_shuffle_metadata_is_dataset_stable_and_result_blind():
    routing = _function_source(ROUTING_PATH, "_stateless_candidate_permutation")
    metadata = _class_method_source(
        WRAPPER_PATH, "GeoRouteBackboneWrapper", "forward_with_window_ordinals"
    )
    thumos = (ROOT / "opentad/datasets/thumos.py").read_text()
    formatting = (ROOT / "opentad/datasets/transforms/formatting.py").read_text()
    assert "window * 2862933555777941757" in routing
    assert "tubelet * 1442695040888963407" in routing
    assert "torch.rand" not in routing and "counter" not in routing
    assert thumos.count("window_ordinal=int(index)") == 2
    assert '"window_ordinal"' in formatting
    for forbidden in (
        '"gt_segments"',
        '"gt_labels"',
        '"prediction"',
        '"teacher"',
        '"oracle"',
        '"raw_prediction"',
    ):
        assert forbidden in metadata
    assert "_successful_update_index" not in metadata
    assert "torch" not in sys.modules


def test_r3_successful_update_dual_and_recovery_lifecycle_are_real():
    commit = _class_method_source(
        WRAPPER_PATH, "GeoRouteBackboneWrapper", "commit_successful_update"
    )
    finish = _class_method_source(
        WRAPPER_PATH, "GeoRouteBackboneWrapper", "finish_training_epoch"
    )
    capture = _function_source(TRAIN_PATH, "_capture_zoomtoken_training_state")
    restore = _function_source(TRAIN_PATH, "_restore_zoomtoken_training_state")
    engine = ENGINE_PATH.read_text()
    assert engine.index("commit_successful_update(successful_update_index)") < engine.index(
        "successful_update_index += 1"
    )
    assert engine.index("successful_update_index += 1") < engine.index(
        "finish_training_epoch(curr_epoch, successful_update_index)"
    )
    assert "self.r3_epoch_g_sum.add_(self._pending_r3_epoch_g)" in commit
    assert "self.r3_epoch_successful_updates.add_(1)" in commit
    assert "self.r3_dual_lambda.copy_" in finish
    assert ".clamp(-4.0, 4.0)" in finish
    for binding in (
        "export_r3_recovery_state",
        'training_state["r3_dual_state"]',
        'r3_dual_state.get("last_completed_update")',
        'r3_dual_state.get("last_completed_epoch")',
    ):
        assert binding in capture
    assert "restore_r3_recovery_state" in restore
    assert "next_successful_update_index - 1" in restore
    assert "torch" not in sys.modules


def test_eight_leaf_configs_preserve_recipe_and_have_immutable_identities():
    common = COMMON_PATH.read_text()
    launcher = LAUNCHER_PATH.read_text()
    assert "train=dict(batch_size=2)" in common
    assert "local_batch_size=1" in common and "global_batch_size=2" in common
    assert "end_epoch=60" in (CONFIG_DIR / "e2e_thumos_videomae_s_768x1_160_adapter.py").read_text()
    observed_workdirs = set()
    for name, (arm, support) in CONFIG_IDENTITIES.items():
        source = (CONFIG_DIR / name).read_text()
        assert '_base_ = ["./georoute_official_prebackbone_bc_common_seed42_v001.py"]' in source
        assert f'official_bc_arm = "{arm}"' in source
        assert f'georoute_official_support="{support}"' in source
        assert f'arm_surface="{arm}"' in source
        assert "source_commit=None" in source
        assert "seed=42" in source
        assert "checkpoint_interval=5" in source and "keep_latest=3" in source
        for forbidden in ("dataset =", "scheduler =", "solver =", "post_processing ="):
            assert forbidden not in source
        work_dir = next(
            node.value.value
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "work_dir" for target in node.targets)
        )
        assert work_dir.endswith("_unbound") and work_dir not in observed_workdirs
        observed_workdirs.add(work_dir)
        assert name in launcher and f"{arm})" in launcher
    assert "6x8" not in "\n".join((CONFIG_DIR / name).read_text() for name in CONFIG_IDENTITIES)
    assert "8x6" not in "\n".join((CONFIG_DIR / name).read_text() for name in CONFIG_IDENTITIES)
    for forbidden in ("batch_size=", "optimizer.", "scheduler.", "solver.", "post_processing."):
        assert forbidden not in launcher
    assert "torch" not in sys.modules


def test_r234_runtime_geometry_tie_st_shuffle_and_dynamic_k():
    if os.environ.get("ZOOMTOKEN_R234_RUNTIME_CHECK") != "1":
        return
    import torch
    from opentad.models.backbones.georoute_routing import (
        select_continuous_strict_rectangle,
        select_rectangle_constrained_qbase_8x8,
        select_rectangle_core_outside_qbase_7x7,
    )

    valid = torch.ones((1, 2, 100), dtype=torch.bool)
    geometry = torch.zeros((1, 2, 4), dtype=torch.float64, requires_grad=True)
    q_base = torch.arange(100, dtype=torch.float64).view(1, 1, 100).expand(1, 2, 100).clone().requires_grad_()
    r2 = select_rectangle_constrained_qbase_8x8(
        geometry, q_base, training=True, temperature=0.5, valid_mask=valid
    )
    assert r2["indices"].shape == (1, 2, 48)
    assert torch.equal(r2["st_gate"].detach(), torch.ones_like(r2["st_gate"]))
    r2["st_gate"].sum().backward(retain_graph=True)
    assert geometry.grad is not None and q_base.grad is not None

    shuffled_a = select_rectangle_constrained_qbase_8x8(
        geometry.detach(), q_base.detach(), training=False, temperature=0.5,
        valid_mask=valid, shuffle_seed=42, window_ordinals=torch.tensor([17]),
    )
    shuffled_b = select_rectangle_constrained_qbase_8x8(
        geometry.detach(), -q_base.detach(), training=False, temperature=0.5,
        valid_mask=valid, shuffle_seed=42, window_ordinals=torch.tensor([17]),
    )
    assert torch.equal(shuffled_a["indices"], shuffled_b["indices"])

    r4 = select_rectangle_core_outside_qbase_7x7(
        geometry.detach(), q_base.detach(), training=True, temperature=0.5, valid_mask=valid
    )
    assert r4["block_top_left_row_col"].tolist() == [[[1, 1], [1, 1]]]
    assert r4["core_indices"].shape[-1] == 49 and r4["outside_indices"].shape[-1] == 15
    assert r4["indices"].shape[-1] == 64

    extent_logit = math.log(((0.8 - 0.02) / 0.98) / (1.0 - (0.8 - 0.02) / 0.98))
    dynamic_logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    dynamic_logits[0, 0, 2:] = extent_logit
    dynamic_logits[0, 1, 2:] = -100.0
    r3 = select_continuous_strict_rectangle(
        dynamic_logits, training=True, valid_mask=valid,
        soft_temperature=0.025, area_shift_tubelets=0,
    )
    assert r3["k_per_tubelet"].tolist() == [[64, 0]]
    assert r3["physical_indices"].shape == (1, 64)
    assert torch.equal(r3["st_gate"].detach(), torch.ones_like(r3["st_gate"]))


def test_r234_target_wrapper_executes_prepatch_one_ragged_with_masked_zero(monkeypatch):
    if os.environ.get("ZOOMTOKEN_R234_WRAPPER_RUNTIME_CHECK") != "1":
        return
    from types import SimpleNamespace
    import torch
    from torch import nn
    import opentad.models.backbones.georoute_wrapper as wrapper_module
    from opentad.models.backbones.georoute_wrapper import GeoRouteBackboneWrapper

    native = torch.arange(200, dtype=torch.uint8).reshape(1, 2, 100, 1, 1, 1, 1).expand(1, 2, 100, 3, 2, 1, 1).clone()
    physical = torch.arange(64, dtype=torch.long).view(1, 64)
    monkeypatch.setattr(wrapper_module, "extract_native_tubelets", lambda source, **kwargs: (native, (10, 10), (0, 0), torch.ones((1, 2, 100), dtype=torch.bool)))
    monkeypatch.setattr(wrapper_module, "_normalize_uint8_video", lambda value, mean, std: value)

    class FakeBackbone:
        embed_dims = 3
        def __init__(self):
            self.native_ragged_forward_invocations = 0
            self.latest_native_packed_summary = None
        def forward_native_ragged(self, selected_native, physical_indices, **kwargs):
            assert torch.equal(physical_indices, physical)
            assert torch.equal(selected_native[:, :, 0, 0, 0, 0].long(), physical)
            self.native_ragged_forward_invocations += 1
            count = int(physical_indices.shape[1])
            self.latest_native_packed_summary = dict(
                schema_version="videomae_native_ragged_v1", execution_mode="true_clip_ragged_no_padding",
                window_token_budget=count, requested_physical_tokens_per_window=count,
                unique_physical_tokens_per_window=count, padded_heavy_tokens_per_window=0,
                executed_patch_tokens_per_window=count, heavy_backbone_forward_count=1,
                dense_adapter_forward_count=0,
            )
            return physical_indices.float().unsqueeze(-1).expand(-1, -1, 3)

    class FakeAdapter:
        def forward_ragged(self, selected_features, selected_scores, geometry, selected_coordinates, tubelet_indices, **kwargs):
            assert selected_features.shape == (1, 64, 3)
            pooled = selected_features.mean(dim=1).view(1, 3, 1)
            return torch.cat((pooled, torch.zeros_like(pooled)), dim=-1), torch.tensor([[True, False]])

    wrapper = GeoRouteBackboneWrapper.__new__(GeoRouteBackboneWrapper)
    nn.Module.__init__(wrapper)
    backbone = FakeBackbone()
    wrapper.model = SimpleNamespace(backbone=backbone)
    wrapper.sparse_adapter = FakeAdapter()
    wrapper.patch_size = 1
    wrapper.tubelet_size = 2
    wrapper.output_length = 4
    wrapper.absolute_position_enabled = True
    wrapper.official_support = "continuous_rect_dynamic"
    wrapper.source_mean = torch.zeros(1)
    wrapper.source_std = torch.ones(1)
    wrapper.r3_dual_lambda = torch.tensor(0.0)
    wrapper._pending_regularization = None
    wrapper._pending_score_function = None
    wrapper._pending_dynamic_auxiliary = None
    wrapper._pending_r3_epoch_g = None
    wrapper._pending_r3_update_index = None
    wrapper.latest_georoute_audit = None
    wrapper.latest_heavy_valid_mask = None
    wrapper.set_norm_layer = lambda: None
    wrapper._validate_official_fixed_support_input = lambda frames: frames[:, 0]
    wrapper._official_fixed_support_route = lambda source, **kwargs: dict(
        schema_version="georoute_continuous_strict_rectangle_routing_v1", mode="r3",
        geometry=torch.tensor([[[0.5, 0.5, 0.8, 0.8]]]).expand(1, 2, 4),
        physical_indices=physical, st_gate=torch.ones((1, 64)),
        k_per_tubelet=torch.tensor([[64, 0]]), soft_count_sum=torch.tensor(128.0),
        valid_tubelet_count=torch.tensor(2), area_shift_tubelets=0,
    )
    wrapper.eval()
    output = GeoRouteBackboneWrapper._forward_official_fixed_support(
        wrapper, torch.zeros((1, 1, 3, 4, 1, 1), dtype=torch.uint8), None
    )
    assert output.shape == (1, 3, 4)
    assert backbone.native_ragged_forward_invocations == 1
    assert wrapper.latest_heavy_valid_mask.tolist() == [[True, False]]
    assert wrapper.latest_georoute_audit["packed"]["padded_heavy_tokens_per_window"] == 0


def test_r3_target_recovery_roundtrip_restores_dual_and_update_identity(
    tmp_path, monkeypatch
):
    if os.environ.get("ZOOMTOKEN_R234_RECOVERY_RUNTIME_CHECK") != "1":
        return
    import copy
    import random
    from types import SimpleNamespace
    import numpy as np
    import torch
    import tools.train as train_module

    class Sampler:
        epoch = 4
        def set_epoch(self, epoch):
            self.epoch = int(epoch)

    class Loader:
        sampler = Sampler()
        def __len__(self):
            return 7

    class Backbone:
        def __init__(self):
            self.restored = None
        def export_r3_recovery_state(self):
            return dict(
                schema_version="zoomtoken_r3_dual_state_v001",
                dual_lambda=0.75,
                epoch_g_sum=0.0,
                epoch_successful_updates=0,
                last_completed_update=9,
                last_completed_epoch=4,
            )
        def restore_r3_recovery_state(self, state):
            self.restored = copy.deepcopy(state)

    args = SimpleNamespace(
        rank=0, world_size=1, local_rank=0, seed=42,
        config=str(tmp_path / "r3.py"),
    )
    cfg = SimpleNamespace(work_dir=str(tmp_path / "cell"))
    loader = Loader()
    backbone = Backbone()
    model = SimpleNamespace(module=SimpleNamespace(backbone=backbone))
    contract = dict(
        arm_surface="R3", seed=42,
        source_commit="1" * 40,
    )
    cuda_state = torch.arange(16, dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: cuda_state.clone())
    restored_cuda = {}
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: restored_cuda.update(state=state.clone(), device=device),
    )
    monkeypatch.setattr(
        train_module.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = train_module._capture_zoomtoken_training_state(
        args, cfg, loader, 4, contract, 10, model=model
    )
    assert state["r3_dual_state"]["dual_lambda"] == 0.75
    checkpoint = dict(checkpoint_role="recovery", epoch=4, training_state=state)
    backbone.restored = None
    next_update = train_module._restore_zoomtoken_training_state(
        checkpoint, args, cfg, loader, contract, model=model
    )
    assert next_update == 10 and loader.sampler.epoch == 5
    assert backbone.restored == state["r3_dual_state"]
    assert restored_cuda["device"] == 0

    wrong = copy.deepcopy(checkpoint)
    wrong["training_state"]["source_commit"] = "2" * 40
    _assert_value_error(
        lambda: train_module._restore_zoomtoken_training_state(
            wrong, args, cfg, loader, contract, model=model
        )
    )
