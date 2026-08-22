import ast
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ROUTING_PATH = ROOT / "opentad/models/backbones/georoute_routing.py"
WRAPPER_PATH = ROOT / "opentad/models/backbones/georoute_wrapper.py"
CONFIG_PATH = (
    ROOT
    / "configs/adatad/thumos/"
    "georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"
)
COMMON_PATH = (
    ROOT
    / "configs/adatad/thumos/"
    "georoute_official_prebackbone_bc_common_seed42_v001.py"
)
LAUNCHER_PATH = ROOT / "scripts/run_zoomtoken_official_prebackbone_bc_n16r4.sh"


def _function_source(path, name):
    source = path.read_text()
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, node)


def _strict_rectangle_block_functions():
    namespace = {}
    for name in (
        "_validate_strict_rectangle_8x8_blocks",
        "strict_rectangle_8x8_blocks",
    ):
        source = _function_source(ROUTING_PATH, name)
        exec(compile(source, str(ROUTING_PATH), "exec"), namespace)
    return (
        namespace["_validate_strict_rectangle_8x8_blocks"],
        namespace["strict_rectangle_8x8_blocks"],
    )


def _assert_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected fail-closed ValueError")


def test_r1_has_exactly_nine_complete_hole_free_row_major_blocks():
    validate, build = _strict_rectangle_block_functions()
    blocks = build()
    assert len(blocks) == 9
    assert all(len(block) == len(set(block)) == 64 for block in blocks)
    assert blocks[0][:8] == tuple(range(8))
    assert blocks[4][:8] == tuple(range(11, 19))
    assert blocks[-1][-8:] == tuple(range(92, 100))
    assert validate(blocks) == blocks

    _assert_value_error(lambda: build(grid_height=9))
    _assert_value_error(lambda: validate(blocks[:-1]))
    duplicate_token = list(blocks[0])
    duplicate_token[-1] = duplicate_token[0]
    _assert_value_error(
        lambda: validate((tuple(duplicate_token),) + tuple(blocks[1:]))
    )
    _assert_value_error(lambda: validate(tuple(blocks[:-1]) + (blocks[0],)))
    assert "torch" not in sys.modules


def test_r1_routing_source_is_categorical_not_token_topk():
    primitive = _function_source(ROUTING_PATH, "select_strict_rectangle_8x8")
    assert primitive is not None
    assert "center = 0.4 + 0.2 * torch.sigmoid" in primitive
    assert "/ (0.1**2)" in primitive
    assert "torch.argmax(block_logits, dim=-1)" in primitive
    assert "F.softmax(block_logits / float(temperature), dim=-1)" in primitive
    assert "hard_categorical + (probability - probability.detach())" in primitive
    assert "torch.matmul(" in primitive
    assert "strict_rectangle_8x8_blocks()" in primitive
    assert "topk" not in primitive.lower()
    assert "q_base" not in primitive
    assert "residual" not in primitive
    assert '"padded_token_count": 0' in primitive
    assert "torch" not in sys.modules


def test_r1_wrapper_keeps_raw_native_order_one_ragged_and_zero_padding_audit():
    source = WRAPPER_PATH.read_text()
    tree = ast.parse(source)
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GeoRouteBackboneWrapper"
    )
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef)
    }
    route = methods["_official_fixed_support_route"]
    forward = methods["_forward_official_fixed_support"]
    assert route is not None and forward is not None
    assert route.index("self.scout.geometry_head(") < route.index(
        "select_strict_rectangle_8x8("
    )
    assert "select_exact_k(" in route
    strict_branch = route[route.index('if self.official_support == "strict_rect8x8"') :]
    strict_branch = strict_branch[: strict_branch.index("else:")]
    assert "select_exact_k(" not in strict_branch
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
    assert '"padded_heavy_tokens_per_window": 0' in forward
    assert '"dummy_tokens_used": False' in forward
    assert '"raw_native_gather_before_patch_embedding": True' in forward
    assert '"uses_gt_for_route": False' in forward
    assert '"uses_teacher": False' in forward
    assert '"uses_oracle": False' in forward
    assert '"uses_raw_prediction": False' in forward
    assert "torch" not in sys.modules


def test_r1_config_inherits_common_and_changes_only_frozen_metadata():
    config = CONFIG_PATH.read_text()
    common = COMMON_PATH.read_text()
    assignment_names = {
        target.id
        for node in ast.parse(config).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignment_names == {
        "_base_",
        "official_bc_arm",
        "model",
        "strict_rectangle_topology",
        "workflow",
        "zoomtoken_recovery",
        "zoomtoken_p1_config",
        "work_dir",
    }
    assert (
        '_base_ = ["./georoute_official_prebackbone_bc_common_seed42_v001.py"]'
        in config
    )
    assert 'georoute_official_support="strict_rect8x8"' in config
    for forbidden in (
        "dataset =",
        "optimizer =",
        "scheduler =",
        "solver =",
        "post_processing =",
    ):
        assert forbidden not in config
    assert "end_epoch=60" in (
        ROOT
        / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
    ).read_text()
    assert "train=dict(batch_size=2)" in common
    assert "local_batch_size=1" in common and "global_batch_size=2" in common
    assert 'type="AdamW"' not in config
    assert "amp=" not in config and "ema=" not in config
    assert "torch" not in sys.modules


def test_r1_recovery_and_launcher_are_same_cell_and_do_not_override_recipe():
    config = CONFIG_PATH.read_text()
    launcher = LAUNCHER_PATH.read_text()
    train = (ROOT / "tools/train.py").read_text()
    checkpoint = (ROOT / "opentad/utils/checkpoint.py").read_text()
    assert 'arm_surface="R1"' in config
    assert 'checkpoint_interval=5' in config
    assert 'checkpoint_policy="recovery_latest3_plus_final"' in config
    assert "keep_latest=3" in config and "full_state=True" in config
    assert "source_commit=None" in config
    assert 'ZOOMTOKEN_RECOVERY_ARMS = {"DN", "G", "R1"}' in train
    assert '"state_dict": model.state_dict()' in checkpoint
    assert '"state_dict_ema": model_ema.module.state_dict()' in checkpoint
    assert '"training_state": dict(training_state)' in checkpoint
    assert '"checkpoint_role": "final_ema"' in checkpoint
    assert '"checkpoint_role": "final_raw"' in checkpoint
    assert '"zoomtoken_p1_config.source_commit=${EXPECTED_COMMIT}"' in launcher
    assert "georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py" in launcher
    assert "same-cell checkpoint/recovery_epoch_<N>.pth" in launcher
    assert 'resume_args=(--resume "${RESUME}")' in launcher
    for forbidden in (
        "batch_size=",
        "optimizer.",
        "scheduler.",
        "solver.",
        "post_processing.",
        "workflow.",
    ):
        assert forbidden not in launcher
    assert "torch" not in sys.modules


def test_r1_runtime_geometry_tie_st_and_order_contract():
    if os.environ.get("ZOOMTOKEN_R1_RUNTIME_CHECK") != "1":
        return

    import torch

    from opentad.models.backbones.georoute_routing import (
        select_strict_rectangle_8x8,
    )

    logits = torch.tensor(
        [[[0.7, -0.4, 0.0, 0.0], [0.7, -0.4, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    route = select_strict_rectangle_8x8(
        logits,
        training=True,
        temperature=0.5,
        valid_mask=torch.ones((1, 2, 100), dtype=torch.bool),
    )
    assert route["indices"].shape == (1, 2, 64)
    assert torch.equal(
        route["indices"][0, 0],
        torch.tensor(_strict_rectangle_block_functions()[1]()[4], dtype=torch.long),
    )
    assert route["block_top_left_row_col"].tolist() == [[[1, 1], [1, 1]]]
    assert torch.equal(route["st_gate"].detach(), torch.ones_like(route["st_gate"]))
    route["st_gate"].sum().backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad[..., :2]) > 0
    assert torch.count_nonzero(logits.grad[..., 2:]) == 0

    tie_logits = torch.zeros((1, 1, 4), dtype=torch.float64)
    tie_logits[..., 0] = math.log(0.25 / 0.75)
    tie = select_strict_rectangle_8x8(
        tie_logits,
        training=False,
        temperature=0.5,
        valid_mask=torch.ones((1, 1, 100), dtype=torch.bool),
    )
    assert tie["block_top_left_row_col"].tolist() == [[[1, 0]]]
    assert torch.equal(
        tie["indices"][0, 0],
        torch.tensor(_strict_rectangle_block_functions()[1]()[3], dtype=torch.long),
    )


def test_r1_target_wrapper_runs_prepatch_gather_then_one_ragged(monkeypatch):
    if os.environ.get("ZOOMTOKEN_R1_WRAPPER_RUNTIME_CHECK") != "1":
        return

    from types import SimpleNamespace

    import torch
    from torch import nn

    import opentad.models.backbones.georoute_wrapper as wrapper_module
    from opentad.models.backbones.georoute_wrapper import GeoRouteBackboneWrapper

    _validate, build = _strict_rectangle_block_functions()
    spatial = torch.tensor(build()[4], dtype=torch.long).view(1, 1, 64).expand(1, 2, 64)
    physical = torch.cat((spatial[:, 0], spatial[:, 1] + 100), dim=1)

    native = torch.arange(200, dtype=torch.uint8).reshape(
        1, 2, 100, 1, 1, 1, 1
    ).expand(1, 2, 100, 3, 2, 1, 1).clone()

    def fake_extract(source, *, patch_size, tubelet_size):
        assert patch_size == 1 and tubelet_size == 2
        return native, (10, 10), (0, 0), torch.ones((1, 2, 100), dtype=torch.bool)

    monkeypatch.setattr(wrapper_module, "extract_native_tubelets", fake_extract)
    monkeypatch.setattr(
        wrapper_module,
        "_normalize_uint8_video",
        lambda value, mean, std: value,
    )

    class FakeBackbone:
        embed_dims = 3

        def __init__(self):
            self.native_ragged_forward_invocations = 0
            self.latest_native_packed_summary = None

        def forward_native_ragged(
            self,
            selected_native,
            physical_indices,
            *,
            total_tubelets,
            source_grid_hw,
            use_absolute_position,
        ):
            assert total_tubelets == 2 and source_grid_hw == (10, 10)
            assert use_absolute_position
            assert torch.equal(physical_indices, physical)
            observed = selected_native[:, :, 0, 0, 0, 0].to(torch.long)
            assert torch.equal(observed, physical_indices)
            self.native_ragged_forward_invocations += 1
            selected_count = int(physical_indices.shape[1])
            self.latest_native_packed_summary = {
                "schema_version": "videomae_native_ragged_v1",
                "execution_mode": "true_clip_ragged_no_padding",
                "window_token_budget": selected_count,
                "requested_physical_tokens_per_window": selected_count,
                "unique_physical_tokens_per_window": selected_count,
                "padded_heavy_tokens_per_window": 0,
                "executed_patch_tokens_per_window": selected_count,
                "heavy_backbone_forward_count": 1,
                "dense_adapter_forward_count": 0,
            }
            return physical_indices.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)

    class FakeSparseAdapter:
        def forward_ragged(
            self,
            selected_features,
            selected_scores,
            geometry,
            selected_coordinates,
            tubelet_indices,
            **kwargs,
        ):
            assert selected_features.shape == (1, 128, 3)
            assert torch.count_nonzero(selected_scores) == 0
            assert selected_coordinates.shape == (1, 128, 2)
            assert kwargs == {
                "use_absolute_coordinates": False,
                "use_roi_relative_coordinates": False,
                "use_geometry_projection": False,
                "pooling_mode": "uniform_selected",
            }
            pooled = selected_features.reshape(1, 2, 64, 3).mean(dim=2)
            return pooled.transpose(1, 2), torch.ones((1, 2), dtype=torch.bool)

    wrapper = GeoRouteBackboneWrapper.__new__(GeoRouteBackboneWrapper)
    nn.Module.__init__(wrapper)
    backbone = FakeBackbone()
    wrapper.model = SimpleNamespace(backbone=backbone)
    wrapper.sparse_adapter = FakeSparseAdapter()
    wrapper.patch_size = 1
    wrapper.tubelet_size = 2
    wrapper.output_length = 4
    wrapper.absolute_position_enabled = True
    wrapper.official_support = "strict_rect8x8"
    wrapper.source_mean = torch.zeros(1)
    wrapper.source_std = torch.ones(1)
    wrapper._pending_regularization = None
    wrapper._pending_score_function = None
    wrapper._pending_dynamic_auxiliary = None
    wrapper.latest_georoute_audit = None
    wrapper.latest_heavy_valid_mask = None
    wrapper.set_norm_layer = lambda: None
    wrapper._validate_official_fixed_support_input = lambda frames: frames[:, 0]
    wrapper._official_fixed_support_route = lambda source, **kwargs: {
        "geometry": torch.tensor([[[0.5, 0.5, 0.8, 0.8]]]).expand(1, 2, 4),
        "spatial_indices": spatial,
        "st_gate": torch.ones((1, 2, 64)),
        "routing_schema": "georoute_strict_rectangle_8x8_routing_v1",
        "candidate_top_left_row_col": torch.tensor(
            [[row, col] for row in (0, 1, 2) for col in (0, 1, 2)]
        ),
        "block_top_left_row_col": torch.tensor([[[1, 1], [1, 1]]]),
        "block_size_hw": (8, 8),
        "candidate_count": 9,
        "hole_count": 0,
    }
    wrapper.eval()
    output = GeoRouteBackboneWrapper._forward_official_fixed_support(
        wrapper,
        torch.zeros((1, 1, 3, 4, 1, 1), dtype=torch.uint8),
        None,
    )
    assert output.shape == (1, 3, 4)
    assert backbone.native_ragged_forward_invocations == 1
    assert wrapper.latest_georoute_audit["packed"]["padded_heavy_tokens_per_window"] == 0
    assert wrapper.latest_georoute_audit["strict_rectangle"]["dummy_tokens_used"] is False


def test_r1_target_checkpoint_roundtrip_full_state_and_lineage(tmp_path):
    if os.environ.get("ZOOMTOKEN_R1_RECOVERY_RUNTIME_CHECK") != "1":
        return

    import torch

    from opentad.utils.checkpoint import save_checkpoint

    class Ema:
        def __init__(self, module):
            self.module = module

    class StateHolder:
        def __init__(self, value):
            self.value = value

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state):
            self.value = state["value"]

    model = torch.nn.Linear(2, 2)
    ema_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        ema_model.weight.fill_(7.0)
        ema_model.bias.fill_(3.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = StateHolder(11)
    lineage = {
        "schema_version": "zoomtoken_same_cell_recovery_v001",
        "arm_surface": "R1",
        "seed": 42,
        "source_commit": "7" * 40,
        "config_path": "/canonical/r1.py",
        "work_dir": str(tmp_path.resolve()),
        "completed_epoch": 59,
        "next_epoch": 60,
        "next_successful_update_index": 1234,
    }
    recovery_lineage = dict(
        lineage,
        completed_epoch=4,
        next_epoch=5,
        next_successful_update_index=321,
    )
    save_checkpoint(
        model,
        Ema(ema_model),
        optimizer,
        scheduler,
        4,
        work_dir=str(tmp_path),
        scaler=scaler,
        training_state=recovery_lineage,
        checkpoint_role="recovery",
        recovery_keep_latest=3,
    )
    recovery = torch.load(
        tmp_path / "checkpoint/recovery_epoch_4.pth",
        map_location="cpu",
    )
    assert recovery["checkpoint_role"] == "recovery"
    assert recovery["training_state"] == recovery_lineage
    assert {
        "state_dict",
        "state_dict_ema",
        "optimizer",
        "scheduler",
        "scaler",
    }.issubset(recovery)

    save_checkpoint(
        model,
        Ema(ema_model),
        optimizer,
        scheduler,
        59,
        work_dir=str(tmp_path),
        scaler=scaler,
        training_state=lineage,
        checkpoint_role="final",
    )
    checkpoint_dir = tmp_path / "checkpoint"
    full = torch.load(checkpoint_dir / "epoch_59.pth", map_location="cpu")
    ema = torch.load(checkpoint_dir / "final_ema.pth", map_location="cpu")
    raw = torch.load(checkpoint_dir / "final_raw.pth", map_location="cpu")
    assert {
        "state_dict",
        "state_dict_ema",
        "optimizer",
        "scheduler",
        "scaler",
        "training_state",
    }.issubset(full)
    assert full["checkpoint_role"] == "final"
    assert ema["checkpoint_role"] == "final_ema"
    assert raw["checkpoint_role"] == "final_raw"
    assert full["training_state"] == ema["training_state"] == raw["training_state"] == lineage
    assert torch.equal(ema["state_dict_ema"]["weight"], full["state_dict_ema"]["weight"])
    assert torch.equal(raw["state_dict"]["weight"], full["state_dict"]["weight"])

    restored_model = torch.nn.Linear(2, 2)
    restored_ema = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    restored_scaler = StateHolder(0)
    restored_model.load_state_dict(full["state_dict"])
    restored_ema.load_state_dict(full["state_dict_ema"])
    restored_optimizer.load_state_dict(full["optimizer"])
    restored_scheduler.load_state_dict(full["scheduler"])
    restored_scaler.load_state_dict(full["scaler"])
    assert restored_scaler.value == 11
    assert torch.equal(restored_ema.weight, ema_model.weight)

    try:
        save_checkpoint(
            model,
            Ema(ema_model),
            optimizer,
            scheduler,
            59,
            work_dir=str(tmp_path),
            scaler=scaler,
            training_state=lineage,
            checkpoint_role="final",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("R1 final artifacts must be immutable")
    after = torch.load(checkpoint_dir / "final_ema.pth", map_location="cpu")
    assert torch.equal(after["state_dict_ema"]["weight"], ema["state_dict_ema"]["weight"])
