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


def _block_indices(top_row, top_col):
    return [
        row * 10 + col
        for row in range(top_row, top_row + 8)
        for col in range(top_col, top_col + 8)
    ]


def _function_source(path, name):
    source = path.read_text()
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, node)


def test_r1_has_exactly_nine_complete_hole_free_row_major_blocks():
    blocks = [_block_indices(row, col) for row in (0, 1, 2) for col in (0, 1, 2)]
    assert len(blocks) == 9
    assert all(len(block) == len(set(block)) == 64 for block in blocks)
    assert blocks[0][:8] == list(range(8))
    assert blocks[4][:8] == list(range(11, 19))
    assert blocks[-1][-8:] == list(range(92, 100))
    for block, (top_row, top_col) in zip(
        blocks,
        ((row, col) for row in (0, 1, 2) for col in (0, 1, 2)),
    ):
        expected = {
            row * 10 + col
            for row in range(top_row, top_row + 8)
            for col in range(top_col, top_col + 8)
        }
        assert set(block) == expected


def test_r1_routing_source_is_categorical_not_token_topk():
    primitive = _function_source(ROUTING_PATH, "select_strict_rectangle_8x8")
    assert primitive is not None
    assert "center = 0.4 + 0.2 * torch.sigmoid" in primitive
    assert "/ (0.1**2)" in primitive
    assert "torch.argmax(block_logits, dim=-1)" in primitive
    assert "F.softmax(block_logits / float(temperature), dim=-1)" in primitive
    assert "hard_categorical + (probability - probability.detach())" in primitive
    assert "torch.matmul(" in primitive
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
    assert 'ZOOMTOKEN_RECOVERY_ARMS = {"DN", "G", "R1"}' in train
    assert '"state_dict": model.state_dict()' in checkpoint
    assert '"state_dict_ema": model_ema.module.state_dict()' in checkpoint
    assert '"training_state": dict(training_state)' in checkpoint
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
        torch.tensor(_block_indices(1, 1), dtype=torch.long),
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
        torch.tensor(_block_indices(1, 0), dtype=torch.long),
    )
