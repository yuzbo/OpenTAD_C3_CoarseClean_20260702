from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll is unstable; Linux runs this gate", allow_module_level=True)

import torch
import torch.nn as nn
from mmengine.config import Config

from opentad.datasets.transforms.end_to_end import LoadFrames
from opentad.models.duca.acquisition import SparseTemporalGrid
from opentad.models.selectors.duca_online_frame_selector import (
    DucaOnlineFrameSelector,
    _DEFAULT_METADATA_KEYS,
)
from tools.bata.validate_duca_cellcf_fixed384 import validate_config


CONFIG_ROOT = "configs/adatad/thumos"


def test_all_cellcf_configs_are_fail_closed_and_single_factor_matched() -> None:
    for variant in ("uniform", "transition_beta0", "cellcf"):
        assert validate_config(variant)["ok"] is True

    uniform = Config.fromfile(f"{CONFIG_ROOT}/duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py")
    beta0 = Config.fromfile(f"{CONFIG_ROOT}/duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py")
    utility = Config.fromfile(f"{CONFIG_ROOT}/duca_cellcf_fixed384_official_adatad_backend_full_train.py")
    for cfg in (uniform, beta0, utility):
        selector = cfg.model.frame_selector
        assert selector.acquisition_policy == "local_cell_deformation"
        assert selector.max_unselected_hole is None
        assert selector.detector_gradient_mode == "none"
        assert "policy_alpha" not in selector.loss_weight_schedule
        assert cfg.workflow.checkpoint_interval == 5
        assert cfg.workflow.end_epoch == 132
    assert uniform.model.frame_selector.local_cell_force_exact_uniform is True
    assert beta0.model.frame_selector.local_cell_force_exact_uniform is False
    assert utility.model.frame_selector.local_cell_force_exact_uniform is False
    assert uniform.model.frame_selector.counterfactual_utility_distillation_weight == 0.0
    assert beta0.model.frame_selector.counterfactual_utility_distillation_weight == 0.0
    assert utility.model.frame_selector.counterfactual_utility_distillation_weight > 0.0


def test_canonical_predecode_uniform_matches_round_endpoint_contract() -> None:
    transform = LoadFrames(method="exact_uniform_fixed_subsample")
    for valid_len in range(1, 769):
        positions = transform._canonical_exact_uniform_positions(valid_len, 384)
        assert positions.shape[0] == min(valid_len, 384)
        assert positions[0] == 0
        assert positions[-1] == valid_len - 1
        assert len(set(int(value) for value in positions.tolist())) == positions.shape[0]


def test_acquisition_positions_are_separate_from_fixed_detector_grid_geometry() -> None:
    selector = object.__new__(DucaOnlineFrameSelector)
    nn.Module.__init__(selector)
    selector.metadata_keys = dict(_DEFAULT_METADATA_KEYS)
    selector.selected_positions_unit = "original_time_index"
    selector.detector_output_coordinate_space = "selected_axis_index"
    actual = torch.tensor([[0, 4, 8, 11]], dtype=torch.long)
    anchors = torch.tensor([[0, 3, 7, 11]], dtype=torch.long)
    dense_mask = torch.zeros((1, 12), dtype=torch.bool)
    dense_mask[0, actual[0]] = True
    grid = SparseTemporalGrid(
        selected_positions=actual,
        selected_mask=dense_mask,
        original_length=12,
        valid_len=torch.tensor([12]),
        budget=4,
        requested_budget=torch.tensor([4]),
        effective_budget=torch.tensor([4]),
        detector_input_length=torch.tensor([4]),
    ).validate()
    meta = selector._write_metas(
        [{}],
        grid,
        detector_grid_positions=anchors,
        actionness_source_name="synthetic",
    )[0]
    assert meta["duca_online_selected_positions"] == [0, 4, 8, 11]
    assert meta["duca_acquisition_positions"] == [0, 4, 8, 11]
    assert meta["duca_detector_grid_positions"] == [0, 3, 7, 11]
    assert meta["selected_axis_to_true_time_dense_index"] == [0, 3, 7, 11]
    assert meta["duca_online_selected_axis_remap"]["acquisition_positions"] == [0, 4, 8, 11]
