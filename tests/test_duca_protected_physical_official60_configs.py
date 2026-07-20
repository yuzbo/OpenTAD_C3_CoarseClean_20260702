from __future__ import annotations

import copy
from pathlib import Path

from mmengine.config import Config

import opentad.datasets  # noqa: F401 - registers OpenTAD data transforms.
from opentad.models import build_detector
from opentad.models.selectors.duca_protected_e2e_frame_selector import (
    DucaProtectedE2EFrameSelector,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"
CONTRACT = "duca_protected_e2e_physical_v1"
CONFIGS = {
    "exact_uniform": "duca_protected_physical_exact_uniform_fixed384_official60.py",
    "transition_no_bridge": "duca_protected_physical_transition_no_bridge_fixed384_official60.py",
    "protected_e2e": "duca_protected_physical_e2e_fixed384_official60.py",
    "protected_e2e_rho001": "duca_protected_physical_e2e_rho001_fixed384_official60.py",
}


def _load_all():
    return {
        arm: Config.fromfile(str(CONFIG_ROOT / filename))
        for arm, filename in CONFIGS.items()
    }


def _normalized_model(cfg):
    model = copy.deepcopy(cfg.model.to_dict())
    selector = model["frame_selector"]
    selector.pop("arm")
    source = selector.get("actionness_source_cfg")
    if source is None:
        selector["actionness_source_cfg"] = "ARM_CONTROLLED_COARSE_SOURCE"
    else:
        source = dict(source)
        source.pop("policy_hidden_gradient_scope")
        selector["actionness_source_cfg"] = "ARM_CONTROLLED_COARSE_SOURCE"
    return model


def test_four_arm_configs_have_one_shared_training_and_detector_protocol():
    configs = _load_all()
    reference = configs["protected_e2e"]
    for arm, cfg in configs.items():
        assert cfg.model.frame_selector.arm == arm
        assert cfg.duca_variant_contract.variant == arm
        assert cfg.workflow.end_epoch == 60
        assert cfg.workflow.checkpoint_interval == 5
        assert cfg.workflow.val_eval_interval == -1
        assert cfg.workflow.primary_checkpoint_epoch == 59
        assert cfg.workflow.primary_checkpoint_state_key == "state_dict_ema"
        assert cfg.workflow.seal_eval_dataloaders_during_training is True
        assert cfg.workflow.derive_train_loader_contract is True
        assert cfg.solver.static_graph is False
        assert cfg.solver.find_unused_parameters is True
        assert cfg.model.backbone.backbone.total_frames == 384
        assert cfg.model.backbone.backbone.with_cp is False
        assert cfg.model.projection.max_seq_len == 384
        assert cfg.model.rpn_head.physical_grid_actionformer.contract == CONTRACT
        assert cfg.model.rpn_head.type == "ActionFormerHead"
        assert cfg.model.frame_selector.type == "DucaProtectedE2EFrameSelector"
        assert cfg.model.frame_selector.budget == 384
        assert cfg.model.frame_selector.dense_window_size == 768
        assert cfg.model.frame_selector.coarse_hidden_dim == 96
        assert cfg.model.frame_selector.selector_hidden_dim == 64
        train_pipeline = cfg.dataset.train.pipeline
        load_frames = next(
            item for item in train_pipeline if item["type"] == "LoadFrames"
        )
        assert load_frames["emit_boundary_validity"] is True
        collect_train = next(
            item for item in train_pipeline if item["type"] == "Collect"
        )
        assert "gt_boundary_validity" in collect_train["keys"]
        assert cfg.dataset.to_dict() == reference.dataset.to_dict()
        assert cfg.optimizer.to_dict() == reference.optimizer.to_dict()
        assert cfg.scheduler.to_dict() == reference.scheduler.to_dict()
        assert _normalized_model(cfg) == _normalized_model(reference)


def test_data_pipeline_exposes_physical_axis_without_selected_axis_targets():
    configs = _load_all()
    for cfg in configs.values():
        assert cfg.dataset.val is None
        for split in ("train", "test"):
            pipeline = cfg.dataset[split].pipeline
            collect = next(item for item in pipeline if item["type"] == "Collect")
            assert "frame_inds" in collect["meta_keys"]
            assert "avg_fps" in collect["meta_keys"]
            assert "selected_dense_indices" not in collect["meta_keys"]
        assert cfg.duca_protected_physical_contract.selected_axis_gt_remap is False
        assert cfg.duca_protected_physical_contract.detector_axis == (
            "native_dense_physical_candidate_axis"
        )
        assert (
            cfg.duca_protected_physical_contract.backbone_tail_padding
            == "replicate_last_selected"
        )


def test_each_config_builds_the_registered_final_selector_and_official_head():
    configs = _load_all()
    for arm, cfg in configs.items():
        build_cfg = copy.deepcopy(cfg.model)
        # This structural test must not silently depend on a machine-local
        # checkpoint. Formal CUDA gates bind and hash the real pretrained file.
        build_cfg.backbone.custom.pretrain = None
        model = build_detector(build_cfg)
        selector = model.frame_selector
        assert isinstance(selector, DucaProtectedE2EFrameSelector)
        assert selector.arm == arm
        assert model.rpn_head.protected_physical_grid is True
        assert model.rpn_head.physical_grid_contract == CONTRACT
        if arm == "exact_uniform":
            assert selector.raw_actionness_source is None
            assert selector.transition_scorer is None
        else:
            assert selector.raw_actionness_source is not None
            assert selector.transition_scorer is not None
            assert selector.raw_actionness_source.probe_model == "official-action-seg"
            assert (
                selector.raw_actionness_source._provenance_override[
                    "official_action_seg_backend"
                ]
                == "official_asformer"
            )
        expected_scope = (
            "asformer_last_encoder_layer"
            if arm == "protected_e2e_rho001"
            else "none"
        )
        if selector.raw_actionness_source is not None:
            assert (
                selector.raw_actionness_source.policy_hidden_gradient_scope
                == expected_scope
            )
