from __future__ import annotations

import copy
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.run_georoute_dynamic_stage1_p0 import (
    REQUIRED_GRADIENT_COMPONENTS,
    REQUIRED_LOSS_KEYS,
    build_dynamic_stage1_p0_report,
    validate_dynamic_stage1_p0_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_payload():
    losses = {name: 0.25 for name in REQUIRED_LOSS_KEYS}
    losses["georoute_geometry_regularization_loss"] = 0.0
    return {
        "schema_version": "georoute_dynamic_stage1_cuda_p0_v1",
        "status": "PASS_NO_PERFORMANCE_P0",
        "source": {
            "commit": "a" * 40,
            "expected_commit": "a" * 40,
            "origin_ref": "a" * 40,
            "head_matches_expected": True,
            "origin_ref_matches_expected": True,
            "tree_clean": True,
        },
        "slurm": {
            "job_id": "123",
            "logical_device": "cuda:0",
            "visible_device_count": 1,
        },
        "config": {
            "native_spatial_candidates": 4,
        },
        "losses": losses,
        "gradient": {
            "all_gradients_finite": True,
            "required_components": sorted(REQUIRED_GRADIENT_COMPONENTS),
            "nonzero_components": sorted(REQUIRED_GRADIENT_COMPONENTS),
            "missing_required_components": [],
        },
        "amp": {
            "autocast_dtype": "torch.float16",
            "loss_scale_before": 256.0,
            "loss_scale_after": 256.0,
            "optimizer_update_succeeded": True,
        },
        "backbone_audit": {
            "routing_schema": "georoute_dynamic_global_routing_v2",
            "route_mode": "dynamic_scnr",
            "policy_estimator": "straight_through",
            "roi_modifier_geometry": (
                "signed_ellipse_with_semiaxes_half_decoded_full_extent"
            ),
            "scout_policy_stop_gradient": True,
            "proxy_inference_enabled": False,
            "proxy_updates_scout_stem": False,
            "proxy_updates_heavy_backbone": False,
            "window_budget_is_global": True,
            "independent_count_head": False,
            "fixed_context_quota": False,
            "fixed_per_tubelet_k": False,
            "k_t_allows_zero": True,
            "zero_carrier_mode": "masked_zero",
            "heavy_valid_mask_matches_k_t": True,
            "uses_gt_for_route": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
            "window_token_budget": 3,
            "requested_physical_tokens_per_window": 3,
            "unique_physical_tokens_per_window": 3,
            "executed_patch_tokens_per_window": 3,
            "padded_heavy_tokens_per_window": 0,
            "heavy_backbone_forward_count": 1,
            "k_per_tubelet": [[3, 0]],
            "role_counts_per_window": [[1, 1, 1]],
            "proxy_soft_budget_sum": [3.0],
            "output_shape": [1, 384, 768],
            "packed": {
                "schema_version": "videomae_native_ragged_v1",
                "execution_mode": "true_clip_ragged_no_padding",
                "adapter_execution": "coordinate_lineage_true_ragged",
                "padded_heavy_tokens_per_window": 0,
                "executed_patch_tokens_per_window": 3,
                "dense_adapter_forward_count": 0,
                "clip_token_counts": [[3]],
                "attention_pairs_per_window": [9],
            },
        },
        "scope": {
            "synthetic_inputs_only": True,
            "dataset_loaded": False,
            "metric_computed": False,
            "prediction_written": False,
            "checkpoint_written": False,
            "performance_claim_allowed": False,
            "official_test_opened": False,
        },
    }


def test_dynamic_stage1_p0_report_accepts_exact_no_performance_receipt():
    validate_dynamic_stage1_p0_report(
        build_dynamic_stage1_p0_report(_valid_payload())
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("backbone_audit", "unique_physical_tokens_per_window"),
            2,
            "exact global B",
        ),
        (
            ("backbone_audit", "padded_heavy_tokens_per_window"),
            1,
            "padding or dummy",
        ),
        (
            ("backbone_audit", "packed", "attention_pairs_per_window"),
            [8],
            "sum_c b_c",
        ),
        (("scope", "metric_computed"), True, "no-performance scope"),
    ],
)
def test_dynamic_stage1_p0_report_rejects_false_evidence(path, value, message):
    payload = copy.deepcopy(_valid_payload())
    destination = payload
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_dynamic_stage1_p0_report(
            build_dynamic_stage1_p0_report(payload)
        )


def test_dynamic_stage1_config_is_global_budget_support_only():
    cfg = Config.fromfile(
        str(
            ROOT
            / "configs"
            / "adatad"
            / "thumos"
            / "georoute_dynamic_scnr_stage1_base.py"
        )
    )
    custom = cfg.model.backbone.custom
    assert custom.georoute_route_mode == "dynamic_scnr"
    assert custom.georoute_window_token_budget == 384 * 64
    assert custom.georoute_zero_carrier_mode == "masked_zero"
    assert custom.georoute_roi_extent_floor_mode == "native_cells"
    assert custom.georoute_absolute_coordinates_enabled is False
    assert custom.georoute_roi_relative_coordinates_enabled is False
    assert custom.georoute_geometry_projection_enabled is False
    assert custom.georoute_geometry_smoothness_weight == 0.0
    assert custom.georoute_area_prior_weight == 0.0
    assert cfg.workflow.require_successful_update_hook is True
    assert cfg.solver.static_graph is False


def test_dynamic_stage1_slurm_launcher_preserves_scheduler_gpu_mapping():
    text = (
        ROOT / "scripts" / "run_georoute_dynamic_stage1_p0_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in text
    assert "--device cuda:0" in text
    assert "CUDA_VISIBLE_DEVICES=" not in text
    assert "--mem=" not in text
    assert "run_georoute_dynamic_stage1_p0" in text
