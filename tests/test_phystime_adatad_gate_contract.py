import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from opentad.datasets.transforms.end_to_end import LoadFrames
from opentad.datasets.transforms.formatting import Collect


ROOT = Path(__file__).resolve().parents[1]
GATE_TOOL = ROOT / "tools" / "bata" / "run_phystime_adatad_real_gate.py"
GATE_LAUNCHER = ROOT / "scripts" / "run_phystime_adatad_gate_gpu1.sh"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("phystime_adatad_real_gate", GATE_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_report():
    variant = {
        "decoded_frame_count": 384,
        "valid_observation_count": 384,
        "backbone_feature_length": 384,
        "inference_backbone_feature_length": 384,
        "adapter_gradient_nonzero": True,
        "detector_gradient_nonzero": True,
        "finite_loss": True,
        "finite_predictions": True,
        "optimizer_coverage": True,
    }
    return {
        "schema_version": "phystime_adatad_real_gate_v1",
        "gate_pass": True,
        "input_source": "raw_thumos_mp4",
        "logical_window": 768,
        "decoded_frame_count": 384,
        "backbone_feature_length": 384,
        "selected_index_checksum_match": True,
        "adapter_gradient_nonzero": True,
        "projection_gradient_nonzero": True,
        "classification_gradient_nonzero": True,
        "regression_gradient_nonzero": True,
        "endpoint_gradient_nonzero": True,
        "prediction_time_unit": "seconds",
        "uses_preextracted_features": False,
        "variants": {
            "selected_axis": dict(variant),
            "physical_grid": dict(variant),
            "phystime": {
                **variant,
                "projection_gradient_nonzero": True,
                "classification_gradient_nonzero": True,
                "regression_gradient_nonzero": True,
                "endpoint_gradient_nonzero": True,
            },
        },
    }


def test_gate_requires_raw_video_and_all_three_configs():
    module = _load_gate_module()
    assert set(module.GATE_CONFIGS) == {"selected_axis", "physical_grid", "phystime"}
    assert module.GATE_CONFIGS["selected_axis"].name == "selected_axis_adatad_sparse_k384.py"
    assert module.GATE_CONFIGS["physical_grid"].name == "physical_grid_adatad_sparse_k384.py"
    assert module.GATE_CONFIGS["phystime"].name == "phystime_adatad_sparse_k384.py"

    source = GATE_TOOL.read_text(encoding="utf-8")
    assert "build_dataset" in source
    assert "build_detector" in source
    assert "build_optimizer" in source
    assert "DecordDecode" in source
    assert "losses[\"cost\"].backward()" in source
    assert "forward_test" in source
    assert "LoadFeats" in source


def test_random_fixed_sampling_exposes_auditable_raw_frame_indices():
    loader = LoadFrames(
        method="random_fixed_subsample",
        method_base="sliding_window",
        keep_ratio=0.5,
        target_len=384,
        source_len=768,
        remap_gt_to_selected_axis=False,
    )
    results = loader(
        {
            "video_name": "gate_sample",
            "total_frames": 4000,
            "avg_fps": 30.0,
            "snippet_stride": 4,
            "window_size": 768,
            "feature_start_idx": 20,
            "feature_end_idx": 787,
        }
    )
    valid_count = int(results["masks"].sum().item())
    expected = results["frame_inds"][:valid_count].astype(int).tolist()
    assert results["selected_raw_frame_indices"] == expected

    results["imgs"] = torch.zeros(1, 3, 384, 1, 1)
    meta = Collect(inputs="imgs", keys=["masks"])(results)["metas"]
    assert meta["selected_raw_frame_indices"] == expected


def test_gate_report_validation_is_fail_closed():
    module = _load_gate_module()
    report = _valid_report()
    module.validate_gate_report(report)

    for key in (
        "selected_index_checksum_match",
        "adapter_gradient_nonzero",
        "projection_gradient_nonzero",
        "classification_gradient_nonzero",
        "regression_gradient_nonzero",
        "endpoint_gradient_nonzero",
    ):
        broken = copy.deepcopy(report)
        broken[key] = False
        with pytest.raises(RuntimeError, match=key):
            module.validate_gate_report(broken)

    broken = copy.deepcopy(report)
    broken["variants"]["physical_grid"]["backbone_feature_length"] = 768
    with pytest.raises(RuntimeError, match="physical_grid.backbone_feature_length"):
        module.validate_gate_report(broken)


def test_gate_launcher_requires_gpu1_raw_paths_and_checkpoint():
    text = GATE_LAUNCHER.read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]]' in text
    assert "OPENTAD_THUMOS14_ANNOTATION" in text
    assert "OPENTAD_THUMOS14_CLASS_MAP" in text
    assert "OPENTAD_THUMOS14_TRAIN_VIDEOS" in text
    assert "OPENTAD_THUMOS14_TEST_VIDEOS" in text
    assert "PHYSTIME_VIDEOMAE_CHECKPOINT" in text
    assert "SLURM_JOB_ID" in text
    assert "run_phystime_adatad_real_gate.py" in text
    assert "PHYSTIME_FEATURE_PATH" not in text
    assert "LoadFeats" not in text
