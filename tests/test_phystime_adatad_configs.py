import hashlib
import json
from pathlib import Path

import numpy as np
from mmengine.config import Config

from opentad.datasets.transforms.end_to_end import LoadFrames
from tools.bata.validate_phystime_adatad_track import validate_track


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "selected": ROOT / "configs/adatad/thumos/selected_axis_adatad_sparse_k384.py",
    "physical": ROOT / "configs/adatad/thumos/physical_grid_adatad_sparse_k384.py",
    "phystime": ROOT / "configs/adatad/thumos/phystime_adatad_sparse_k384.py",
}
SDPQ_FEATURE = ROOT / "configs/adatad/thumos/phystime_sdpq_i3d_feature_gate0b.py"
SDPQ_NATIVE = ROOT / "configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py"


def load_frame_step(cfg, split):
    return next(step for step in cfg.dataset[split].pipeline if step["type"] == "LoadFrames")


def pipeline_types(cfg, split):
    return [step["type"] for step in cfg.dataset[split].pipeline]


def _pipeline_step(cfg, split, step_type):
    return next(step for step in cfg.dataset[split].pipeline if step["type"] == step_type)


def checksum(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def test_sparse_configs_share_raw_sampling_contract():
    cfgs = {name: Config.fromfile(path) for name, path in CONFIGS.items()}

    assert {cfg.window_size for cfg in cfgs.values()} == {384}
    assert {cfg.dense_window_size for cfg in cfgs.values()} == {768}
    for split in ("train", "val", "test"):
        steps = {name: load_frame_step(cfg, split) for name, cfg in cfgs.items()}
        assert {step["method"] for step in steps.values()} == {"random_fixed_subsample"}
        assert {step["target_len"] for step in steps.values()} == {384}
        assert {step["keep_ratio"] for step in steps.values()} == {0.5}
        if split == "train":
            assert {step["source_len"] for step in steps.values()} == {768}
            assert {step["method_base"] for step in steps.values()} == {"random_trunc"}
        else:
            assert {step["method_base"] for step in steps.values()} == {"sliding_window"}

    assert load_frame_step(cfgs["selected"], "train")["remap_gt_to_selected_axis"] is True
    assert load_frame_step(cfgs["physical"], "train")["remap_gt_to_selected_axis"] is False
    assert load_frame_step(cfgs["phystime"], "train")["remap_gt_to_selected_axis"] is False


def test_sparse_configs_share_backbone_data_and_training_contracts():
    cfgs = {name: Config.fromfile(path) for name, path in CONFIGS.items()}

    assert {cfg.dataset.train.ann_file for cfg in cfgs.values()}.__len__() == 1
    assert {cfg.dataset.train.data_path for cfg in cfgs.values()}.__len__() == 1
    assert {cfg.dataset.val.data_path for cfg in cfgs.values()}.__len__() == 1
    assert {cfg.model.backbone.custom.pretrain for cfg in cfgs.values()}.__len__() == 1
    assert {cfg.model.backbone.backbone.total_frames for cfg in cfgs.values()} == {384}
    assert {cfg.model.backbone.custom.pre_processing_pipeline[0]["t1"] for cfg in cfgs.values()} == {24}
    assert {cfg.model.backbone.custom.post_processing_pipeline[-1]["size"] for cfg in cfgs.values()} == {384}
    assert {canonical(cfg.optimizer) for cfg in cfgs.values()}.__len__() == 1
    assert {canonical(cfg.workflow) for cfg in cfgs.values()}.__len__() == 1
    assert {canonical(cfg.post_processing) for cfg in cfgs.values()}.__len__() == 1


def test_sparse_configs_route_evaluation_to_the_runtime_annotation(monkeypatch, tmp_path):
    annotation = tmp_path / "runtime_thumos_annotations.json"
    monkeypatch.setenv("OPENTAD_THUMOS14_ANNOTATION", str(annotation))

    cfgs = {name: Config.fromfile(path) for name, path in CONFIGS.items()}

    assert {
        Path(cfg.evaluation.ground_truth_filename) for cfg in cfgs.values()
    } == {annotation}
    assert all(cfg.solver.fail_on_non_finite_grad is True for cfg in cfgs.values())
    assert all(cfg.solver.amp_init_scale == 1024.0 for cfg in cfgs.values())
    assert all(cfg.solver.max_consecutive_amp_skips == 4 for cfg in cfgs.values())
    assert all(cfg.solver.max_total_amp_skips_per_epoch == 8 for cfg in cfgs.values())
    assert all(cfg.solver.fp16_compress is False for cfg in cfgs.values())


def test_phystime_changes_only_geometry_projection_and_head():
    cfgs = {name: Config.fromfile(path) for name, path in CONFIGS.items()}

    assert cfgs["phystime"].model.type == "PhysTimeTAD"
    assert cfgs["phystime"].model.backbone.backbone.type == "VisionTransformerAdapter"
    assert cfgs["phystime"].model.projection.type == "PhysTimeMeasureProjection"
    assert cfgs["phystime"].model.rpn_head.type == "PhysTimeHead"
    assert cfgs["phystime"].model.discretization_loss_weight == 0.0
    assert cfgs["selected"].model.rpn_head.type == "ActionFormerHead"
    assert cfgs["physical"].model.rpn_head.type == "ActionFormerHead"
    assert cfgs["physical"].model.rpn_head.physical_grid_actionformer.enabled is True

    for split in ("train", "val", "test"):
        types = pipeline_types(cfgs["phystime"], split)
        assert types.index("BuildPhysTimeRawFrameGeometry") < types.index("mmaction.DecordDecode")
        assert "LoadFeats" not in types
        assert "BuildPhysTimeRawFrameGeometry" not in pipeline_types(cfgs["selected"], split)
        assert "BuildPhysTimeRawFrameGeometry" not in pipeline_types(cfgs["physical"], split)


def test_all_heads_select_identical_frames_for_same_window():
    sample_results = {
        "video_name": "video_test_0000001",
        "total_frames": 4000,
        "avg_fps": 30.0,
        "snippet_stride": 4,
        "window_size": 768,
        "feature_start_idx": 20,
        "feature_end_idx": 787,
    }
    selected = []
    for remap in (True, False, False):
        loader = LoadFrames(
            method="random_fixed_subsample",
            method_base="sliding_window",
            target_len=384,
            keep_ratio=0.5,
            remap_gt_to_selected_axis=remap,
        )
        selected.append(loader(dict(sample_results))["frame_inds"])

    assert len({checksum(item) for item in selected}) == 1


def test_track_validator_emits_one_shared_sampling_contract(tmp_path):
    output = tmp_path / "phystime_adatad_contract.json"

    payload = validate_track(config_paths=CONFIGS, output=output)

    assert payload["contract_pass"] is True
    assert payload["raw_video_only"] is True
    assert payload["target_len"] == 384
    assert payload["dense_window_size"] == 768
    assert len(payload["sampling_contract_sha256"]) == 64
    assert len(set(payload["resolved_config_sha256"].values())) == 3
    assert output.is_file()


def test_sdpq_configs_use_support_decoupled_head_and_native_g1b_has_no_interpolation():
    feature_cfg = Config.fromfile(SDPQ_FEATURE, lazy_import=False)
    native_cfg = Config.fromfile(SDPQ_NATIVE, lazy_import=False)

    for cfg in (feature_cfg, native_cfg):
        assert cfg.model.type == "PhysTimeTAD"
        assert cfg.model.projection.type == "PhysTimeMeasureProjection"
        assert cfg.model.projection.keep_uncovered_queries is True
        assert cfg.model.projection.use_null_evidence is True
        assert cfg.model.rpn_head.type == "SupportDecoupledPhysicalQueryHead"
        assert cfg.model.rpn_head.center_sample_radius == 2.0

    assert native_cfg.raw_observation_count == 384
    assert native_cfg.native_token_count == 192
    assert native_cfg.model.native_temporal_geometry.expected_raw_count == 384
    assert native_cfg.model.native_temporal_geometry.expected_token_count == 192
    post_types = [step["type"] for step in native_cfg.model.backbone.custom.post_processing_pipeline]
    assert post_types == ["Reduce", "Rearrange"]
    assert "Interpolate" not in post_types
    for split in ("train", "val", "test"):
        assert _pipeline_step(native_cfg, split, "BuildPhysTimeNativeTubeletGeometry")[
            "coordinate_mode"
        ] == "physical_time_seconds"
