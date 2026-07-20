import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mmengine.config import Config, ConfigDict

_TORCH_PROBE = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
if _TORCH_PROBE.returncode != 0:
    pytest.skip(
        "PyTorch runtime is unavailable in this Python environment",
        allow_module_level=True,
    )

import torch

from opentad.cores.phystime_decode_replay_capture import (
    build_decode_replay_collector,
    decode_replay_effective_config_sha256,
)
import tools.bata.replay_phystime_decode_cross as replay
import tools.bata.validate_phystime_decode_cross_replay as validator


ROOT = Path(__file__).resolve().parents[1]
_ACTIONFORMER_HELPER = None


def _load_actionformer_test_helper():
    global _ACTIONFORMER_HELPER
    if _ACTIONFORMER_HELPER is not None:
        return _ACTIONFORMER_HELPER
    path = ROOT / "tests" / "test_c3_physical_grid_actionformer_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "phystime_decode_cross_actionformer_helper",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ACTIONFORMER_HELPER = module
    return module


def _make_capture_head():
    helper = _load_actionformer_test_helper()
    return helper._make_head(
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            eps=1.0e-6,
            positions_key="phystime_g1a_axis_positions_sec",
            selected_count_keys=["phystime_native_valid_count"],
            axis_start_key="phystime_g1a_axis_start_sec",
            axis_end_key="phystime_g1a_axis_end_sec",
        )
    )


def _capture_meta(coordinate_mode="physical_time_seconds"):
    physical = [0.0, 3.0, 10.0]
    uniform = [0.0, 5.0, 10.0]
    return {
        "video_name": "synthetic",
        "duration": 10.0,
        "prediction_time_unit": "seconds",
        "phystime_native_coordinate_mode": coordinate_mode,
        "phystime_native_valid_count": 3,
        "phystime_native_token_count": 3,
        "phystime_raw_observation_count": 6,
        "phystime_uniform_rank_timestamps_sec": uniform,
        "phystime_native_token_timestamps_sec": physical,
        "phystime_g1a_axis_positions_sec": (
            physical
            if coordinate_mode == "physical_time_seconds"
            else uniform
        ),
        "phystime_g1a_axis_start_sec": 0.0,
        "phystime_g1a_axis_end_sec": 10.0,
        "phystime_selected_raw_frame_indices": [0, 2, 4, 6, 8, 10],
        "phystime_raw_selected_dense_indices": [0, 2, 4, 6, 8, 10],
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
    }


def _features_and_mask():
    return [torch.zeros(1, 2, 4)], [
        torch.tensor([[True, True, True, False]])
    ]


def test_capture_is_opt_in_and_does_not_change_forward_outputs():
    head = _make_capture_head()
    features, masks = _features_and_mask()
    ordinary_proposals, ordinary_scores = head.forward_test(
        features,
        masks,
        metas=[_capture_meta()],
    )
    with pytest.raises(RuntimeError, match="not enabled"):
        head.consume_decode_replay_state()

    head.enable_decode_replay_capture(True)
    captured_proposals, captured_scores = head.forward_test(
        features,
        masks,
        metas=[_capture_meta()],
    )
    assert torch.equal(ordinary_proposals[0], captured_proposals[0])
    assert torch.equal(ordinary_scores[0], captured_scores[0])
    state = head.consume_decode_replay_state()
    assert state["cls_logits"].shape == (1, 4, 2)
    assert state["reg_distances"].shape == (1, 4, 2)
    assert state["native_proposals"].shape == (1, 4, 2)
    assert state["base_mask"].tolist() == [[True, True, True, False]]
    assert state["native_mask"].tolist() == [[True, True, True, False]]
    assert state["metadata"][0]["video_name"] == "synthetic"
    assert state["source_tensor_dtypes"]["cls_logits"].startswith("torch.")
    head.enable_decode_replay_capture(False)


def test_capture_refuses_to_overwrite_unconsumed_batch():
    head = _make_capture_head()
    features, masks = _features_and_mask()
    head.enable_decode_replay_capture(True)
    head.forward_test(features, masks, metas=[_capture_meta()])
    with pytest.raises(RuntimeError, match="previous batch"):
        head.forward_test(features, masks, metas=[_capture_meta()])
    head.consume_decode_replay_state()
    head.enable_decode_replay_capture(False)


def test_collector_writes_pickle_free_hashed_artifact(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "epoch_59.pth"
    checkpoint.write_bytes(b"frozen")
    monkeypatch.setenv("PHYSTIME_EXPECTED_COMMIT", "a" * 40)
    monkeypatch.setenv("PHYSTIME_EXPECTED_TREE", "b" * 40)
    monkeypatch.setenv("PHYSTIME_SOURCE_COMMIT", "c" * 40)
    monkeypatch.setenv("PHYSTIME_SOURCE_TREE", "d" * 40)
    monkeypatch.setenv("PHYSTIME_CHECKPOINT_PATH", str(checkpoint))

    head = _make_capture_head()
    model = SimpleNamespace(rpn_head=head)
    cfg = Config(
        {
            "work_dir": str(tmp_path),
            "inference": {
                "phystime_decode_replay_capture": {
                    "enabled": True,
                    "train_axis": "physical_time_seconds",
                    "expected_native_coordinate_mode": (
                        "physical_time_seconds"
                    ),
                    "weights_source": "ema",
                }
            },
        }
    )
    collector = build_decode_replay_collector(
        model=model,
        cfg=cfg,
        external_cls=["a", "b"],
        world_size=1,
        rank=0,
        evaluation_epoch=59,
    )
    features, masks = _features_and_mask()
    head.forward_test(features, masks, metas=[_capture_meta()])
    collector.collect_latest_batch()
    artifact = collector.finalize()

    with np.load(artifact["artifact_path"], allow_pickle=False) as archive:
        assert set(archive.files) == replay.REQUIRED_ARRAYS
        assert archive["cls_logits"].dtype == np.float32
        assert archive["base_mask"].dtype == np.bool_
    manifest = json.loads(
        Path(artifact["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["checkpoint"]["sha256"] == replay.sha256_file(checkpoint)
    assert manifest["runtime"]["commit"] == "a" * 40
    assert manifest["source"]["commit"] == "c" * 40
    assert manifest["weights_source"] == "ema"
    assert manifest["window_count"] == 1
    assert manifest["capture_memory"]["within_budget"] is True
    assert (
        manifest["capture_memory"]["estimated_peak_tensor_bytes"]
        <= manifest["capture_memory"]["max_in_memory_bytes"]
    )
    assert manifest["source_tensor_dtypes"]["cls_scores"].startswith("torch.")
    assert len(manifest["observation_sequence_sha256"]) == 64
    assert (
        replay.canonical_sha256(
            [manifest["windows"][0]["observation_binding_sha256"]]
        )
        == manifest["observation_sequence_sha256"]
    )


def _synthetic_dense_arrays():
    base_points = np.asarray(
        [
            [0.0, 0.0, 100.0, 1.0],
            [1.0, 0.0, 100.0, 1.0],
            [2.0, 0.0, 100.0, 1.0],
            [3.0, 0.0, 100.0, 1.0],
        ],
        dtype=np.float32,
    )
    axis = np.asarray([[0.0, 2.0, 5.0, np.nan]], dtype=np.float32)
    points = replay.build_axis_points(
        torch.from_numpy(base_points),
        torch.from_numpy(axis[0, :3].copy()),
        0.0,
        6.0,
    )
    reg = torch.tensor(
        [[[0.25, 0.5], [0.5, 0.25], [1.0, 1.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    proposals = torch.stack(
        (
            points[:, 0] - reg[0, :, 0] * points[:, 3],
            points[:, 0] + reg[0, :, 1] * points[:, 3],
        ),
        dim=-1,
    ).unsqueeze(0)
    return {
        "base_points": base_points,
        "reg_distances": reg.numpy(),
        "base_mask": np.asarray([[True, True, True, True]]),
        "native_mask": np.asarray([[True, True, True, False]]),
        "native_points": points.unsqueeze(0).numpy(),
        "native_proposals": proposals.numpy(),
        "cls_scores": np.full((1, 4, 2), 0.5, dtype=np.float32),
        "uniform_axis_sec": axis.copy(),
        "physical_axis_sec": axis.copy(),
        "native_valid_count": np.asarray([3], dtype=np.int32),
        "domain_sec": np.asarray([[0.0, 6.0]], dtype=np.float64),
    }


def test_identical_axes_produce_identical_dense_replay():
    arrays = _synthetic_dense_arrays()
    uniform = replay.decode_axis(
        arrays,
        "uniform_rank_seconds",
        "uniform_rank_seconds",
    )
    physical = replay.decode_axis(
        arrays,
        "physical_time_seconds",
        "uniform_rank_seconds",
    )
    assert np.array_equal(
        uniform["dense_proposals"],
        physical["dense_proposals"],
    )
    recomputed, mask, _ = validator.recompute_dense_decode(
        arrays,
        "physical_time_seconds",
    )
    assert np.allclose(
        recomputed,
        physical["dense_proposals"],
        rtol=0.0,
        atol=1.0e-6,
    )
    assert np.array_equal(mask, arrays["native_mask"])
    assert physical["uses_captured_production_proposals"] is False


def test_native_capture_is_audit_reference_not_decode_substitution():
    arrays = _synthetic_dense_arrays()
    arrays["native_proposals"] = arrays["native_proposals"].copy()
    arrays["native_proposals"][0, 0, 0] += 5.0e-5
    decoded = replay.decode_axis(
        arrays,
        "uniform_rank_seconds",
        "uniform_rank_seconds",
    )
    assert not np.array_equal(
        decoded["dense_proposals"],
        arrays["native_proposals"],
    )
    assert (
        decoded["native_proposal_reconstruction_max_abs_error"]
        == pytest.approx(5.0e-5, abs=1.0e-7)
    )
    assert decoded["uses_captured_production_proposals"] is False


def test_collector_fails_closed_on_memory_budget(tmp_path, monkeypatch):
    checkpoint = tmp_path / "epoch_59.pth"
    checkpoint.write_bytes(b"frozen")
    monkeypatch.setenv("PHYSTIME_EXPECTED_COMMIT", "a" * 40)
    monkeypatch.setenv("PHYSTIME_EXPECTED_TREE", "b" * 40)
    monkeypatch.setenv("PHYSTIME_SOURCE_COMMIT", "c" * 40)
    monkeypatch.setenv("PHYSTIME_SOURCE_TREE", "d" * 40)
    monkeypatch.setenv("PHYSTIME_CHECKPOINT_PATH", str(checkpoint))
    head = _make_capture_head()
    collector = build_decode_replay_collector(
        model=SimpleNamespace(rpn_head=head),
        cfg=Config(
            {
                "work_dir": str(tmp_path),
                "inference": {
                    "phystime_decode_replay_capture": {
                        "enabled": True,
                        "train_axis": "physical_time_seconds",
                        "expected_native_coordinate_mode": (
                            "physical_time_seconds"
                        ),
                        "weights_source": "ema",
                        "max_in_memory_bytes": 1,
                    }
                },
            }
        ),
        external_cls=["a", "b"],
        world_size=1,
        rank=0,
        evaluation_epoch=59,
    )
    features, masks = _features_and_mask()
    head.forward_test(features, masks, metas=[_capture_meta()])
    with pytest.raises(RuntimeError, match="capture budget exceeded"):
        collector.collect_latest_batch()


def test_decode_replay_configs_change_only_capture_contract():
    selected = Config.fromfile(
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "phystime_g1a_selected_axis_native_j192_decode_replay.py",
        lazy_import=False,
    )
    physical = Config.fromfile(
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "phystime_g1a_physical_metric_native_j192_decode_replay.py",
        lazy_import=False,
    )
    assert (
        selected.inference.phystime_decode_replay_capture.train_axis
        == "uniform_rank_seconds"
    )
    assert (
        physical.inference.phystime_decode_replay_capture.train_axis
        == "physical_time_seconds"
    )
    for cfg in (selected, physical):
        capture = cfg.inference.phystime_decode_replay_capture
        assert capture.enabled is True
        assert capture.weights_source == "must_be_overridden"
        assert capture.max_in_memory_bytes == 8589934592
        assert cfg.post_processing.round_before_cross_window_nms is False
        assert cfg.post_processing.round_after_cross_window_nms is False
        assert cfg.post_processing.filter_invalid_proposals is True


def test_effective_config_hash_ignores_only_runtime_derived_fields():
    base = Config(
        {
            "inference": {"load_from_raw_predictions": False},
            "post_processing": {"nms": {"iou_threshold": 0.4}},
            "model": {"type": "Detector"},
        }
    )
    runtime = Config(base.to_dict())
    runtime.inference.folder = "/tmp/dynamic-output"
    runtime.post_processing.sliding_window = True
    assert decode_replay_effective_config_sha256(
        base
    ) == decode_replay_effective_config_sha256(runtime)

    changed = Config(base.to_dict())
    changed.post_processing.nms.iou_threshold = 0.5
    assert decode_replay_effective_config_sha256(
        base
    ) != decode_replay_effective_config_sha256(changed)
