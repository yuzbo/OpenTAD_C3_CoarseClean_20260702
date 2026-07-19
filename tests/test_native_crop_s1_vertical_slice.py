from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/adatad/thumos/native_crop_s1_center_videomae_s_768x1_adapter.py"
)


def coordinate_image(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(height, dtype=np.uint16),
        np.arange(width, dtype=np.uint16),
        indexing="ij",
    )
    return np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(
        np.uint8
    )


def test_source_native_crop_is_exact_and_never_interpolated():
    from opentad.datasets.transforms.native_crop import crop_source_native_uint8

    source = coordinate_image(180, 320)
    crop, record = crop_source_native_uint8(
        source, crop_size=128, allow_padding=False
    )
    x0, y0, x1, y1 = record["source_box_xyxy"]
    assert np.array_equal(crop, source[y0:y1, x0:x1])
    assert record["local_interpolation"] is False
    assert record["padding_ltrb"] == [0, 0, 0, 0]
    assert record["valid_pixel_fraction"] == 1.0


def test_source_native_crop_fails_closed_before_padding():
    from opentad.datasets.transforms.native_crop import crop_source_native_uint8

    source = coordinate_image(96, 160)
    with pytest.raises(ValueError, match="smaller"):
        crop_source_native_uint8(
            source, crop_size=128, allow_padding=False
        )
    padded, record = crop_source_native_uint8(
        source, crop_size=128, allow_padding=True
    )
    assert padded.shape == (128, 128, 3)
    assert record["padding_ltrb"] == [0, 0, 0, 32]
    assert np.array_equal(padded[:96], source[:, 16:144])


def test_transform_keeps_uint8_and_is_gt_invariant():
    from opentad.datasets.transforms.native_crop import NativeCropSourceViews

    transform = NativeCropSourceViews(global_size=96, local_size=128)
    frames = [coordinate_image(180, 320) for _ in range(4)]
    first = transform(
        {
            "imgs": frames,
            "gt_segments": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "gt_labels": np.asarray([0], dtype=np.int64),
        }
    )
    second = transform(
        {
            "imgs": frames,
            "gt_segments": np.asarray([[100.0, 200.0]], dtype=np.float32),
            "gt_labels": np.asarray([19], dtype=np.int64),
        }
    )
    assert first["native_crop_inputs"]["global"].shape == (1, 3, 4, 96, 96)
    assert first["native_crop_inputs"]["local"].shape == (1, 3, 4, 128, 128)
    assert first["native_crop_inputs"]["global"].dtype == np.uint8
    assert first["native_crop_inputs"]["local"].dtype == np.uint8
    assert np.array_equal(
        first["native_crop_inputs"]["local"],
        second["native_crop_inputs"]["local"],
    )
    geometry = first["native_crop_geometry"]
    assert geometry["source_hw"] == [180, 320]
    assert geometry["source_float_video_materialized"] is False
    assert geometry["uses_gt"] is False
    assert geometry["uses_teacher"] is False
    assert geometry["uses_oracle"] is False
    assert geometry["uses_test_evidence"] is False


def test_global_letterbox_retains_rectangular_source_context():
    from opentad.datasets.transforms.native_crop import letterbox_global_uint8

    source = coordinate_image(180, 320)
    global_view, record = letterbox_global_uint8(source, output_size=96)
    assert global_view.shape == (96, 96, 3)
    x0, y0, x1, y1 = record["content_box_xyxy"]
    assert (x1 - x0, y1 - y0) == (96, 54)
    assert y0 > 0
    assert record["global_interpolation"] == "bilinear"


def test_config_is_development_only_and_crop_precedes_resize():
    from opentad.utils.training_guard import assert_detector_training_allowed
    from tools.bata.run_native_crop_s1_precheck import validate_native_crop_config

    cfg = Config.fromfile(str(CONFIG_PATH))
    audit = validate_native_crop_config(cfg)
    assert audit["development_only"] is True
    assert audit["official_test_dataset_materialized"] is False
    assert cfg.dataset.test is None
    assert cfg.dataset.val.window_overlap_ratio == 0.5
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_gate_launcher_binds_commit_and_never_overrides_slurm_gpu():
    launcher = (
        ROOT / "scripts/run_native_crop_s1_gate_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "NATIVE_CROP_S1_EXPECTED_COMMIT" in launcher
    assert "git -C \"${ROOT}\" rev-parse HEAD" in launcher
    assert "--untracked-files=all" in launcher
    assert "ls-files --error-unmatch" in launcher
    assert "--expected-commit \"${EXPECTED_COMMIT}\"" in launcher
    assert "--geometry-census \"${OUT_ROOT}/geometry_census.json\"" in launcher
    assert "export CUDA_VISIBLE_DEVICES" not in launcher
    assert "CUDA_VISIBLE_DEVICES=" not in launcher


def test_precheck_cli_wires_geometry_census(monkeypatch, tmp_path):
    import tools.bata.run_native_crop_s1_precheck as precheck

    captured = {}

    def fake_run_full_precheck(**kwargs):
        captured.update(kwargs)
        return {
            "precheck_sha256": "a" * 64,
            "projection_input_shape": [1, 384, 768],
        }

    monkeypatch.setattr(precheck, "run_full_precheck", fake_run_full_precheck)
    output = tmp_path / "precheck.json"
    geometry = tmp_path / "geometry.json"
    assert (
        precheck.main(
            [
                "--config",
                str(CONFIG_PATH),
                "--expected-commit",
                "a" * 40,
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--geometry-census",
                str(geometry),
                "--annotation",
                str(tmp_path / "development.json"),
                "--class-map",
                str(tmp_path / "class_map.txt"),
                "--video-root",
                str(tmp_path / "videos"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured["geometry_census_path"] == geometry
    assert captured["expected_commit"] == "a" * 40
    assert output.is_file()


def test_cost_schema_requires_full_stack_measurement():
    from tools.bata.native_crop_s1_contract import (
        NATIVE_CROP_COST_STAGES,
        build_cost_schema,
    )

    schema = build_cost_schema(global_size=96, local_size=128)
    assert schema["measurement_status"] == "not_measured"
    assert schema["required_stages"] == list(NATIVE_CROP_COST_STAGES)
    assert schema["teacher_search_cost_separate"] is True
    assert schema["full_stack_claim_allowed"] is False
    assert schema["view_budget"]["total_view_pixels_per_frame"] == 25600


def test_geometry_summary_reports_padding_and_relative_view():
    from tools.bata.native_crop_s1_geometry_census import summarize_records

    records = [
        {"height": 180, "width": 320},
        {"height": 96, "width": 128},
    ]
    summary = summarize_records(records, [96, 128])
    assert summary["video_count"] == 2
    assert summary["crop_sizes"]["96"]["no_padding_rate"] == 1.0
    assert summary["crop_sizes"]["128"]["no_padding_rate"] == 0.5
    assert summary["crop_sizes"]["128"]["padding_count"] == 1
    single_narrow = summarize_records(
        [{"height": 96, "width": 320}],
        [128],
    )
    crop_128 = single_narrow["crop_sizes"]["128"]
    assert crop_128["valid_pixels_if_padded"]["median"] == 0.75
    assert crop_128["crop_area_over_source"]["median"] == 0.4


def test_census_source_reprobe_rejects_forged_geometry_and_replaced_file(
    monkeypatch,
    tmp_path,
):
    import tools.bata.run_native_crop_s1_precheck as precheck

    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "video_training_0000001.mp4"
    video_path.write_bytes(b"original")
    actual = {
        "width": 320,
        "height": 180,
        "rotation_degrees": 0,
        "nb_frames": "768",
        "avg_frame_rate": "30/1",
    }
    monkeypatch.setattr(
        precheck,
        "probe_video_geometry",
        lambda _path: dict(actual),
    )
    record = {
        "path": str(video_path.resolve()),
        "file_size_bytes": video_path.stat().st_size,
        **actual,
    }
    forged = dict(record, width=640)
    with pytest.raises(ValueError, match="differs from the current source"):
        precheck.audit_census_record_source(
            forged,
            expected_path=video_path,
            video_root=video_root,
        )
    video_path.write_bytes(b"replacement-with-different-size")
    with pytest.raises(ValueError, match="source size changed"):
        precheck.audit_census_record_source(
            record,
            expected_path=video_path,
            video_root=video_root,
        )


def test_geometry_manifest_rejects_a_resigned_foreign_split():
    from tools.bata.native_crop_s1_contract import (
        finalize_self_hash,
        validate_development_only_manifest,
    )

    manifest = finalize_self_hash({
        "splits": {
            "fit": ["fit_a"],
            "gate": ["gate_a"],
            "test": ["test_a"],
        },
    }, "manifest_sha256")
    with pytest.raises(ValueError, match="frozen S1 manifest"):
        validate_development_only_manifest(manifest)


def test_geometry_census_rejects_an_underspecified_self_signed_report(tmp_path):
    from tools.bata.native_crop_s1_contract import finalize_self_hash
    from tools.bata.run_native_crop_s1_precheck import audit_geometry_census

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    census_path = tmp_path / "census.json"
    payload = finalize_self_hash(
        {
            "manifest_path": str(manifest_path.resolve()),
            "video_root": str(video_root.resolve()),
            "sealed_test_files_probed": 0,
            "summary": {
                "combined": {
                    "video_count": 200,
                    "crop_sizes": {
                        "128": {
                            "no_padding_count": 200,
                        }
                    },
                }
            },
        },
        "census_sha256",
    )
    census_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        audit_geometry_census(
            census_path,
            manifest_path=manifest_path,
            video_root=video_root,
        )


def test_48_by_8_tubelet_order_and_exact_2x_interpolation():
    torch = pytest.importorskip("torch")
    from opentad.models.backbones.native_crop_wrapper import (
        deterministic_linear_2x,
        flatten_chunk_tubelets,
    )

    chunk_sentinel = torch.arange(48 * 8, dtype=torch.float32).reshape(
        48, 1, 8
    )
    flattened = flatten_chunk_tubelets(
        chunk_sentinel,
        source_batch=1,
        chunk_num=48,
    )
    assert flattened.shape == (1, 1, 384)
    assert torch.equal(
        flattened.flatten(),
        torch.arange(384, dtype=torch.float32),
    )
    upsampled = deterministic_linear_2x(flattened)
    expected = torch.nn.functional.interpolate(
        flattened,
        size=768,
        mode="linear",
        align_corners=False,
    )
    assert upsampled.shape == (1, 1, 768)
    assert torch.equal(upsampled, expected)


def test_shared_wrapper_preserves_time_axis_and_backward():
    torch = pytest.importorskip("torch")
    from opentad.models.backbones.native_crop_wrapper import (
        NativeCropBackboneWrapper,
        NativeCropFeatureFusion,
    )

    class FakePreprocessor:
        def preprocess(self, tensors, data_samples=None, training=False):
            del data_samples, training
            return torch.stack(tensors).float(), None

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value):
            pooled = torch.nn.functional.avg_pool3d(
                value.mean(dim=1, keepdim=True),
                kernel_size=(2, 16, 16),
                stride=(2, 16, 16),
            )
            return pooled.repeat(1, 4, 1, 1, 1) * self.scale

    class FakeRecognizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FakeBackbone()
            self.data_preprocessor = FakePreprocessor()

    class ChunkIntoTwo:
        def __call__(self, payload):
            frames = payload["frames"]
            batch, num_seg, channels, time, height, width = frames.shape
            assert num_seg == 1 and time == 32
            frames = (
                frames.reshape(batch, num_seg, channels, 2, 16, height, width)
                .permute(0, 3, 1, 2, 4, 5, 6)
                .reshape(batch * 2, num_seg, channels, 16, height, width)
            )
            return {"frames": frames}

    wrapper = NativeCropBackboneWrapper.__new__(NativeCropBackboneWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.global_key = "global"
    wrapper.local_key = "local"
    wrapper.expected_global_size = 96
    wrapper.expected_local_size = 128
    wrapper.chunk_num = 2
    wrapper.expected_intermediate_length = 16
    wrapper.output_length = 32
    wrapper.fusion_mode = "fixed_mean"
    wrapper.model = FakeRecognizer()
    wrapper.pre_processing_pipeline = ChunkIntoTwo()
    wrapper.post_processing_pipeline = None
    wrapper.norm_eval = False
    wrapper.freeze_backbone = False
    wrapper.use_temporal_checkpointing = False
    wrapper.fusion = NativeCropFeatureFusion("fixed_mean")
    wrapper.latest_native_crop_audit = None

    inputs = {
        "global": torch.randint(
            0, 256, (1, 1, 3, 32, 96, 96), dtype=torch.uint8
        ),
        "local": torch.randint(
            0, 256, (1, 1, 3, 32, 128, 128), dtype=torch.uint8
        ),
    }
    output = wrapper(inputs)
    assert output.shape == (1, 4, 32)
    output.square().mean().backward()
    assert wrapper.model.backbone.scale.grad is not None
    assert torch.isfinite(wrapper.model.backbone.scale.grad)
    assert torch.count_nonzero(wrapper.model.backbone.scale.grad)
    assert wrapper.latest_native_crop_audit["shared_backbone_instances"] == 1
    with pytest.raises(ValueError, match="fail-closed"):
        wrapper({**inputs, "teacher": torch.tensor(1)})


def test_structured_views_collate_to_backbone_contract():
    torch = pytest.importorskip("torch")
    from opentad.datasets.builder import collate

    sample = {
        "inputs": {
            "global": np.zeros((1, 3, 8, 96, 96), dtype=np.uint8),
            "local": np.zeros((1, 3, 8, 128, 128), dtype=np.uint8),
        },
        "masks": torch.ones(8, dtype=torch.bool),
        "metas": {"video_name": "development_only"},
    }
    batch = collate([sample])
    assert batch["inputs"]["global"].shape == (1, 1, 3, 8, 96, 96)
    assert batch["inputs"]["local"].shape == (1, 1, 3, 8, 128, 128)
    assert batch["inputs"]["global"].dtype == torch.uint8
    assert batch["inputs"]["local"].dtype == torch.uint8
    assert batch["masks"].shape == (1, 8)


def test_videomae_runtime_position_grids_and_checkpoint_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mmcv")
    pytest.importorskip("mmaction")
    from opentad.models.backbones.vit_adapter import VisionTransformerAdapter

    model = VisionTransformerAdapter(
        img_size=224,
        patch_size=16,
        embed_dims=48,
        depth=1,
        num_heads=3,
        mlp_ratio=2,
        qkv_bias=True,
        num_frames=4,
        drop_path_rate=0.0,
        return_feat_map=True,
        with_cp=True,
        total_frames=4,
        adapter_index=[0],
    )
    for height, width in (
        (96, 96),
        (112, 112),
        (128, 128),
        (96, 128),
        (128, 96),
    ):
        model.zero_grad(set_to_none=True)
        inputs = torch.randn(
            1, 3, 4, height, width, requires_grad=True
        )
        output = model(inputs)
        assert output.shape == (
            1,
            48,
            2,
            height // 16,
            width // 16,
        )
        output.square().mean().backward()
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()


def test_local_transform_ignores_all_supervision_values():
    from opentad.datasets.transforms.native_crop import NativeCropSourceViews

    transform = NativeCropSourceViews()
    frames = [coordinate_image(180, 320) for _ in range(2)]
    base = {"imgs": frames}
    supervised = {
        **copy.deepcopy(base),
        "gt_segments": np.asarray([[1.0, 2.0]]),
        "teacher": {"boxes": [[0, 0, 1, 1]]},
        "oracle": "different",
    }
    plain = transform(copy.deepcopy(base))
    with_supervision = transform(supervised)
    assert np.array_equal(
        plain["native_crop_inputs"]["local"],
        with_supervision["native_crop_inputs"]["local"],
    )
