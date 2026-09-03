import ast
import json
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    ARMS,
    SEEDS,
    build_full_data_bundle,
    config_path,
    validate_matrix,
    validate_parameter_fairness,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_profile import (
    FullOperatorLedger,
    batched_matmul_fma2,
    compare_c_exec_receipts,
    convolution_fma2,
    linear_fma2,
    validate_c_exec_comparison,
)
from tools.bata.trace_d2s_patad_full_operator import (
    _sort_comparison_upper_bound,
)


ROOT = Path(__file__).resolve().parents[1]


def test_full_matrix_and_parameter_surface_are_frozen():
    receipt = validate_matrix(ROOT)
    assert receipt["cell_count"] == 9
    assert [cell["arm"] for cell in receipt["cells"]] == [
        arm for arm in ARMS for _ in SEEDS
    ]
    assert [cell["seed"] for cell in receipt["cells"]] == [
        seed for _ in ARMS for seed in SEEDS
    ]
    assert receipt["training_identities"] == 200
    assert receipt["evaluation_videos"] == 211
    assert receipt["evaluation_ordered_windows"] == 792
    assert receipt["successful_updates_per_cell"] == 6000
    validate_parameter_fairness(ROOT)


def test_a0_crop_is_source_native_fixed_center_and_label_free():
    source_height, source_width, crop = 180, 320, 128
    x0 = (source_width - crop) // 2
    y0 = (source_height - crop) // 2
    assert [x0, y0, x0 + crop, y0 + crop] == [96, 26, 224, 154]
    cfg = Config.fromfile(config_path(ROOT, "U128-A0", 4407))
    binding = cfg.continuous_roi_s2_v3_full200_compute
    assert binding.canonical_crop_xyxy == [96, 26, 224, 154]
    assert binding.crop_policy == "source_native_fixed_center"
    pipeline_text = repr(cfg.dataset.train.pipeline).lower()
    assert "gt_segments" not in repr(
        next(step for step in cfg.dataset.train.pipeline if step.type == "NativeCropSourceViews")
    ).lower()
    assert "candidate" not in pipeline_text
    assert "metric" not in pipeline_text


def test_u128_a0_fusion_has_no_parameters():
    path = ROOT / "opentad/models/backbones/native_crop_wrapper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fusion = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeCropFeatureFusion"
    )
    init = next(
        node
        for node in fusion.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assigned_attributes = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert assigned_attributes == {"mode"}


def test_full_data_bundle_is_complete_label_free_and_deterministic(tmp_path):
    database = {}
    media_root = tmp_path / "video"
    for index in range(200):
        video_id = f"train_{index:03d}"
        database[video_id] = {
            "subset": "training",
            "frame": 768,
            "duration": 100.0,
            "annotations": [
                {"label": "Action", "segment": [0.0, float(index + 1)]}
            ],
        }
        path = media_root / "training" / f"{video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    for index in range(211):
        video_id = f"validation_{index:03d}"
        snippet_count = 1536 if index < 159 else 1152
        database[video_id] = {
            "subset": "validation",
            "frame": (snippet_count - 1) * 4 + 1,
            "duration": 100.0,
            "annotations": [
                {"label": "Action", "segment": [10.0, 20.0]}
            ],
        }
        path = media_root / "validation" / f"{video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"database": database}), encoding="utf-8")
    class_map = tmp_path / "class_map.txt"
    class_map.write_text("Action\n", encoding="utf-8")

    manifest = build_full_data_bundle(
        annotation, class_map, media_root, tmp_path / "manifest"
    )
    assert manifest["training"]["identity_count"] == 200
    assert manifest["evaluation"]["video_count"] == 211
    assert manifest["evaluation"]["ordered_window_count"] == 792
    assert manifest["media"]["count"] == 411
    assert manifest["short_q1"]["q1_float64_hex"] == float(50.75).hex()
    heldout = json.loads(
        (tmp_path / "manifest" / "heldout_inference_annotation.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        set(row) == {"subset", "frame", "duration"}
        for row in heldout["database"].values()
    )


def test_full_operator_ledger_is_integer_complete_and_never_uses_latency():
    assert linear_fma2(batch=2, positions=3, inputs=4, outputs=5) == 240
    assert convolution_fma2(
        batch=1,
        output_positions=8,
        output_channels=4,
        input_channels_per_group=3,
        kernel_elements=3,
    ) == 576
    assert batched_matmul_fma2(batches=2, m=3, n=4, k=5) == 240

    identity = {
        "candidate_commit": "a" * 40,
        "protocol_sha256": "b" * 64,
        "evaluation_manifest_sha256": "c" * 64,
        "checkpoint_policy": "epoch_59_state_dict_ema",
        "dtype": "float16",
        "batch_size": 1,
        "ordered_window_count": 792,
    }
    boundary = {
        "start": "first_arm_dependent_decoded_rgb_transform",
        "end": "pre_nms_raw_detections",
        "nms_called": False,
        "evaluator_called": False,
    }
    counts = {"D160": 1000, "G96": 800, "U128-A0": 900}
    receipts = {}
    for arm, count in counts.items():
        ledger = FullOperatorLedger(arm=arm)
        ledger.add_automatic(
            event_id=f"{arm}/backbone/mm",
            operator="aten.mm",
            integer_operations=count,
        )
        receipts[arm] = ledger.receipt(
            execution_identity=identity,
            boundary_trace=boundary,
        )
    comparison = compare_c_exec_receipts(receipts)
    assert comparison["primary_exact_10u_le_9d"]
    assert comparison["g96_not_more_than_candidate"]
    assert comparison["gate_uses_latency_or_memory"] is False
    assert validate_c_exec_comparison(comparison) == comparison

    tampered = dict(comparison)
    tampered["primary_exact_10u_le_9d"] = False
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_c_exec_comparison(tampered)


def test_runtime_trace_sort_bound_is_integer_and_shape_aware():
    assert _sort_comparison_upper_bound([2, 16], 3) == 384


def test_full_operator_ledger_fails_closed_on_unknown_or_duplicate_operator():
    ledger = FullOperatorLedger(arm="D160")
    ledger.mark_unsupported("aten.some_new_fused_op")
    with pytest.raises(RuntimeError, match="FAILED_C_EXEC_INCOMPLETE"):
        ledger.receipt(
            execution_identity={},
            boundary_trace={
                "start": "first_arm_dependent_decoded_rgb_transform",
                "end": "pre_nms_raw_detections",
                "nms_called": False,
                "evaluator_called": False,
            },
        )
