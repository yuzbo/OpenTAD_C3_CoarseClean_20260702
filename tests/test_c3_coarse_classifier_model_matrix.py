from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.bata import c3_coarse_classifier_model_matrix as model_zoo
from tools.bata import train_paction_acquisition_policy as train_policy

MODEL_MATRIX = model_zoo.MODEL_MATRIX
iter_matrix = model_zoo.iter_matrix


def test_first_wave_matrix_has_broad_model_families() -> None:
    entries = list(iter_matrix(tier="first_wave"))
    families = {entry["family"] for entry in entries}
    assert "image_backbone_temporal_head" in families
    assert "native_video_classifier" in families
    assert "video_transformer_teacher" in families
    assert len(entries) >= 10


def test_model_zoo_covers_every_supported_coarse_probe_model() -> None:
    from tools.bata import train_lowres_action_probe as probe

    assert hasattr(model_zoo, "MODEL_ZOO")
    assert hasattr(model_zoo, "iter_model_zoo")
    zoo = list(model_zoo.iter_model_zoo(tier="all", include_optional=True))
    by_probe_model = {}
    for entry in zoo:
        by_probe_model.setdefault(entry["probe_model"], []).append(entry)

    assert {"c3-reader", "mobilenetv3", "temporal-tcn", "official-action-seg", "matrix-zoo"} <= set(by_probe_model)

    reader_types = {entry.get("reader_type") for entry in by_probe_model["c3-reader"]}
    assert set(probe.SUPPORTED_C3_READER_TYPES) <= reader_types

    tcn_variants = {entry.get("tcn_variant") for entry in by_probe_model["temporal-tcn"]}
    assert set(probe.SUPPORTED_TCN_VARIANTS) <= tcn_variants
    default_tcn_variants = {
        entry.get("tcn_variant")
        for entry in model_zoo.iter_model_zoo(tier="first_wave")
        if entry.get("probe_model") == "temporal-tcn"
    }
    assert set(model_zoo.DEFAULT_TCN_VARIANTS) <= default_tcn_variants
    assert not (set(model_zoo.LOCAL_EXPERIMENTAL_TCN_VARIANTS) & default_tcn_variants)

    official_backends = {
        entry.get("official_action_seg_backend") for entry in by_probe_model["official-action-seg"]
    }
    assert set(probe.SUPPORTED_OFFICIAL_ACTION_SEG_BACKENDS) <= official_backends

    matrix_ids = {entry.get("matrix_model_id") for entry in by_probe_model["matrix-zoo"]}
    assert {entry["id"] for entry in MODEL_MATRIX} <= matrix_ids

    assert len({entry["id"] for entry in model_zoo.MODEL_ZOO}) == len(model_zoo.MODEL_ZOO)


def test_model_zoo_entries_report_launch_arguments_and_claim_boundaries() -> None:
    assert hasattr(model_zoo, "MODEL_ZOO")
    for entry in model_zoo.MODEL_ZOO:
        assert entry["route_label"] == "C3_MAINLINE_OPTIMIZATION"
        assert entry["route_family"] == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
        assert entry["selection_signal"] == "binary_frame_actionness"
        assert entry["deploy_time_candidate"] is True
        assert entry["uses_gt_at_test"] is False
        assert entry["uses_teacher_at_test"] is False
        assert entry["uses_raw_prediction_cache"] is False
        assert isinstance(entry.get("train_lowres_action_probe_args"), list)
        assert entry["train_lowres_action_probe_args"]


def test_matrix_entries_are_c3_diagnostic_only_and_have_selection_rationale() -> None:
    ids = set()
    for entry in MODEL_MATRIX:
        assert entry["id"] not in ids
        ids.add(entry["id"])
        assert entry["tier"] in {"first_wave", "second_wave"}
        assert entry["backend"] in {"timm", "torchvision_video", "pytorchvideo_hub", "hf_snapshot"}
        assert isinstance(entry.get("why"), str) and len(entry["why"]) >= 32
        assert isinstance(entry.get("intended_head"), str) and len(entry["intended_head"]) >= 8
        assert isinstance(entry.get("compute_class"), str) and entry["compute_class"]
        assert "DIVERGENT_INNOVATION" not in json.dumps(entry)


def test_second_wave_is_not_default_downloaded() -> None:
    second_wave = [entry for entry in MODEL_MATRIX if entry["tier"] == "second_wave"]
    assert second_wave
    assert all(not entry["default_download"] for entry in second_wave)


def test_cli_dry_run_writes_download_status(tmp_path: Path) -> None:
    output = tmp_path / "matrix.json"
    script = Path("tools/bata/c3_coarse_classifier_model_matrix.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--download",
            "--dry-run",
            "--tier",
            "first_wave",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["route_label"] == "C3_MAINLINE_OPTIMIZATION"
    assert payload["diagnostic_only"] is True
    assert payload["no_detector_training"] is True
    assert payload["no_detector_eval"] is True
    assert payload["download_results"]
    assert {item["status"] for item in payload["download_results"]} == {"dry_run"}
    assert payload["schema_version"] == "c3_coarse_classifier_model_matrix_v1"
    assert proc.stderr == ""


def test_acquisition_policy_zoo_registers_paction_value_and_dynamic_budget() -> None:
    assert hasattr(model_zoo, "ACQUISITION_POLICY_ZOO")
    assert hasattr(model_zoo, "iter_acquisition_policy_zoo")

    policies = list(model_zoo.iter_acquisition_policy_zoo())
    ids = {entry["id"] for entry in policies}

    assert "paction_gap_loss_value_selector" in ids
    assert "paction_gap_loss_dynamic_budget_policy" in ids
    for entry in policies:
        assert entry["route_label"] == "C3_MAINLINE_OPTIMIZATION"
        assert entry["route_family"] == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
        assert entry["selection_signal"] == "p_action"
        assert entry["uses_gt_at_test"] is False
        assert entry["uses_teacher_at_test"] is False
        assert entry["uses_raw_prediction_cache"] is False
        assert entry["gap_control"] == "learned_gap_hole_loss_no_uniform_fill"
        assert entry["uses_uniform_scaffold"] is False
        assert entry["uses_uniform_fill"] is False
        assert "large_gap" in entry["loss_terms"]
        assert "temporal_hole" in entry["loss_terms"]
        assert entry["policy_model"] == "PActionDynamicAcquisitionPolicy"
        assert isinstance(entry["train_paction_acquisition_policy_args"], list)
        assert entry["train_paction_acquisition_policy_args"]

    dynamic = next(entry for entry in policies if entry["id"] == "paction_gap_loss_dynamic_budget_policy")
    assert dynamic["dynamic_budget"] is True
    assert dynamic["budget_buckets"] == [128, 192, 256, 320, 384, 512, 768]


def test_paction_policy_training_cli_accepts_policy_zoo_argument_aliases() -> None:
    parser = train_policy.build_arg_parser()
    args = parser.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--out-dir",
            "out",
            "--budget-buckets",
            "2",
            "4",
            "8",
            "--policy-mode",
            "dynamic_budget",
            "--selection-strategy",
            "learned_paction_gap_loss_dynamic_budget",
        ]
    )

    assert args.dynamic_budget_buckets == [2, 4, 8]
    assert args.policy_mode == "dynamic_budget"


def test_cli_can_print_acquisition_policy_zoo() -> None:
    script = Path("tools/bata/c3_coarse_classifier_model_matrix.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--print-policy-zoo"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["acquisition_policy_zoo_schema_version"] == "c3_paction_acquisition_policy_zoo_v1"
    assert payload["acquisition_policy_zoo_count"] >= 2
    assert {entry["id"] for entry in payload["acquisition_policy_zoo_entries"]} >= {
        "paction_gap_loss_value_selector",
        "paction_gap_loss_dynamic_budget_policy",
    }
    assert proc.stderr == ""
