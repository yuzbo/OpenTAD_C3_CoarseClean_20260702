import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (
    LEAF_ORDERS,
    PHYSICAL_WINDOWS,
    STUDY_ID,
    WARMUP_WINDOWS_PER_PASS,
    WINDOW_BUDGET,
    build_runtime_identity,
    canonical_sha256,
    exclusive_write_json,
    leaf_sequence,
    read_json_object,
    validate_afterany_dependency,
    validate_execution_binding_receipt,
    validate_jobgraph,
    validate_leaf_rows,
    validate_paired_configs,
    validate_population_manifest,
    validate_preregistration,
    validate_scheduler_job,
    validate_tracked_config,
    validate_runtime_identity_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_leaf_orders_and_full_population_warmup():
    assert list(LEAF_ORDERS.values()) == ["ABBA"] * 4 + ["BAAB"] * 4
    assert leaf_sequence("L01") == (
        "none_control",
        "residual_window_center",
        "residual_window_center",
        "none_control",
    )
    assert leaf_sequence("L08") == (
        "residual_window_center",
        "none_control",
        "none_control",
        "residual_window_center",
    )
    assert WARMUP_WINDOWS_PER_PASS == PHYSICAL_WINDOWS == 136
    with pytest.raises(ValueError):
        leaf_sequence("L01", "BAAB")
    with pytest.raises(ValueError):
        leaf_sequence("L09")


def test_tracked_configs_have_one_explicit_treatment_difference():
    control = validate_tracked_config(ROOT, "A")
    centered = validate_tracked_config(ROOT, "B")
    assert control["calibration_mode"] == "none"
    assert centered["calibration_mode"] == "residual_window_center"
    assert control["path"] != centered["path"]
    assert control["canonical_config_sha256"] != centered["canonical_config_sha256"]
    validate_paired_configs(ROOT)


def test_tracked_population_is_exact_unique_ordered_gate_manifest():
    path = ROOT / "research-wiki/experiments/zoomtoken_scnr_steady_cost_population_v001.json"
    manifest = validate_population_manifest(read_json_object(path, label="population"))
    assert manifest["study_id"] == STUDY_ID
    assert len(manifest["windows"]) == 136
    assert len({row["physical_window_id"] for row in manifest["windows"]}) == 136
    assert len({row["video_id"] for row in manifest["windows"]}) == 40
    duplicate = copy.deepcopy(manifest)
    duplicate["windows"][1] = copy.deepcopy(duplicate["windows"][0])
    with pytest.raises(ValueError):
        validate_population_manifest(duplicate)


def test_preregistration_and_jobgraph_are_frozen():
    prereg = read_json_object(
        ROOT / "research-wiki/experiments/zoomtoken_scnr_steady_cost_prereg_v001.json",
        label="prereg",
    )
    graph = read_json_object(
        ROOT / "research-wiki/experiments/zoomtoken_scnr_steady_cost_jobgraph_v001.json",
        label="jobgraph",
    )
    validate_preregistration(prereg)
    validated = validate_jobgraph(graph)
    assert validated["finalizer"]["dependency"] == "afterany"
    assert validated["finalizer"]["writes_authoritative_decision"] is False
    assert all(leaf["automatic_retry"] is False for leaf in validated["leaves"])


def _leaf_rows(leaf_id="L01"):
    manifest = read_json_object(
        ROOT / "research-wiki/experiments/zoomtoken_scnr_steady_cost_population_v001.json",
        label="population",
    )
    rows = []
    for pass_index, arm in enumerate(leaf_sequence(leaf_id)):
        for window in manifest["windows"]:
            row = {
                "leaf_id": leaf_id,
                "pass_index": pass_index,
                "sample_ordinal": window["ordinal"],
                "loader_ordinal": window["ordinal"],
                "arm": arm,
                "video_id": window["video_id"],
                "physical_window_id": window["physical_window_id"],
                "window_id": f"{window['physical_window_id']}#{window['ordinal']}",
                "measurement_phase": "measured",
                "warmup": False,
                "exact_window_budget": WINDOW_BUDGET,
                "selected_physical_tokens": WINDOW_BUDGET,
                "executed_physical_tokens": WINDOW_BUDGET,
                "duplicate_selected_physical_tokens": 0,
                "padded_heavy_tokens": 0,
                "route_audit": {
                    "physical_indices_sha256": "0" * 64,
                    "k_t_min": 0,
                    "k_t_max": 128,
                    "k_t_zero_count": 1,
                    "role_counts": {"heavy": WINDOW_BUDGET},
                    "attention_pairs": WINDOW_BUDGET,
                    "clip_token_counts": [WINDOW_BUDGET],
                    "exact_window_budget": WINDOW_BUDGET,
                    "padded_heavy_tokens": 0,
                    "branch_calibration_mode": (
                        "none" if arm == "none_control" else "residual_window_center"
                    ),
                },
                "input_pipeline_serial_ms": 0.1,
                "h2d_ms": 0.1,
                "decode_to_window_output_wall_ms": 1.0,
                "model_forward_ms": 0.5,
                "postprocess_ms": 0.1,
                "final_video_nms_ms": 0.1,
                "end_to_end_serial_ms": 1.0,
                "peak_gpu_allocated_mb": 1.0,
                "peak_gpu_reserved_mb": 1.0,
                "gross_gpu_energy_j_per_sample": 1.0,
            }
            row["sample_sha256"] = canonical_sha256(row)
            rows.append(row)
    return rows


def test_leaf_rows_exclude_warmup_and_require_exact_route_budget():
    rows = _leaf_rows()
    assert len(validate_leaf_rows(rows, leaf_id="L01")) == 4 * 136
    contaminated = copy.deepcopy(rows)
    contaminated[0]["warmup"] = True
    with pytest.raises(ValueError):
        validate_leaf_rows(contaminated, leaf_id="L01")
    padded = copy.deepcopy(rows)
    padded[0]["padded_heavy_tokens"] = 1
    padded[0]["sample_sha256"] = canonical_sha256(
        {key: value for key, value in padded[0].items() if key != "sample_sha256"}
    )
    with pytest.raises(ValueError):
        validate_leaf_rows(padded, leaf_id="L01")


def test_scheduler_confirmation_is_exact_completed_zero_zero():
    validate_scheduler_job(
        {"job_id": "123", "state": "COMPLETED", "exit_code": "0:0"},
        expected_job_id="123",
        label="Job 9",
    )
    for bad in (
        {"job_id": "124", "state": "COMPLETED", "exit_code": "0:0"},
        {"job_id": "123", "state": "FAILED", "exit_code": "1:0"},
        {"job_id": "123", "state": "RUNNING", "exit_code": "0:0"},
    ):
        with pytest.raises(ValueError):
            validate_scheduler_job(bad, expected_job_id="123", label="Job 9")


def test_execution_and_runtime_identity_mutations_fail_closed():
    expected_binding = {
        "schema_version": "zoomtoken_scnr_steady_cost_execution_binding_v001",
        "study_id": STUDY_ID,
        "arm": "A",
        "variant": "none_control",
        "calibration_mode": "none",
        "tracked_config_path": "control.py",
        "legacy_calibration_mode": "none",
        "checkpoint_receipt": {"path": "/run/control.pth", "sha256": "a" * 64},
        "bound_accuracy_config_receipt": {
            "path": "/run/control.py",
            "sha256": "b" * 64,
        },
    }
    assert (
        validate_execution_binding_receipt(expected_binding, expected=expected_binding)
        == expected_binding
    )
    mutated_binding = copy.deepcopy(expected_binding)
    mutated_binding["checkpoint_receipt"]["path"] = "/run/other.pth"
    with pytest.raises(ValueError):
        validate_execution_binding_receipt(mutated_binding, expected=expected_binding)

    pre_run = {
        "runtime": {
            "python_version": "3.10.0",
            "numpy_version": "1.23.5",
            "gpu_constraint_or_sku": "A100",
        }
    }
    hardware = {"gpu_name": "NVIDIA A100-SXM4-80GB"}
    software = {"python": "3.10.0", "packages": {"numpy": "1.23.5"}}
    runtime = build_runtime_identity(
        pre_run,
        hardware=hardware,
        software=software,
        slurm_job_constraints="A100",
    )
    validate_runtime_identity_receipt(
        runtime, pre_run=pre_run, hardware=hardware, software=software
    )
    mutated_runtime = copy.deepcopy(runtime)
    mutated_runtime["numpy_version"] = "2.0.0"
    with pytest.raises(ValueError):
        validate_runtime_identity_receipt(
            mutated_runtime, pre_run=pre_run, hardware=hardware, software=software
        )


def test_job9_dependency_requires_exact_ordered_afterany_parents():
    parents = [str(index) for index in range(101, 109)]
    row = {
        "job_id": "200",
        "state": "RUNNING",
        "exit_code": "0:0",
        "dependency": "afterany:" + ":".join(parents),
    }
    validate_afterany_dependency(
        row, expected_job_id="200", expected_parent_job_ids=parents
    )
    for dependency in (
        "",
        "afterok:" + ":".join(parents),
        "afterany:" + ":".join(reversed(parents)),
        "afterany:" + ":".join(parents[:-1]),
    ):
        bad = {**row, "dependency": dependency}
        with pytest.raises(ValueError):
            validate_afterany_dependency(
                bad, expected_job_id="200", expected_parent_job_ids=parents
            )


def test_authoritative_json_publication_has_one_winner(tmp_path):
    target = tmp_path / "final_decision.json"

    def publish(index):
        try:
            exclusive_write_json(target, {"writer": index})
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(publish, range(8)))
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7
    assert read_json_object(target, label="exclusive result")["writer"] in range(8)
