from __future__ import annotations

import copy
import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_official_comparable_contract import (
    FORMAL_DEVELOPMENT_ARM_ORDER,
    P1_CONDITIONAL_MODIFIER_MAP,
    P1_DEVELOPMENT_SEED,
    P1_FIRST_SCREEN_ARM_ORDER,
    P1_MATCHED_RUNNER_ARM_ORDER,
    P1_WINDOW_TOKEN_BUDGET,
    p1_arm_spec,
    p1_source_config_relative_path,
    require_clean_formal_checkout,
)
from tools.bata.georoute_p1_runtime_attestor import (
    _NVIDIA_QUERY_FIELDS,
    build_runtime_attestation,
    validate_runtime_attestation,
)
from tools.bata.georoute_official_development_stage_runner import (
    summarize_formal_telemetry,
    validate_p1_q_routing_audit,
)
from tools.bata.finalize_georoute_official_development import (
    P1_GT_BINS,
    P1_UNMATCHED_PREDICTION_BINS,
    _match_p1_video,
)
from tools.bata.georoute_experiment_contract import canonical_sha256
from tools.bata import deploy_georoute_official_development as p1_deployer
from tools.bata import georoute_official_development_stage_runner as p1_stage_runner
from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (
    P1_COST_LEAF_SPECS,
    P1_COST_RATIO_LIMIT,
    P1_DENSE_PHYSICAL_TOKENS,
    analyze_p1_cost_leaves,
    p1_cost_leaf_sequence,
)

def _load_surface(relative_path: str) -> dict:
    return runpy.run_path(str(ROOT / relative_path))


def _observations() -> dict:
    gpu = {
        "uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "name": "NVIDIA A100 80GB PCIe",
        "pci.bus_id": "00000000:17:00.0",
        "pci.device_id": "0x20b110de",
        "pci.sub_device_id": "0x145f10de",
        "memory.total": "81920",
        "compute_cap": "8.0",
        "mig.mode.current": "Disabled",
        "driver_version": "535.183.01",
        "persistence_mode": "Enabled",
        "clocks.applications.graphics": "1410",
        "clocks.applications.memory": "1215",
        "power.limit": "250.0",
    }
    second = dict(gpu)
    second["uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
    second["pci.bus_id"] = "00000000:65:00.0"
    return {
        "schema_version": "georoute_p1_runtime_observations_v001",
        "collector": "georoute_p1_runtime_attestor.collect_runtime_observations",
        "collector_uses_framework_import": False,
        "collector_initializes_cuda": False,
        "gpu_query_fields": list(_NVIDIA_QUERY_FIELDS),
        "gpu_rows": [gpu, second],
        "allocation": {
            "slurm_job_present": True,
            "cuda_visible_device_count": 2,
            "cuda_visible_devices_overridden_by_attestor": False,
        },
        "nvidia_smi_cuda_version": "11.8",
        "nvml_version": "12.535.183",
        "container": {
            "path": "/container/image.sif",
            "digest": "sha256:" + "a" * 64,
            "active_runtime_path_verified": True,
        },
        "dependency_lock": {
            "path": "/container/conda-lock.yml",
            "sha256": "b" * 64,
        },
        "python": {"implementation": "CPython", "version": "3.10.14"},
        "framework": {
            "name": "torch",
            "version": "2.0.1",
            "build_cuda": "11.8",
            "git_version": "c" * 40,
            "version_metadata_path": "/env/torch/version.py",
            "version_metadata_sha256": "d" * 64,
        },
        "cuda_library_versions": {
            "nvidia-cublas-cu11": "11.11.3.6",
            "nvidia-cudnn-cu11": "8.5.0.96",
        },
        "kernel_release": "5.15.0",
    }


def _q_k_t() -> list[int]:
    return [0, 128, *([64] * 382)]


def _q_roles() -> tuple[list[list[int]], dict[str, int]]:
    rows = [
        [value - 2 * (value // 4), value // 4, value // 4]
        for value in _q_k_t()
    ]
    names = ("context", "roi", "residual")
    return rows, {
        name: sum(row[index] for row in rows) for index, name in enumerate(names)
    }


def _q_ragged() -> dict:
    clip_counts = [512] * 48
    return {
        "clip_token_counts": clip_counts,
        "attention_pairs": sum(value**2 for value in clip_counts),
        "requested_physical_tokens": P1_WINDOW_TOKEN_BUDGET,
        "unique_physical_tokens": P1_WINDOW_TOKEN_BUDGET,
        "padded_heavy_tokens": 0,
        "executed_patch_tokens": P1_WINDOW_TOKEN_BUDGET,
        "ragged_attention_bucket_call_count": 2,
        "ragged_mlp_bucket_call_count": 2,
    }


def _q_branch_calibration() -> dict:
    return {
        "schema_version": "scnr_dynamic_branch_calibration_window_v1",
        "mode": "none",
        "target": "delta_residual",
        "scope": "disabled",
        "valid_candidate_count": 84_480,
        "residual_valid_mean_before": 0.0,
        "residual_valid_mean_after": 0.0,
        "changes_q_base": False,
        "changes_delta_roi": False,
        "changes_context_zero_modifier": False,
        "changes_budget_or_role_quota": False,
        "mean_detached": False,
    }


def _q_routing_audit() -> dict:
    k_t = _q_k_t()
    role_rows, role_counts = _q_roles()
    clip_counts = [512] * 48
    return {
        "routing_schema": "georoute_dynamic_global_routing_v2",
        "route_mode": "dynamic_scnr",
        "policy_estimator": "straight_through",
        "target_k": None,
        "window_token_budget": P1_WINDOW_TOKEN_BUDGET,
        "window_budget_is_global": True,
        "fixed_per_tubelet_k": False,
        "k_t_allows_zero": True,
        "k_per_tubelet": [k_t],
        "k_t_min": 0,
        "k_t_max": 128,
        "k_t_zero_count": 1,
        "role_counts": role_counts,
        "role_counts_per_window": [
            [sum(row[index] for row in role_rows) for index in range(3)]
        ],
        "zero_carrier_mode": "masked_zero",
        "heavy_valid_mask_matches_k_t": True,
        "dynamic_roi_modifier_enabled": False,
        "dynamic_residual_modifier_enabled": False,
        "branch_calibration": _q_branch_calibration(),
        "source_grid_hw": [11, 20],
        "requested_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
        "unique_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
        "padded_heavy_tokens_per_window": 0,
        "executed_patch_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
        "heavy_backbone_forward_count": 1,
        "physical_indices_sha256": "a" * 64,
        "heavy_valid_mask_sha256": "b" * 64,
        "diagnostic_telemetry_enabled": True,
        "uses_grid_sample": False,
        "uses_resized_local_crop": False,
        "uses_gt_for_route": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_test_evidence": False,
        "packed": {
            "schema_version": "videomae_native_ragged_v1",
            "execution_mode": "true_clip_ragged_no_padding",
            "adapter_execution": "coordinate_lineage_true_ragged",
            "window_token_budget": P1_WINDOW_TOKEN_BUDGET,
            "requested_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "unique_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "padded_heavy_tokens_per_window": 0,
            "executed_patch_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "heavy_backbone_forward_count": 1,
            "dense_adapter_forward_count": 0,
            "clip_token_counts": [clip_counts],
            "attention_pairs_per_window": [
                sum(value**2 for value in clip_counts)
            ],
        },
    }


def _q_route_telemetry() -> dict:
    k_t = _q_k_t()
    role_rows, role_counts = _q_roles()
    histogram = {"0": 1, "64": 382, "128": 1}
    return {
        "schema_version": "georoute_dynamic_diagnostic_window_telemetry_v1",
        "measurement_scope": "accuracy_replay_only_excluded_from_timed_cost",
        "batch_size": 1,
        "tubelet_count": 384,
        "item_count": 220,
        "source_grid_hw": [11, 20],
        "window_token_budget": P1_WINDOW_TOKEN_BUDGET,
        "target_k": None,
        "selected_physical_index_sha256": "c" * 64,
        "k_t": {
            "values": k_t,
            "min": 0,
            "max": 128,
            "zero_count": 1,
            "histogram": histogram,
        },
        "roles": {
            "order": ["context", "roi", "residual"],
            "aggregate_counts": role_counts,
            "per_tubelet_counts": role_rows,
        },
        "branch_calibration": _q_branch_calibration(),
        "ragged_execution": _q_ragged(),
        "role_assignment_changes_execution": False,
        "gt_for_route_used": False,
        "teacher_used": False,
        "oracle_used": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }


def _formal_telemetry_payload(route: dict) -> dict:
    descriptor = {
        "dataset_index": 0,
        "rank": 0,
        "local_batch_index": 0,
        "video_id": "video_validation_0000001",
        "window_center_count": 1,
        "window_center_first": 128.0,
        "window_center_last": 128.0,
    }
    descriptor_sha256 = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    record = {
        **descriptor,
        "window_descriptor_sha256": descriptor_sha256,
        "route": route,
    }
    population_descriptor = {
        **descriptor,
        "window_descriptor_sha256": descriptor_sha256,
    }
    population_sha256 = hashlib.sha256(
        json.dumps(
            [population_descriptor],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "georoute_formal_development_telemetry_v1",
        "development_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
        "world_size": 2,
        "local_batch_size": 1,
        "dataset_count": 1,
        "record_count": 1,
        "unique_dataset_count": 1,
        "sampler_padding_count": 0,
        "population_sha256": population_sha256,
        "records": [record],
    }


def _summarize_fixture(payload: dict, *, arm: str) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "telemetry.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return summarize_formal_telemetry(path, arm=arm)


def _p1_cost_route(arm: str) -> dict:
    selected = P1_DENSE_PHYSICAL_TOKENS if arm in {"DO", "DN"} else 24_576
    route = {
        "arm": arm,
        "route_mode": {
            "DO": "dense",
            "DN": "dense",
            "U": "uniform",
            "R": "random",
            "Q": "dynamic_scnr",
        }[arm],
        "target_k": None if arm in {"DO", "DN", "Q"} else 64,
        "dynamic_k_t": arm == "Q",
        "selected_physical_tokens": selected,
        "executed_physical_tokens": selected,
        "duplicate_selected_physical_tokens": 0,
        "padded_heavy_tokens": 0,
        "uses_gt_for_route": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_test_evidence": False,
    }
    if arm == "Q":
        clip_counts = [512] * 48
        route.update(
            {
                "k_t_min": 0,
                "k_t_max": 128,
                "k_t_zero_count": 1,
                "clip_token_counts": clip_counts,
                "attention_pairs": sum(value**2 for value in clip_counts),
                "physical_indices_sha256": "e" * 64,
            }
        )
    return route


def _p1_cost_rows(leaf_id: str) -> list[dict]:
    rows = []
    for pass_index, arm in enumerate(p1_cost_leaf_sequence(leaf_id)):
        ratio = 0.85 if arm == "Q" else 1.0
        for ordinal in range(136):
            video_id = f"video_validation_{ordinal % 40:07d}"
            selected = P1_DENSE_PHYSICAL_TOKENS if arm in {"DO", "DN"} else 24_576
            row = {
                "schema_version": "zoomtoken_p1_cost_sample_v001",
                "leaf_id": leaf_id,
                "pass_index": pass_index,
                "arm": arm,
                "sample_ordinal": ordinal,
                "loader_ordinal": ordinal,
                "measurement_phase": "measured",
                "warmup": False,
                "video_id": video_id,
                "physical_window_id": f"{video_id}:{ordinal}",
                "window_id": f"{video_id}:{ordinal}#{ordinal}",
                "exact_window_budget": 24_576,
                "selected_physical_tokens": selected,
                "executed_physical_tokens": selected,
                "duplicate_selected_physical_tokens": 0,
                "padded_heavy_tokens": 0,
                "input_pipeline_serial_ms": 10.0 * ratio,
                "h2d_ms": 2.0 * ratio,
                "decode_to_window_output_wall_ms": 90.0 * ratio,
                "model_forward_ms": 60.0 * ratio,
                "postprocess_ms": 10.0 * ratio,
                "final_video_nms_ms": 10.0 * ratio,
                "end_to_end_serial_ms": 100.0 * ratio,
                "peak_gpu_allocated_mb": 1000.0,
                "peak_gpu_reserved_mb": 1200.0,
                "gross_gpu_energy_j_per_sample": 10.0 * ratio,
                "energy_window_monotonic_s": [
                    float(pass_index * 1000 + ordinal + 1),
                    float(pass_index * 1000 + ordinal + 2),
                ],
                "nms_energy_window_monotonic_s": [
                    float(pass_index * 1000 + 500),
                    float(pass_index * 1000 + 501),
                ],
                "route_audit": _p1_cost_route(arm),
            }
            row["sample_sha256"] = canonical_sha256(row)
            rows.append(row)
    return rows


def _p1_release_receipt_fixture(directory: Path) -> dict:
    job_ids = tuple(str(910_000 + index) for index in range(15))
    seed_key = str(P1_DEVELOPMENT_SEED)
    runtime_preflight_job = job_ids[0]
    stage_jobs = {
        arm: {seed_key: job_ids[1 + index]}
        for index, arm in enumerate(P1_FIRST_SCREEN_ARM_ORDER)
    }
    cost_jobs = {
        leaf_id: job_ids[6 + index]
        for index, leaf_id in enumerate(P1_COST_LEAF_SPECS)
    }
    predecessor_ids = [
        runtime_preflight_job,
        *(stage_jobs[arm][seed_key] for arm in P1_FIRST_SCREEN_ARM_ORDER),
        *(cost_jobs[leaf_id] for leaf_id in P1_COST_LEAF_SPECS),
    ]
    deployment = {
        "schema_version": p1_deployer.FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
        "study_id": p1_deployer.P1_STUDY_ID,
        "mode": "p1",
        "status": "SUBMITTED_ZOOMTOKEN_P1_DNURQ_MATRIX",
        "runtime_commit": "4" * 40,
        "arms": list(P1_FIRST_SCREEN_ARM_ORDER),
        "seed": P1_DEVELOPMENT_SEED,
        "seeds": [P1_DEVELOPMENT_SEED],
        "accuracy_cells": 5,
        "cost_leaves": 8,
        "jobs": {
            "runtime_preflight": runtime_preflight_job,
            "stage": stage_jobs,
            "cost": cost_jobs,
            "finalizer": job_ids[-1],
        },
        "dependency_policy": {
            "all_fifteen_jobs_held_until_receipts_immutable": True,
            "accuracy_afterany_runtime_preflight": True,
            "cost_afterany_runtime_preflight_and_source_stages": True,
            "finalizer_afterany_all_fourteen_predecessors": True,
            "release_all_fifteen_atomically": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "requeue_allowed": False,
        },
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    deployment["deployment_sha256"] = canonical_sha256(deployment)
    deployment_path = directory / "deployment.json"
    deployment_path.write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    submission = {
        "schema_version": p1_deployer.FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
        "study_id": p1_deployer.P1_STUDY_ID,
        "mode": "p1",
        "status": "SUBMITTED_P1_FINALIZER_AFTERANY",
        "runtime_commit": deployment["runtime_commit"],
        "deployment_file_sha256": p1_deployer.sha256_file(deployment_path),
        "finalizer_job_id": job_ids[-1],
        "dependency_type": "afterany",
        "predecessor_job_ids": predecessor_ids,
    }
    submission["receipt_sha256"] = canonical_sha256(submission)
    submission_path = directory / "finalizer_submission.json"
    submission_path.write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "job_ids": job_ids,
        "deployment": deployment,
        "deployment_path": deployment_path,
        "submission": submission,
        "submission_path": submission_path,
    }


class ZoomTokenP1StaticContractTest(unittest.TestCase):
    def test_first_screen_enum_and_conditional_controls_are_frozen(self):
        self.assertEqual(
            FORMAL_DEVELOPMENT_ARM_ORDER,
            (
                "dense_native",
                "fixed_lattice",
                "random",
                "residual_st_rep_off",
                "residual_pl_rep_off",
            ),
        )
        self.assertEqual(P1_FIRST_SCREEN_ARM_ORDER, ("DO", "DN", "U", "R", "Q"))
        self.assertEqual(P1_MATCHED_RUNNER_ARM_ORDER, ("DN", "U", "R", "Q"))
        self.assertEqual(P1_DEVELOPMENT_SEED, 3407)
        self.assertEqual(P1_WINDOW_TOKEN_BUDGET, 24_576)
        self.assertEqual(
            P1_CONDITIONAL_MODIFIER_MAP,
            {
                "Q": {
                    "roi_modifier_enabled": False,
                    "residual_modifier_enabled": False,
                    "branch_calibration": "none",
                },
                "G": {
                    "roi_modifier_enabled": True,
                    "residual_modifier_enabled": False,
                    "branch_calibration": "none",
                },
                "N": {
                    "roi_modifier_enabled": False,
                    "residual_modifier_enabled": True,
                    "branch_calibration": "residual_window_center",
                },
                "F": {
                    "roi_modifier_enabled": True,
                    "residual_modifier_enabled": True,
                    "branch_calibration": "residual_window_center",
                },
            },
        )
        self.assertEqual(p1_arm_spec("Q")["route_mode"], "dynamic_scnr")
        self.assertTrue(p1_arm_spec("Q")["dynamic_k_t"])
        self.assertEqual(p1_arm_spec("U")["tokens_per_tubelet"], 64)
        self.assertEqual(p1_arm_spec("R")["tokens_per_tubelet"], 64)
        self.assertEqual(p1_arm_spec("DN")["route_mode"], "dense")

    def test_only_dn_and_q_tracked_surfaces_materialize_the_first_screen(self):
        dn_relative = p1_source_config_relative_path("DN")
        q_relative = p1_source_config_relative_path("Q")
        self.assertEqual(p1_source_config_relative_path("U"), q_relative)
        self.assertEqual(p1_source_config_relative_path("R"), q_relative)
        dn = _load_surface(dn_relative)
        q = _load_surface(q_relative)
        self.assertEqual(dn["zoomtoken_p1_config"]["arm_surface"], "DN")

        self.assertFalse(dn["zoomtoken_p1_config"]["routing_enabled"])
        self.assertTrue(dn["zoomtoken_p1_config"]["full_native_spatial_compute"])
        self.assertEqual(
            dn["zoomtoken_p1_config"]["executed_token_contract"],
            "full_native_spatial_support",
        )
        self.assertFalse(
            dn["zoomtoken_p1_config"]["window_token_budget_applies_to_execution"]
        )
        self.assertEqual(q["zoomtoken_p1_config"]["arm_surface"], "Q")
        self.assertEqual(q["zoomtoken_p1_config"]["exact_window_budget"], 24_576)
        self.assertTrue(q["zoomtoken_p1_config"]["q_dynamic_k_t"])
        self.assertFalse(q["zoomtoken_p1_config"]["fixed_per_tubelet_quota"])
        self.assertFalse(q["zoomtoken_p1_config"]["conditional_controls_open"])
        custom = q["model"]["backbone"]["custom"]
        self.assertFalse(custom["georoute_dynamic_roi_modifier_enabled"])
        self.assertFalse(custom["georoute_dynamic_residual_modifier_enabled"])
        self.assertEqual(custom["georoute_zero_carrier_mode"], "masked_zero")
        materialized = sorted(
            path.name
            for path in (ROOT / "configs" / "adatad" / "thumos").glob(
                "georoute_p1_*_seed3407_v001.py"
            )
        )
        self.assertEqual(
            materialized,
            [
                "georoute_p1_dn_seed3407_v001.py",
                "georoute_p1_q_seed3407_v001.py",
            ],
        )
        runner = (
            ROOT
            / "tools"
            / "bata"
            / "georoute_official_development_stage_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("*P1_MATCHED_RUNNER_ARM_ORDER", runner)

    def test_p1_deployment_binds_the_official_do_config_hash(self):
        protocol = {
            "current_source_bridge": {
                "official_config_sha256": "official",
                "georoute_source_config_sha256": "georoute",
            }
        }
        self.assertEqual(
            p1_deployer._deployment_source_config_sha256(
                mode="p1", protocol=protocol
            ),
            "official",
        )
        self.assertEqual(
            p1_deployer._deployment_source_config_sha256(
                mode="formal", protocol=protocol
            ),
            "georoute",
        )

    def test_p1_stage_uses_the_n16r4_site_default_memory_request(self):
        completed = mock.Mock(returncode=0, stdout="123\n", stderr="")
        with mock.patch.object(
            p1_deployer.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(
                p1_deployer._sbatch(
                    name="probe",
                    script=Path("stage.sh"),
                    logs=Path("logs"),
                    exports={},
                    stage=True,
                    test_only=True,
                ),
                "TEST_ONLY_PASS",
            )
        command = run.call_args.args[0]
        self.assertIn("--gpus", command)
        self.assertIn("--cpus-per-task", command)
        self.assertNotIn("--mem", command)

        launcher = (
            ROOT / "scripts" / "run_georoute_official_development_stage_slurm.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(launcher.index("source /etc/profile"), launcher.index("set -euo pipefail"))
        self.assertIn("module load apptainer/1.2.4", launcher)
        self.assertIn(
            "--env GEOROUTE_SOURCE_IDENTITY_VERIFIED=1",
            launcher,
        )
        self.assertIn(
            "container entry lacks outer source identity verification",
            launcher,
        )
        self.assertNotIn("--mem=", launcher)

    def test_wrapper_switches_stop_at_the_production_selector_boundary(self):
        source = (
            ROOT / "opentad" / "models" / "backbones" / "georoute_wrapper.py"
        ).read_text(encoding="utf-8")
        dynamic_start = source.index("    def _forward_dynamic_scnr(")
        forward_start = source.index("    def forward(", dynamic_start)
        dynamic_body = source[dynamic_start:forward_start]
        selector_call = dynamic_body.index("select_dynamic_global_exact_budget(")
        heavy_call = dynamic_body.index("forward_native_ragged(")
        boundary_end = dynamic_body.index(
            "        selected_native = self._gather_selected_native_physical("
        )
        routing_boundary = dynamic_body[:boundary_end]
        self.assertIn("georoute_dynamic_roi_modifier_enabled", source)
        self.assertIn("georoute_dynamic_residual_modifier_enabled", source)
        self.assertIn("delta_roi = torch.zeros_like(delta_roi)", dynamic_body)
        self.assertIn(
            "delta_residual_raw = torch.zeros_like(delta_residual_raw)",
            dynamic_body,
        )
        self.assertLess(selector_call, heavy_call)
        self.assertIn('physical_indices = route["physical_indices"]', routing_boundary)
        self.assertNotIn("FPN", routing_boundary)
        self.assertNotIn("detector", routing_boundary.lower())

    def test_runtime_attestor_requires_exact_preflight_leaf_class(self):
        self.assertNotIn("torch", sys.modules)
        observations = _observations()
        for row in observations["gpu_rows"]:
            row["mig.mode.current"] = "[N/A]"
        preflight = build_runtime_attestation(observations, phase="preflight")
        leaf = build_runtime_attestation(_observations(), phase="leaf")
        validate_runtime_attestation(preflight)
        with self.assertRaisesRegex(ValueError, "differs from preflight"):
            validate_runtime_attestation(leaf, reference=preflight)
        self.assertEqual(leaf["runtime_class"]["gpu"]["mig_mode"], "Disabled")
        matching_leaf_observations = _observations()
        for row in matching_leaf_observations["gpu_rows"]:
            row["mig.mode.current"] = "[N/A]"
        matching_leaf_observations["kernel_release"] = "5.15.0-other-site-patch"
        matching_leaf = build_runtime_attestation(
            matching_leaf_observations,
            phase="leaf",
        )
        validate_runtime_attestation(matching_leaf, reference=preflight)
        self.assertEqual(
            matching_leaf["raw_observations"]["kernel_release"],
            "5.15.0-other-site-patch",
        )
        self.assertNotIn("kernel_release", matching_leaf["runtime_class"])
        self.assertEqual(preflight["runtime_class"]["gpu"]["mig_mode"], "N/A")
        self.assertNotIn("uuid", preflight["runtime_class"]["gpu"])
        self.assertNotIn("pci_bus_id", preflight["runtime_class"]["gpu"])
        self.assertFalse(preflight["node_name_recorded"])

        changed = _observations()
        changed["gpu_rows"][0]["memory.total"] = "40960"
        changed["gpu_rows"][1]["memory.total"] = "40960"
        changed_leaf = build_runtime_attestation(changed, phase="leaf")
        with self.assertRaisesRegex(ValueError, "differs from preflight"):
            validate_runtime_attestation(changed_leaf, reference=preflight)

        mutated = copy.deepcopy(leaf)
        mutated["runtime_class_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "attestation is invalid"):
            validate_runtime_attestation(mutated, reference=preflight)

    def test_stage_runner_consumes_outer_source_identity_without_inner_git(self):
        expected = "a" * 40
        with (
            mock.patch.dict(
                p1_stage_runner.os.environ,
                {
                    "GEOROUTE_SOURCE_IDENTITY_VERIFIED": "1",
                    "GEOROUTE_EXPECTED_COMMIT": expected,
                },
                clear=False,
            ),
            mock.patch.object(p1_stage_runner.subprocess, "run") as run,
        ):
            self.assertEqual(p1_stage_runner._current_commit(), expected)
            p1_stage_runner._assert_clean_source_snapshot()
        run.assert_not_called()

    def test_train_and_test_consume_outer_source_identity_without_inner_git(self):
        expected = "b" * 40
        with (
            mock.patch.dict(
                p1_stage_runner.os.environ,
                {
                    "GEOROUTE_SOURCE_IDENTITY_VERIFIED": "1",
                    "GEOROUTE_EXPECTED_COMMIT": expected,
                },
                clear=False,
            ),
            mock.patch(
                "tools.bata.georoute_official_comparable_contract.subprocess.run"
            ) as run,
        ):
            require_clean_formal_checkout(expected_commit=expected, root=ROOT)
        run.assert_not_called()

    def test_accuracy_storage_preflight_keeps_bound_work_dir_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            run_root.mkdir()
            cell_root = run_root / "p1_accuracy" / "DO_seed3407"
            with mock.patch.object(
                p1_stage_runner,
                "storage_capacity_receipt",
                return_value={"status": "PASS_STORAGE_PREFLIGHT"},
            ):
                receipt_path = p1_stage_runner._write_accuracy_storage_preflight(
                    run_root=run_root,
                    cell_root=cell_root,
                    arm="DO",
                    seed=3407,
                )
            self.assertFalse(cell_root.exists())
            self.assertEqual(
                receipt_path,
                run_root / "control" / "storage_preflights" / "DO_seed3407.json",
            )
            self.assertEqual(
                json.loads(receipt_path.read_text(encoding="utf-8")),
                {"status": "PASS_STORAGE_PREFLIGHT"},
            )

    def test_q_postrun_accepts_global_dynamic_budget_without_target_k(self):
        self.assertNotIn("torch", sys.modules)
        validate_p1_q_routing_audit(_q_routing_audit())
        summary = _summarize_fixture(
            _formal_telemetry_payload(_q_route_telemetry()),
            arm="Q",
        )
        self.assertEqual(
            summary["schema_version"],
            "georoute_p1_q_telemetry_summary_v001",
        )
        self.assertIsNone(summary["target_k"])
        self.assertEqual(summary["window_token_budget"], 24_576)
        self.assertTrue(summary["window_budget_is_global"])
        self.assertTrue(summary["dynamic_k_t"])
        self.assertEqual(summary["k_t"]["min"], 0)
        self.assertEqual(summary["k_t"]["max"], 128)
        self.assertEqual(summary["k_t"]["zero_count"], 1)
        self.assertEqual(summary["unique_physical_tokens_per_window"], 24_576)
        self.assertEqual(summary["padded_heavy_tokens_per_window"], 0)
        self.assertEqual(summary["ragged_execution"], "true_clip_ragged_no_padding")
        self.assertNotIn("torch", sys.modules)

    def test_q_postrun_rejects_fixed_k_duplicates_padding_and_broken_ragged(self):
        cases = {}

        fixed_target = _formal_telemetry_payload(_q_route_telemetry())
        fixed_target["records"][0]["route"]["target_k"] = 64
        cases["fixed_target_k"] = fixed_target

        duplicate = _formal_telemetry_payload(_q_route_telemetry())
        duplicate["records"][0]["route"]["ragged_execution"][
            "unique_physical_tokens"
        ] = 24_575
        cases["nonunique_physical_tokens"] = duplicate

        padded = _formal_telemetry_payload(_q_route_telemetry())
        padded["records"][0]["route"]["ragged_execution"][
            "padded_heavy_tokens"
        ] = 1
        cases["padded_heavy_token"] = padded

        broken_k_t = _formal_telemetry_payload(_q_route_telemetry())
        broken_k_t["records"][0]["route"]["k_t"]["values"][0] = 1
        cases["wrong_global_k_t_sum"] = broken_k_t

        broken_pairs = _formal_telemetry_payload(_q_route_telemetry())
        broken_pairs["records"][0]["route"]["ragged_execution"][
            "attention_pairs"
        ] += 1
        cases["broken_attention_pairs"] = broken_pairs

        leaked = _formal_telemetry_payload(_q_route_telemetry())
        leaked["records"][0]["route"]["teacher_used"] = True
        cases["teacher_route_leak"] = leaked

        for label, payload in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                _summarize_fixture(payload, arm="Q")
        self.assertNotIn("torch", sys.modules)

    def test_legacy_uniform_postrun_fallback_remains_exact_k(self):
        legacy_route = {
            "role_counts": {"uniform": 64},
            "target_k": 64,
            "selected_index_sha256": "d" * 64,
        }
        summary = _summarize_fixture(
            _formal_telemetry_payload(legacy_route),
            arm="U",
        )
        self.assertEqual(summary["target_k"], 64)
        self.assertEqual(summary["role_counts"], {"uniform": 64})
        self.assertNotIn("torch", sys.modules)

    def test_p1_deployer_and_launcher_freeze_one_held_fifteen_job_entry(self):
        deployer = (
            ROOT / "tools" / "bata" / "deploy_georoute_official_development.py"
        ).read_text(encoding="utf-8")
        launcher = (
            ROOT / "scripts" / "run_georoute_official_development_stage_slurm.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--mode", choices=("formal", "p1")', deployer)
        self.assertIn("additional_jobs=15", deployer)
        self.assertIn('"runtime_preflight": runtime_preflight_job', deployer)
        self.assertIn('"cost": cost_jobs', deployer)
        self.assertIn("finalizer_afterany_all_fourteen_predecessors", deployer)
        self.assertIn("release_all_fifteen_atomically", deployer)
        self.assertIn("_release_p1_jobs_from_receipts(", deployer)
        self.assertIn("_cancel_p1_jobs_and_verify(", deployer)
        self.assertLess(
            launcher.index("georoute_p1_runtime_attestor"),
            launcher.index("import numpy"),
        )
        self.assertIn("apptainer exec --nv", launcher)
        self.assertIn("--phase preflight", launcher)
        self.assertIn("--phase leaf", launcher)
        self.assertIn("--task cost", launcher)

    def test_p1_receipt_mutation_cancels_before_release(self):
        self.assertNotIn("torch", sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            fixture = _p1_release_receipt_fixture(Path(directory))
            submission = copy.deepcopy(fixture["submission"])
            submission["predecessor_job_ids"][0] = "999999"
            submission.pop("receipt_sha256")
            submission["receipt_sha256"] = canonical_sha256(submission)
            fixture["submission_path"].write_text(
                json.dumps(submission, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(p1_deployer, "_release_p1_jobs_checked") as release,
                mock.patch.object(
                    p1_deployer, "_cancel_p1_jobs_and_verify"
                ) as cancel,
            ):
                with self.assertRaisesRegex(ValueError, "before release"):
                    p1_deployer._release_p1_jobs_from_receipts(
                        fixture["deployment_path"],
                        fixture["submission_path"],
                        expected_commit=fixture["deployment"]["runtime_commit"],
                        expected_submitted=fixture["job_ids"],
                        expected_deployment_sha256=fixture["deployment"][
                            "deployment_sha256"
                        ],
                        expected_submission_sha256=fixture["submission"][
                            "receipt_sha256"
                        ],
                    )
            release.assert_not_called()
            cancel.assert_called_once_with(fixture["job_ids"])
        self.assertNotIn("torch", sys.modules)

    def test_p1_release_failure_cancels_the_complete_population(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _p1_release_receipt_fixture(Path(directory))
            with (
                mock.patch.object(
                    p1_deployer,
                    "_release_p1_jobs_checked",
                    side_effect=RuntimeError("partial release"),
                ) as release,
                mock.patch.object(
                    p1_deployer, "_cancel_p1_jobs_and_verify"
                ) as cancel,
            ):
                with self.assertRaisesRegex(RuntimeError, "partial release"):
                    p1_deployer._release_p1_jobs_from_receipts(
                        fixture["deployment_path"],
                        fixture["submission_path"],
                        expected_commit=fixture["deployment"]["runtime_commit"],
                        expected_submitted=fixture["job_ids"],
                        expected_deployment_sha256=fixture["deployment"][
                            "deployment_sha256"
                        ],
                        expected_submission_sha256=fixture["submission"][
                            "receipt_sha256"
                        ],
                    )
            release.assert_called_once_with(fixture["job_ids"])
            cancel.assert_called_once_with(fixture["job_ids"])
        self.assertNotIn("torch", sys.modules)

    def test_p1_cancellation_command_failure_is_fail_closed(self):
        job_ids = ("910000", "910001")
        responses = [
            subprocess.CompletedProcess(
                ["scancel", *job_ids],
                returncode=1,
                stdout="",
                stderr="permission denied",
            ),
            subprocess.CompletedProcess(
                ["squeue"],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]
        with mock.patch.object(
            p1_deployer.subprocess,
            "run",
            side_effect=responses,
        ) as run_control:
            with self.assertRaisesRegex(RuntimeError, "scancel failed"):
                p1_deployer._cancel_p1_jobs_and_verify(
                    job_ids,
                    max_polls=1,
                    poll_interval_seconds=0.0,
                )
        self.assertEqual(run_control.call_count, 2)
        self.assertNotIn("torch", sys.modules)

    def test_p1_bounded_cancellation_rejects_surviving_jobs(self):
        job_ids = ("910000", "910001")
        responses = [
            subprocess.CompletedProcess(
                ["scancel", *job_ids],
                returncode=0,
                stdout="",
                stderr="",
            ),
            *[
                subprocess.CompletedProcess(
                    ["squeue"],
                    returncode=0,
                    stdout="910000\n",
                    stderr="",
                )
                for _ in range(2)
            ],
        ]
        with mock.patch.object(
            p1_deployer.subprocess,
            "run",
            side_effect=responses,
        ) as run_control:
            with self.assertRaisesRegex(RuntimeError, "surviving job IDs: 910000"):
                p1_deployer._cancel_p1_jobs_and_verify(
                    job_ids,
                    max_polls=2,
                    poll_interval_seconds=0.0,
                )
        self.assertEqual(run_control.call_count, 3)
        self.assertNotIn("torch", sys.modules)

    def test_p1_cost_contract_is_eight_leaves_and_dn_gate_is_literal(self):
        self.assertNotIn("torch", sys.modules)
        self.assertEqual(
            tuple(P1_COST_LEAF_SPECS),
            (
                "DO_ABBA",
                "DO_BAAB",
                "DN_ABBA",
                "DN_BAAB",
                "U_ABBA",
                "U_BAAB",
                "R_ABBA",
                "R_BAAB",
            ),
        )
        self.assertEqual(p1_cost_leaf_sequence("DN_ABBA"), ("DN", "Q", "Q", "DN"))
        self.assertEqual(p1_cost_leaf_sequence("R_BAAB"), ("Q", "R", "R", "Q"))
        leaves = {leaf_id: _p1_cost_rows(leaf_id) for leaf_id in P1_COST_LEAF_SPECS}
        analysis = analyze_p1_cost_leaves(leaves, bootstrap_replicates=25)
        self.assertEqual(P1_COST_RATIO_LIMIT, 0.85)
        self.assertEqual(analysis["dense_denominator"], "DN")
        self.assertTrue(analysis["q_over_dn_cost_gate_passed"])
        self.assertTrue(analysis["comparisons"]["DO"]["report_only"])
        self.assertFalse(analysis["comparisons"]["DN"]["report_only"])
        for metric in analysis["comparisons"]["DN"]["metrics"].values():
            self.assertAlmostEqual(metric["one_sided_95_upper_bound"], 0.85)
            self.assertTrue(metric["upper_bound_le_0_85"])
            self.assertEqual(metric["tolerance"], 0.0)
        for leaf_id in ("DN_ABBA", "DN_BAAB"):
            for row in leaves[leaf_id]:
                if row["arm"] == "Q":
                    row["end_to_end_serial_ms"] = 86.0
                    row["gross_gpu_energy_j_per_sample"] = 8.6
                    row.pop("sample_sha256")
                    row["sample_sha256"] = canonical_sha256(row)
        failed = analyze_p1_cost_leaves(leaves, bootstrap_replicates=10)
        self.assertFalse(failed["q_over_dn_cost_gate_passed"])
        self.assertNotIn("torch", sys.modules)

    def test_report_only_matching_freezes_short_boundary_and_bin_precedence(self):
        ground_truth = [
            {"id": "g-hit", "label": "A", "start": 0.0, "end": 5.0},
            {"id": "g-start", "label": "A", "start": 10.0, "end": 20.0},
            {"id": "g-class", "label": "B", "start": 30.0, "end": 35.0},
        ]
        predictions = [
            {
                "id": "p-hit",
                "label": "A",
                "start": 0.0,
                "end": 5.0,
                "score": 0.9,
            },
            {
                "id": "p-start",
                "label": "A",
                "start": 4.0,
                "end": 20.0,
                "score": 0.8,
            },
            {
                "id": "p-duplicate",
                "label": "A",
                "start": 0.0,
                "end": 5.0,
                "score": 0.1,
            },
            {
                "id": "p-class",
                "label": "C",
                "start": 30.0,
                "end": 35.0,
                "score": 0.7,
            },
            {
                "id": "p-other",
                "label": "D",
                "start": 50.0,
                "end": 51.0,
                "score": 0.6,
            },
        ]
        report = _match_p1_video(ground_truth, predictions)
        self.assertEqual(tuple(report["gt_bins"]), P1_GT_BINS)
        self.assertEqual(
            tuple(report["unmatched_prediction_bins"]),
            P1_UNMATCHED_PREDICTION_BINS,
        )
        self.assertEqual(report["gt_bins"]["HIT_070"], 1)
        self.assertEqual(report["gt_bins"]["START_LIMITED"], 1)
        self.assertEqual(report["gt_bins"]["CLASS_CONFUSION"], 1)
        self.assertEqual(report["unmatched_prediction_bins"]["DUPLICATE_FP"], 1)
        self.assertEqual(report["unmatched_prediction_bins"]["CLASS_CONFUSION_FP"], 1)
        self.assertEqual(report["unmatched_prediction_bins"]["OTHER_FP"], 1)
        self.assertEqual(report["short_gt_count"], 2)
        self.assertEqual(report["short_hit_070_count"], 1)

    def test_p1_finalizer_names_only_the_frozen_terminal_decisions(self):
        source = (
            ROOT / "tools" / "bata" / "finalize_georoute_official_development.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"NO_SURVIVOR_INVALID_P1"', source)
        self.assertIn('"STOP_Q_CORE_P1"', source)
        self.assertIn('"Q_CORE_P1_SURVIVES"', source)
        self.assertIn('"dense_denominator": "DN"', source)
        self.assertIn('"do_mandatory_report_only": True', source)
        self.assertIn('"short_actions_affect_gate": False', source)
        self.assertIn('"boundary_diagnostics_affect_gate": False', source)
        self.assertIn('"high_iou_decomposition_affects_gate": False', source)


if __name__ == "__main__":
    unittest.main()
