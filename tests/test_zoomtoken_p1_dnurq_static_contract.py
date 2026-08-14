from __future__ import annotations

import copy
import hashlib
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path


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
        preflight = build_runtime_attestation(_observations(), phase="preflight")
        leaf = build_runtime_attestation(_observations(), phase="leaf")
        validate_runtime_attestation(preflight)
        validate_runtime_attestation(leaf, reference=preflight)
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


if __name__ == "__main__":
    unittest.main()
