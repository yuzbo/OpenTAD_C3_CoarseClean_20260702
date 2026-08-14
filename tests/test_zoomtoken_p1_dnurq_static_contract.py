from __future__ import annotations

import copy
import runpy
import sys
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


if __name__ == "__main__":
    unittest.main()
