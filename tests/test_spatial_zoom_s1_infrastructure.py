from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import pytest
from mmengine.config import Config
import tools.bata.spatial_zoom_s1_power as s1_power
import tools.bata.spatial_zoom_s1_profile_recovery as s1_profile_recovery
import tools.bata.spatial_zoom_s1_test_open as s1_test_open

from tools.bata.analyze_spatial_zoom_s1_results import (
    DetectionCorpus,
    _class_ap,
    _map_vector,
    _paired_bayesian_weights,
    _simultaneous_max_t_lower_bounds,
    aggregate_s1_runs,
    assert_official_evaluator_parity,
    evaluate_corpus,
    seal_s1_result_report,
    validate_s1_result_report_envelope,
)
from tools.bata.spatial_zoom_s1_contract import (
    S1_PRETRAINED_CHECKPOINT_FILENAME,
    S1_PRETRAINED_CHECKPOINT_SHA256,
    S1_PROFILE_ORDER_SEED,
    S1_RESOLUTIONS,
    build_s1_profile_order,
    build_s1_manifest,
    canonical_sha256,
    sha256_file,
    stable_id_hash,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_evidence import (
    S1_TEST_RUNTIME_EVIDENCE_FIELDS,
    _validate_test_runtime_evidence,
    write_s1_gate_evidence,
)
from tools.bata.spatial_zoom_s1_evidence import (
    validate_s1_checkpoint_metadata_for_binding,
)
from tools.bata.spatial_zoom_s1_cost import (
    S1_PROFILE_PROTOCOL,
    build_profile_summary,
    compare_resolution_profiles,
    make_profile_exposure_id,
)
from tools.bata.run_spatial_zoom_s1_precheck import (
    S1_EXPECTED_UNUSED_TRAINABLE_PARAMETERS,
    _register_opentad_runtime_modules,
    _validate_expected_unused_trainable_parameters,
    _validate_gradient_coverage_evidence,
    _validate_interpolation_calls,
    _validate_pretrained_load_audit,
    build_precheck_spec,
    run_precheck,
    validate_precheck_certificate,
)
from tools.bata.profile_spatial_zoom_s1 import (
    _dataset_exposure_topology,
    _hardware_identity,
    _sample_identity,
    create_profile_attempt_marker,
    validate_profile_attempt_marker,
    validate_profile_order_ready,
)
from tools.bata.spatial_zoom_s1_profile_recovery import (
    S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE,
    S1_BUFFERED_SIDECAR_PROFILE_RECOVERY_SCHEMA,
    S1_BUFFERED_SIDECAR_RECOVERY_REASON,
    S1_BUFFERED_TRACE_PUBLICATION_MODE,
    S1_CHAINED_PROFILE_RECOVERY_SCHEMA,
    S1_CHAINED_RECOVERY_REASON,
    S1_PROFILE_FAILURE_SIGNATURE,
    S1_POWER_FAILURE_SIGNATURE,
    S1_PROFILE_RECOVERY_REASON,
    S1_PROFILE_RECOVERY_SCHEMA,
    S1_SCHEMA_COMPAT_RECOVERY_REASON,
    S1_SIDECAR_PROFILE_RECOVERY_SCHEMA,
    S1_SIDECAR_RECOVERY_REASON,
    S1_STEP_RUNTIME_RECOVERY_REASON,
    profile_campaign_prefix,
    validate_profile_recovery_certificate,
)
from tools.bata.spatial_zoom_s1_power import (
    S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA,
    S1_POWER_BUFFERED_SIDECAR_RESULT_SCHEMA,
    S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE,
    S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX,
    S1_POWER_SIDECAR_ATTEMPT_SCHEMA,
    S1_POWER_SIDECAR_BACKEND,
    S1_POWER_SIDECAR_RESULT_SCHEMA,
    NvmlPowerSampler,
    NvmlSidecarPowerSampler,
    _load_sidecar_trace,
    run_nvml_sidecar,
    salvage_nvml_sidecar_attempt,
    summarize_power_cadence,
    validate_nvml_sidecar_attempt,
    validate_nvml_sidecar_cadence_failure,
)
from tools.bata.spatial_zoom_s1_sidecar_gate import (
    build_sidecar_gate_evidence,
    load_sidecar_gate_evidence,
    sidecar_gate_hardware_class,
    sidecar_gate_path,
    validate_sidecar_gate_runtime_identity,
    write_sidecar_gate_evidence,
)
from tools.bata.select_spatial_zoom_s1_checkpoint import (
    select_s1_checkpoint,
    validate_checkpoint_selection,
)
from tools.bata.spatial_zoom_s1_training import (
    S1_MIN_FREE_STORAGE_BYTES,
    S1_CHECKPOINT_METADATA_SCHEMA,
    S1_CHECKPOINT_SIDECAR_SCHEMA,
    bind_s1_training_config,
    build_s1_experiment_identity,
    build_s1_checkpoint_metadata,
    checkpoint_sidecar_path,
    require_slurm_memory_limit_mb,
    require_slurm_single_gpu_allocation,
    should_save_s1_checkpoint,
    validate_bound_s1_training_config,
    validate_s1_checkpoint_sidecar,
)
from tools.bata.spatial_zoom_s1_test_open import (
    _shared_experiment_identity,
    _shared_precheck_identity,
    create_global_test_open_marker,
    recover_global_test_open_certificate,
    validate_global_test_open_marker,
)
from tools.bata.validate_spatial_zoom_s1 import (
    CONFIG_PATHS,
    validate_config_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def _annotation_fixture() -> dict:
    database = {}
    labels = ("A", "B")
    for index in range(12):
        label = labels[index % 2]
        database[f"fit_video_{index:02d}"] = {
            "subset": "training",
            "duration": 20.0,
            "frame": 600,
            "annotations": [
                {"label": label, "segment": [0.0, 1.0]},
                {"label": label, "segment": [3.0, 5.0]},
                {"label": label, "segment": [7.0, 10.0]},
                {"label": label, "segment": [12.0, 16.0]},
            ],
        }
    for index in range(4):
        database[f"test_video_{index:02d}"] = {
            "subset": "validation",
            "duration": 20.0,
            "frame": 600,
            "annotations": [
                {"label": labels[index % 2], "segment": [0.0, 1.0]},
                {"label": labels[index % 2], "segment": [3.0, 5.0]},
                {"label": labels[index % 2], "segment": [7.0, 10.0]},
                {"label": labels[index % 2], "segment": [12.0, 16.0]},
            ],
        }
    return {"database": database}


def _write_annotation(tmp_path: Path) -> Path:
    path = tmp_path / "thumos_fixture.json"
    path.write_text(json.dumps(_annotation_fixture()), encoding="utf-8")
    return path


def test_dense_resolution_configs_are_a_strict_matched_matrix() -> None:
    summary = validate_config_matrix(
        {resolution: ROOT / path for resolution, path in CONFIG_PATHS.items()}
    )

    assert summary["status"] == "PASS"
    assert tuple(summary["resolutions"]) == S1_RESOLUTIONS
    assert summary["only_spatial_resolution_differs"] is True
    assert summary["official_dense160_matched"] is True
    assert summary["temporal_protocol_matched"] is True
    assert summary["model_optimizer_evaluator_matched"] is True
    assert summary["protocol_fingerprint"]
    for resolution in S1_RESOLUTIONS:
        row = summary["configs"][str(resolution)]
        assert row["runtime_resolution"] == resolution
        assert row["window_size"] == 768
        assert row["tubelet_points"] == 384


def test_precheck_specs_cover_native_and_interpolated_position_grids() -> None:
    specs = {
        resolution: build_precheck_spec(ROOT / CONFIG_PATHS[resolution])
        for resolution in S1_RESOLUTIONS
    }
    assert specs[160]["runtime_grid"] == [10, 10]
    assert specs[224]["runtime_grid"] == [14, 14]
    assert specs[256]["runtime_grid"] == [16, 16]
    assert specs[160]["position_interpolation_expected"] is True
    assert specs[224]["position_interpolation_expected"] is False
    assert specs[256]["position_interpolation_expected"] is True
    assert specs[256]["clip_output_shape"] == [1, 384, 8, 16, 16]
    assert specs[256]["full_detector_feature_shape"] == [1, 384, 768]
    _validate_interpolation_calls(specs[160], [[10, 10]])
    _validate_interpolation_calls(specs[224], [])
    _validate_interpolation_calls(specs[256], [[16, 16]])
    with pytest.raises(AssertionError, match="exactly match"):
        _validate_interpolation_calls(specs[224], [[13, 13]])
    with pytest.raises(AssertionError, match="exactly match"):
        _validate_interpolation_calls(specs[256], [[16, 16], [16, 16]])


def test_pretrained_load_audit_fails_closed_on_partial_or_unverified_state() -> None:
    expected_sha = "a" * 64
    complete = {
        "verified": True,
        "checkpoint_sha256": expected_sha,
        "model_core_parameter_count": 144,
        "loaded_core_parameter_count": 144,
        "model_core_parameter_numel": 21_000_000,
        "loaded_core_parameter_numel": 21_000_000,
        "core_parameter_numel_coverage": 1.0,
        "core_keyset_sha256": "b" * 64,
        "missing_core_keys": [],
        "shape_mismatch_core_keys": [],
        "value_mismatch_core_keys": [],
    }
    _validate_pretrained_load_audit(complete, expected_sha256=expected_sha)
    partial = copy.deepcopy(complete)
    partial["loaded_core_parameter_count"] -= 1
    with pytest.raises(ValueError, match="incomplete"):
        _validate_pretrained_load_audit(partial, expected_sha256=expected_sha)
    forged = copy.deepcopy(complete)
    forged["verified"] = False
    with pytest.raises(ValueError, match="no verified"):
        _validate_pretrained_load_audit(forged, expected_sha256=expected_sha)


def test_formal_precheck_rejects_partial_matrix_and_non_cuda_execution() -> None:
    paths = [ROOT / CONFIG_PATHS[value] for value in S1_RESOLUTIONS]
    static_certificate = run_precheck(paths, mode="static", device="cpu", amp=False)
    assert (
        validate_precheck_certificate(static_certificate, require_full=False)["status"]
        == "PASS"
    )
    with pytest.raises(ValueError, match="complete audited 3-config matrix"):
        run_precheck(paths[:2], mode="full", device="cuda:0", amp=True)
    with pytest.raises(ValueError, match="requires cuda:0"):
        run_precheck(paths, mode="full", device="cpu", amp=False)
    with pytest.raises(ValueError, match="frozen contract"):
        run_precheck(
            paths,
            mode="full",
            device="cuda:0",
            amp=True,
            expected_pretrained_sha256="0" * 64,
        )


def test_formal_precheck_registers_model_pipeline_transforms(monkeypatch) -> None:
    imported = []
    monkeypatch.setattr(
        "tools.bata.run_spatial_zoom_s1_precheck.importlib.import_module",
        imported.append,
    )
    _register_opentad_runtime_modules()
    assert imported == ["opentad.datasets", "opentad.models.backbones"]


def _valid_gradient_coverage_evidence() -> dict:
    expected = sorted(S1_EXPECTED_UNUSED_TRAINABLE_PARAMETERS)
    return {
        "trainable_parameter_tensors": 12,
        "gradient_required_parameter_tensors": 10,
        "expected_unused_trainable_parameters": expected,
        "observed_missing_gradient_parameters": expected,
        "finite_gradient_tensors": 10,
        "nonzero_gradient_tensors": 6,
        "gradient_coverage": {
            "backbone": {
                "trainable_parameter_tensors": 6,
                "gradient_required_parameter_tensors": 4,
                "expected_unused_trainable_parameter_tensors": 2,
                "gradient_tensors": 4,
                "nonzero_gradient_tensors": 2,
                "all_present_gradients_finite": True,
            },
            "projection": {
                "trainable_parameter_tensors": 3,
                "gradient_required_parameter_tensors": 3,
                "expected_unused_trainable_parameter_tensors": 0,
                "gradient_tensors": 3,
                "nonzero_gradient_tensors": 2,
                "all_present_gradients_finite": True,
            },
            "rpn_head": {
                "trainable_parameter_tensors": 3,
                "gradient_required_parameter_tensors": 3,
                "expected_unused_trainable_parameter_tensors": 0,
                "gradient_tensors": 3,
                "nonzero_gradient_tensors": 2,
                "all_present_gradients_finite": True,
            },
        },
    }


def test_full_precheck_only_allows_exact_videomae_fc_norm_bypass() -> None:
    expected = sorted(S1_EXPECTED_UNUSED_TRAINABLE_PARAMETERS)
    assert expected == [
        "backbone.model.backbone.fc_norm.bias",
        "backbone.model.backbone.fc_norm.weight",
    ]
    assert _validate_expected_unused_trainable_parameters(expected) == expected
    with pytest.raises(RuntimeError, match="missing_expected"):
        _validate_expected_unused_trainable_parameters(expected[:1])
    with pytest.raises(RuntimeError, match="unexpected"):
        _validate_expected_unused_trainable_parameters(
            expected + ["backbone.unexpected.weight"]
        )


def test_full_precheck_gradient_evidence_fails_closed_on_contract_drift() -> None:
    evidence = _valid_gradient_coverage_evidence()
    _validate_gradient_coverage_evidence(evidence)

    missing_allowlisted = copy.deepcopy(evidence)
    missing_allowlisted["observed_missing_gradient_parameters"] = [
        S1_EXPECTED_UNUSED_TRAINABLE_PARAMETERS[0]
    ]
    with pytest.raises(ValueError, match="incomplete detector-gradient evidence"):
        _validate_gradient_coverage_evidence(missing_allowlisted)

    unknown_disconnect = copy.deepcopy(evidence)
    unknown_disconnect["observed_missing_gradient_parameters"].append(
        "projection.unexpected.weight"
    )
    with pytest.raises(ValueError, match="incomplete detector-gradient evidence"):
        _validate_gradient_coverage_evidence(unknown_disconnect)

    forged_allowlist = copy.deepcopy(evidence)
    forged_allowlist["expected_unused_trainable_parameters"].append(
        "projection.unexpected.weight"
    )
    with pytest.raises(ValueError, match="incomplete detector-gradient evidence"):
        _validate_gradient_coverage_evidence(forged_allowlist)

    incomplete_backbone = copy.deepcopy(evidence)
    incomplete_backbone["gradient_coverage"]["backbone"]["gradient_tensors"] = 3
    with pytest.raises(ValueError, match="invalid backbone gradient coverage"):
        _validate_gradient_coverage_evidence(incomplete_backbone)

    forged_component_count = copy.deepcopy(evidence)
    forged_component_count["gradient_coverage"]["backbone"][
        "expected_unused_trainable_parameter_tensors"
    ] = 3
    with pytest.raises(ValueError, match="invalid backbone gradient coverage"):
        _validate_gradient_coverage_evidence(forged_component_count)

    underreported_component = copy.deepcopy(evidence)
    for key in (
        "trainable_parameter_tensors",
        "gradient_required_parameter_tensors",
        "gradient_tensors",
    ):
        underreported_component["gradient_coverage"]["backbone"][key] -= 1
    with pytest.raises(ValueError, match="component gradient totals"):
        _validate_gradient_coverage_evidence(underreported_component)


def test_videomae_fc_norm_is_bypassed_only_for_dense_tad_feature_maps() -> None:
    source = (ROOT / "opentad" / "models" / "backbones" / "vit_adapter.py").read_text(
        encoding="utf-8"
    )
    dense_return = source.index("if self.return_feat_map:")
    classification_norm = source.index("if self.fc_norm is not None:", dense_return)
    assert dense_return < classification_norm
    for resolution in S1_RESOLUTIONS:
        cfg = Config.fromfile(str(ROOT / CONFIG_PATHS[resolution]))
        assert cfg.model.backbone.backbone.return_feat_map is True


def test_formal_s1_accepts_slurm_assigned_single_gpu(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "6")
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    assert require_slurm_single_gpu_allocation() == "6"

    monkeypatch.setenv("SLURM_JOB_GPUS", "6,7")
    monkeypatch.setenv("SLURM_STEP_GPUS", "7")
    assert require_slurm_single_gpu_allocation() == "7"

    monkeypatch.setenv("SLURM_STEP_GPUS", "3")
    with pytest.raises(RuntimeError, match="not a member"):
        require_slurm_single_gpu_allocation()

    monkeypatch.setenv("SLURM_STEP_GPUS", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="exactly one Slurm-visible"):
        require_slurm_single_gpu_allocation()


def test_s1_test_runtime_evidence_binds_the_recovery_certificate(
    tmp_path: Path, monkeypatch
) -> None:
    recovery_path = tmp_path / "recovery_certificate.json"
    recovery_path.write_text('{"sealed": true}\n', encoding="utf-8")
    binding = {"code_commit": "a" * 40}
    recovery = {
        "profile_code_commit": "b" * 40,
        "certificate_sha256": "c" * 64,
        "campaign_id": "runtime-campaign",
        "formal_test_runtime_mode": (
            s1_profile_recovery.S1_STEP_SCOPED_TEST_RUNTIME_MODE
        ),
    }

    def fake_load(path, *, binding: dict, verify_checkout: bool):
        assert Path(path).resolve() == recovery_path.resolve()
        assert binding["code_commit"] == "a" * 40
        assert verify_checkout is False
        return copy.deepcopy(recovery)

    monkeypatch.setattr(
        s1_profile_recovery,
        "load_profile_recovery_certificate",
        fake_load,
    )
    payload = {
        "formal_test_runtime_mode": (
            s1_profile_recovery.S1_STEP_SCOPED_TEST_RUNTIME_MODE
        ),
        "training_code_commit": "a" * 40,
        "test_runtime_code_commit": "b" * 40,
        "profile_recovery_certificate_path": str(recovery_path.resolve()),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": "c" * 64,
        "profile_recovery_campaign_id": "runtime-campaign",
    }
    marker = {
        key: value for key, value in payload.items() if key != "training_code_commit"
    }
    assert set(payload) == set(S1_TEST_RUNTIME_EVIDENCE_FIELDS)
    assert (
        _validate_test_runtime_evidence(
            payload,
            binding=binding,
            marker=marker,
        )
        == recovery
    )

    incomplete = dict(payload)
    incomplete.pop("profile_recovery_campaign_id")
    with pytest.raises(ValueError, match="incomplete"):
        _validate_test_runtime_evidence(incomplete, binding=binding)

    forged = dict(payload)
    forged["test_runtime_code_commit"] = "d" * 40
    with pytest.raises(ValueError, match="test_runtime_code_commit"):
        _validate_test_runtime_evidence(forged, binding=binding)


def test_formal_s1_reads_the_tightest_finite_step_memory_limit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_GPUS", "7")
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    cgroup_root = tmp_path / "cgroup"
    relative = Path("system.slice/slurmstepd.scope/job_123/step_0/user/task_0")
    task_root = cgroup_root / relative
    task_root.mkdir(parents=True)
    (task_root / "memory.max").write_text("max\n", encoding="utf-8")
    (task_root.parent / "memory.max").write_text("100663296000\n", encoding="utf-8")
    (task_root.parent.parent.parent / "memory.max").write_text(
        "130442854400\n", encoding="utf-8"
    )
    proc_cgroup = tmp_path / "proc_self_cgroup"
    proc_cgroup.write_text(f"0::/{relative.as_posix()}\n", encoding="utf-8")

    assert (
        require_slurm_memory_limit_mb(
            minimum_mb=90000,
            proc_cgroup_path=proc_cgroup,
            cgroup_root=cgroup_root,
        )
        == 96000
    )
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "124400")
    assert (
        require_slurm_memory_limit_mb(
            minimum_mb=90000,
            proc_cgroup_path=proc_cgroup,
            cgroup_root=cgroup_root,
        )
        == 96000
    )
    with pytest.raises(RuntimeError, match="below the required headroom"):
        require_slurm_memory_limit_mb(
            minimum_mb=96001,
            proc_cgroup_path=proc_cgroup,
            cgroup_root=cgroup_root,
        )
    empty_cgroup_root = tmp_path / "empty_cgroup"
    empty_cgroup_root.mkdir()
    with pytest.raises(RuntimeError, match="cgroup v2"):
        require_slurm_memory_limit_mb(
            minimum_mb=90000,
            proc_cgroup_path=proc_cgroup,
            cgroup_root=empty_cgroup_root,
        )
    v1_proc_cgroup = tmp_path / "proc_self_cgroup_v1"
    v1_proc_cgroup.write_text(f"5:memory:/{relative.as_posix()}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cgroup v2"):
        require_slurm_memory_limit_mb(
            minimum_mb=90000,
            proc_cgroup_path=v1_proc_cgroup,
            cgroup_root=cgroup_root,
        )


def test_hardware_identity_binds_logical_cuda_uuid_to_step_gpu(
    monkeypatch,
) -> None:
    properties = SimpleNamespace(
        name="S1 GPU",
        total_memory=48 * 1024**3,
        major=8,
        minor=0,
        multi_processor_count=108,
    )
    fake_cuda = SimpleNamespace(
        get_device_properties=lambda _device: properties,
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    expected_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    monkeypatch.setattr(
        "tools.bata.profile_spatial_zoom_s1._cuda_driver_device_uuid_hex",
        lambda _ordinal=0: "aaaaaaaabbbbccccddddeeeeeeeeeeee",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_GPUS", "1,2")
    monkeypatch.setenv("SLURM_STEP_GPUS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "124400")
    nvidia_smi = SimpleNamespace(
        returncode=0,
        stdout=(
            f"{expected_uuid}, 00000000:01:00.0, 550.54, Enabled, Default, "
            "300.00, 1410, 1215\n"
        ),
        stderr="",
    )

    def fake_nvidia_smi(command, **_kwargs):
        assert command[-2:] == ["-i", "0"]
        return nvidia_smi

    monkeypatch.setattr(
        "tools.bata.profile_spatial_zoom_s1.subprocess.run",
        fake_nvidia_smi,
    )
    identity = _hardware_identity(
        fake_torch,
        "cuda:0",
        physical_gpu_id="1",
        allocated_cpu_ids=(0, 1, 2, 3, 4),
        detector_cpu_ids=(0, 1, 2, 3),
        sidecar_cpu_id=4,
        memory_limit_mb=96000,
    )
    assert identity["cuda_runtime_device_ordinal"] == 0
    assert (
        identity["cuda_runtime_device_uuid_hex"] == "aaaaaaaabbbbccccddddeeeeeeeeeeee"
    )
    assert identity["cuda_visible_device_uuid"] == expected_uuid
    assert identity["nvidia_smi"]["uuid"] == expected_uuid
    assert identity["physical_gpu_id"] == "1"
    assert identity["nvidia_smi_query_selector"] == "0"
    assert identity["slurm_gpu_scope"]["step_id"] == "0"
    assert identity["slurm_resources"]["effective_step_memory_limit_mb"] == 96000

    monkeypatch.setattr(
        "tools.bata.profile_spatial_zoom_s1._cuda_driver_device_uuid_hex",
        lambda _ordinal=0: "11111111222233334444555555555555",
    )
    with pytest.raises(RuntimeError, match="logical cuda:0 UUID differs"):
        _hardware_identity(
            fake_torch,
            "cuda:0",
            physical_gpu_id="1",
            allocated_cpu_ids=(0, 1, 2, 3, 4),
            detector_cpu_ids=(0, 1, 2, 3),
            sidecar_cpu_id=4,
            memory_limit_mb=96000,
        )


def test_s1_slurm_launchers_use_kernel_assigned_rendezvous_ports() -> None:
    for filename in (
        "run_spatial_zoom_s1_train_slurm.sh",
        "run_spatial_zoom_s1_test_profile_slurm.sh",
    ):
        text = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "--standalone" not in text
        assert "--master_port=" not in text
        assert "--rdzv_backend=c10d" in text
        assert "--rdzv_endpoint=127.0.0.1:0" in text
        assert "${SLURM_JOB_ID}" in text
    post = (ROOT / "scripts" / "run_spatial_zoom_s1_test_profile_slurm.sh").read_text(
        encoding="utf-8"
    )
    assert "SPATIAL_ZOOM_S1_TRAINING_SOURCE_ROOT" in post
    assert "SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT" in post
    assert "SPATIAL_ZOOM_S1_PROFILE_RECOVERY" in post
    assert "SPATIAL_ZOOM_S1_PREFLIGHT_ONLY" in post
    assert "SPATIAL_ZOOM_S1_MATRIX_DRY_RUN" in post
    assert "--matrix-start-receipt" in post
    assert "build_test_matrix_binding" in post
    assert "reuse validated test evidence" in post
    assert 'cd "${TRAINING_ROOT}"' in post
    assert '"${ROOT}/tools/test.py" "${BOUND_CONFIG}"' in post
    assert "--s1-profile-recovery-certificate" in post
    assert 'PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in post
    assert '"${ROOT}/tools/bata/profile_spatial_zoom_s1.py"' in post
    environment_activation = post.index(
        'source "${BASE}/conda_envs/opentad/bin/activate"'
    )
    first_python = post.index("python")
    assert environment_activation < first_python
    preflight_exit = post.index("PREFLIGHT PASS")
    test_open = post.index('"${ROOT}/tools/test.py" "${BOUND_CONFIG}"')
    profile_open = post.index('"${ROOT}/tools/bata/profile_spatial_zoom_s1.py"')
    assert preflight_exit < test_open < profile_open
    matrix = (
        ROOT / "scripts" / "run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT" in matrix
    assert "profile source root differs from the certificate-bound commit" in matrix
    assert "256:3408 224:3409" in matrix
    assert "build_s1_profile_order" in matrix
    assert "CUDA_VISIBLE_DEVICES=" not in matrix
    assert 'mkdir "${MATRIX_LOCK_DIR}"' in matrix
    assert "matrix.started.json" in matrix
    assert "build_profile_matrix_start_receipt" in matrix
    assert "validate_profile_matrix_start_receipt" in matrix
    assert "build_profile_matrix_completion_receipt" in matrix
    assert "validate_profile_matrix_completion_receipt" in matrix
    assert "canonical_matrix_completion_path" in matrix
    assert "refusing a concurrent or repeated matrix" in matrix
    assert "validate_sidecar_gate_evidence" in matrix
    matrix_preflight = matrix.index("SPATIAL_ZOOM_S1_MATRIX_DRY_RUN=1")
    matrix_lock = matrix.index('mkdir "${MATRIX_LOCK_DIR}"')
    assert matrix_preflight < matrix_lock
    assert matrix.index("for cell in ${FROZEN_ORDER}", matrix_preflight) < matrix_lock
    gate = (
        ROOT / "scripts" / "run_spatial_zoom_s1_power_sidecar_gate_slurm.sh"
    ).read_text(encoding="utf-8")
    gate_memory_preflight = gate.index("require_slurm_memory_limit_mb")
    gate_namespace = gate.index("sidecar Gate evidence already exists")
    assert gate_memory_preflight < gate_namespace
    cell_memory_preflight = post.index("require_slurm_memory_limit_mb")
    cell_recovery_read = post.index('["training_code_commit"]')
    assert cell_memory_preflight < cell_recovery_read
    for source in (post, gate):
        assert "has_sidecar_runtime_evidence" in source
        assert "failed before the sidecar published attempt evidence" in source
        assert "salvage also failed" in source
        assert "salvage" in source
        assert "|| true" not in source
        assert '-i "${CUDA_VISIBLE_DEVICES}"' in source
        assert '-i "${SCOPED_GPU_ID}"' not in source
    for source in (post, matrix):
        assert "SLURM_STEP_GPUS" in source
        assert "srun --exact" in source
        assert "--gpus=1" in source
        assert "--cpus-per-task=5" in source
        assert "--mem=96000M" in source
    matrix_start_preflight = matrix.index("build_profile_matrix_start_receipt")
    assert matrix_start_preflight < matrix_lock
    test_entrypoint = (ROOT / "tools" / "test.py").read_text(encoding="utf-8")
    assert "--s1-profile-recovery-certificate" in test_entrypoint
    assert "S1_STEP_SCOPED_TEST_RUNTIME_MODE" in test_entrypoint
    assert "load_profile_recovery_certificate" in test_entrypoint
    assert "test_runtime_code_commit" in test_entrypoint


@pytest.mark.skipif(os.name == "nt", reason="requires Linux Slurm shell semantics")
def test_s1_high_memory_launchers_reenter_one_exact_gpu_step(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$@" > "${S1_CAPTURE:?}"\n',
        encoding="utf-8",
    )
    fake_srun.chmod(0o755)
    common_env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_GPUS": "1,2",
        "SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT": str(ROOT),
        "SPATIAL_ZOOM_S1_TRAINING_SOURCE_ROOT": str(ROOT),
        "SPATIAL_ZOOM_S1_RUN_ROOT": str(tmp_path / "run"),
        "SPATIAL_ZOOM_S1_MANIFEST": str(tmp_path / "manifest.json"),
        "SPATIAL_ZOOM_S1_ANNOTATION": str(tmp_path / "annotation.json"),
        "SPATIAL_ZOOM_S1_TEST_OPEN": str(tmp_path / "test_open.json"),
        "SPATIAL_ZOOM_S1_PROFILE_RECOVERY": str(tmp_path / "recovery.json"),
        "SPATIAL_ZOOM_S1_POWER_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "SPATIAL_ZOOM_S1_RESOLUTION": "256",
        "SPATIAL_ZOOM_S1_SEED": "3408",
    }
    common_env.pop("SLURM_STEP_GPUS", None)
    common_env.pop("SPATIAL_ZOOM_S1_SINGLE_GPU_STEP", None)
    expected_prefix = [
        "--exact",
        "--ntasks=1",
        "--gpus=1",
        "--cpus-per-task=5",
        "--mem=96000M",
        "bash",
    ]
    for filename in (
        "run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
        "run_spatial_zoom_s1_test_profile_slurm.sh",
        "run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
    ):
        capture = tmp_path / f"{filename}.args"
        env = {**common_env, "S1_CAPTURE": str(capture)}
        script = (ROOT / "scripts" / filename).resolve()
        completed = subprocess.run(
            ["bash", str(script)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        args = capture.read_text(encoding="utf-8").splitlines()
        assert args == [*expected_prefix, str(script)]


def test_config_validator_rejects_temporal_or_optimizer_drift() -> None:
    configs = {
        resolution: Config.fromfile(str(ROOT / path))
        for resolution, path in CONFIG_PATHS.items()
    }
    broken_temporal = copy.deepcopy(configs)
    broken_temporal[224].dataset.train.pipeline[2].trunc_len = 512
    with pytest.raises(AssertionError, match="temporal window"):
        validate_config_matrix(broken_temporal)

    broken_optimizer = copy.deepcopy(configs)
    broken_optimizer[256].optimizer.lr = 2e-4
    with pytest.raises(AssertionError, match="only permitted spatial fields"):
        validate_config_matrix(broken_optimizer)


def test_manifest_is_deterministic_disjoint_hashed_and_loader_ready(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    first = build_s1_manifest(annotation)
    second = build_s1_manifest(annotation)

    assert first == second
    checked = validate_s1_manifest(first, annotation_path=annotation)
    fit = set(checked["splits"]["fit"])
    gate = set(checked["splits"]["gate"])
    test = set(checked["splits"]["test"])
    assert fit and gate and test
    assert fit.isdisjoint(gate)
    assert (fit | gate).isdisjoint(test)
    assert checked["seeds"]["training"] == [3407, 3408, 3409]
    assert checked["bootstrap"] == {
        "unit": "paired_bayesian_video_cluster",
        "paired": True,
        "replicates": 10_000,
        "recompute_full_class_ap": True,
        "positive_video_weights": True,
        "support_rejection": False,
        "inferential_target": (
            "Bayesian bootstrap over the empirical video-cluster distribution "
            "with fixed class support and weighted AP"
        ),
        "simultaneous_correction": "max_t_for_224_and_256",
    }
    assert checked["duration_quartiles_seconds"]["q1"] > 0.0
    assert checked["manifest_sha256"]
    assert checked["pretrained_checkpoint"] == {
        "filename": S1_PRETRAINED_CHECKPOINT_FILENAME,
        "sha256": S1_PRETRAINED_CHECKPOINT_SHA256,
        "source": (
            "Kinetics-400 VideoMAE-S checkpoint used by the official-derived "
            "AdaTAD config"
        ),
    }
    assert checked["split_hashes"]["fit"] != checked["split_hashes"]["gate"]
    forged_bootstrap = copy.deepcopy(first)
    forged_bootstrap["bootstrap"]["support_rejection"] = True
    forged_bootstrap.pop("manifest_sha256")
    forged_bootstrap["manifest_sha256"] = canonical_sha256(forged_bootstrap)
    with pytest.raises(ValueError, match="Bayesian bootstrap protocol"):
        validate_s1_manifest(forged_bootstrap)
    with pytest.raises(ValueError, match="frozen"):
        build_s1_manifest(annotation, gate_ratio=0.25)
    with pytest.raises(ValueError, match="frozen"):
        build_s1_manifest(annotation, split_seed=1)


def test_manifest_rejects_a_rehashed_alternative_fit_gate_partition(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    forged = copy.deepcopy(manifest)
    gate_id = forged["splits"]["gate"][0]
    parity = int(gate_id.rsplit("_", 1)[1]) % 2
    fit_id = next(
        video_id
        for video_id in forged["splits"]["fit"]
        if int(video_id.rsplit("_", 1)[1]) % 2 == parity
    )
    forged["splits"]["gate"] = sorted(
        (set(forged["splits"]["gate"]) - {gate_id}) | {fit_id}
    )
    forged["splits"]["fit"] = sorted(
        (set(forged["splits"]["fit"]) - {fit_id}) | {gate_id}
    )
    forged["split_hashes"] = {
        name: stable_id_hash(values) for name, values in forged["splits"].items()
    }
    forged.pop("manifest_sha256")
    forged["manifest_sha256"] = canonical_sha256(forged)

    with pytest.raises(ValueError, match="deterministic frozen protocol"):
        validate_s1_manifest(forged, annotation_path=annotation)


def test_test_open_requires_one_shared_precheck_and_pretrained_identity() -> None:
    binding = {
        "precheck_file_sha256": "a" * 64,
        "precheck_sha256": "b" * 64,
        "pretrained_checkpoint_sha256": S1_PRETRAINED_CHECKPOINT_SHA256,
    }
    assert _shared_precheck_identity([binding] * 9) == binding
    drifted = copy.deepcopy(binding)
    drifted["precheck_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="do not share one precheck identity"):
        _shared_precheck_identity([binding] * 8 + [drifted])


def test_formal_experiment_namespace_is_unique_and_test_open_is_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root = (tmp_path / "sealed_study_v1").resolve()
    monkeypatch.setattr(
        s1_test_open,
        "S1_CANONICAL_STUDY_ROOT",
        str(study_root),
    )
    precheck_a = tmp_path / "precheck_a.json"
    precheck_b = tmp_path / "precheck_b.json"
    semantic_precheck = {"precheck_sha256": "e" * 64, "status": "PASS"}
    precheck_a.write_text(
        json.dumps(semantic_precheck, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    precheck_b.write_text(
        json.dumps(semantic_precheck, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    assert sha256_file(precheck_a) != sha256_file(precheck_b)
    identity = build_s1_experiment_identity(
        manifest_sha256="a" * 64,
        code_commit="b" * 40,
        protocol_fingerprint="c" * 64,
        precheck_file_sha256=sha256_file(precheck_a),
        precheck_sha256="e" * 64,
        pretrained_checkpoint_sha256=S1_PRETRAINED_CHECKPOINT_SHA256,
    )
    same = build_s1_experiment_identity(
        manifest_sha256="a" * 64,
        code_commit="b" * 40,
        protocol_fingerprint="c" * 64,
        precheck_file_sha256=sha256_file(precheck_a),
        precheck_sha256="e" * 64,
        pretrained_checkpoint_sha256=S1_PRETRAINED_CHECKPOINT_SHA256,
    )
    changed = build_s1_experiment_identity(
        manifest_sha256="a" * 64,
        code_commit="b" * 40,
        protocol_fingerprint="c" * 64,
        precheck_file_sha256=sha256_file(precheck_b),
        precheck_sha256="e" * 64,
        pretrained_checkpoint_sha256=S1_PRETRAINED_CHECKPOINT_SHA256,
    )
    assert identity == same
    assert identity["experiment_namespace"] != changed["experiment_namespace"]
    local_root = tmp_path / identity["experiment_namespace"]
    shared_binding = {
        "experiment_namespace": identity["experiment_namespace"],
        "canonical_experiment_root": str(local_root),
    }
    assert _shared_experiment_identity([shared_binding] * 9) == shared_binding
    selection_paths = [tmp_path / f"selection-{index}.json" for index in range(9)]
    checkpoint_paths = [tmp_path / f"checkpoint-{index}.pth" for index in range(9)]
    manifest_path = tmp_path / "manifest.json"
    annotation_path = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selection_rows = []
    for index, (selection_path, checkpoint_path) in enumerate(
        zip(selection_paths, checkpoint_paths)
    ):
        selection_path.write_text(
            json.dumps({"resolution": index, "seed": index}), encoding="utf-8"
        )
        checkpoint_path.write_bytes(f"checkpoint-{index}".encode("ascii"))
        selection_rows.append(
            {
                "selection_path": str(selection_path),
                "selection_file_sha256": sha256_file(selection_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
    certificate = {
        **shared_binding,
        "canonical_study_root": str(study_root),
        "global_test_open_marker_path": str(
            study_root / "test_open" / "test_open_issued.json"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "annotation_sha256": manifest["annotation_sha256"],
        "code_commit": "b" * 40,
        "precheck_file_sha256": sha256_file(precheck_a),
        "precheck_sha256": "e" * 64,
        "pretrained_checkpoint_sha256": S1_PRETRAINED_CHECKPOINT_SHA256,
        "manifest_path": str(manifest_path),
        "annotation_path": str(annotation_path),
        "selection_matrix": selection_rows,
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    marker_path, marker = create_global_test_open_marker(certificate)
    assert validate_global_test_open_marker(certificate) == marker
    recovered_output = local_root / "test_open" / "test_open_certificate.json"
    recovered = recover_global_test_open_certificate(
        output_path=recovered_output,
        manifest_path=manifest_path,
        annotation_path=annotation_path,
        selection_paths=selection_paths,
    )
    assert recovered == certificate
    assert json.loads(recovered_output.read_text(encoding="utf-8")) == certificate
    selection_paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="current selection differs"):
        recover_global_test_open_certificate(
            output_path=recovered_output,
            manifest_path=manifest_path,
            annotation_path=annotation_path,
            selection_paths=selection_paths,
        )
    with pytest.raises(FileExistsError):
        create_global_test_open_marker(certificate)
    rerun_certificate = {
        **certificate,
        "experiment_namespace": changed["experiment_namespace"],
        "canonical_experiment_root": changed["canonical_experiment_root"],
        "precheck_file_sha256": sha256_file(precheck_b),
    }
    rerun_certificate.pop("certificate_sha256", None)
    rerun_certificate["certificate_sha256"] = canonical_sha256(rerun_certificate)
    with pytest.raises(FileExistsError):
        create_global_test_open_marker(rerun_certificate)
    changed_root = {**rerun_certificate, "canonical_study_root": str(tmp_path / "x")}
    changed_root.pop("certificate_sha256", None)
    changed_root["certificate_sha256"] = canonical_sha256(changed_root)
    with pytest.raises(ValueError, match="canonical study root"):
        create_global_test_open_marker(changed_root)
    assert marker_path.is_file()


def test_profile_attempt_marker_is_atomic_and_self_hashed(tmp_path: Path) -> None:
    path = tmp_path / "dense160_seed3407.started.json"
    marker = create_profile_attempt_marker(
        path,
        {"resolution": 160, "seed": 3407, "canonical_output_prefix": "x"},
    )
    assert validate_profile_attempt_marker(path) == marker
    with pytest.raises(FileExistsError):
        create_profile_attempt_marker(path, {"resolution": 160, "seed": 3407})


def test_profile_recovery_certificate_preserves_failed_attempt_and_scope(
    tmp_path: Path,
) -> None:
    canonical_root = (tmp_path / "canonical").resolve()
    binding = {
        "code_commit": "a" * 40,
        "experiment_namespace": "s1-experiment",
        "canonical_experiment_root": str(canonical_root),
        "work_dir": str((canonical_root / "dense256" / "seed3408").resolve()),
        "manifest_sha256": "b" * 64,
        "protocol_fingerprint": "c" * 64,
        "precheck_file_sha256": "d" * 64,
        "precheck_sha256": "e" * 64,
        "pretrained_checkpoint_sha256": "f" * 64,
    }
    marker_path = tmp_path / "failed.started.json"
    marker = {
        "schema_version": "spatial_zoom_s1_profile_attempt_v4",
        "resolution": int(build_s1_profile_order()[0]["resolution"]),
        "seed": int(build_s1_profile_order()[0]["seed"]),
        "code_commit": binding["code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
        "test_open_certificate_sha256": "1" * 64,
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    failure_log = tmp_path / "slurm.err"
    failure_log.write_text(S1_PROFILE_FAILURE_SIGNATURE + "\n", encoding="utf-8")
    changed_paths = (
        "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
        "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
        "tests/test_spatial_zoom_s1_infrastructure.py",
        "tools/bata/analyze_spatial_zoom_s1_results.py",
        "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
        "tools/bata/preflight_spatial_zoom_s1_profile.py",
        "tools/bata/profile_spatial_zoom_s1.py",
        "tools/bata/run_spatial_zoom_s1_precheck.py",
        "tools/bata/spatial_zoom_s1_cost.py",
        "tools/bata/spatial_zoom_s1_profile_recovery.py",
        "tools/bata/spatial_zoom_s1_training.py",
    )
    basis = {
        "schema_version": S1_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_PROFILE_RECOVERY_REASON,
        "failure_signature": S1_PROFILE_FAILURE_SIGNATURE,
        "failed_job_id": "1167257",
        "training_code_commit": binding["code_commit"],
        "profile_code_commit": "2" * 40,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": marker["test_open_certificate_sha256"],
        "superseded_marker_path": str(marker_path.resolve()),
        "superseded_marker_file_sha256": sha256_file(marker_path),
        "superseded_marker_sha256": marker["marker_sha256"],
        "failure_log_path": str(failure_log.resolve()),
        "failure_log_sha256": sha256_file(failure_log),
        "expected_loader_exposure_count": 792,
        "expected_physical_window_count": 791,
        "expected_duplicate_physical_window_ids": ["video_test_0001431:7680"],
        "changed_files": [
            {"status": "M", "path": path, "file_sha256": "3" * 64}
            for path in changed_paths
        ],
        "repair_scope": "profile_identity_and_postprocessing_only",
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
    }
    campaign_id = canonical_sha256(basis)[:16]
    certificate = {
        **basis,
        "campaign_id": campaign_id,
        "campaign_root": str(canonical_root / "profile_campaigns" / campaign_id),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    checked = validate_profile_recovery_certificate(
        certificate, binding=binding, verify_checkout=False
    )
    prefix = profile_campaign_prefix(checked, resolution=160, seed=3407)
    assert checked["preserve_superseded_attempt"] is True
    assert prefix.name == "dense160_seed3407"
    assert marker_path.is_file()

    forged = copy.deepcopy(certificate)
    forged["expected_physical_window_count"] = 790
    forged.pop("certificate_sha256")
    forged["certificate_sha256"] = canonical_sha256(forged)
    with pytest.raises(ValueError, match="campaign identity|exposure topology"):
        validate_profile_recovery_certificate(
            forged, binding=binding, verify_checkout=False
        )


def test_chained_profile_recovery_binds_power_failure_and_diagnostic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical_root = (tmp_path / "canonical").resolve()
    binding = {
        "code_commit": "a" * 40,
        "experiment_namespace": "s1-experiment",
        "canonical_experiment_root": str(canonical_root),
        "work_dir": str((canonical_root / "dense256" / "seed3408").resolve()),
        "manifest_sha256": "b" * 64,
        "protocol_fingerprint": "c" * 64,
        "precheck_file_sha256": "d" * 64,
        "precheck_sha256": "e" * 64,
        "pretrained_checkpoint_sha256": "f" * 64,
    }
    first = build_s1_profile_order()[0]
    original_marker_path = tmp_path / "original.started.json"
    original_marker = {
        "schema_version": "spatial_zoom_s1_profile_attempt_v4",
        "resolution": int(first["resolution"]),
        "seed": int(first["seed"]),
        "code_commit": binding["code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
        "test_open_certificate_sha256": "1" * 64,
    }
    original_marker["marker_sha256"] = canonical_sha256(original_marker)
    original_marker_path.write_text(
        json.dumps(original_marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_log = tmp_path / "original.log"
    original_log.write_text(S1_PROFILE_FAILURE_SIGNATURE + "\n", encoding="utf-8")
    required_v1 = (
        "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
        "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
        "tests/test_spatial_zoom_s1_infrastructure.py",
        "tools/bata/analyze_spatial_zoom_s1_results.py",
        "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
        "tools/bata/preflight_spatial_zoom_s1_profile.py",
        "tools/bata/profile_spatial_zoom_s1.py",
        "tools/bata/run_spatial_zoom_s1_precheck.py",
        "tools/bata/spatial_zoom_s1_cost.py",
        "tools/bata/spatial_zoom_s1_profile_recovery.py",
        "tools/bata/spatial_zoom_s1_training.py",
    )
    parent_basis = {
        "schema_version": S1_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_PROFILE_RECOVERY_REASON,
        "failure_signature": S1_PROFILE_FAILURE_SIGNATURE,
        "failed_job_id": "1167257",
        "training_code_commit": binding["code_commit"],
        "profile_code_commit": "2" * 40,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": original_marker["test_open_certificate_sha256"],
        "superseded_marker_path": str(original_marker_path.resolve()),
        "superseded_marker_file_sha256": sha256_file(original_marker_path),
        "superseded_marker_sha256": original_marker["marker_sha256"],
        "failure_log_path": str(original_log.resolve()),
        "failure_log_sha256": sha256_file(original_log),
        "expected_loader_exposure_count": 792,
        "expected_physical_window_count": 791,
        "expected_duplicate_physical_window_ids": ["video_test_0001431:7680"],
        "changed_files": [
            {"status": "M", "path": path, "file_sha256": "3" * 64}
            for path in required_v1
        ],
        "repair_scope": "profile_identity_and_postprocessing_only",
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
    }
    parent_id = canonical_sha256(parent_basis)[:16]
    parent = {
        **parent_basis,
        "campaign_id": parent_id,
        "campaign_root": str(canonical_root / "profile_campaigns" / parent_id),
    }
    parent["certificate_sha256"] = canonical_sha256(parent)
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(
        json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    power_marker_path = tmp_path / "power.started.json"
    power_marker = {
        **original_marker,
        "schema_version": "spatial_zoom_s1_profile_attempt_v5",
        "profile_code_commit": parent["profile_code_commit"],
        "profile_recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_recovery_campaign_id": parent["campaign_id"],
    }
    power_marker.pop("marker_sha256")
    power_marker["marker_sha256"] = canonical_sha256(power_marker)
    power_marker_path.write_text(
        json.dumps(power_marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    power_log = tmp_path / "power.log"
    power_log.write_text(S1_POWER_FAILURE_SIGNATURE + "\n", encoding="utf-8")
    diagnostic_path = tmp_path / "power_diagnostic.json"
    diagnostic = {
        "schema_version": "spatial_zoom_s1_power_sampler_diagnostic_v1",
        "reads_test_data": False,
        "paper_claim_allowed": False,
        "code_commit": "6" * 40,
        "node": "g0059",
        "gpu_uuid": "GPU-allocated",
        "target_interval_ms": 20,
        "duration_seconds_per_backend": 10.0,
        "slurm_job_id": "1167536",
        "backends": [
            {
                "backend": "nvidia-smi-persistent-loop-ms",
                "status": "FAIL",
                "cadence": {"formal_cadence_pass": False},
            },
            {
                "backend": "nvml-persistent-poll-v1",
                "status": "PASS",
                "cadence": {
                    "formal_cadence_pass": True,
                    "max_gap_ms": 57.7,
                    "max_gap_limit_ms": 100.0,
                },
            },
        ],
    }
    diagnostic["diagnostic_sha256"] = canonical_sha256(diagnostic)
    diagnostic_path.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    chain_paths = (*required_v1, "tools/bata/spatial_zoom_s1_power.py")
    chain_basis = {
        **parent_basis,
        "schema_version": S1_CHAINED_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_CHAINED_RECOVERY_REASON,
        "profile_code_commit": "4" * 40,
        "changed_files": [
            {"status": "M", "path": path, "file_sha256": "5" * 64}
            for path in chain_paths
        ],
        "repair_scope": "profile_identity_power_sampling_and_postprocessing_only",
        "preserve_recovery_chain": True,
        "superseded_recovery_certificate_path": str(parent_path.resolve()),
        "superseded_recovery_certificate_file_sha256": sha256_file(parent_path),
        "superseded_recovery_certificate_sha256": parent["certificate_sha256"],
        "superseded_recovery_campaign_id": parent["campaign_id"],
        "superseded_recovery_profile_code_commit": parent["profile_code_commit"],
        "power_failure_signature": S1_POWER_FAILURE_SIGNATURE,
        "power_failed_job_id": "1167516",
        "power_failure_marker_path": str(power_marker_path.resolve()),
        "power_failure_marker_file_sha256": sha256_file(power_marker_path),
        "power_failure_marker_sha256": power_marker["marker_sha256"],
        "power_failure_log_path": str(power_log.resolve()),
        "power_failure_log_sha256": sha256_file(power_log),
        "power_diagnostic_path": str(diagnostic_path.resolve()),
        "power_diagnostic_file_sha256": sha256_file(diagnostic_path),
        "power_diagnostic_sha256": diagnostic["diagnostic_sha256"],
        "power_diagnostic_job_id": diagnostic["slurm_job_id"],
        "power_diagnostic_code_commit": diagnostic["code_commit"],
        "power_sampler_backend": "nvml-persistent-poll-v1",
    }
    chain_id = canonical_sha256(chain_basis)[:16]
    chain = {
        **chain_basis,
        "campaign_id": chain_id,
        "campaign_root": str(canonical_root / "profile_campaigns" / chain_id),
    }
    chain["certificate_sha256"] = canonical_sha256(chain)
    checked = validate_profile_recovery_certificate(
        chain, binding=binding, verify_checkout=False
    )
    assert checked["preserve_recovery_chain"] is True
    assert checked["power_sampler_backend"] == "nvml-persistent-poll-v1"
    assert checked["schema_version"] == S1_CHAINED_PROFILE_RECOVERY_SCHEMA

    chain_path = tmp_path / "chain.json"
    chain_path.write_text(
        json.dumps(chain, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    legacy_test_evidence_path = (
        Path(binding["work_dir"]) / "gpu1_id0" / "test_evidence" / "test.evidence.json"
    )
    legacy_test_evidence = {"schema_version": "spatial_zoom_s1_test_evidence_v4"}
    legacy_test_evidence["evidence_sha256"] = canonical_sha256(legacy_test_evidence)
    legacy_test_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_test_evidence_path.write_text(
        json.dumps(legacy_test_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_marker_path = tmp_path / "sidecar.started.json"
    sidecar_marker = {
        **original_marker,
        "schema_version": "spatial_zoom_s1_profile_attempt_v5",
        "profile_code_commit": chain["profile_code_commit"],
        "profile_recovery_certificate_sha256": chain["certificate_sha256"],
        "profile_recovery_campaign_id": chain["campaign_id"],
        "test_evidence_sha256": legacy_test_evidence["evidence_sha256"],
    }
    sidecar_marker.pop("marker_sha256")
    sidecar_marker["marker_sha256"] = canonical_sha256(sidecar_marker)
    sidecar_marker_path.write_text(
        json.dumps(sidecar_marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_log = tmp_path / "sidecar.log"
    sidecar_log.write_text(S1_POWER_FAILURE_SIGNATURE + "\n", encoding="utf-8")
    sidecar_paths = (
        *chain_paths,
        "docs/superpowers/specs/2026-07-17-spatial-zoom-s1-power-sidecar-design.md",
        "scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
        "tests/test_spatial_zoom_s1_matrix.py",
        "tools/bata/spatial_zoom_s1_matrix.py",
        "tools/bata/spatial_zoom_s1_sidecar_gate.py",
    )
    sidecar_basis = {
        **parent_basis,
        "schema_version": S1_SIDECAR_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_SIDECAR_RECOVERY_REASON,
        "profile_code_commit": "7" * 40,
        "legacy_unbound_test_resolution": int(
            build_s1_profile_order()[0]["resolution"]
        ),
        "legacy_unbound_test_seed": int(build_s1_profile_order()[0]["seed"]),
        "legacy_unbound_test_evidence_path": str(legacy_test_evidence_path.resolve()),
        "legacy_unbound_test_evidence_file_sha256": sha256_file(
            legacy_test_evidence_path
        ),
        "legacy_unbound_test_evidence_sha256": legacy_test_evidence["evidence_sha256"],
        "changed_files": [
            {"status": "M", "path": path, "file_sha256": "8" * 64}
            for path in sidecar_paths
        ],
        "repair_scope": "out_of_process_power_sidecar_and_failure_evidence_only",
        "preserve_recovery_chain": True,
        "superseded_recovery_certificate_path": str(chain_path.resolve()),
        "superseded_recovery_certificate_file_sha256": sha256_file(chain_path),
        "superseded_recovery_certificate_sha256": chain["certificate_sha256"],
        "superseded_recovery_campaign_id": chain["campaign_id"],
        "superseded_recovery_profile_code_commit": chain["profile_code_commit"],
        "sidecar_power_failure_signature": S1_POWER_FAILURE_SIGNATURE,
        "sidecar_power_failed_job_id": "1167538",
        "sidecar_power_failure_marker_path": str(sidecar_marker_path.resolve()),
        "sidecar_power_failure_marker_file_sha256": sha256_file(sidecar_marker_path),
        "sidecar_power_failure_marker_sha256": sidecar_marker["marker_sha256"],
        "sidecar_power_failure_log_path": str(sidecar_log.resolve()),
        "sidecar_power_failure_log_sha256": sha256_file(sidecar_log),
        "power_sampler_backend": "nvml-sidecar-process-v1",
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }
    sidecar_id = canonical_sha256(sidecar_basis)[:16]
    sidecar = {
        **sidecar_basis,
        "campaign_id": sidecar_id,
        "campaign_root": str(canonical_root / "profile_campaigns" / sidecar_id),
    }
    sidecar["certificate_sha256"] = canonical_sha256(sidecar)
    sidecar_checked = validate_profile_recovery_certificate(
        sidecar, binding=binding, verify_checkout=False
    )
    assert sidecar_checked["power_sampler_backend"] == "nvml-sidecar-process-v1"
    assert sidecar_checked["sidecar_power_failed_job_id"] == "1167538"
    tampered_cpu = copy.deepcopy(sidecar)
    tampered_cpu["allocated_cpu_count"] = 4
    for key in ("certificate_sha256", "campaign_id", "campaign_root"):
        tampered_cpu.pop(key)
    tampered_id = canonical_sha256(tampered_cpu)[:16]
    tampered_cpu["campaign_id"] = tampered_id
    tampered_cpu["campaign_root"] = str(
        canonical_root / "profile_campaigns" / tampered_id
    )
    tampered_cpu["certificate_sha256"] = canonical_sha256(tampered_cpu)
    with pytest.raises(ValueError, match="sidecar recovery contract mismatch"):
        validate_profile_recovery_certificate(
            tampered_cpu, binding=binding, verify_checkout=False
        )

    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed_prefix = (
        Path(sidecar["campaign_root"]) / "dense256" / "seed3408" / "dense256_seed3408"
    )
    attempt_report_path, attempt_trace_path, attempt = _write_valid_sidecar_attempt(
        failed_prefix,
        expected_uuid="GPU-S1",
        timestamps_ns=(
            1_000_000_000,
            1_020_000_000,
            1_170_000_000,
            1_190_000_000,
        ),
        cadence_failure=True,
    )
    parent_failure_path = Path(f"{failed_prefix}.power_parent_failure.json")
    parent_failure = {
        "schema_version": "spatial_zoom_s1_profile_parent_failure_v1",
        "status": "FAIL",
        "paper_claim_allowed": False,
        "power_attempt_sha256": attempt["attempt_sha256"],
        "power_attempt_report_file_sha256": sha256_file(attempt_report_path),
        "power_attempt_trace_file_sha256": sha256_file(attempt_trace_path),
    }
    parent_failure["parent_failure_sha256"] = canonical_sha256(parent_failure)
    parent_failure_path.write_text(
        json.dumps(parent_failure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matrix_start_path = tmp_path / "matrix.started.json"
    matrix_start = {
        "slurm_job_id": "1168823",
        "slurm_step_id": "0",
        "step_gpu_uuid": "GPU-S1",
        "profile_code_commit": sidecar["profile_code_commit"],
        "profile_recovery_certificate_sha256": sidecar["certificate_sha256"],
        "profile_recovery_campaign_id": sidecar["campaign_id"],
        "frozen_order": build_s1_profile_order(),
    }
    matrix_start["matrix_sha256"] = canonical_sha256(matrix_start)
    matrix_start_path.write_text(
        json.dumps(matrix_start, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    buffered_marker_path = tmp_path / "buffered.started.json"
    buffered_marker = {
        **original_marker,
        "schema_version": "spatial_zoom_s1_profile_attempt_v7",
        "profile_code_commit": sidecar["profile_code_commit"],
        "profile_recovery_certificate_sha256": sidecar["certificate_sha256"],
        "profile_recovery_campaign_id": sidecar["campaign_id"],
        "test_evidence_sha256": legacy_test_evidence["evidence_sha256"],
        "canonical_output_prefix": str(failed_prefix.resolve()),
        "power_sampler_backend": "nvml-sidecar-process-v1",
        "gate_only": False,
        "slurm_job_id": "1168823",
        "slurm_step_id": "0",
        "step_gpu_uuid": "GPU-S1",
        "matrix_start_receipt_path": str(matrix_start_path.resolve()),
        "matrix_start_receipt_file_sha256": sha256_file(matrix_start_path),
        "matrix_sha256": matrix_start["matrix_sha256"],
    }
    buffered_marker.pop("marker_sha256")
    buffered_marker["marker_sha256"] = canonical_sha256(buffered_marker)
    buffered_marker_path.write_text(
        json.dumps(buffered_marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    buffered_log = tmp_path / "buffered.log"
    buffered_log.write_text(
        S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE + "\n",
        encoding="utf-8",
    )
    parent_delta = [
        {
            "status": "M",
            "path": "tools/bata/spatial_zoom_s1_power.py",
            "file_sha256": "2" * 64,
        }
    ]
    monkeypatch.setattr(
        s1_profile_recovery,
        "_changed_files_between_commits",
        lambda *_args, **_kwargs: copy.deepcopy(parent_delta),
    )
    monkeypatch.setattr(
        s1_profile_recovery,
        "_git_file_sha256",
        lambda commit, _path: (
            "1" * 64 if commit == sidecar["profile_code_commit"] else "2" * 64
        ),
    )
    matrix_validation_calls = []

    def validate_matrix(path, *, parent_recovery_path):
        matrix_validation_calls.append(
            (Path(path).resolve(), Path(parent_recovery_path).resolve())
        )
        checked = json.loads(Path(path).read_text(encoding="utf-8"))
        return checked

    monkeypatch.setattr(
        s1_profile_recovery,
        "_validate_v3_matrix_start_receipt",
        validate_matrix,
    )
    buffered_basis = {
        **parent_basis,
        "schema_version": S1_BUFFERED_SIDECAR_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_BUFFERED_SIDECAR_RECOVERY_REASON,
        "profile_code_commit": "9" * 40,
        "legacy_unbound_test_resolution": int(first["resolution"]),
        "legacy_unbound_test_seed": int(first["seed"]),
        "legacy_unbound_test_evidence_path": str(legacy_test_evidence_path.resolve()),
        "legacy_unbound_test_evidence_file_sha256": sha256_file(
            legacy_test_evidence_path
        ),
        "legacy_unbound_test_evidence_sha256": legacy_test_evidence["evidence_sha256"],
        "changed_files": [
            {"status": "M", "path": path, "file_sha256": "a" * 64}
            for path in sidecar_paths
        ],
        "parent_to_current_changed_files": parent_delta,
        "parent_sampling_implementation_sha256": "1" * 64,
        "sampling_implementation_sha256": "2" * 64,
        "repair_scope": "buffered_sidecar_trace_publication_only",
        "preserve_recovery_chain": True,
        "superseded_recovery_certificate_path": str(sidecar_path.resolve()),
        "superseded_recovery_certificate_file_sha256": sha256_file(sidecar_path),
        "superseded_recovery_certificate_sha256": sidecar["certificate_sha256"],
        "superseded_recovery_campaign_id": sidecar["campaign_id"],
        "superseded_recovery_profile_code_commit": sidecar["profile_code_commit"],
        "buffered_sidecar_failure_signature": (S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE),
        "buffered_sidecar_failed_job_id": "1168823",
        "buffered_sidecar_failed_slurm_step_id": "0",
        "buffered_sidecar_failed_gpu_uuid": "GPU-S1",
        "buffered_sidecar_failure_marker_path": str(buffered_marker_path.resolve()),
        "buffered_sidecar_failure_marker_file_sha256": sha256_file(
            buffered_marker_path
        ),
        "buffered_sidecar_failure_marker_sha256": buffered_marker["marker_sha256"],
        "buffered_sidecar_failure_log_path": str(buffered_log.resolve()),
        "buffered_sidecar_failure_log_sha256": sha256_file(buffered_log),
        "buffered_sidecar_attempt_report_path": str(attempt_report_path.resolve()),
        "buffered_sidecar_attempt_report_file_sha256": sha256_file(attempt_report_path),
        "buffered_sidecar_attempt_sha256": attempt["attempt_sha256"],
        "buffered_sidecar_attempt_trace_path": str(attempt_trace_path.resolve()),
        "buffered_sidecar_attempt_trace_file_sha256": sha256_file(attempt_trace_path),
        "buffered_sidecar_parent_failure_path": str(parent_failure_path.resolve()),
        "buffered_sidecar_parent_failure_file_sha256": sha256_file(parent_failure_path),
        "buffered_sidecar_parent_failure_sha256": parent_failure[
            "parent_failure_sha256"
        ],
        "buffered_sidecar_matrix_start_path": str(matrix_start_path.resolve()),
        "buffered_sidecar_matrix_start_file_sha256": sha256_file(matrix_start_path),
        "buffered_sidecar_matrix_sha256": matrix_start["matrix_sha256"],
        "power_sampler_backend": "nvml-sidecar-process-v1",
        "trace_publication_mode": S1_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }
    buffered_id = canonical_sha256(buffered_basis)[:16]
    buffered = {
        **buffered_basis,
        "campaign_id": buffered_id,
        "campaign_root": str(canonical_root / "profile_campaigns" / buffered_id),
    }
    buffered["certificate_sha256"] = canonical_sha256(buffered)
    buffered_checked = validate_profile_recovery_certificate(
        buffered, binding=binding, verify_checkout=False
    )
    assert buffered_checked["buffered_sidecar_failed_job_id"] == "1168823"
    assert buffered_checked["trace_io_inside_sampling_loop"] is False
    assert matrix_validation_calls == [
        (matrix_start_path.resolve(), sidecar_path.resolve())
    ]

    no_sampling_change = copy.deepcopy(buffered)
    for key in ("certificate_sha256", "campaign_id", "campaign_root"):
        no_sampling_change.pop(key)
    no_sampling_change["sampling_implementation_sha256"] = no_sampling_change[
        "parent_sampling_implementation_sha256"
    ]
    no_sampling_change_id = canonical_sha256(no_sampling_change)[:16]
    no_sampling_change["campaign_id"] = no_sampling_change_id
    no_sampling_change["campaign_root"] = str(
        canonical_root / "profile_campaigns" / no_sampling_change_id
    )
    no_sampling_change["certificate_sha256"] = canonical_sha256(no_sampling_change)
    with pytest.raises(ValueError, match="sampling implementation binding"):
        validate_profile_recovery_certificate(
            no_sampling_change,
            binding=binding,
            verify_checkout=False,
        )

    tampered_trace_mode = copy.deepcopy(buffered)
    tampered_trace_mode["trace_io_inside_sampling_loop"] = True
    for key in ("certificate_sha256", "campaign_id", "campaign_root"):
        tampered_trace_mode.pop(key)
    tampered_trace_id = canonical_sha256(tampered_trace_mode)[:16]
    tampered_trace_mode["campaign_id"] = tampered_trace_id
    tampered_trace_mode["campaign_root"] = str(
        canonical_root / "profile_campaigns" / tampered_trace_id
    )
    tampered_trace_mode["certificate_sha256"] = canonical_sha256(tampered_trace_mode)
    with pytest.raises(ValueError, match="buffered-sidecar recovery contract"):
        validate_profile_recovery_certificate(
            tampered_trace_mode, binding=binding, verify_checkout=False
        )

    wrong_schema = copy.deepcopy(chain)
    wrong_schema["schema_version"] = S1_PROFILE_RECOVERY_SCHEMA
    wrong_schema.pop("certificate_sha256")
    wrong_schema["certificate_sha256"] = canonical_sha256(wrong_schema)
    with pytest.raises(ValueError, match="schema/reason mismatch"):
        validate_profile_recovery_certificate(
            wrong_schema, binding=binding, verify_checkout=False
        )

    wrong_backend = copy.deepcopy(chain)
    wrong_backend["power_sampler_backend"] = "nvidia-smi-persistent-loop-ms"
    for key in ("certificate_sha256", "campaign_id", "campaign_root"):
        wrong_backend.pop(key)
    wrong_backend_id = canonical_sha256(wrong_backend)[:16]
    wrong_backend["campaign_id"] = wrong_backend_id
    wrong_backend["campaign_root"] = str(
        canonical_root / "profile_campaigns" / wrong_backend_id
    )
    wrong_backend["certificate_sha256"] = canonical_sha256(wrong_backend)
    with pytest.raises(ValueError, match="power diagnostic mismatch"):
        validate_profile_recovery_certificate(
            wrong_backend, binding=binding, verify_checkout=False
        )


def test_profile_recovery_dispatches_v3_parent_to_buffered_builder(
    tmp_path: Path, monkeypatch
) -> None:
    parent_path = tmp_path / "v3.json"
    parent_path.write_text(
        json.dumps({"reason": S1_SIDECAR_RECOVERY_REASON}) + "\n",
        encoding="utf-8",
    )
    sentinel = (tmp_path / "v4.json", {"reason": "buffered"})
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        s1_profile_recovery,
        "build_buffered_sidecar_profile_recovery_certificate",
        fake_builder,
    )
    result = s1_profile_recovery.build_profile_recovery_certificate(
        binding={"code_commit": "a" * 40},
        failed_marker_path=tmp_path / "failed.started.json",
        failure_log_path=tmp_path / "failed.log",
        failed_job_id="1168823",
        expected_exposure_count=792,
        expected_physical_window_count=791,
        expected_duplicate_physical_window_ids=("duplicate",),
        superseded_recovery_certificate_path=parent_path,
        power_diagnostic_path=None,
    )
    assert result == sentinel
    assert captured["superseded_recovery_certificate_path"] == (parent_path.resolve())
    assert captured["failed_job_id"] == "1168823"


def test_profile_schedule_rejects_future_start_and_missing_prior_completion(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    order = build_s1_profile_order()
    binding = {
        "canonical_experiment_root": str(tmp_path / "canonical"),
        "experiment_namespace": "s1",
        "manifest_sha256": manifest["manifest_sha256"],
        "precheck_file_sha256": "a" * 64,
        "precheck_sha256": "b" * 64,
    }
    first, second = order[:2]
    campaign_root = tmp_path / "canonical" / "profile_campaigns" / "campaign"
    recovery_kwargs = {
        "campaign_root": campaign_root,
        "profile_code_commit": "c" * 40,
        "profile_recovery_certificate_sha256": "d" * 64,
        "profile_recovery_campaign_id": "campaign",
    }
    observed, order_sha = validate_profile_order_ready(
        manifest=manifest,
        binding=binding,
        resolution=int(first["resolution"]),
        seed=int(first["seed"]),
        hardware_fingerprint="hardware",
        software_fingerprint="software",
        **recovery_kwargs,
    )
    assert observed == first
    assert order_sha == canonical_sha256(order)
    dry_run_second, _ = validate_profile_order_ready(
        manifest=manifest,
        binding=binding,
        resolution=int(second["resolution"]),
        seed=int(second["seed"]),
        hardware_fingerprint="hardware",
        software_fingerprint="software",
        matrix_dry_run=True,
        **recovery_kwargs,
    )
    assert dry_run_second == second

    future_prefix = (
        campaign_root
        / f"dense{second['resolution']}"
        / f"seed{second['seed']}"
        / f"dense{second['resolution']}_seed{second['seed']}"
    )
    future_marker = future_prefix.with_suffix(".started.json")
    future_marker.parent.mkdir(parents=True, exist_ok=True)
    future_marker.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="started before its turn"):
        validate_profile_order_ready(
            manifest=manifest,
            binding=binding,
            resolution=int(first["resolution"]),
            seed=int(first["seed"]),
            hardware_fingerprint="hardware",
            software_fingerprint="software",
            **recovery_kwargs,
        )
    with pytest.raises(RuntimeError, match="matrix dry-run"):
        validate_profile_order_ready(
            manifest=manifest,
            binding=binding,
            resolution=int(second["resolution"]),
            seed=int(second["seed"]),
            hardware_fingerprint="hardware",
            software_fingerprint="software",
            matrix_dry_run=True,
            **recovery_kwargs,
        )
    future_marker.unlink()
    with pytest.raises(RuntimeError, match="requires completed cell ordinal 0"):
        validate_profile_order_ready(
            manifest=manifest,
            binding=binding,
            resolution=int(second["resolution"]),
            seed=int(second["seed"]),
            hardware_fingerprint="hardware",
            software_fingerprint="software",
            **recovery_kwargs,
        )
    current_prefix = (
        campaign_root
        / f"dense{first['resolution']}"
        / f"seed{first['seed']}"
        / f"dense{first['resolution']}_seed{first['seed']}"
    )
    current_marker = current_prefix.with_suffix(".started.json")
    current_marker.parent.mkdir(parents=True, exist_ok=True)
    current_marker.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="was already started"):
        validate_profile_order_ready(
            manifest=manifest,
            binding=binding,
            resolution=int(first["resolution"]),
            seed=int(first["seed"]),
            hardware_fingerprint=None,
            software_fingerprint=None,
            **recovery_kwargs,
        )


def test_manifest_writer_emits_block_lists_with_expected_complements(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    from tools.bata.spatial_zoom_s1_contract import write_s1_manifest_bundle

    paths = write_s1_manifest_bundle(manifest, tmp_path / "bundle")
    fit_blocked = set(paths["fit_block_list"].read_text(encoding="utf-8").splitlines())
    gate_blocked = set(
        paths["gate_block_list"].read_text(encoding="utf-8").splitlines()
    )
    assert fit_blocked == set(manifest["splits"]["gate"])
    assert gate_blocked == set(manifest["splits"]["fit"])
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert saved["manifest_sha256"] == manifest["manifest_sha256"]
    assert write_s1_manifest_bundle(manifest, tmp_path / "bundle") == paths
    paths["fit_block_list"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest artifact differs"):
        write_s1_manifest_bundle(manifest, tmp_path / "bundle")


def test_s1_source_config_is_not_trainable_until_manifest_bound(tmp_path: Path) -> None:
    source = Config.fromfile(str(ROOT / CONFIG_PATHS[160]))
    with pytest.raises(ValueError, match="not directly trainable"):
        validate_bound_s1_training_config(source, seed=3407)

    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bound = bind_s1_training_config(
        source_config_path=ROOT / CONFIG_PATHS[160],
        manifest_path=manifest_path,
        annotation_path=annotation,
        seed=3407,
        work_dir=tmp_path / "run",
    )
    binding = validate_bound_s1_training_config(bound, seed=3407)
    assert set(bound.dataset.train.block_list) == set(manifest["splits"]["gate"])
    assert set(bound.dataset.test.block_list) == set(manifest["splits"]["fit"])
    assert bound.dataset.test.subset_name == "training"
    assert bound.evaluation.subset == "training"
    assert binding["official_test_opened"] is False
    assert binding["formal_precheck_verified"] is False

    tampered = copy.deepcopy(bound)
    tampered.dataset.test.subset_name = "validation"
    with pytest.raises(ValueError, match="modified after manifest binding"):
        validate_bound_s1_training_config(tampered, seed=3407)


def _make_historical_s1_config_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = (tmp_path / "historical-source").resolve()
    shutil.copytree(ROOT / "configs", repository / "configs")
    commands = (
        ("git", "init", "--quiet"),
        ("git", "config", "user.email", "s1-test@example.invalid"),
        ("git", "config", "user.name", "S1 Test"),
        ("git", "add", "configs"),
        ("git", "commit", "--quiet", "-m", "historical S1 configs"),
    )
    for command in commands:
        subprocess.run(command, cwd=repository, check=True)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def test_precheck_certificate_can_be_rebuilt_from_its_historical_repository(
    tmp_path: Path,
) -> None:
    paths = [ROOT / CONFIG_PATHS[value] for value in S1_RESOLUTIONS]
    certificate = run_precheck(paths, mode="static", device="cpu", amp=False)
    repository, commit = _make_historical_s1_config_repository(tmp_path)
    historical = copy.deepcopy(certificate)
    historical["code_commit"] = commit
    for row in historical["rows"]:
        resolution = int(row["spec"]["resolution"])
        row["spec"]["config"] = str((repository / CONFIG_PATHS[resolution]).resolve())
    historical.pop("precheck_sha256")
    historical["precheck_sha256"] = canonical_sha256(historical)
    checked = validate_precheck_certificate(
        historical,
        require_full=False,
        repository_root=repository,
        expected_code_commit=commit,
    )
    assert checked["code_commit"] == commit
    with pytest.raises(ValueError, match="repository commit mismatch"):
        validate_precheck_certificate(
            historical,
            require_full=False,
            repository_root=repository,
            expected_code_commit="0" * 40,
        )


def test_bound_config_accepts_only_clean_commit_exact_historical_source(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bound = bind_s1_training_config(
        source_config_path=ROOT / CONFIG_PATHS[160],
        manifest_path=manifest_path,
        annotation_path=annotation,
        seed=3407,
        work_dir=tmp_path / "run",
    )
    repository, commit = _make_historical_s1_config_repository(tmp_path)
    historical = copy.deepcopy(bound)
    historical.spatial_zoom_s1_runtime_binding.source_config_path = str(
        (repository / CONFIG_PATHS[160]).resolve()
    )
    historical.spatial_zoom_s1_runtime_binding.code_commit = commit
    checked = validate_bound_s1_training_config(historical, seed=3407)
    assert checked["code_commit"] == commit

    dirty_path = repository / "dirty.txt"
    dirty_path.write_text("untracked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must remain clean"):
        validate_bound_s1_training_config(historical, seed=3407)
    dirty_path.unlink()

    wrong_commit = copy.deepcopy(historical)
    wrong_commit.spatial_zoom_s1_runtime_binding.code_commit = "0" * 40
    with pytest.raises(RuntimeError, match="recorded commit"):
        validate_bound_s1_training_config(wrong_commit, seed=3407)

    wrong_source = copy.deepcopy(historical)
    wrong_source.spatial_zoom_s1_runtime_binding.source_config_path = str(
        repository / "configs" / "not-the-audited-config.py"
    )
    with pytest.raises(ValueError, match="audited config path"):
        validate_bound_s1_training_config(wrong_source, seed=3407)


def test_s1_runtime_components_receive_copies_of_the_bound_config(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bound = bind_s1_training_config(
        source_config_path=ROOT / CONFIG_PATHS[160],
        manifest_path=manifest_path,
        annotation_path=annotation,
        seed=3407,
        work_dir=tmp_path / "run",
    )
    binding = validate_bound_s1_training_config(bound, seed=3407)

    optimizer_cfg = copy.deepcopy(bound.optimizer)
    scheduler_cfg = copy.deepcopy(bound.scheduler)
    optimizer_cfg.pop("type")
    scheduler_cfg.pop("type")
    scheduler_cfg.warmup_epoch *= 80
    inference_cfg = copy.deepcopy(bound.inference)
    post_processing_cfg = copy.deepcopy(bound.post_processing)
    inference_cfg["folder"] = str(tmp_path / "outputs")
    post_processing_cfg.sliding_window = True
    validate_bound_s1_training_config(bound, seed=3407)

    metadata = build_s1_checkpoint_metadata(
        bound,
        seed=3407,
        epoch=0,
        successful_updates=1,
        train_batches_per_epoch=1,
    )
    assert metadata["bound_config_sha256"] == canonical_sha256(bound.to_dict())
    assert metadata["manifest_sha256"] == binding["manifest_sha256"]

    train_source = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    assert "build_optimizer(copy.deepcopy(cfg.optimizer)" in train_source
    assert "copy.deepcopy(cfg.scheduler), optimizer" in train_source
    assert "build_s1_checkpoint_metadata(\n                        cfg," in train_source
    test_engine_source = (ROOT / "opentad" / "cores" / "test_engine.py").read_text(
        encoding="utf-8"
    )
    assert "inference_cfg = copy.deepcopy(cfg.inference)" in test_engine_source
    assert (
        "post_processing_cfg = copy.deepcopy(cfg.post_processing)" in test_engine_source
    )
    assert 'cfg.inference["folder"] =' not in test_engine_source
    assert "cfg.post_processing.sliding_window =" not in test_engine_source


def test_checkpoint_metadata_rejects_a_rehashed_wrong_pretrained_identity(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bound = bind_s1_training_config(
        source_config_path=ROOT / CONFIG_PATHS[160],
        manifest_path=manifest_path,
        annotation_path=annotation,
        seed=3407,
        work_dir=tmp_path / "run",
    )
    binding = validate_bound_s1_training_config(bound, seed=3407)
    metadata = build_s1_checkpoint_metadata(
        bound,
        seed=3407,
        epoch=0,
        successful_updates=1,
        train_batches_per_epoch=1,
        amp_skipped_attempts=2,
        max_amp_retries_per_batch=8,
        max_amp_retries_observed=1,
    )
    assert metadata["schema_version"] == S1_CHECKPOINT_METADATA_SCHEMA
    assert metadata["optimizer_attempts"] == 3
    assert metadata["amp_skipped_attempts"] == 2
    assert metadata["max_amp_retries_per_batch"] == 8
    assert metadata["max_amp_retries_observed"] == 1
    forged = copy.deepcopy(metadata)
    forged["pretrained_checkpoint_sha256"] = "0" * 64
    forged.pop("metadata_sha256")
    forged["metadata_sha256"] = canonical_sha256(forged)
    with pytest.raises(ValueError, match="pretrained_checkpoint_sha256"):
        validate_s1_checkpoint_metadata_for_binding(
            forged, binding=binding, epoch=0, cfg=bound
        )

    checkpoint = tmp_path / "epoch_0.pth"
    checkpoint.write_bytes(b"forged-checkpoint")
    sidecar = {
        "schema_version": S1_CHECKPOINT_SIDECAR_SCHEMA,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "experiment_metadata": forged,
    }
    sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
    checkpoint_sidecar_path(checkpoint).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="repository-frozen"):
        validate_s1_checkpoint_sidecar(checkpoint)


def test_real_checkpoint_writer_uses_current_s1_sidecar_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bound = bind_s1_training_config(
        source_config_path=ROOT / CONFIG_PATHS[160],
        manifest_path=manifest_path,
        annotation_path=annotation,
        seed=3407,
        work_dir=tmp_path / "run",
    )
    metadata = build_s1_checkpoint_metadata(
        bound,
        seed=3407,
        epoch=0,
        successful_updates=1,
        train_batches_per_epoch=1,
    )
    fake_torch = SimpleNamespace(
        save=lambda _payload, path: Path(path).write_bytes(b"s1-checkpoint")
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location(
        "s1_checkpoint_writer_under_test",
        ROOT / "opentad" / "utils" / "checkpoint.py",
    )
    assert spec is not None and spec.loader is not None
    checkpoint_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint_module)

    stateful = SimpleNamespace(state_dict=lambda: {"value": 1})

    checkpoint_module.save_checkpoint(
        stateful,
        None,
        stateful,
        stateful,
        0,
        work_dir=str(tmp_path),
        experiment_metadata=metadata,
        experiment_sidecar_schema=S1_CHECKPOINT_SIDECAR_SCHEMA,
    )

    checkpoint = tmp_path / "checkpoint" / "epoch_0.pth"
    sidecar = validate_s1_checkpoint_sidecar(
        checkpoint,
        expected_metadata=metadata,
    )
    assert sidecar["schema_version"] == S1_CHECKPOINT_SIDECAR_SCHEMA


def test_failed_atomic_checkpoint_write_removes_partial_temp_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_partial_write(_payload, path) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated storage failure")

    fake_torch = SimpleNamespace(save=fail_after_partial_write)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location(
        "s1_checkpoint_failure_writer_under_test",
        ROOT / "opentad" / "utils" / "checkpoint.py",
    )
    assert spec is not None and spec.loader is not None
    checkpoint_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint_module)
    stateful = SimpleNamespace(state_dict=lambda: {"value": 1})

    with pytest.raises(RuntimeError, match="simulated storage failure"):
        checkpoint_module.save_checkpoint(
            stateful,
            None,
            stateful,
            stateful,
            41,
            work_dir=str(tmp_path),
            experiment_metadata={"epoch": 41},
            experiment_sidecar_schema=S1_CHECKPOINT_SIDECAR_SCHEMA,
        )

    checkpoint_dir = tmp_path / "checkpoint"
    assert not (checkpoint_dir / "epoch_41.pth").exists()
    assert not (checkpoint_dir / "epoch_41.pth.tmp").exists()
    assert not (checkpoint_dir / "epoch_41.pth.metadata.json").exists()
    assert not (checkpoint_dir / "epoch_41.pth.metadata.json.tmp").exists()


def test_failed_sidecar_publish_rolls_back_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        save=lambda _payload, path: Path(path).write_bytes(b"checkpoint")
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location(
        "s1_checkpoint_publish_failure_writer_under_test",
        ROOT / "opentad" / "utils" / "checkpoint.py",
    )
    assert spec is not None and spec.loader is not None
    checkpoint_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint_module)
    real_replace = checkpoint_module.os.replace
    replace_calls = 0

    def fail_second_replace(source, destination) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated sidecar publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_second_replace)
    stateful = SimpleNamespace(state_dict=lambda: {"value": 1})

    with pytest.raises(OSError, match="sidecar publication failure"):
        checkpoint_module.save_checkpoint(
            stateful,
            None,
            stateful,
            stateful,
            41,
            work_dir=str(tmp_path),
            experiment_metadata={"epoch": 41},
            experiment_sidecar_schema=S1_CHECKPOINT_SIDECAR_SCHEMA,
        )

    checkpoint_dir = tmp_path / "checkpoint"
    assert not (checkpoint_dir / "epoch_41.pth").exists()
    assert not (checkpoint_dir / "epoch_41.pth.tmp").exists()
    assert not (checkpoint_dir / "epoch_41.pth.metadata.json").exists()
    assert not (checkpoint_dir / "epoch_41.pth.metadata.json.tmp").exists()


def test_formal_s1_persists_only_gate_eligible_checkpoints() -> None:
    binding = {"eligible_checkpoint_epochs": list(range(41, 60, 2))}
    assert not should_save_s1_checkpoint(epoch=39, binding=binding)
    assert should_save_s1_checkpoint(epoch=41, binding=binding)
    assert should_save_s1_checkpoint(epoch=59, binding=binding)
    assert not should_save_s1_checkpoint(epoch=58, binding=binding)
    with pytest.raises(ValueError, match="non-empty and unique"):
        should_save_s1_checkpoint(
            epoch=41, binding={"eligible_checkpoint_epochs": [41, 41]}
        )

    train_source = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    assert "should_save_s1_checkpoint(" in train_source
    launcher = (ROOT / "scripts" / "run_spatial_zoom_s1_train_slurm.sh").read_text(
        encoding="utf-8"
    )
    assert S1_MIN_FREE_STORAGE_BYTES == 96 * 1024**3
    assert "S1_MIN_FREE_STORAGE_BYTES" in launcher
    assert 'df -Pk "${STORAGE_PROBE_PATH}"' in launcher


def test_checkpoint_selection_recomputes_gate_metric_and_uses_earliest_tie(
    tmp_path: Path,
) -> None:
    annotation = _write_annotation(tmp_path)
    manifest = build_s1_manifest(annotation)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    database = _annotation_fixture()["database"]
    work_dir = tmp_path / "run"
    materialized_path = tmp_path / "bound.py"
    bound = bind_s1_training_config(
        source_config_path=ROOT / CONFIG_PATHS[160],
        manifest_path=manifest_path,
        annotation_path=annotation,
        seed=3407,
        work_dir=work_dir,
    )
    bound.dump(str(materialized_path))
    bound = Config.fromfile(str(materialized_path))
    validate_bound_s1_training_config(bound, seed=3407)

    exact_results = {}
    for video_id in manifest["splits"]["gate"]:
        rows = []
        for item in database[video_id]["annotations"]:
            start, end = item["segment"]
            rows.append({"label": item["label"], "segment": [start, end], "score": 0.9})
        exact_results[video_id] = rows

    evidence_paths = []
    for epoch in bound.spatial_zoom_s1_runtime_binding.eligible_checkpoint_epochs:
        checkpoint = work_dir / "checkpoint" / f"epoch_{epoch}.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{epoch}".encode("ascii"))
        metadata = build_s1_checkpoint_metadata(
            bound,
            seed=3407,
            epoch=int(epoch),
            successful_updates=int(epoch) + 1,
            train_batches_per_epoch=1,
        )
        sidecar = {
            "schema_version": S1_CHECKPOINT_SIDECAR_SCHEMA,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "experiment_metadata": metadata,
        }
        sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
        checkpoint_sidecar_path(checkpoint).write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        evidence_paths.append(
            write_s1_gate_evidence(
                result_dict=exact_results,
                evaluated_video_ids=manifest["splits"]["gate"],
                cfg=bound,
                epoch=int(epoch),
            )
        )

    report = select_s1_checkpoint(
        config_path=materialized_path,
        seed=3407,
        evidence_paths=evidence_paths,
    )

    assert report["selected"]["epoch"] == min(
        bound.spatial_zoom_s1_runtime_binding.eligible_checkpoint_epochs
    )
    assert report["official_test_read"] is False
    assert report["selection_sha256"]
    with pytest.raises(ValueError, match="cover every eligible checkpoint"):
        select_s1_checkpoint(
            config_path=materialized_path,
            seed=3407,
            evidence_paths=evidence_paths[:-1],
        )
    validate_checkpoint_selection(
        report,
        config=bound,
        seed=3407,
        manifest=manifest,
        checkpoint_path=report["selected"]["checkpoint_path"],
        protocol_fingerprint=bound.spatial_zoom_s1_runtime_binding.protocol_fingerprint,
    )
    forged = copy.deepcopy(report)
    forged["candidates"][-1]["gate_average_map"] = 999.0
    forged.pop("selection_sha256")
    forged["selection_sha256"] = canonical_sha256(forged)
    with pytest.raises(ValueError, match="recomputed gate evidence"):
        validate_checkpoint_selection(
            forged,
            config=bound,
            seed=3407,
            manifest=manifest,
            checkpoint_path=report["selected"]["checkpoint_path"],
            protocol_fingerprint=bound.spatial_zoom_s1_runtime_binding.protocol_fingerprint,
        )
    selected_prediction = Path(report["selected"]["prediction_path"])
    selected_prediction.write_text('{"results": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_checkpoint_selection(
            report,
            config=bound,
            seed=3407,
            manifest=manifest,
            checkpoint_path=report["selected"]["checkpoint_path"],
            protocol_fingerprint=bound.spatial_zoom_s1_runtime_binding.protocol_fingerprint,
        )


def _corpus(score_shift: float = 0.0, boundary_shift: float = 0.0) -> DetectionCorpus:
    gt = {
        "A": {
            "v1": [(0.0, 2.0)],
            "v2": [(4.0, 8.0)],
        },
        "B": {
            "v1": [(10.0, 11.0)],
            "v2": [(12.0, 16.0)],
        },
    }
    predictions = {
        "A": {
            "v1": [(0.95 + score_shift, 0.0 + boundary_shift, 2.0 + boundary_shift)],
            "v2": [(0.90 + score_shift, 4.0 + boundary_shift, 8.0 + boundary_shift)],
        },
        "B": {
            "v1": [(0.85 + score_shift, 10.0 + boundary_shift, 11.0 + boundary_shift)],
            "v2": [(0.80 + score_shift, 12.0 + boundary_shift, 16.0 + boundary_shift)],
        },
    }
    return DetectionCorpus(gt=gt, predictions=predictions, video_ids=("v1", "v2"))


def test_result_evaluator_recomputes_ap_duration_and_boundary_metrics() -> None:
    assert_official_evaluator_parity(_corpus(), tiou_thresholds=(0.3, 0.5, 0.7))
    assert_official_evaluator_parity(
        _corpus(), tiou_thresholds=(0.7,), duration_bounds=(0.0, 1.5)
    )
    tie_corpus = DetectionCorpus(
        gt={"A": {"v1": [(0.0, 1.0)], "v2": [(0.0, 1.0)]}},
        predictions={
            "A": {
                "v1": [(0.5, 2.0, 3.0)],
                "v2": [(0.5, 0.0, 1.0)],
            }
        },
        video_ids=("v1", "v2"),
    )
    assert_official_evaluator_parity(tie_corpus, tiou_thresholds=(0.5,))
    exact = evaluate_corpus(
        _corpus(),
        video_sample=("v1", "v2"),
        tiou_thresholds=(0.3, 0.5, 0.7),
        duration_quartiles=(1.5, 2.5, 3.5),
    )
    shifted = evaluate_corpus(
        _corpus(boundary_shift=0.8),
        video_sample=("v1", "v2"),
        tiou_thresholds=(0.3, 0.5, 0.7),
        duration_quartiles=(1.5, 2.5, 3.5),
    )

    assert exact["average_map"] == pytest.approx(100.0)
    assert exact["map_at"]["0.7"] == pytest.approx(100.0)
    assert exact["duration_map"]["short"]["0.7"] == pytest.approx(100.0)
    assert exact["boundary_error"]["start_mae_seconds"] == pytest.approx(0.0)
    assert shifted["map_at"]["0.7"] < exact["map_at"]["0.7"]
    assert shifted["boundary_error"]["start_mae_seconds"] > 0.0


def test_result_evaluator_matches_official_zero_length_prediction_policy(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotation.json"
    prediction_path = tmp_path / "prediction.json"
    annotation_path.write_text(
        json.dumps(
            {
                "database": {
                    "v1": {
                        "subset": "training",
                        "annotations": [{"label": "A", "segment": [0.0, 1.0]}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    prediction_path.write_text(
        json.dumps(
            {
                "results": {
                    "v1": [
                        {"label": "A", "segment": [-0.0, 0.0], "score": 0.9},
                        {"label": "A", "segment": [0.0, 1.0], "score": 0.8},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    corpus = DetectionCorpus.from_files(
        ground_truth_path=annotation_path,
        prediction_path=prediction_path,
        subset="training",
        video_ids=("v1",),
    )
    assert_official_evaluator_parity(corpus, tiou_thresholds=(0.5,))
    map_at = _map_vector(
        corpus,
        video_sample=corpus.video_ids,
        tiou_thresholds=(0.5,),
        required_labels=("A",),
    )
    assert map_at[0] == pytest.approx(50.0)

    for invalid_segment in ((1.0, 0.0), (0.0, float("nan"))):
        prediction_path.write_text(
            json.dumps(
                {
                    "results": {
                        "v1": [
                            {
                                "label": "A",
                                "segment": list(invalid_segment),
                                "score": 0.9,
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-finite or invalid segment"):
            DetectionCorpus.from_files(
                ground_truth_path=annotation_path,
                prediction_path=prediction_path,
                subset="training",
                video_ids=("v1",),
            )


def test_short_duration_ap_keeps_long_false_positives() -> None:
    corpus = DetectionCorpus(
        gt={
            "A": {
                "short": [(0.0, 1.0)],
                "medium": [(0.0, 3.0)],
                "long": [(0.0, 5.0)],
            }
        },
        predictions={
            "A": {
                "short": [(0.99, 0.0, 10.0), (0.50, 0.0, 1.0)],
                "medium": [(0.90, 0.0, 3.0)],
                "long": [(0.80, 0.0, 5.0)],
            }
        },
        video_ids=("short", "medium", "long"),
    )
    metrics = evaluate_corpus(
        corpus,
        video_sample=corpus.video_ids,
        tiou_thresholds=(0.7,),
        duration_quartiles=(1.5, 3.5, 4.5),
    )
    assert metrics["duration_map"]["short"]["0.7"] == pytest.approx(25.0)


def test_s1_aggregator_uses_paired_video_bootstrap_and_applies_gate_rules() -> None:
    runs = []
    for seed in (3407, 3408, 3409):
        runs.append(
            {"resolution": 160, "seed": seed, "corpus": _corpus(boundary_shift=0.8)}
        )
        runs.append({"resolution": 224, "seed": seed, "corpus": _corpus()})
        runs.append(
            {"resolution": 256, "seed": seed, "corpus": _corpus(boundary_shift=0.4)}
        )

    report = aggregate_s1_runs(
        runs,
        duration_quartiles=(1.5, 2.5, 3.5),
        tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
        bootstrap_replicates=32,
        bootstrap_seed=3407001,
        require_three_seeds=True,
    )

    assert report["status"] == "GO"
    assert report["bootstrap"]["unit"] == "paired_bayesian_video_cluster"
    assert report["bootstrap"]["paired"] is True
    assert report["bootstrap"]["recomputes_full_class_ap"] is True
    assert report["bootstrap"]["support_rejection"] is False
    assert report["resolutions"]["224"]["gate"]["all_conditions"] is True
    assert report["simultaneous_max_t"]["metric"] == "high_tiou_headroom"
    assert report["baseline_dense160"]["boundary_error"]["matched_gt_mean"] > 0
    assert (
        report["resolutions"]["224"]["metrics_per_seed"]["3407"]["boundary_error"][
            "start_mae_seconds"
        ]
        == 0.0
    )


def test_bayesian_video_weights_preserve_all_one_ap_and_integer_multiplicity() -> None:
    corpus = DetectionCorpus(
        gt={"A": {"v1": [(0.0, 1.0)], "v2": [(0.0, 1.0)]}},
        predictions={
            "A": {
                "v1": [(0.9, 0.0, 1.0)],
                "v2": [(0.8, 0.0, 2.0), (0.7, 0.0, 1.0)],
            }
        },
        video_ids=("v1", "v2"),
    )
    unweighted = _map_vector(
        corpus,
        video_sample=corpus.video_ids,
        tiou_thresholds=(0.7,),
    )
    all_one = _map_vector(
        corpus,
        video_sample=corpus.video_ids,
        tiou_thresholds=(0.7,),
        video_weights={"v1": 1.0, "v2": 1.0},
    )
    integer_weighted = _class_ap(
        gt_by_video=corpus.gt["A"],
        pred_by_video=corpus.predictions["A"],
        video_sample=corpus.video_ids,
        tiou_thresholds=(0.7,),
        duration_bounds=None,
        video_weights={"v1": 2.0, "v2": 1.0},
    )
    explicit_clusters = _class_ap(
        gt_by_video=corpus.gt["A"],
        pred_by_video=corpus.predictions["A"],
        video_sample=("v1", "v1", "v2"),
        tiou_thresholds=(0.7,),
        duration_bounds=None,
    )
    assert all_one == pytest.approx(unweighted)
    assert integer_weighted == pytest.approx(explicit_clusters)


def test_bayesian_bootstrap_keeps_rare_class_support_without_rejection() -> None:
    corpus = DetectionCorpus(
        gt={
            "common": {"v1": [(0.0, 2.0)], "v2": [(0.0, 2.0)]},
            "rare": {"v3": [(0.0, 0.5)]},
        },
        predictions={
            "common": {
                "v1": [(0.9, 0.0, 2.0)],
                "v2": [(0.8, 0.0, 2.0)],
            },
            "rare": {"v3": [(0.7, 0.0, 0.5)]},
        },
        video_ids=("v1", "v2", "v3"),
    )
    weights = _paired_bayesian_weights(corpus, replicates=10_000, seed=3407001)
    assert weights.shape == (10_000, 3)
    assert np.isfinite(weights).all()
    assert np.all(weights > 0.0)
    for row in weights[:32]:
        metric = _map_vector(
            corpus,
            video_sample=corpus.video_ids,
            tiou_thresholds=(0.7,),
            video_weights={
                video_id: float(weight)
                for video_id, weight in zip(corpus.video_ids, row)
            },
        )
        assert np.isfinite(metric).all()


def test_deterministic_temporal_interpolation_matches_linear_forward_and_backward() -> (
    None
):
    if sys.platform == "win32":
        pytest.skip("the project Torch DLL runtime is unavailable on this Windows host")
    try:
        import torch
        import torch.nn.functional as functional
        from opentad.datasets.transforms.end_to_end import Interpolate
    except OSError as exc:
        pytest.skip(f"local Torch runtime unavailable: {exc}")

    reference_input = torch.randn(2, 3, 384, dtype=torch.float64, requires_grad=True)
    deterministic_input = reference_input.detach().clone().requires_grad_(True)
    reference = functional.interpolate(
        reference_input, size=768, mode="linear", align_corners=False
    )
    deterministic = Interpolate._linear_2x(deterministic_input)
    assert torch.allclose(reference, deterministic, atol=1e-12, rtol=1e-12)
    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    deterministic.backward(gradient)
    assert torch.allclose(
        reference_input.grad,
        deterministic_input.grad,
        atol=1e-12,
        rtol=1e-12,
    )


def test_formal_s1_entrypoints_request_strict_determinism() -> None:
    train_source = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    test_source = (ROOT / "tools" / "test.py").read_text(encoding="utf-8")
    profile_source = (ROOT / "tools" / "bata" / "profile_spatial_zoom_s1.py").read_text(
        encoding="utf-8"
    )
    assert "deterministic_warn_only=s1_binding is None" in train_source
    assert "deterministic_warn_only=s1_binding is None" in test_source
    assert "set_seed(int(args.seed), deterministic_warn_only=False)" in profile_source


def test_full_precheck_preserves_prediction_shape_evidence_before_release() -> None:
    source = (ROOT / "tools" / "bata" / "run_spatial_zoom_s1_precheck.py").read_text(
        encoding="utf-8"
    )
    capture = source.index("prediction_container_length = len(predictions)")
    release = source.index("del predictions", capture)
    publish = source.index(
        '"prediction_container_length": prediction_container_length', release
    )
    assert capture < release < publish
    assert "len(predictions)" not in source[release:publish]


def test_simultaneous_max_t_uses_the_upper_bootstrap_pivot_for_lower_bounds() -> None:
    observed = np.asarray([1.5, 2.0], dtype=np.float64)
    bootstrap = np.asarray(
        [
            [1.40, 1.90],
            [1.45, 1.95],
            [1.55, 2.05],
            [1.60, 2.10],
            [1.70, 2.25],
            [3.80, 2.40],
        ],
        dtype=np.float64,
    )
    lower, standard_error, critical = _simultaneous_max_t_lower_bounds(
        observed, bootstrap
    )
    pivots = (bootstrap - observed) / standard_error
    expected_critical = max(0.0, float(np.quantile(np.max(pivots, axis=1), 0.95)))
    reversed_critical = max(0.0, float(np.quantile(np.max(-pivots, axis=1), 0.95)))
    assert critical == pytest.approx(expected_critical)
    assert lower == pytest.approx(observed - expected_critical * standard_error)
    assert critical != pytest.approx(reversed_critical)


def test_formal_result_report_rejects_rehashed_manual_go_kill_edits() -> None:
    sources = [
        {
            "resolution": resolution,
            "seed": seed,
            "descriptor_path": f"/{resolution}/{seed}.json",
            "descriptor_file_sha256": f"{resolution}-{seed}-file",
            "descriptor_sha256": f"{resolution}-{seed}-internal",
        }
        for resolution in (160, 224, 256)
        for seed in (3407, 3408, 3409)
    ]
    sealed = seal_s1_result_report(
        {
            "schema_version": "core",
            "status": "GO",
            "resolution_decision": {"selected_resolution": 224},
        },
        source_descriptors=sources,
        global_identity={"experiment_namespace": "s1"},
    )
    assert validate_s1_result_report_envelope(sealed) == sealed
    forged = copy.deepcopy(sealed)
    forged["status"] = "KILL"
    forged["resolution_decision"]["selected_resolution"] = 256
    forged.pop("report_sha256")
    forged["report_sha256"] = canonical_sha256(forged)
    with pytest.raises(ValueError, match="deterministic rebuild"):
        validate_s1_result_report_envelope(forged, expected_report=sealed)


def _profile_metadata(resolution: int, seed: int = 3407) -> dict:
    hardware_identity = {
        "node": "s1-node-a",
        "machine": "x86_64",
        "cpu_model": "S1 CPU",
        "logical_cpu_count": 64,
        "system_memory_bytes": 256 * 1024**3,
        "gpu_name": "S1 GPU",
        "gpu_total_memory": 48 * 1024**3,
        "gpu_compute_capability": [8, 0],
        "gpu_multi_processor_count": 108,
        "physical_gpu_id": "1",
        "scoped_gpu_id": "1",
        "step_gpu_uuid": "GPU-S1",
        "cuda_visible_device_uuid": "GPU-S1",
        "cuda_visible_devices": "0",
        "cuda_visible_nvml_index": 1,
        "nvidia_smi": {
            "uuid": "GPU-S1",
            "pci.bus_id": "00000000:01:00.0",
            "driver_version": "550.54",
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "power.limit": "300.00",
            "clocks.max.sm": "1410",
            "clocks.max.memory": "1215",
        },
        "slurm_gpu_scope": {
            "job_id": "123",
            "step_id": "0",
            "job_gpus": "1,2",
            "step_gpus": "1",
            "scoped_gpu_id": "1",
            "cuda_visible_devices": "0",
        },
        "slurm_resources": {
            "slurm_job_id": "123",
            "slurm_step_id": "0",
            "slurm_job_gpus": "1,2",
            "slurm_step_gpus": "1",
            "scoped_gpu_id": "1",
            "step_gpu_id": "1",
            "step_gpu_uuid": "GPU-S1",
            "cuda_visible_devices": "0",
            "cpus_per_task": 5,
            "outer_job_mem_per_node_mb": 124400,
            "effective_step_memory_limit_mb": 96000,
            "memory_limit_source": "tightest_finite_cgroup_v2_or_slurm",
            "allocated_cpu_ids": [0, 1, 2, 3, 4],
            "detector_cpu_ids": [0, 1, 2, 3],
            "sidecar_cpu_id": 4,
            "detector_process_affinity": [0, 1, 2, 3],
        },
    }
    software_identity = {
        "python": "3.10.0",
        "torch": "2.1.0",
        "cuda_runtime": "11.8",
        "ffmpeg": "ffmpeg version s1",
    }
    profile_order = build_s1_profile_order()
    profile_order_entry = next(
        row
        for row in profile_order
        if int(row["resolution"]) == int(resolution) and int(row["seed"]) == int(seed)
    )
    physical_manifest = ["video-0:0", "video-1:0"]
    exposure_manifest = [
        make_profile_exposure_id(physical_id, ordinal)
        for ordinal, physical_id in enumerate(physical_manifest)
    ]
    return {
        "method": f"dense{resolution}",
        "resolution": resolution,
        "protocol": S1_PROFILE_PROTOCOL,
        "protocol_fingerprint": "matched-s1",
        "manifest_sha256": "manifest",
        "hardware_identity": hardware_identity,
        "hardware_fingerprint": canonical_sha256(hardware_identity),
        "software_identity": software_identity,
        "software_fingerprint": canonical_sha256(software_identity),
        "config_commit": "deadbeef",
        "experiment_namespace": "s1-experiment",
        "canonical_experiment_root": "/s1/canonical",
        "checkpoint_sha256": f"checkpoint-{resolution}",
        "pretrained_checkpoint_sha256": S1_PRETRAINED_CHECKPOINT_SHA256,
        "checkpoint_epoch": 59,
        "trained_checkpoint": True,
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 5,
        "amp": True,
        "power_sampling_enabled": True,
        "power_sampler_backend": "nvml-sidecar-process-v1",
        "trace_publication_mode": S1_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
        "formal_profile": False,
        "split": "test",
        "seed": seed,
        "sample_manifest_sha256": canonical_sha256(exposure_manifest),
        "physical_window_manifest_sha256": canonical_sha256(physical_manifest),
        "loader_exposure_count": 2,
        "physical_window_count": 2,
        "duplicate_physical_window_exposure_count": 0,
        "max_physical_window_multiplicity": 1,
        "test_open_certificate_sha256": "test-open",
        "test_evidence_sha256": "test-evidence",
        "test_open_marker_sha256": "test-marker",
        "precheck_file_sha256": "precheck-file",
        "precheck_sha256": "precheck",
        "power_gpu_id": "1",
        "power_interval_ms": 20,
        "video_count": 2,
        "world_size": 1,
        "execution_wrapper": "torchrun_ddp_world1",
        "result_finalizer": "opentad.cores.test_engine.gather_ddp_results",
        "profile_attempt_marker_path": "dense.started.json",
        "profile_attempt_marker_file_sha256": "marker-file",
        "profile_attempt_marker_sha256": "marker",
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": canonical_sha256(profile_order),
        "profile_order_ordinal": int(profile_order_entry["ordinal"]),
        "profile_code_commit": "f" * 40,
        "profile_recovery_certificate_path": "/s1/canonical/recovery.json",
        "profile_recovery_certificate_file_sha256": "recovery-file",
        "profile_recovery_certificate_sha256": "recovery-internal",
        "profile_recovery_campaign_id": "campaign",
        "power_attempt_report_path": "dense.power_attempt.json",
        "power_attempt_report_file_sha256": "d" * 64,
        "power_attempt_sha256": "d" * 64,
        "power_attempt_trace_path": "dense.power_attempt.jsonl",
        "power_attempt_trace_file_sha256": "d" * 64,
        "power_attempt_cadence": {
            "sample_count": 10,
            "duration_ms": 180.0,
            "finite_nonnegative": True,
            "strictly_increasing": True,
            "min_gap_ms": 20.0,
            "median_gap_ms": 20.0,
            "p95_gap_ms": 20.0,
            "max_gap_ms": 20.0,
            "max_gap_limit_ms": 100.0,
            "formal_cadence_pass": True,
        },
        "allocated_cpu_ids": [0, 1, 2, 3, 4],
        "detector_cpu_ids": [0, 1, 2, 3],
        "sidecar_cpu_id": 4,
        "sidecar_gate_evidence_path": "/s1/canonical/sidecar_gate.json",
        "sidecar_gate_evidence_file_sha256": "d" * 64,
        "sidecar_gate_sha256": "d" * 64,
        "matrix_start_receipt_path": "/s1/canonical/matrix.lock/matrix.started.json",
        "matrix_start_receipt_file_sha256": "d" * 64,
        "matrix_sha256": "d" * 64,
        "slurm_job_id": "123",
        "slurm_step_id": "0",
        "step_gpu_uuid": "GPU-S1",
    }


def _write_valid_sidecar_attempt(
    prefix: Path,
    *,
    expected_uuid: str = "GPU-S1",
    buffered: bool = False,
    timestamps_ns: tuple[int, ...] | None = None,
    cadence_failure: bool = False,
) -> tuple[Path, Path, dict]:
    trace_path = Path(f"{prefix}.power_attempt.jsonl")
    report_path = Path(f"{prefix}.power_attempt.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if timestamps_ns is None:
        timestamps_ns = tuple(1_000_000_000 + index * 20_000_000 for index in range(4))
    trace_path.write_text(
        "".join(
            json.dumps(
                {
                    "sequence": index,
                    "monotonic_ns": timestamp_ns,
                    "power_w": 120.0 + index,
                },
                sort_keys=True,
            )
            + "\n"
            for index, timestamp_ns in enumerate(timestamps_ns)
        ),
        encoding="utf-8",
    )
    clock = {
        "clock": "time.monotonic_ns",
        "implementation": "clock_gettime(CLOCK_MONOTONIC)",
        "monotonic": True,
        "adjustable": False,
        "resolution_seconds": 1e-9,
    }
    pid_record = {
        "schema_version": "spatial_zoom_s1_power_sidecar_pid_v1",
        "pid": 1234,
        "parent_pid": 1200,
        "expected_uuid": expected_uuid,
        "sidecar_cpu_id": 4,
        "actual_cpu_affinity": [4],
        "allocated_cpu_ids": [0, 1, 2, 3, 4],
        "clock_identity": clock,
    }
    pid_record["pid_sha256"] = canonical_sha256(pid_record)
    ready_record = {
        "schema_version": "spatial_zoom_s1_power_sidecar_ready_v1",
        "pid": 1234,
        "parent_pid": 1200,
        "expected_uuid": expected_uuid,
        "actual_uuid": expected_uuid,
        "sidecar_cpu_id": 4,
        "actual_cpu_affinity": [4],
        "allocated_cpu_ids": [0, 1, 2, 3, 4],
        "interval_ms": 20,
        "clock_identity": clock,
        "first_sample_monotonic_ns": int(timestamps_ns[0]),
    }
    ready_record["ready_sha256"] = canonical_sha256(ready_record)
    result_record = {
        "schema_version": (
            S1_POWER_BUFFERED_SIDECAR_RESULT_SCHEMA
            if buffered
            else S1_POWER_SIDECAR_RESULT_SCHEMA
        ),
        "status": "PASS",
        "error": None,
        "pid": 1234,
        "parent_pid": 1200,
        "expected_uuid": expected_uuid,
        "actual_uuid": expected_uuid,
        "sidecar_cpu_id": 4,
        "actual_cpu_affinity": [4],
        "allocated_cpu_ids": [0, 1, 2, 3, 4],
        "interval_ms": 20,
        "clock_identity": clock,
        "sample_count": len(timestamps_ns),
        "started_monotonic_ns": max(1, int(timestamps_ns[0]) - 1_000_000),
        "finished_monotonic_ns": int(timestamps_ns[-1]) + 1_000_000,
        "trace_sha256": sha256_file(trace_path),
    }
    if buffered:
        result_record.update(
            {
                "trace_publication_mode": S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE,
                "trace_io_inside_sampling_loop": False,
            }
        )
    result_record["result_sha256"] = canonical_sha256(result_record)
    cadence = summarize_power_cadence(
        _load_sidecar_trace(trace_path.read_bytes()), target_interval_ms=20
    )
    report = {
        "schema_version": (
            S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA
            if buffered
            else S1_POWER_SIDECAR_ATTEMPT_SCHEMA
        ),
        "backend": S1_POWER_SIDECAR_BACKEND,
        "status": "PASS",
        "error": None,
        "created_utc": "2026-07-17T00:00:00+00:00",
        "expected_uuid": expected_uuid,
        "interval_ms": 20,
        "allocated_cpu_ids": [0, 1, 2, 3, 4],
        "detector_cpu_ids": [0, 1, 2, 3],
        "sidecar_cpu_id": 4,
        "process_pid": 1234,
        "process_exit_code": 0,
        "clock_identity": clock,
        "pid_record": pid_record,
        "ready_record": ready_record,
        "result_record": result_record,
        "cadence": cadence,
        "trace_path": str(trace_path.resolve()),
        "trace_file_sha256": sha256_file(trace_path),
        "stdout_sha256": "0" * 64,
        "stderr_sha256": "0" * 64,
    }
    if cadence_failure:
        report["status"] = "FAIL"
        report["error"] = (
            f"{S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX} "
            f"max_gap_ms={cadence['max_gap_ms']}"
        )
    if buffered:
        report.update(
            {
                "trace_publication_mode": S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE,
                "trace_io_inside_sampling_loop": False,
            }
        )
    report["attempt_sha256"] = canonical_sha256(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, trace_path, report


def _profile_sample(scale: float, index: int) -> dict:
    physical_window_id = f"video-{index}:0"
    return {
        "input_pipeline_serial_ms": 20.0 * scale,
        "h2d_ms": 5.0 * scale,
        "model_forward_ms": 100.0 * scale,
        "postprocess_ms": 10.0 * scale,
        "backbone_wrapper_ms": 80.0 * scale,
        "heavy_backbone_ms": 70.0 * scale,
        "projection_ms": 8.0 * scale,
        "neck_ms": 4.0 * scale,
        "head_ms": 5.0 * scale,
        "decode_to_window_output_wall_ms": 140.0 * scale,
        "final_video_nms_ms": 2.0 * scale,
        "end_to_end_serial_ms": 142.0 * scale,
        "video_id": f"video-{index}",
        "physical_window_id": physical_window_id,
        "loader_ordinal": index,
        "window_id": make_profile_exposure_id(physical_window_id, index),
        "peak_gpu_allocated_mb": 4096.0 * scale,
        "peak_gpu_reserved_mb": 5120.0 * scale,
        "gpu_energy_j": 30.0 * scale,
    }


def test_profile_identity_keeps_duplicate_physical_windows_as_unique_exposures() -> (
    None
):
    batch = {"metas": [{"video_name": "video", "window_start_frame": 7680}]}
    first = _sample_identity(batch, 0)
    second = _sample_identity(batch, 1)
    assert first["physical_window_id"] == second["physical_window_id"]
    assert first["window_id"] != second["window_id"]

    dataset = SimpleNamespace(
        data_list=[
            ["video", {}, {}, np.asarray([7680, 7684])],
            ["video", {}, {}, np.asarray([7680, 7684])],
        ]
    )
    topology = _dataset_exposure_topology(dataset)
    assert topology == {
        "physical_manifest": ["video:7680", "video:7680"],
        "loader_exposure_count": 2,
        "physical_window_count": 1,
        "duplicate_physical_window_exposure_count": 1,
        "max_physical_window_multiplicity": 2,
        "duplicate_physical_window_ids": ["video:7680"],
    }

    samples = [_profile_sample(1.0, 0), _profile_sample(1.1, 1)]
    samples[1].update(
        {
            "video_id": samples[0]["video_id"],
            "physical_window_id": samples[0]["physical_window_id"],
            "window_id": make_profile_exposure_id(samples[0]["physical_window_id"], 1),
        }
    )
    metadata = _profile_metadata(160)
    physical_manifest = [sample["physical_window_id"] for sample in samples]
    metadata.update(
        {
            "sample_manifest_sha256": canonical_sha256(
                [sample["window_id"] for sample in samples]
            ),
            "physical_window_manifest_sha256": canonical_sha256(physical_manifest),
            "physical_window_count": 1,
            "duplicate_physical_window_exposure_count": 1,
            "max_physical_window_multiplicity": 2,
            "video_count": 1,
        }
    )
    report = build_profile_summary(samples, metadata=metadata)
    assert report["sample_count"] == 2
    assert report["physical_window_count"] == 1


def test_formal_profile_accepts_all_exposures_with_one_physical_duplicate() -> None:
    samples = [_profile_sample(1.0, index) for index in range(200)]
    samples[-1].update(
        {
            "video_id": samples[0]["video_id"],
            "physical_window_id": samples[0]["physical_window_id"],
            "window_id": make_profile_exposure_id(
                samples[0]["physical_window_id"], 199
            ),
        }
    )
    metadata = _profile_metadata(160)
    metadata.update(
        {
            "formal_profile": True,
            "warmup_samples": 50,
            "config_commit": "c" * 40,
            "profile_code_commit": "e" * 40,
            "video_count": 199,
            "sample_manifest_sha256": canonical_sha256(
                [sample["window_id"] for sample in samples]
            ),
            "physical_window_manifest_sha256": canonical_sha256(
                [sample["physical_window_id"] for sample in samples]
            ),
            "loader_exposure_count": 200,
            "physical_window_count": 199,
            "duplicate_physical_window_exposure_count": 1,
            "max_physical_window_multiplicity": 2,
        }
    )
    for key in (
        "protocol_fingerprint",
        "manifest_sha256",
        "checkpoint_sha256",
        "pretrained_checkpoint_sha256",
        "test_open_certificate_sha256",
        "test_evidence_sha256",
        "test_open_marker_sha256",
        "precheck_file_sha256",
        "precheck_sha256",
        "profile_attempt_marker_file_sha256",
        "profile_attempt_marker_sha256",
        "profile_recovery_certificate_file_sha256",
        "profile_recovery_certificate_sha256",
    ):
        metadata[key] = "d" * 64
    report = build_profile_summary(
        samples,
        metadata=metadata,
        power_trace=[
            {"timestamp_ms": 0.0, "power_w": 200.0},
            {"timestamp_ms": 20.0, "power_w": 201.0},
        ],
    )
    assert report["sample_count"] == 200
    assert report["physical_window_count"] == 199


def test_power_cadence_diagnostic_preserves_the_formal_gap_threshold() -> None:
    regular = summarize_power_cadence(
        [(0.00, 200.0), (0.02, 201.0), (0.04, 202.0)],
        target_interval_ms=20,
    )
    assert regular["formal_cadence_pass"] is True
    assert regular["max_gap_limit_ms"] == 100.0
    assert regular["sample_count"] == 3

    sparse = summarize_power_cadence(
        [(0.00, 200.0), (0.12, 201.0), (0.14, 202.0)],
        target_interval_ms=20,
    )
    assert sparse["formal_cadence_pass"] is False
    assert sparse["max_gap_ms"] == pytest.approx(120.0)


def test_power_cadence_diagnostic_rejects_nonmonotonic_or_nonfinite_samples() -> None:
    nonmonotonic = summarize_power_cadence(
        [(1.0, 200.0), (1.0, 201.0)], target_interval_ms=20
    )
    assert nonmonotonic["strictly_increasing"] is False
    assert nonmonotonic["formal_cadence_pass"] is False

    nonfinite = summarize_power_cadence(
        [(1.0, 200.0), (1.02, float("nan"))], target_interval_ms=20
    )
    assert nonfinite["finite_nonnegative"] is False
    assert nonfinite["formal_cadence_pass"] is False


def test_power_sampler_diagnostic_is_slurm_local_and_test_blind() -> None:
    source = (
        ROOT / "scripts" / "run_spatial_zoom_s1_power_sampler_diag_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "SLURM_GPUS_ON_NODE" in source
    assert '--physical-gpu-id "${SLURM_JOB_GPUS}"' in source
    assert "CUDA_VISIBLE_DEVICES=" not in source
    assert "annotation" not in source.lower()
    assert "test_evidence" not in source.lower()
    profile_source = (ROOT / "tools" / "bata" / "profile_spatial_zoom_s1.py").read_text(
        encoding="utf-8"
    )
    assert "NvmlSidecarPowerSampler as PowerSampler" in profile_source
    assert 'expected_uuid=hardware_identity["nvidia_smi"]["uuid"]' in profile_source
    assert "sidecar_cpu_id=sidecar_cpu_id" in profile_source
    assert "local_gpu_index=" not in profile_source


def test_sidecar_gate_and_matrix_launchers_freeze_resources_and_order() -> None:
    gate_source = (
        ROOT / "scripts" / "run_spatial_zoom_s1_power_sidecar_gate_slurm.sh"
    ).read_text(encoding="utf-8")
    cell_source = (
        ROOT / "scripts" / "run_spatial_zoom_s1_test_profile_slurm.sh"
    ).read_text(encoding="utf-8")
    matrix_source = (
        ROOT / "scripts" / "run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh"
    ).read_text(encoding="utf-8")
    preflight_source = (
        ROOT / "tools" / "bata" / "preflight_spatial_zoom_s1_profile.py"
    ).read_text(encoding="utf-8")

    for source in (gate_source, cell_source, matrix_source):
        assert 'SLURM_CPUS_PER_TASK:-}" == "5"' in source
        assert "SLURM_STEP_GPUS" in source
        assert "srun --exact" in source
        assert "--gpus=1" in source
        assert "--cpus-per-task=5" in source
        assert "--mem=96000M" in source
        assert "SLURM_MEM_PER_NODE" not in source
        assert "CUDA_VISIBLE_DEVICES=" not in source
        assert "export PYTHONNOUSERSITE=1" in source
    for source in (gate_source, cell_source):
        assert 'DETECTOR_CPUS="${CPU_ARRAY[0]},${CPU_ARRAY[1]}' in source
        assert 'SIDECAR_CPU="${CPU_ARRAY[4]}"' in source
        assert 'taskset -c "${DETECTOR_CPUS}"' in source
        assert "--power-scratch-root" in source
        assert 'spatial_zoom_s1_power.py" salvage' in source
    assert "--sidecar-gate" in gate_source
    assert "--samples 0" in gate_source
    assert "TEST_EVIDENCE_SHA_BEFORE" in gate_source
    assert "Gate published a paper profile" in gate_source
    assert "FROZEN_ORDER=" in matrix_source
    assert "refusing to duplicate an already-started sidecar matrix" in matrix_source
    assert "require_slurm_memory_limit_mb" in matrix_source
    assert "sched_getaffinity" in preflight_source
    assert "set(detector_cpu_ids) | {sidecar_cpu_id}" in preflight_source


def test_nvml_sampler_resolves_the_slurm_device_by_uuid(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeNvml:
        def initialize(self) -> None:
            calls["initialized"] = True

        def handle_by_uuid(self, uuid: str) -> str:
            calls["requested_uuid"] = uuid
            return "allocated-handle"

        def uuid(self, handle: str) -> str:
            calls["checked_handle"] = handle
            return "GPU-allocated"

        def power_w(self, handle: str) -> float:
            calls["sampled_handle"] = handle
            return 123.0

        def shutdown(self) -> None:
            calls["shutdown"] = True

    monkeypatch.setattr(s1_power, "_Nvml", FakeNvml)
    sampler = NvmlPowerSampler(expected_uuid="GPU-allocated", interval_ms=20)
    sampler.start()
    sampler.stop()
    assert calls == {
        "initialized": True,
        "requested_uuid": "GPU-allocated",
        "checked_handle": "allocated-handle",
        "sampled_handle": "allocated-handle",
        "shutdown": True,
    }
    assert sampler.samples and sampler.samples[0][1] == 123.0


def test_sidecar_core_publishes_ordered_monotonic_trace(
    tmp_path: Path, monkeypatch
) -> None:
    affinity = {0, 1, 2, 3, 4}
    scratch = tmp_path / "sidecar"
    trace = scratch / "power.jsonl"
    ready = scratch / "ready.json"
    result = scratch / "result.json"
    sample_calls = 0

    class FakeNvml:
        def initialize(self) -> None:
            pass

        def handle_by_uuid(self, uuid: str) -> str:
            assert uuid == "GPU-allocated"
            return "handle"

        def uuid(self, handle: str) -> str:
            assert handle == "handle"
            return "GPU-allocated"

        def power_w(self, handle: str) -> float:
            nonlocal sample_calls
            assert handle == "handle"
            assert not trace.exists()
            sample_calls += 1
            return 125.0

        def shutdown(self) -> None:
            pass

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        nonlocal affinity
        affinity = set(cpus)

    monkeypatch.setattr(s1_power, "_Nvml", FakeNvml)
    monkeypatch.setattr(s1_power.os, "sched_setaffinity", set_affinity, raising=False)
    monkeypatch.setattr(
        s1_power.os, "sched_getaffinity", lambda _pid: set(affinity), raising=False
    )
    assert (
        run_nvml_sidecar(
            expected_uuid="GPU-allocated",
            interval_ms=20,
            trace_path=trace,
            ready_path=ready,
            result_path=result,
            sidecar_cpu_id=4,
            allocated_cpu_ids=(0, 1, 2, 3, 4),
            stop_after_samples=3,
        )
        == 0
    )
    samples = _load_sidecar_trace(trace.read_bytes())
    assert sample_calls == 3
    assert len(samples) == 3
    assert all(power == 125.0 for _, power in samples)
    result_record = json.loads(result.read_text(encoding="utf-8"))
    result_hash = result_record.pop("result_sha256")
    assert canonical_sha256(result_record) == result_hash
    assert result_record["trace_sha256"] == sha256_file(trace)


def test_sidecar_failure_preserves_raw_trace_and_self_hashed_report(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        s1_power.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2, 3},
        raising=False,
    )
    sampler = NvmlSidecarPowerSampler(
        expected_uuid="GPU-allocated",
        interval_ms=20,
        scratch_dir=tmp_path / "scratch",
        attempt_prefix=tmp_path / "campaign" / "dense256_seed3408",
        sidecar_cpu_id=4,
        detector_cpu_ids=(0, 1, 2, 3),
        allocated_cpu_ids=(0, 1, 2, 3, 4),
    )
    sampler.scratch_dir.mkdir(parents=True)
    trace_payload = (
        json.dumps(
            {"sequence": 0, "monotonic_ns": 1_000_000_000, "power_w": 120.0},
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {"sequence": 1, "monotonic_ns": 1_250_000_000, "power_w": 121.0},
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    sampler._trace_path.write_bytes(trace_payload)
    result = {
        "schema_version": "spatial_zoom_s1_power_sidecar_result_v1",
        "status": "PASS",
        "expected_uuid": "GPU-allocated",
        "actual_uuid": "GPU-allocated",
        "sample_count": 2,
        "trace_sha256": sha256_file(sampler._trace_path),
    }
    result["result_sha256"] = canonical_sha256(result)
    sampler._result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sampler._stdout_path.write_text("", encoding="utf-8")
    sampler._stderr_path.write_text("", encoding="utf-8")
    sampler._process = SimpleNamespace(pid=123, poll=lambda: 0)
    with pytest.raises(RuntimeError, match="cadence failed"):
        sampler._finalize_attempt(forced_error=None)
    assert sampler.attempt_trace_path.read_bytes() == trace_payload
    report = json.loads(sampler.attempt_report_path.read_text(encoding="utf-8"))
    report_hash = report.pop("attempt_sha256")
    assert canonical_sha256(report) == report_hash
    assert report["status"] == "FAIL"
    assert report["cadence"]["max_gap_ms"] == pytest.approx(250.0)


def test_launcher_salvage_seals_trace_after_detector_crash(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    trace = scratch / "power.jsonl"
    trace.write_text(
        json.dumps(
            {"sequence": 0, "monotonic_ns": 1_000_000_000, "power_w": 120.0},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (scratch / "stdout.log").write_text("", encoding="utf-8")
    (scratch / "stderr.log").write_text("worker OOM\n", encoding="utf-8")
    prefix = tmp_path / "campaign" / "dense256_seed3408"
    report = salvage_nvml_sidecar_attempt(
        scratch_dir=scratch,
        attempt_prefix=prefix,
        expected_uuid="GPU-allocated",
        interval_ms=20,
        sidecar_cpu_id=4,
        detector_cpu_ids=(0, 1, 2, 3),
        allocated_cpu_ids=(0, 1, 2, 3, 4),
    )
    assert report["status"] == "FAIL"
    assert report["salvaged_after_parent_failure"] is True
    shared_trace = Path(f"{prefix}.power_attempt.jsonl")
    shared_report = Path(f"{prefix}.power_attempt.json")
    parent_failure = Path(f"{prefix}.power_parent_failure.json")
    assert shared_trace.read_bytes() == trace.read_bytes()
    assert shared_report.is_file()
    assert parent_failure.is_file()
    assert (
        salvage_nvml_sidecar_attempt(
            scratch_dir=scratch,
            attempt_prefix=prefix,
            expected_uuid="GPU-allocated",
            interval_ms=20,
            sidecar_cpu_id=4,
            detector_cpu_ids=(0, 1, 2, 3),
            allocated_cpu_ids=(0, 1, 2, 3, 4),
        )
        == report
    )


def test_launcher_salvage_preserves_completed_attempt_and_seals_parent_failure(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    prefix = tmp_path / "campaign" / "dense256_seed3408"
    shared_trace = Path(f"{prefix}.power_attempt.jsonl")
    shared_report = Path(f"{prefix}.power_attempt.json")
    shared_trace.parent.mkdir(parents=True)
    shared_trace.write_text(
        json.dumps(
            {"sequence": 0, "monotonic_ns": 1_000_000_000, "power_w": 120.0},
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {"sequence": 1, "monotonic_ns": 1_020_000_000, "power_w": 121.0},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    attempt = {
        "schema_version": S1_POWER_SIDECAR_ATTEMPT_SCHEMA,
        "backend": S1_POWER_SIDECAR_BACKEND,
        "status": "PASS",
        "trace_file_sha256": sha256_file(shared_trace),
    }
    attempt["attempt_sha256"] = canonical_sha256(attempt)
    shared_report.write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt_bytes_before = shared_report.read_bytes()
    trace_bytes_before = shared_trace.read_bytes()

    failure = salvage_nvml_sidecar_attempt(
        scratch_dir=scratch,
        attempt_prefix=prefix,
        expected_uuid="GPU-allocated",
        interval_ms=20,
        sidecar_cpu_id=4,
        detector_cpu_ids=(0, 1, 2, 3),
        allocated_cpu_ids=(0, 1, 2, 3, 4),
    )

    assert failure["status"] == "FAIL"
    assert failure["power_attempt_status"] == "PASS"
    assert failure["power_attempt_sha256"] == attempt["attempt_sha256"]
    assert shared_report.read_bytes() == attempt_bytes_before
    assert shared_trace.read_bytes() == trace_bytes_before
    assert Path(f"{prefix}.power_parent_failure.json").is_file()


def test_launcher_salvage_completes_trace_only_attempt_without_overwrite(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    prefix = tmp_path / "campaign" / "dense256_seed3408"
    shared_trace = Path(f"{prefix}.power_attempt.jsonl")
    shared_trace.parent.mkdir(parents=True)
    shared_trace.write_text(
        json.dumps(
            {"sequence": 0, "monotonic_ns": 1_000_000_000, "power_w": 120.0},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trace_before = shared_trace.read_bytes()

    failure = salvage_nvml_sidecar_attempt(
        scratch_dir=scratch,
        attempt_prefix=prefix,
        expected_uuid="GPU-allocated",
        interval_ms=20,
        sidecar_cpu_id=4,
        detector_cpu_ids=(0, 1, 2, 3),
        allocated_cpu_ids=(0, 1, 2, 3, 4),
    )

    shared_report = Path(f"{prefix}.power_attempt.json")
    assert shared_trace.read_bytes() == trace_before
    assert shared_report.is_file()
    assert failure["power_attempt_artifacts_complete"] is True
    attempt = json.loads(shared_report.read_text(encoding="utf-8"))
    assert attempt["status"] == "FAIL"
    assert attempt["trace_file_sha256"] == sha256_file(shared_trace)


def test_launcher_salvage_completes_report_only_attempt_from_matching_scratch(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    prefix = tmp_path / "campaign" / "dense256_seed3408"
    shared_report, shared_trace, report = _write_valid_sidecar_attempt(
        prefix,
        expected_uuid="GPU-allocated",
    )
    scratch_trace = scratch / "power.jsonl"
    scratch_trace.write_bytes(shared_trace.read_bytes())
    report_before = shared_report.read_bytes()
    expected_trace = shared_trace.read_bytes()
    shared_trace.unlink()

    failure = salvage_nvml_sidecar_attempt(
        scratch_dir=scratch,
        attempt_prefix=prefix,
        expected_uuid="GPU-allocated",
        interval_ms=20,
        sidecar_cpu_id=4,
        detector_cpu_ids=(0, 1, 2, 3),
        allocated_cpu_ids=(0, 1, 2, 3, 4),
    )

    assert shared_report.read_bytes() == report_before
    assert shared_trace.read_bytes() == expected_trace
    assert failure["power_attempt_artifacts_complete"] is True
    assert failure["power_attempt_sha256"] == report["attempt_sha256"]


def test_salvage_verifies_live_pid_command_before_signalling(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    command_dir = proc_root / "321"
    command_dir.mkdir(parents=True)
    command_path = command_dir / "cmdline"
    command_path.write_bytes(
        b"/env/python\0/repo/tools/bata/spatial_zoom_s1_power.py\0"
        b"sidecar\0--expected-uuid\0GPU-allocated\0"
    )
    pid_record = {
        "schema_version": "spatial_zoom_s1_power_sidecar_pid_v1",
        "pid": 321,
        "parent_pid": 123,
        "expected_uuid": "GPU-allocated",
        "sidecar_cpu_id": 4,
        "actual_cpu_affinity": [4],
        "allocated_cpu_ids": [0, 1, 2, 3, 4],
        "clock_identity": {"clock": "time.monotonic_ns"},
    }
    pid_record["pid_sha256"] = canonical_sha256(pid_record)
    pid, error = s1_power._validated_sidecar_pid_for_salvage(
        pid_record,
        expected_uuid="GPU-allocated",
        sidecar_cpu_id=4,
        allocated_cpu_ids=(0, 1, 2, 3, 4),
        proc_root=proc_root,
    )
    assert (pid, error) == (321, None)

    command_path.write_bytes(b"/usr/bin/python\0unrelated.py\0")
    pid, error = s1_power._validated_sidecar_pid_for_salvage(
        pid_record,
        expected_uuid="GPU-allocated",
        sidecar_cpu_id=4,
        allocated_cpu_ids=(0, 1, 2, 3, 4),
        proc_root=proc_root,
    )
    assert pid is None
    assert error == "live PID does not match the NVML sidecar command"


def test_sidecar_attempt_validator_rejects_swapped_raw_trace(tmp_path: Path) -> None:
    report_a, trace_a, _ = _write_valid_sidecar_attempt(
        tmp_path / "a",
        expected_uuid="GPU-S1",
    )
    _, trace_b, _ = _write_valid_sidecar_attempt(
        tmp_path / "b",
        expected_uuid="GPU-S1",
    )
    rows = [
        json.loads(line) for line in trace_b.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["power_w"] = 999.0
    trace_b.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trace, cadence, clock, or CPU"):
        validate_nvml_sidecar_attempt(
            report_a,
            trace_b,
            expected_uuid="GPU-S1",
            require_pass=True,
        )
    for consumer in (
        "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
        "tools/bata/analyze_spatial_zoom_s1_results.py",
    ):
        source = (ROOT / consumer).read_text(encoding="utf-8")
        assert "validate_nvml_sidecar_attempt(" in source
        assert 'expected_uuid=profile["hardware_identity"]' in source


def test_buffered_sidecar_cadence_failure_requires_healthy_process(
    tmp_path: Path,
) -> None:
    timestamps = (
        1_000_000_000,
        1_020_000_000,
        1_170_000_000,
        1_190_000_000,
    )
    report_path, trace_path, report = _write_valid_sidecar_attempt(
        tmp_path / "cadence-only",
        expected_uuid="GPU-S1",
        buffered=True,
        timestamps_ns=timestamps,
        cadence_failure=True,
    )
    checked = validate_nvml_sidecar_cadence_failure(
        report_path,
        trace_path,
        expected_uuid="GPU-S1",
    )
    assert checked["cadence"]["max_gap_ms"] == pytest.approx(150.0)

    def publish_tampered(path: Path, payload: dict) -> None:
        payload.pop("attempt_sha256", None)
        payload["attempt_sha256"] = canonical_sha256(payload)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for name, mutate in (
        (
            "exit-137",
            lambda payload: payload.update({"process_exit_code": 137}),
        ),
        (
            "wrong-ready-uuid",
            lambda payload: payload["ready_record"].update(
                {"actual_uuid": "GPU-WRONG"}
            ),
        ),
        (
            "missing-result",
            lambda payload: payload.update({"result_record": None}),
        ),
        (
            "wrong-ready-first-sample",
            lambda payload: payload["ready_record"].update(
                {"first_sample_monotonic_ns": timestamps[0] + 1}
            ),
        ),
        (
            "finished-before-last-sample",
            lambda payload: payload["result_record"].update(
                {"finished_monotonic_ns": timestamps[-1] - 1}
            ),
        ),
    ):
        tampered_path = tmp_path / f"{name}.json"
        tampered = copy.deepcopy(report)
        mutate(tampered)
        for record_key, hash_key in (
            ("pid_record", "pid_sha256"),
            ("ready_record", "ready_sha256"),
            ("result_record", "result_sha256"),
        ):
            record = tampered.get(record_key)
            if isinstance(record, dict):
                record.pop(hash_key, None)
                record[hash_key] = canonical_sha256(record)
        publish_tampered(tampered_path, tampered)
        with pytest.raises(ValueError, match="process identity"):
            validate_nvml_sidecar_cadence_failure(
                tampered_path,
                trace_path,
                expected_uuid="GPU-S1",
            )

    extra_error_path = tmp_path / "extra-error.json"
    extra_error = copy.deepcopy(report)
    extra_error["error"] += "; unrelated failure"
    publish_tampered(extra_error_path, extra_error)
    with pytest.raises(ValueError, match="isolated cadence"):
        validate_nvml_sidecar_cadence_failure(
            extra_error_path,
            trace_path,
            expected_uuid="GPU-S1",
        )


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="real sidecar subprocess lifecycle requires Linux CPU affinity",
)
def test_sidecar_sampler_real_subprocess_lifecycle(tmp_path: Path) -> None:
    original_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if len(original_affinity) < 5:
        pytest.skip("real sidecar lifecycle test requires five available CPUs")
    detector_cpus = original_affinity[:4]
    sidecar_cpu = original_affinity[4]
    fake_sidecar = tmp_path / "fake_sidecar.py"
    fake_sidecar.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})

            from tools.bata import spatial_zoom_s1_power as power

            class FakeNvml:
                def initialize(self):
                    pass

                def handle_by_uuid(self, expected_uuid):
                    return expected_uuid

                def uuid(self, handle):
                    return handle

                def power_w(self, _handle):
                    return 125.0

                def shutdown(self):
                    pass

            power._Nvml = FakeNvml
            raise SystemExit(power.cli())
            """
        ).lstrip(),
        encoding="utf-8",
    )
    os.sched_setaffinity(0, set(detector_cpus))
    sampler = None
    try:
        sampler = NvmlSidecarPowerSampler(
            expected_uuid="GPU-S1",
            interval_ms=20,
            scratch_dir=tmp_path / "scratch",
            attempt_prefix=tmp_path / "campaign" / "dense256_seed3408",
            sidecar_cpu_id=sidecar_cpu,
            detector_cpu_ids=detector_cpus,
            allocated_cpu_ids=(*detector_cpus, sidecar_cpu),
            source_path=fake_sidecar,
            startup_timeout_s=5.0,
            stop_timeout_s=5.0,
        )
        sampler.start()
        time.sleep(0.12)
        assert not sampler._trace_path.exists()
        sampler.stop()
        assert sampler._process is not None
        assert sampler._process.poll() == 0
        checked = validate_nvml_sidecar_attempt(
            sampler.attempt_report_path,
            sampler.attempt_trace_path,
            expected_uuid="GPU-S1",
            require_pass=True,
        )
        assert checked["status"] == "PASS"
        assert checked["cadence"]["formal_cadence_pass"] is True
        assert (
            checked["trace_publication_mode"]
            == S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE
        )
        assert checked["trace_io_inside_sampling_loop"] is False
    finally:
        if sampler is not None and sampler._process is not None:
            if sampler._process.poll() is None:
                sampler._process.kill()
                sampler._process.wait(timeout=5.0)
        os.sched_setaffinity(0, set(original_affinity))


@pytest.mark.parametrize(
    ("source", "expected_error", "startup_timeout_s"),
    (
        (
            "import sys\nsys.exit(9)\n",
            "exited before its ready record",
            5.0,
        ),
        (
            "import time\ntime.sleep(30)\n",
            "did not become ready before timeout",
            0.15,
        ),
    ),
)
@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="real sidecar subprocess failures require Linux CPU affinity",
)
def test_sidecar_sampler_process_failure_leaves_no_orphan(
    tmp_path: Path,
    source: str,
    expected_error: str,
    startup_timeout_s: float,
) -> None:
    original_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if len(original_affinity) < 5:
        pytest.skip("real sidecar lifecycle test requires five available CPUs")
    detector_cpus = original_affinity[:4]
    sidecar_cpu = original_affinity[4]
    fake_sidecar = tmp_path / "failing_sidecar.py"
    fake_sidecar.write_text(source, encoding="utf-8")
    os.sched_setaffinity(0, set(detector_cpus))
    sampler = None
    try:
        sampler = NvmlSidecarPowerSampler(
            expected_uuid="GPU-S1",
            interval_ms=20,
            scratch_dir=tmp_path / "scratch",
            attempt_prefix=tmp_path / "campaign" / "dense256_seed3408",
            sidecar_cpu_id=sidecar_cpu,
            detector_cpu_ids=detector_cpus,
            allocated_cpu_ids=(*detector_cpus, sidecar_cpu),
            source_path=fake_sidecar,
            startup_timeout_s=startup_timeout_s,
            stop_timeout_s=0.15,
        )
        with pytest.raises(RuntimeError, match=expected_error):
            sampler.start()
        assert sampler._process is not None
        assert sampler._process.poll() is not None
        report = json.loads(sampler.attempt_report_path.read_text(encoding="utf-8"))
        assert report["status"] == "FAIL"
        assert expected_error in report["error"]
        assert sampler.attempt_trace_path.is_file()
    finally:
        if sampler is not None and sampler._process is not None:
            if sampler._process.poll() is None:
                sampler._process.kill()
                sampler._process.wait(timeout=5.0)
        os.sched_setaffinity(0, set(original_affinity))


@pytest.mark.parametrize(
    "recovery_reason",
    (
        S1_SIDECAR_RECOVERY_REASON,
        S1_BUFFERED_SIDECAR_RECOVERY_REASON,
        S1_STEP_RUNTIME_RECOVERY_REASON,
        S1_SCHEMA_COMPAT_RECOVERY_REASON,
    ),
)
def test_long_sidecar_gate_is_compact_test_reuse_evidence(
    tmp_path: Path, recovery_reason: str
) -> None:
    buffered_recovery = recovery_reason != S1_SIDECAR_RECOVERY_REASON
    campaign_root = (tmp_path / "campaign").resolve()
    recovery = {
        "reason": recovery_reason,
        "campaign_root": str(campaign_root),
        "sidecar_gate_relative_path": "sidecar_gate.json",
        "profile_code_commit": "a" * 40,
        "certificate_sha256": "b" * 64,
        "campaign_id": "campaign",
        "expected_loader_exposure_count": 2,
        "expected_physical_window_count": 2,
        "power_sampler_backend": S1_POWER_SIDECAR_BACKEND,
        "power_max_gap_limit_ms": 100.0,
        "power_target_interval_ms": 20,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
    }
    if buffered_recovery:
        recovery.update(
            {
                "trace_publication_mode": S1_BUFFERED_TRACE_PUBLICATION_MODE,
                "trace_io_inside_sampling_loop": False,
            }
        )
    metadata = _profile_metadata(256, 3408)
    profile = build_profile_summary(
        [_profile_sample(1.0, 0), _profile_sample(1.1, 1)],
        metadata=metadata,
    )
    gate_dir = campaign_root / "sidecar_gate"
    gate_dir.mkdir(parents=True)
    prefix = gate_dir / "dense256_seed3408_long_full_path"
    marker_path = prefix.with_suffix(".started.json")
    marker = {
        "schema_version": "spatial_zoom_s1_profile_attempt_v6",
        "gate_only": True,
    }
    if buffered_recovery:
        marker.update(
            {
                "trace_publication_mode": S1_BUFFERED_TRACE_PUBLICATION_MODE,
                "trace_io_inside_sampling_loop": False,
            }
        )
    marker["marker_sha256"] = canonical_sha256(marker)
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    attempt_report_path, attempt_trace, _ = _write_valid_sidecar_attempt(
        prefix,
        expected_uuid="GPU-S1",
        buffered=buffered_recovery,
    )
    test_evidence_path = tmp_path / "test.evidence.json"
    test_evidence_path.write_text('{"sealed": true}\n', encoding="utf-8")
    evidence = build_sidecar_gate_evidence(
        recovery=recovery,
        profile_report=profile,
        marker_path=marker_path,
        attempt_report_path=attempt_report_path,
        attempt_trace_path=attempt_trace,
        test_evidence_path=test_evidence_path,
        test_evidence_file_sha256_before=sha256_file(test_evidence_path),
        slurm_job_id="12345",
    )
    path = write_sidecar_gate_evidence(evidence, recovery=recovery)
    assert path == sidecar_gate_path(recovery)
    checked = load_sidecar_gate_evidence(path, recovery=recovery)
    assert checked["status"] == "PASS"
    assert checked["trace_publication_mode"] == recovery.get("trace_publication_mode")
    assert checked["trace_io_inside_sampling_loop"] == recovery.get(
        "trace_io_inside_sampling_loop"
    )
    assert checked["published_formal_profile"] is False
    assert not prefix.with_suffix(".summary.json").exists()
    assert checked["hardware_class"] == sidecar_gate_hardware_class(
        profile["hardware_identity"]
    )
    validate_sidecar_gate_runtime_identity(
        checked,
        hardware_identity=profile["hardware_identity"],
        software_fingerprint=profile["software_fingerprint"],
    )
    mismatched_hardware = copy.deepcopy(profile["hardware_identity"])
    mismatched_hardware["gpu_name"] = "different GPU class"
    with pytest.raises(ValueError, match="hardware class"):
        validate_sidecar_gate_runtime_identity(
            checked,
            hardware_identity=mismatched_hardware,
            software_fingerprint=profile["software_fingerprint"],
        )


def test_full_stack_profile_requires_trained_checkpoint_and_matched_protocol() -> None:
    dense160 = build_profile_summary(
        [_profile_sample(1.0, 0), _profile_sample(1.1, 1)],
        metadata=_profile_metadata(160),
    )
    dense224 = build_profile_summary(
        [_profile_sample(1.5, 0), _profile_sample(1.6, 1)],
        metadata=_profile_metadata(224),
    )
    comparison = compare_resolution_profiles(dense160, dense224)
    assert comparison["comparable"] is True
    assert comparison["candidate_resolution"] == 224
    assert (
        comparison["end_to_end_p50_ms"]["candidate"]
        > comparison["end_to_end_p50_ms"]["baseline"]
    )

    random_init = _profile_metadata(256)
    random_init["trained_checkpoint"] = False
    with pytest.raises(ValueError, match="trained checkpoint"):
        build_profile_summary([_profile_sample(2.0, 0)], metadata=random_init)

    forged_identity = _profile_metadata(256)
    forged_identity["hardware_fingerprint"] = "not-derived-from-identity"
    with pytest.raises(ValueError, match="does not match its fingerprint"):
        build_profile_summary([_profile_sample(2.0, 0)], metadata=forged_identity)

    incompatible = copy.deepcopy(dense224)
    incompatible["hardware_identity"] = {
        **incompatible["hardware_identity"],
        "node": "s1-node-b",
    }
    incompatible["hardware_fingerprint"] = canonical_sha256(
        incompatible["hardware_identity"]
    )
    incompatible.pop("profile_sha256")
    incompatible["profile_sha256"] = canonical_sha256(incompatible)
    with pytest.raises(ValueError, match="hardware_identity"):
        compare_resolution_profiles(dense160, incompatible)

    forged_summary = copy.deepcopy(dense224)
    forged_summary["stages"]["end_to_end_serial_ms"]["p50"] = 0.0
    forged_summary.pop("profile_sha256")
    forged_summary["profile_sha256"] = canonical_sha256(forged_summary)
    with pytest.raises(ValueError, match="raw samples"):
        compare_resolution_profiles(dense160, forged_summary)


def test_formal_profile_rejects_sparse_power_trace_and_missing_window_identity() -> (
    None
):
    samples = [_profile_sample(1.0, index) for index in range(200)]
    metadata = _profile_metadata(160)
    metadata.update(
        {
            "formal_profile": True,
            "warmup_samples": 50,
            "config_commit": "c" * 40,
            "video_count": 200,
            "sample_manifest_sha256": canonical_sha256(
                [sample["window_id"] for sample in samples]
            ),
            "physical_window_manifest_sha256": canonical_sha256(
                [sample["physical_window_id"] for sample in samples]
            ),
            "loader_exposure_count": 200,
            "physical_window_count": 200,
            "duplicate_physical_window_exposure_count": 0,
            "max_physical_window_multiplicity": 1,
        }
    )
    for key in (
        "protocol_fingerprint",
        "manifest_sha256",
        "checkpoint_sha256",
        "pretrained_checkpoint_sha256",
        "test_open_certificate_sha256",
        "test_evidence_sha256",
        "test_open_marker_sha256",
        "precheck_file_sha256",
        "precheck_sha256",
        "profile_attempt_marker_file_sha256",
        "profile_attempt_marker_sha256",
        "profile_recovery_certificate_file_sha256",
        "profile_recovery_certificate_sha256",
    ):
        metadata[key] = "d" * 64
    with pytest.raises(ValueError, match="too sparse"):
        build_profile_summary(
            samples,
            metadata=metadata,
            power_trace=[
                {"timestamp_ms": 0.0, "power_w": 200.0},
                {"timestamp_ms": 200.0, "power_w": 210.0},
            ],
        )
    with pytest.raises(ValueError, match="physical start-frame"):
        _sample_identity({"metas": [{"video_name": "v"}]}, 7)


def test_resolution_freeze_uses_matched_full_stack_cost() -> None:
    runs = []
    samples_by_resolution = {
        160: [_profile_sample(1.0, 0), _profile_sample(1.1, 1)],
        224: [_profile_sample(1.5, 0), _profile_sample(1.6, 1)],
        256: [_profile_sample(2.0, 0), _profile_sample(2.1, 1)],
    }
    for seed in (3407, 3408, 3409):
        for resolution, corpus in (
            (160, _corpus(boundary_shift=0.8)),
            (224, _corpus()),
            (256, _corpus(boundary_shift=0.4)),
        ):
            runs.append(
                {
                    "resolution": resolution,
                    "seed": seed,
                    "corpus": corpus,
                    "profile": build_profile_summary(
                        samples_by_resolution[resolution],
                        metadata=_profile_metadata(resolution, seed),
                    ),
                }
            )
    report = aggregate_s1_runs(
        runs,
        duration_quartiles=(1.5, 2.5, 3.5),
        bootstrap_replicates=32,
        bootstrap_seed=3407001,
        require_three_seeds=True,
    )
    assert report["resolution_decision"]["formal_cost_used"] is True
    assert report["resolution_decision"]["selected_resolution"] == 224
    assert "cost_dominated" not in report["resolutions"]["256"]


def test_cost_only_freezes_among_accuracy_passes_and_never_vetoes_s1_go() -> None:
    runs = []
    for seed in (3407, 3408, 3409):
        for resolution, corpus, scale in (
            (160, _corpus(boundary_shift=0.8), 1.0),
            (224, _corpus(boundary_shift=0.8), 1.2),
            (256, _corpus(), 2.0),
        ):
            runs.append(
                {
                    "resolution": resolution,
                    "seed": seed,
                    "corpus": corpus,
                    "profile": build_profile_summary(
                        [_profile_sample(scale, 0), _profile_sample(scale + 0.1, 1)],
                        metadata=_profile_metadata(resolution, seed),
                    ),
                }
            )
    report = aggregate_s1_runs(
        runs,
        duration_quartiles=(1.5, 2.5, 3.5),
        bootstrap_replicates=32,
        bootstrap_seed=3407001,
        require_three_seeds=True,
    )
    assert report["status"] == "GO"
    assert report["resolutions"]["224"]["gate"]["all_conditions"] is False
    assert report["resolutions"]["256"]["gate"]["all_conditions"] is True
    assert report["resolution_decision"]["selected_resolution"] == 256


def test_profile_matrix_rejects_cross_seed_hardware_or_certificate_drift() -> None:
    runs = []
    for seed in (3407, 3408, 3409):
        for resolution, corpus in (
            (160, _corpus(boundary_shift=0.8)),
            (224, _corpus()),
            (256, _corpus(boundary_shift=0.4)),
        ):
            metadata = _profile_metadata(resolution, seed)
            if seed == 3408:
                metadata["hardware_identity"] = {
                    **metadata["hardware_identity"],
                    "node": "s1-node-b",
                }
                metadata["hardware_fingerprint"] = canonical_sha256(
                    metadata["hardware_identity"]
                )
            runs.append(
                {
                    "resolution": resolution,
                    "seed": seed,
                    "corpus": corpus,
                    "profile": build_profile_summary(
                        [_profile_sample(1.0, 0), _profile_sample(1.1, 1)],
                        metadata=metadata,
                    ),
                }
            )
    with pytest.raises(ValueError, match="incompatible hardware_identity"):
        aggregate_s1_runs(
            runs,
            duration_quartiles=(1.5, 2.5, 3.5),
            bootstrap_replicates=16,
            bootstrap_seed=3407001,
            require_three_seeds=True,
        )
