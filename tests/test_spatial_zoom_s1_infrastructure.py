from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mmengine.config import Config
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
from tools.bata.spatial_zoom_s1_evidence import write_s1_gate_evidence
from tools.bata.spatial_zoom_s1_evidence import (
    validate_s1_checkpoint_metadata_for_binding,
)
from tools.bata.spatial_zoom_s1_cost import (
    S1_PROFILE_PROTOCOL,
    build_profile_summary,
    compare_resolution_profiles,
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
    _sample_identity,
    create_profile_attempt_marker,
    validate_profile_attempt_marker,
    validate_profile_order_ready,
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
    assert require_slurm_single_gpu_allocation() == "6"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="exactly one Slurm-visible"):
        require_slurm_single_gpu_allocation()


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
    observed, order_sha = validate_profile_order_ready(
        manifest=manifest,
        binding=binding,
        resolution=int(first["resolution"]),
        seed=int(first["seed"]),
        hardware_fingerprint="hardware",
        software_fingerprint="software",
    )
    assert observed == first
    assert order_sha == canonical_sha256(order)

    future_prefix = (
        Path(binding["canonical_experiment_root"])
        / f"dense{second['resolution']}"
        / f"seed{second['seed']}"
        / "profile"
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
        )
    current_prefix = (
        Path(binding["canonical_experiment_root"])
        / f"dense{first['resolution']}"
        / f"seed{first['seed']}"
        / "profile"
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
    test_engine_source = (
        ROOT / "opentad" / "cores" / "test_engine.py"
    ).read_text(encoding="utf-8")
    assert "inference_cfg = copy.deepcopy(cfg.inference)" in test_engine_source
    assert (
        "post_processing_cfg = copy.deepcopy(cfg.post_processing)"
        in test_engine_source
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
    launcher = (
        ROOT / "scripts" / "run_spatial_zoom_s1_train_slurm.sh"
    ).read_text(encoding="utf-8")
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


def test_deterministic_temporal_interpolation_matches_linear_forward_and_backward() -> None:
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
    profile_source = (
        ROOT / "tools" / "bata" / "profile_spatial_zoom_s1.py"
    ).read_text(encoding="utf-8")
    assert "deterministic_warn_only=s1_binding is None" in train_source
    assert "deterministic_warn_only=s1_binding is None" in test_source
    assert "set_seed(int(args.seed), deterministic_warn_only=False)" in profile_source


def test_full_precheck_preserves_prediction_shape_evidence_before_release() -> None:
    source = (
        ROOT / "tools" / "bata" / "run_spatial_zoom_s1_precheck.py"
    ).read_text(encoding="utf-8")
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
        "gpu": {"uuid": "GPU-S1", "driver_version": "550.54"},
        "cpu": {"model": "S1 CPU", "logical_count": 64},
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
        "formal_profile": False,
        "split": "test",
        "seed": seed,
        "sample_manifest_sha256": canonical_sha256(["video-0:0", "video-1:0"]),
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
    }


def _profile_sample(scale: float, index: int) -> dict:
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
        "window_id": f"video-{index}:0",
        "peak_gpu_allocated_mb": 4096.0 * scale,
        "peak_gpu_reserved_mb": 5120.0 * scale,
        "gpu_energy_j": 30.0 * scale,
    }


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
