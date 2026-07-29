import ast
import hashlib
import inspect
from pathlib import Path

import pytest

import tools.bata.validate_phystime_decode_cross_suite as suite


def _completion_specs(tmp_path):
    return [
        f"{variant}={tmp_path / f'{variant}.json'}"
        for variant in suite.EXPECTED_RUNS
    ]


def test_completion_specs_require_four_distinct_explicit_artifacts(tmp_path):
    specs = _completion_specs(tmp_path)
    parsed = suite.parse_completion_specs(specs)
    assert set(parsed) == set(suite.EXPECTED_RUNS)
    assert all(path.is_absolute() for path in parsed.values())

    with pytest.raises(ValueError, match="exactly"):
        suite.parse_completion_specs(specs[:-1])
    with pytest.raises(ValueError, match="duplicate completion variant"):
        suite.parse_completion_specs(specs + [specs[0]])
    with pytest.raises(ValueError, match="unknown completion variant"):
        suite.parse_completion_specs(
            specs[:-1] + [f"unknown={tmp_path / 'unknown.json'}"]
        )
    duplicate_path = tmp_path / "same.json"
    duplicate_specs = [
        f"{variant}={duplicate_path}"
        for variant in suite.EXPECTED_RUNS
    ]
    with pytest.raises(ValueError, match="multiple variants"):
        suite.parse_completion_specs(duplicate_specs)


def test_artifact_record_checks_path_hash_and_size(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"validation_pass":true}\n')
    record = {
        "path": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "size_bytes": artifact.stat().st_size,
    }
    assert suite.validate_artifact_record(record, "artifact") == artifact.resolve()

    wrong_hash = dict(record, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        suite.validate_artifact_record(wrong_hash, "artifact")

    wrong_size = dict(record, size_bytes=artifact.stat().st_size + 1)
    with pytest.raises(ValueError, match="size mismatch"):
        suite.validate_artifact_record(wrong_size, "artifact")

    other = tmp_path / "other.json"
    other.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path mismatch"):
        suite.validate_artifact_record(
            record,
            "artifact",
            expected_path=other,
        )


def test_checkpoint_binding_compares_artifact_identity_not_record_shape(tmp_path):
    config = tmp_path / "config.py"
    config.write_text("model = dict() \n", encoding="utf-8")
    checkpoint = tmp_path / "epoch_59.pth"
    checkpoint.write_bytes(b"frozen checkpoint")

    def artifact_record(path):
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    config_binding = {
        "canonical_config_sha256": "1" * 64,
        "coordinate_modes": ["uniform_rank_seconds", "physical_time_seconds"],
        "inference_semantic_sha256": "2" * 64,
        "p0_base_inference_semantic_sha256": "3" * 64,
        "dataset_bindings": {"annotation_sha256": "4" * 64},
    }
    preflight_checkpoint = artifact_record(checkpoint)
    gate_checkpoint = {
        "path": str(checkpoint),
        "sha256": preflight_checkpoint["sha256"],
        "epoch": 59,
        "online_tensor_count": 499,
        "ema_tensor_count": 499,
        "online_state_dict_sha256": "5" * 64,
        "ema_state_dict_sha256": "6" * 64,
    }
    preflight = {
        "configs": {
            "selected_axis": {
                **artifact_record(config),
                **config_binding,
            }
        },
        "checkpoints": {"selected_axis": preflight_checkpoint},
    }
    gate = {
        "configs": {"selected_axis": dict(config_binding)},
        "checkpoints": {"selected_axis": gate_checkpoint},
        "real_windows": {
            "selected_online": {
                "checkpoint_state_dict_sha256": gate_checkpoint[
                    "online_state_dict_sha256"
                ]
            }
        },
    }
    manifest = {
        "config": str(config),
        "effective_config_sha256": "7" * 64,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": gate_checkpoint["sha256"],
        "checkpoint_state_dict_sha256": gate_checkpoint[
            "online_state_dict_sha256"
        ],
    }
    suite._validate_config_and_checkpoint_binding(
        variant="selected_online",
        arm="selected_axis",
        weights="online",
        completion_artifacts={"checkpoint": artifact_record(checkpoint)},
        manifest=manifest,
        preflight=preflight,
        gate=gate,
    )

    other_checkpoint = tmp_path / "other_epoch_59.pth"
    other_checkpoint.write_bytes(checkpoint.read_bytes())
    preflight["checkpoints"]["selected_axis"] = artifact_record(other_checkpoint)
    with pytest.raises(ValueError, match="checkpoint binding differs"):
        suite._validate_config_and_checkpoint_binding(
            variant="selected_online",
            arm="selected_axis",
            weights="online",
            completion_artifacts={"checkpoint": artifact_record(checkpoint)},
            manifest=manifest,
            preflight=preflight,
            gate=gate,
        )


@pytest.mark.parametrize(
    "fatal_text",
    [
        "Traceback (most recent call last):\n",
        "CUDA out of memory\n",
        "[PhysTime decode cross] ERROR: mismatch\n",
    ],
)
def test_explicit_log_scan_fails_closed_on_fatal_markers(
    tmp_path,
    fatal_text,
):
    clean = tmp_path / "clean.log"
    clean.write_text("decode cross completed\n", encoding="utf-8")
    report = suite.scan_logs([clean])
    assert report["files_scanned"] == 1
    assert report["fatal_findings"] == []

    fatal = tmp_path / "fatal.log"
    fatal.write_text(fatal_text, encoding="utf-8")
    with pytest.raises(ValueError, match="fatal log markers"):
        suite.scan_logs([fatal])


def _numeric_contract():
    source_dtypes = {
        "cls_logits": "torch.float16",
        "cls_scores": "torch.float16",
        "reg_distances": "torch.float32",
    }
    capture = {
        "schema_version": suite.CAPTURE_SCHEMA,
        "numeric_semantics_version": suite.NUMERIC_SEMANTICS_VERSION,
        "source_amp_enabled": True,
        "source_tensor_dtypes": source_dtypes,
        "array_contract": {
            "cls_scores": {
                "dtype": "float16",
                "stored_numpy_dtype": "float16",
                "semantic_role": "ranking_scores",
                "ordering_sensitive": True,
                "source_torch_dtype": "torch.float16",
                "replay_torch_dtype": "torch.float16",
                "allowed_casts_before_topk": [],
            }
        },
    }
    numeric = {
        "source_amp_enabled": True,
        "source_tensor_dtypes": source_dtypes,
        "numeric_semantics_version": suite.NUMERIC_SEMANTICS_VERSION,
        "score_sort_dtype": "float16",
        "score_sort_device": "cpu",
        "geometry_compute_dtype": "float32",
        "geometry_compute_device": "cpu",
    }
    return capture, numeric


def test_numeric_contract_rejects_score_dtype_widening():
    capture, numeric = _numeric_contract()
    suite.validate_numeric_precision(numeric, capture, "selected_online")

    capture["array_contract"]["cls_scores"]["dtype"] = "float32"
    capture["array_contract"]["cls_scores"]["stored_numpy_dtype"] = "float32"
    numeric["score_sort_dtype"] = "float32"
    with pytest.raises(ValueError, match="widened"):
        suite.validate_numeric_precision(numeric, capture, "selected_online")


def test_suite_api_has_no_live_scheduler_or_environment_dependency():
    signature = inspect.signature(suite.validate_suite)
    assert "run_root" not in signature.parameters
    assert {
        "preflight_path",
        "gate_path",
        "p0_suite_path",
        "p0_gate_path",
        "completion_paths",
        "log_paths",
    }.issubset(signature.parameters)

    tree = ast.parse(Path(suite.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert not (
                isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr in {"environ", "getenv"}
            )
