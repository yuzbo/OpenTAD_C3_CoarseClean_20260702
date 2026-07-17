from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.duca_full_stack_cost import (
    OFFLINE_FULL_WINDOW_PROTOCOL,
    build_profile_summary,
)
from tools.bata.duca_cellcf_training import canonical_sha256, sha256_file
from tools.bata.duca_trained_checkpoint_binding import (
    load_trained_checkpoint_binding,
)
from tools.bata.profile_duca_full_stack_cost import load_cellcf_cost_binding
from tools.bata.summarize_duca_dense_full_stack_cost import (
    summarize_dense_full_stack_cost,
)


def _profile(path: Path, method: str, scale: float, repeat: int = 1) -> None:
    artifact_root = path.parent / path.stem
    artifact_root.mkdir()
    config = artifact_root / "config.py"
    config.write_text("model = dict(type='ActionFormer')\n", encoding="utf-8")
    checkpoint = artifact_root / (
        "epoch_131.pth" if method == "cellcf-fixed384" else "dense_ema.pth"
    )
    checkpoint.write_bytes(b"trained-checkpoint")
    resolved_config_sha256 = "b" * 64
    binding_metadata = {
        "config_path": str(config.resolve()),
        "profile_config_sha256": sha256_file(config),
        "profile_resolved_config_sha256": resolved_config_sha256,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": 131,
        "checkpoint_state_key": "state_dict_ema",
    }
    if method == "dense-adatad":
        training = artifact_root / "training.json"
        evaluation = artifact_root / "evaluation.json"
        training.write_text('{"ok": true}\n', encoding="utf-8")
        evaluation.write_text('{"ok": true}\n', encoding="utf-8")
        binding_path = artifact_root / "checkpoint_binding.json"
        binding_payload = {
            "schema": "opentad_trained_checkpoint_binding_v1",
            "ok": True,
            "task": "offline_temporal_action_detection",
            "role": "dense_adatad_baseline",
            "git_commit": "a" * 40,
            "config_path": str(config.resolve()),
            "config_sha256": sha256_file(config),
            "resolved_config_sha256": resolved_config_sha256,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": 131,
            "checkpoint_state_key": "state_dict_ema",
            "training_evidence_path": str(training.resolve()),
            "training_evidence_sha256": sha256_file(training),
            "evaluation_evidence_path": str(evaluation.resolve()),
            "evaluation_evidence_sha256": sha256_file(evaluation),
        }
        binding_payload["artifact_sha256"] = canonical_sha256(binding_payload)
        binding_path.write_text(json.dumps(binding_payload), encoding="utf-8")
        binding = load_trained_checkpoint_binding(
            binding_path,
            sha256_file(binding_path),
            expected_role="dense_adatad_baseline",
            expected_commit="a" * 40,
            expected_config_path=config,
            expected_config_sha256=sha256_file(config),
            expected_resolved_config_sha256=resolved_config_sha256,
            expected_checkpoint_path=checkpoint,
        )
        binding_metadata.update(
            trained_checkpoint_binding=binding,
            trained_checkpoint_binding_sha256=canonical_sha256(binding),
        )
    else:
        post_run_path = artifact_root / "post_run_evidence.json"
        post_run = {
            "schema": "duca_cellcf_post_run_evidence_v1",
            "ok": True,
            "git_commit": "a" * 40,
            "variant": "cellcf",
            "seed": 0,
            "training_profile": "exposure132",
            "config_sha256": sha256_file(config),
            "resolved_config_sha256": resolved_config_sha256,
            "runtime_config_sha256": "c" * 64,
            "evaluation_runtime_config_sha256": "d" * 64,
            "successful_optimizer_updates": 13200,
            "checkpoint_epoch": 131,
            "checkpoint_state_key": "state_dict_ema",
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_payload_contract": {
                "payload_reopened": True,
                "epoch": 131,
            },
        }
        post_run["artifact_chain_sha256"] = canonical_sha256(post_run)
        post_run_path.write_text(json.dumps(post_run), encoding="utf-8")
        binding = load_cellcf_cost_binding(
            post_run_path,
            sha256_file(post_run_path),
            expected_checkpoint_path=checkpoint,
            expected_commit="a" * 40,
        )
        binding_metadata.update(
            cellcf_cost_binding=binding,
            cellcf_cost_binding_sha256=canonical_sha256(binding),
        )
    metadata = {
        "method": method,
        "protocol": OFFLINE_FULL_WINDOW_PROTOCOL,
        "hardware_fingerprint": "same-gpu",
        "host_fingerprint": "same-host",
        "software_fingerprint": "same-software",
        "config_commit": "a" * 40,
        "trained_commit": "a" * 40,
        "profile_session_id": "slurm-123",
        "profile_pair_id": f"repeat-{repeat}",
        "profile_repeat_index": repeat,
        "profile_order_position": (
            (1 if repeat % 2 == 1 else 2)
            if method == "dense-adatad"
            else (2 if repeat % 2 == 1 else 1)
        ),
        "tracked_tree_clean": True,
        "dataset_fingerprint": "same-dataset",
        "source_dataset_fingerprint": "same-source",
        "inference_fingerprint": "same-inference",
        "detector_stack_fingerprint": "same-detector",
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 20,
        "amp": True,
        "uses_ema": True,
        "random_init": False,
        "power_sampling_enabled": True,
        "power_interval_ms": 20,
        "power_gpu_id": "GPU-1",
        **binding_metadata,
    }
    jitter = 1.0 + repeat * 0.001
    sample = {
        "input_pipeline_serial_ms": 10.0 * scale * jitter,
        "h2d_ms": 2.0 * scale * jitter,
        "model_forward_ms": 50.0 * scale * jitter,
        "postprocess_ms": 3.0 * scale * jitter,
        "frame_selector_total_ms": 5.0 if method == "cellcf-fixed384" else 0.0,
        "coarse_probe_ms": 2.0 if method == "cellcf-fixed384" else 0.0,
        "backbone_wrapper_total_ms": 35.0 * scale * jitter,
        "heavy_backbone_ms": 30.0 * scale * jitter,
        "projection_ms": 4.0 * scale * jitter,
        "neck_ms": 2.0 * scale * jitter,
        "head_ms": 3.0 * scale * jitter,
        "selected_count": 384 if method == "cellcf-fixed384" else 768,
    }
    samples = []
    for index in range(500):
        factor = 1.0 + index * 0.00001
        samples.append(
            {
                key: (
                    value * factor
                    if key.endswith("_ms") and isinstance(value, (int, float))
                    else value
                )
                for key, value in sample.items()
            }
        )
    report = build_profile_summary(
        samples,
        metadata=metadata,
    )
    path.write_text(json.dumps(report), encoding="utf-8")


def _rebuild_with_raw_samples(payload: dict) -> dict:
    return build_profile_summary(
        payload["raw_samples"],
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "schema_version",
                "sample_count",
                "stage_semantics",
                "stages",
                "selected_count",
                "resources",
                "energy",
                "claims",
                "raw_samples",
            }
        },
    )


def test_dense_full_stack_summary_requires_all_repeat_cost_gates(
    tmp_path: Path,
) -> None:
    dense = []
    cellcf = []
    for repeat in range(1, 4):
        dense_path = tmp_path / f"dense_{repeat}.json"
        cellcf_path = tmp_path / f"cellcf_{repeat}.json"
        _profile(dense_path, "dense-adatad", 2.0, repeat)
        _profile(cellcf_path, "cellcf-fixed384", 1.0, repeat)
        dense.append(dense_path)
        cellcf.append(cellcf_path)

    payload = summarize_dense_full_stack_cost(dense, cellcf)

    assert payload["dense_full_stack_baseline_included"] is True
    assert payload["aggregate"]["median_latency_saving_fraction"] == pytest.approx(
        0.5
    )
    assert payload["aggregate"]["all_repeat_cost_gates_pass"] is True
    assert payload["inference_cost_measurement_claim_allowed"] is True
    assert payload["paper_cost_claim_allowed"] is False
    assert (
        "dense_training_and_evaluation_semantic_validation_not_integrated"
        in payload["paper_cost_claim_blockers"]
    )
    assert payload["training_inference_break_even"]["available"] is False


def test_dense_full_stack_summary_rejects_fewer_than_three_repeats(
    tmp_path: Path,
) -> None:
    dense_path = tmp_path / "dense.json"
    cellcf_path = tmp_path / "cellcf.json"
    _profile(dense_path, "dense-adatad", 2.0)
    _profile(cellcf_path, "cellcf-fixed384", 1.0)

    with pytest.raises(ValueError, match="at least three repeats"):
        summarize_dense_full_stack_cost([dense_path], [cellcf_path])


def test_dense_full_stack_summary_reopens_checkpoint_evidence(
    tmp_path: Path,
) -> None:
    dense = []
    cellcf = []
    for repeat in range(1, 4):
        dense_path = tmp_path / f"dense_{repeat}.json"
        cellcf_path = tmp_path / f"cellcf_{repeat}.json"
        _profile(dense_path, "dense-adatad", 2.0, repeat)
        _profile(cellcf_path, "cellcf-fixed384", 1.0, repeat)
        dense.append(dense_path)
        cellcf.append(cellcf_path)

    payload = json.loads(dense[0].read_text(encoding="utf-8"))
    payload.pop("trained_checkpoint_binding")
    dense[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="trained-checkpoint binding"):
        summarize_dense_full_stack_cost(dense, cellcf)


def test_dense_full_stack_runner_is_alternating_and_hash_bound() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_duca_cellcf_dense_full_stack_cost.sh"
    ).read_text(encoding="utf-8")

    assert "repeat % 2" in source
    assert "--checkpoint-evidence-sha256" in source
    assert "--post-run-evidence-sha256" in source
    assert "--trained-commit" in source
    assert "DUCA_CELLCF_TRAINED_CONFIG" in source
    assert "duca_cellcf_require_external_path" in source
    assert "--sample-power" in source
    assert "REPEATS" in source
    assert '[[ "${SAMPLES}" -ge 500 ]]' in source
    assert '[[ "${WARMUP}" -ge 20 ]]' in source
    assert ".summary.json" in source
    assert "refusing to overwrite an existing cost root" in source
    assert "--profile-repeat-index" in source
    assert "--profile-order-position" in source


def test_dense_full_stack_summary_rejects_too_few_samples(
    tmp_path: Path,
) -> None:
    dense = []
    cellcf = []
    for repeat in range(1, 4):
        dense_path = tmp_path / f"dense_{repeat}.json"
        cellcf_path = tmp_path / f"cellcf_{repeat}.json"
        _profile(dense_path, "dense-adatad", 2.0, repeat)
        _profile(cellcf_path, "cellcf-fixed384", 1.0, repeat)
        dense.append(dense_path)
        cellcf.append(cellcf_path)
    payload = json.loads(dense[0].read_text(encoding="utf-8"))
    payload["raw_samples"] = payload["raw_samples"][:499]
    dense[0].write_text(
        json.dumps(_rebuild_with_raw_samples(payload)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 500 measured samples"):
        summarize_dense_full_stack_cost(dense, cellcf)


def test_dense_full_stack_summary_rejects_too_few_warmup_samples(
    tmp_path: Path,
) -> None:
    dense = []
    cellcf = []
    for repeat in range(1, 4):
        dense_path = tmp_path / f"dense_{repeat}.json"
        cellcf_path = tmp_path / f"cellcf_{repeat}.json"
        _profile(dense_path, "dense-adatad", 2.0, repeat)
        _profile(cellcf_path, "cellcf-fixed384", 1.0, repeat)
        dense.append(dense_path)
        cellcf.append(cellcf_path)
    payload = json.loads(dense[0].read_text(encoding="utf-8"))
    payload["warmup_samples"] = 19
    dense[0].write_text(
        json.dumps(_rebuild_with_raw_samples(payload)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 20 warmup samples"):
        summarize_dense_full_stack_cost(dense, cellcf)


def test_dense_full_stack_summary_rejects_reused_repeat_file(
    tmp_path: Path,
) -> None:
    dense_path = tmp_path / "dense.json"
    cellcf_path = tmp_path / "cellcf.json"
    _profile(dense_path, "dense-adatad", 2.0, 1)
    _profile(cellcf_path, "cellcf-fixed384", 1.0, 1)

    with pytest.raises(ValueError, match="repeat paths contain duplicates"):
        summarize_dense_full_stack_cost(
            [dense_path, dense_path, dense_path],
            [cellcf_path, cellcf_path, cellcf_path],
        )


def test_dense_full_stack_summary_rejects_copied_raw_samples_with_new_metadata(
    tmp_path: Path,
) -> None:
    dense = []
    cellcf = []
    for repeat in range(1, 4):
        dense_path = tmp_path / f"dense_{repeat}.json"
        cellcf_path = tmp_path / f"cellcf_{repeat}.json"
        _profile(dense_path, "dense-adatad", 2.0, repeat)
        _profile(cellcf_path, "cellcf-fixed384", 1.0, repeat)
        dense.append(dense_path)
        cellcf.append(cellcf_path)

    copied = json.loads(dense[0].read_text(encoding="utf-8"))
    for repeat, path in ((2, dense[1]), (3, dense[2])):
        payload = json.loads(json.dumps(copied))
        payload["profile_repeat_index"] = repeat
        payload["profile_pair_id"] = f"repeat-{repeat}"
        payload["profile_order_position"] = 2 if repeat % 2 == 0 else 1
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="raw sample multiset"):
        summarize_dense_full_stack_cost(dense, cellcf)


def test_dense_full_stack_summary_rejects_permuted_raw_sample_copy(
    tmp_path: Path,
) -> None:
    dense = []
    cellcf = []
    for repeat in range(1, 4):
        dense_path = tmp_path / f"dense_{repeat}.json"
        cellcf_path = tmp_path / f"cellcf_{repeat}.json"
        _profile(dense_path, "dense-adatad", 2.0, repeat)
        _profile(cellcf_path, "cellcf-fixed384", 1.0, repeat)
        dense.append(dense_path)
        cellcf.append(cellcf_path)

    copied = json.loads(dense[0].read_text(encoding="utf-8"))
    copied["raw_samples"] = list(reversed(copied["raw_samples"]))
    copied["profile_repeat_index"] = 2
    copied["profile_pair_id"] = "repeat-2"
    copied["profile_order_position"] = 2
    dense[1].write_text(
        json.dumps(_rebuild_with_raw_samples(copied)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw sample multiset"):
        summarize_dense_full_stack_cost(dense, cellcf)
