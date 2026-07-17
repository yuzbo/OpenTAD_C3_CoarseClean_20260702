from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.bata.duca_full_stack_cost import (
    OFFLINE_FULL_WINDOW_PROTOCOL,
    build_profile_summary,
)
from tools.bata.duca_cellcf_protocol import protocol_for_name
from tools.bata.profile_duca_full_stack_cost import (
    _canonical_sha256,
    _sha256_file,
    build_arg_parser,
    load_cellcf_cost_binding,
    validate_loaded_checkpoint_binding,
)
from tools.bata.summarize_duca_cellcf_cost import summarize


COMMIT = "a" * 40
EXPOSURE_PROTOCOL = protocol_for_name("exposure132")
CELLCF_TERMINAL_EPOCH = EXPOSURE_PROTOCOL.terminal_epoch
CELLCF_TERMINAL_STATE_KEY = EXPOSURE_PROTOCOL.terminal_state_key


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _post_run_evidence(
    tmp_path: Path,
    *,
    checkpoint_name: str = "epoch_131.pth",
    checkpoint_epoch: int | None = None,
    training_profile: str = "exposure132",
) -> tuple[Path, str, Path]:
    protocol = protocol_for_name(training_profile)
    if checkpoint_epoch is None:
        checkpoint_epoch = protocol.terminal_epoch
    checkpoint = tmp_path / "work" / "checkpoint" / checkpoint_name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"terminal-cellcf-checkpoint")
    payload = {
        "schema": "duca_cellcf_post_run_evidence_v1",
        "ok": True,
        "variant": "cellcf",
        "task": "offline_temporal_action_detection",
        "git_commit": COMMIT,
        "seed": 0,
        "training_profile": protocol.name,
        "config_sha256": "b" * 64,
        "resolved_config_sha256": "c" * 64,
        "runtime_config_sha256": "d" * 64,
        "evaluation_runtime_config_sha256": "e" * 64,
        "successful_optimizer_updates": (
            protocol.expected_successful_optimizer_updates
        ),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_state_key": CELLCF_TERMINAL_STATE_KEY,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_payload_contract": {
            "payload_reopened": True,
            "epoch": checkpoint_epoch,
        },
    }
    payload["artifact_chain_sha256"] = _canonical_sha256(payload)
    evidence = tmp_path / "post_run_evidence.json"
    _write_json(evidence, payload)
    return evidence, _sha256_file(evidence), checkpoint


def _profile(
    binding: dict,
    method: str,
    *,
    end_to_end_ms: float,
    repeat: int,
) -> dict:
    is_cellcf = method == "cellcf-fixed384"
    selector_ms = 3.0 if is_cellcf else 0.0
    probe_ms = 1.0 if is_cellcf else 0.0
    metadata = {
        "method": method,
        "protocol": OFFLINE_FULL_WINDOW_PROTOCOL,
        "hardware_fingerprint": "same-gpu",
        "host_fingerprint": "same-host",
        "software_fingerprint": "same-software",
        "config_commit": binding["git_commit"],
        "profile_session_id": "slurm-test",
        "profile_pair_id": f"repeat-{repeat}",
        "profile_repeat_index": repeat,
        "profile_order_position": (
            (1 if repeat % 2 == 1 else 2)
            if is_cellcf
            else (2 if repeat % 2 == 1 else 1)
        ),
        "tracked_tree_clean": True,
        "dataset_fingerprint": "same-profile-dataset",
        "source_dataset_fingerprint": "same-source-dataset",
        "inference_fingerprint": "same-inference",
        "detector_stack_fingerprint": "same-downstream-detector",
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 20,
        "amp": True,
        "random_init": False,
        "uses_ema": True,
        "power_sampling_enabled": True,
        "power_interval_ms": 20,
        "power_gpu_id": "GPU-1",
        "checkpoint_path": binding["checkpoint_path"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "checkpoint_epoch": binding["checkpoint_epoch"],
        "checkpoint_state_key": binding["checkpoint_state_key"],
        "checkpoint_dropped_prefixes": [] if is_cellcf else ["frame_selector."],
        "checkpoint_dropped_key_count": 0 if is_cellcf else 17,
        "profile_config_sha256": binding["config_sha256"] if is_cellcf else "f" * 64,
        "profile_resolved_config_sha256": (
            binding["resolved_config_sha256"] if is_cellcf else "1" * 64
        ),
        "cellcf_cost_binding": copy.deepcopy(binding),
        "cellcf_cost_binding_sha256": _canonical_sha256(binding),
        "weight_source": "cellcf_trained_terminal_state_dict_ema",
        "frontend_variant": "cellcf" if is_cellcf else "bare_exact_uniform_lower_bound",
        "dense_full_stack_savings_claimed": False,
    }
    sample = {
        "input_pipeline_serial_ms": 1.0,
        "h2d_ms": 1.0,
        "model_forward_ms": end_to_end_ms - 3.0,
        "postprocess_ms": 1.0,
        "frame_selector_total_ms": selector_ms,
        "coarse_probe_ms": probe_ms,
        "backbone_wrapper_total_ms": 3.0,
        "heavy_backbone_ms": 2.0,
        "projection_ms": 1.0,
        "neck_ms": 0.5,
        "head_ms": 0.5,
        "selected_count": 384,
    }
    final_sample = {
        key: (
            value * 1.001
            if key.endswith("_ms") and isinstance(value, (int, float))
            else value
        )
        for key, value in sample.items()
    }
    return build_profile_summary(
        [sample] * 499 + [final_sample],
        metadata=metadata,
    )


def _profile_pair(tmp_path: Path, binding: dict) -> tuple[list[str], list[str]]:
    cellcf_paths = []
    bare_paths = []
    for repeat_index in range(1, 4):
        cellcf_path = tmp_path / f"cellcf_{repeat_index}.json"
        bare_path = tmp_path / f"bare_{repeat_index}.json"
        _write_json(
            cellcf_path,
            _profile(
                binding,
                "cellcf-fixed384",
                end_to_end_ms=11.0 + repeat_index,
                repeat=repeat_index,
            ),
        )
        _write_json(
            bare_path,
            _profile(
                binding,
                "bare-uniform384",
                end_to_end_ms=9.0 + repeat_index,
                repeat=repeat_index,
            ),
        )
        cellcf_paths.append(str(cellcf_path))
        bare_paths.append(str(bare_path))
    return cellcf_paths, bare_paths


def test_binding_reopens_hash_bound_terminal_cellcf_evidence(tmp_path: Path) -> None:
    evidence, evidence_sha, checkpoint = _post_run_evidence(tmp_path)

    binding = load_cellcf_cost_binding(
        evidence,
        evidence_sha,
        expected_checkpoint_path=checkpoint,
        expected_commit=COMMIT,
    )

    assert binding["post_run_evidence_sha256"] == evidence_sha
    assert binding["variant"] == "cellcf"
    assert binding["seed"] == 0
    assert binding["checkpoint_path"] == str(checkpoint.resolve())
    assert binding["checkpoint_epoch"] == CELLCF_TERMINAL_EPOCH
    assert binding["checkpoint_state_key"] == CELLCF_TERMINAL_STATE_KEY
    assert binding["runtime_config_sha256"] == "d" * 64
    assert binding["evaluation_runtime_config_sha256"] == "e" * 64
    assert binding["training_profile"] == "exposure132"


def test_binding_rejects_post_run_evidence_tampered_after_sha_freeze(tmp_path: Path) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["seed"] = 9
    _write_json(evidence, payload)

    with pytest.raises(ValueError, match="post-run evidence SHA256 mismatch"):
        load_cellcf_cost_binding(evidence, evidence_sha)


def test_binding_rejects_terminal_checkpoint_tamper(tmp_path: Path) -> None:
    evidence, evidence_sha, checkpoint = _post_run_evidence(tmp_path)
    checkpoint.write_bytes(b"tampered-checkpoint")

    with pytest.raises(ValueError, match="terminal checkpoint SHA256 differs"):
        load_cellcf_cost_binding(evidence, evidence_sha)


def test_binding_rejects_earlier_checkpoint_path_even_with_forged_terminal_fields(tmp_path: Path) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(
        tmp_path,
        checkpoint_name="epoch_130.pth",
        checkpoint_epoch=CELLCF_TERMINAL_EPOCH,
    )

    with pytest.raises(ValueError, match="exact epoch_131 checkpoint"):
        load_cellcf_cost_binding(evidence, evidence_sha)


def test_loaded_checkpoint_rejects_earlier_payload_epoch(tmp_path: Path) -> None:
    evidence, evidence_sha, checkpoint = _post_run_evidence(tmp_path)
    binding = load_cellcf_cost_binding(evidence, evidence_sha)
    loaded = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "checkpoint_epoch": 130,
        "checkpoint_state_key": CELLCF_TERMINAL_STATE_KEY,
    }

    with pytest.raises(ValueError, match="checkpoint_epoch"):
        validate_loaded_checkpoint_binding(loaded, binding)


@pytest.mark.parametrize("tampered_arm", ["cellcf", "bare"])
def test_summary_rejects_either_profile_arm_when_its_binding_is_tampered(
    tmp_path: Path,
    tampered_arm: str,
) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(tmp_path)
    binding = load_cellcf_cost_binding(evidence, evidence_sha)
    cellcf_paths, bare_paths = _profile_pair(tmp_path, binding)
    target = Path(cellcf_paths[1] if tampered_arm == "cellcf" else bare_paths[1])
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["cellcf_cost_binding"]["seed"] = 7
    payload["cellcf_cost_binding_sha256"] = _canonical_sha256(payload["cellcf_cost_binding"])
    _write_json(target, payload)

    with pytest.raises(ValueError, match="differs from the frozen CellCF cost binding"):
        summarize(
            cellcf_paths,
            bare_paths,
            post_run_evidence_path=evidence,
            post_run_evidence_sha256=evidence_sha,
        )


def test_summary_emits_complete_pass_frontend_only_cost_evidence(tmp_path: Path) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(tmp_path)
    binding = load_cellcf_cost_binding(evidence, evidence_sha)
    cellcf_paths, bare_paths = _profile_pair(tmp_path, binding)

    payload = summarize(
        cellcf_paths,
        bare_paths,
        post_run_evidence_path=evidence,
        post_run_evidence_sha256=evidence_sha,
    )

    assert payload["ok"] is True
    assert payload["status"] == "complete"
    assert payload["pass"] is True
    assert payload["variant"] == "cellcf"
    assert payload["checkpoint_epoch"] == 131
    assert payload["checkpoint_state_key"] == "state_dict_ema"
    assert payload["paired_repeat_order_required"] is True
    assert [
        record["first"]
        for record in payload["paired_repeat_order_receipt"]
    ] == [
        "cellcf-fixed384",
        "bare-uniform384",
        "cellcf-fixed384",
    ]
    assert payload["comparison_contract"] == {
        "weights": "same_cellcf_trained_terminal_state_dict_ema",
        "candidate_frontend": "cellcf",
        "reference_frontend": "bare_exact_uniform_lower_bound",
        "downstream_detector_weights_matched": True,
        "dense_full_stack_baseline_included": False,
        "dense_full_stack_savings_claimed": False,
    }
    assert payload["dense_full_stack_savings_claimed"] is False
    assert payload["dense_baseline_still_required"] is True
    unsigned = dict(payload)
    evidence_hash = unsigned.pop("evidence_sha256")
    assert evidence_hash == _canonical_sha256(unsigned)


def test_official60_cost_binding_and_summary_use_epoch_59(
    tmp_path: Path,
) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(
        tmp_path,
        checkpoint_name="epoch_59.pth",
        training_profile="official60",
    )
    binding = load_cellcf_cost_binding(evidence, evidence_sha)
    cellcf_paths, bare_paths = _profile_pair(tmp_path, binding)

    payload = summarize(
        cellcf_paths,
        bare_paths,
        post_run_evidence_path=evidence,
        post_run_evidence_sha256=evidence_sha,
    )

    assert binding["training_profile"] == "official60"
    assert binding["checkpoint_epoch"] == 59
    assert payload["training_profile"] == "official60"
    assert payload["training_protocol"]["end_epoch"] == 60
    assert payload["checkpoint_epoch"] == 59
    assert payload["checkpoint_state_key"] == "state_dict_ema"


def test_summary_rejects_stage_summary_tampered_without_raw_samples(
    tmp_path: Path,
) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(tmp_path)
    binding = load_cellcf_cost_binding(evidence, evidence_sha)
    cellcf_paths, bare_paths = _profile_pair(tmp_path, binding)
    target = Path(cellcf_paths[0])
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["stages"]["end_to_end_serial_ms"]["p50"] += 1.0
    _write_json(target, payload)

    with pytest.raises(ValueError, match="does not reconstruct exactly"):
        summarize(
            cellcf_paths,
            bare_paths,
            post_run_evidence_path=evidence,
            post_run_evidence_sha256=evidence_sha,
        )


def test_summary_rejects_copied_raw_samples_with_new_repeat_metadata(
    tmp_path: Path,
) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(tmp_path)
    binding = load_cellcf_cost_binding(evidence, evidence_sha)
    cellcf_paths, bare_paths = _profile_pair(tmp_path, binding)
    copied = json.loads(Path(cellcf_paths[0]).read_text(encoding="utf-8"))
    copied["profile_repeat_index"] = 2
    copied["profile_pair_id"] = "repeat-2"
    copied["profile_order_position"] = 2
    _write_json(Path(cellcf_paths[1]), copied)

    with pytest.raises(ValueError, match="raw sample multiset"):
        summarize(
            cellcf_paths,
            bare_paths,
            post_run_evidence_path=evidence,
            post_run_evidence_sha256=evidence_sha,
        )


def test_summary_rejects_permuted_copy_of_the_same_raw_sample_multiset(
    tmp_path: Path,
) -> None:
    evidence, evidence_sha, _ = _post_run_evidence(tmp_path)
    binding = load_cellcf_cost_binding(evidence, evidence_sha)
    cellcf_paths, bare_paths = _profile_pair(tmp_path, binding)
    copied = json.loads(Path(cellcf_paths[0]).read_text(encoding="utf-8"))
    copied["raw_samples"] = list(reversed(copied["raw_samples"]))
    copied["profile_repeat_index"] = 2
    copied["profile_pair_id"] = "repeat-2"
    copied["profile_order_position"] = 2
    _write_json(Path(cellcf_paths[1]), copied)

    with pytest.raises(ValueError, match="raw sample multiset"):
        summarize(
            cellcf_paths,
            bare_paths,
            post_run_evidence_path=evidence,
            post_run_evidence_sha256=evidence_sha,
        )


def test_formal_profile_cli_and_launcher_require_post_run_path_and_sha(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "config.py",
            "--checkpoint",
            "epoch_131.pth",
            "--use-ema",
            "--method-name",
            "cellcf-fixed384",
            "--output-prefix",
            "cost/cellcf",
        ]
    )
    with pytest.raises(ValueError, match="post-run evidence"):
        args.validate()

    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "run_duca_cellcf_cost_pair.sh").read_text(encoding="utf-8")
    assert "DUCA_CELLCF_POST_RUN_EVIDENCE_JSON" in launcher
    assert "DUCA_CELLCF_POST_RUN_EVIDENCE_SHA256" in launcher
    assert launcher.count('--post-run-evidence "${POST_RUN_EVIDENCE}"') == 3
    assert launcher.count('--post-run-evidence-sha256 "${POST_RUN_EVIDENCE_SHA256}"') == 3
    assert ".summary.json" in launcher
    assert "--profile-repeat-index" in launcher
    assert "--profile-order-position" in launcher
    assert "repeat % 2" in launcher
    assert "refusing to overwrite an existing cost root" in launcher
