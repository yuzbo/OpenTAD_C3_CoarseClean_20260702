from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "duca_cellcf_real_loader_cuda_gate_v1"
SYNTHETIC_GATE_SCHEMA = "duca_cellcf_synthetic_gate_v1"
CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "duca_cellcf_fixed384_official_adatad_backend_full_train.py"
)
AUDITED_PATHS = (
    "opentad/cores/train_engine.py",
    "opentad/cores/optimizer.py",
    "opentad/cores/scheduler.py",
    "opentad/datasets/builder.py",
    "opentad/datasets/thumos.py",
    "opentad/datasets/transforms/end_to_end.py",
    "opentad/models/backbones/backbone_wrapper.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/duca/acquisition.py",
    "opentad/models/duca/counterfactual_utility.py",
    "opentad/models/duca/structured_selection.py",
    "opentad/models/duca/transition_only.py",
    "opentad/models/selectors/duca_online_frame_selector.py",
    "opentad/models/utils/truetime_geometry.py",
    "opentad/utils/ema.py",
    "opentad/utils/training_guard.py",
    "configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",
    "configs/_base_/models/actionformer.py",
    "configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py",
    "configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py",
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py",
    "tools/bata/run_duca_cellcf_synthetic_gate.py",
    "tools/bata/run_duca_cellcf_real_loader_cuda_gate.py",
    "tools/bata/validate_duca_cellcf_real_loader_gate.py",
    "tools/bata/duca_cellcf_protocol.py",
    "tools/bata/duca_cellcf_training.py",
    "tools/bata/finalize_duca_cellcf_run.py",
    "tools/bata/validate_duca_cellcf_ddp_pilot.py",
    "tools/bata/validate_duca_cellcf_fixed384.py",
    "tools/bata/validate_duca_cellcf_suite.py",
    "tools/train.py",
    "tools/test.py",
    "tests/test_duca_cellcf_real_loader_gate_contract.py",
)


class GateArtifactFailure(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateArtifactFailure(message)


def _path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _require_hashed_file(payload: Mapping[str, Any], path_key: str, hash_key: str, label: str) -> Path:
    path = _path(payload.get(path_key, ""))
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(payload.get(hash_key) == _sha256(path), f"{label} hash drifted")
    return path


def validate_real_loader_gate_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str | None = None,
    require_clean: bool = False,
) -> dict[str, Any]:
    artifact_path = _path(path)
    _require(artifact_path.is_file(), f"real-loader gate artifact is missing: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "real-loader gate artifact must be a JSON object")
    artifact_sha256 = _sha256(artifact_path)
    if expected_sha256 is not None:
        _require(artifact_sha256 == str(expected_sha256), "real-loader gate artifact SHA-256 mismatch")
    _require(payload.get("schema") == SCHEMA, "real-loader gate artifact schema mismatch")
    _require(payload.get("ok") is True and payload.get("fail_closed") is True, "real-loader gate did not pass fail closed")
    _require(payload.get("git_commit") == expected_commit, "real-loader gate artifact is stale")
    _require(payload.get("git_tree_clean") is True, "real-loader gate was not produced from a clean tree")
    if require_clean:
        _require(_git("rev-parse", "HEAD") == expected_commit, "current checkout differs from the gate commit")
        _require(not _git("status", "--porcelain", "--untracked-files=normal"), "current checkout is dirty")

    final_binding = payload.get("final_clean_binding")
    _require(isinstance(final_binding, Mapping), "real-loader gate lacks its final clean binding")
    for key in (
        "git_commit_unchanged",
        "git_tree_unchanged",
        "git_tree_clean_after_gate",
        "audited_hashes_unchanged",
    ):
        _require(final_binding.get(key) is True, f"real-loader gate final binding failed: {key}")

    audited = payload.get("audited_file_sha256")
    _require(isinstance(audited, Mapping), "real-loader gate lacks audited file hashes")
    _require(set(audited) == set(AUDITED_PATHS), "real-loader gate audited surface drifted")
    for relative_path in AUDITED_PATHS:
        _require(audited.get(relative_path) == _sha256(ROOT / relative_path), f"audited file hash drifted: {relative_path}")

    synthetic_path = _require_hashed_file(
        payload, "synthetic_gate_path", "synthetic_gate_sha256", "bound synthetic gate"
    )
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    _require(synthetic.get("schema") == SYNTHETIC_GATE_SCHEMA and synthetic.get("ok") is True, "bound synthetic gate is invalid")
    _require(synthetic.get("git_commit") == expected_commit, "bound synthetic gate is stale")
    _require(synthetic.get("real_dataset_loader_executed") is False, "synthetic gate provenance is dishonest")
    training_profile = payload.get("config_contract", {}).get("training_profile")
    _require(
        synthetic.get("training_profile") == training_profile,
        "synthetic and real-loader gates use different training profiles",
    )
    synthetic_binding = payload.get("synthetic_gate_binding")
    _require(
        isinstance(synthetic_binding, Mapping)
        and synthetic_binding.get("training_profile") == training_profile,
        "real-loader gate synthetic binding lost its training profile",
    )

    config_path = _require_hashed_file(payload, "config_path", "config_sha256", "CellCF main config")
    _require(config_path == (ROOT / CONFIG_DEFAULT).resolve(), "real-loader gate used another config")
    assets = payload.get("assets")
    _require(isinstance(assets, Mapping), "real-loader gate lacks external asset bindings")
    for key in ("videomae_checkpoint", "official_asformer_source"):
        asset = assets.get(key)
        _require(isinstance(asset, Mapping), f"real-loader gate lacks {key} binding")
        _require_hashed_file(asset, "path", "sha256", f"bound {key}")

    slurm = payload.get("slurm_cuda_binding")
    _require(isinstance(slurm, Mapping), "real-loader gate lacks Slurm CUDA binding")
    _require(str(slurm.get("slurm_job_id", "")).isdigit(), "real-loader gate lacks a numeric Slurm job id")
    _require(slurm.get("logical_device") == "cuda:0", "real-loader gate did not use logical cuda:0")
    _require(slurm.get("logical_cuda_device_count") == 1, "real-loader gate did not use one logical GPU")
    _require(slurm.get("physical_gpu_index_assumed") is False, "real-loader gate assumed a physical GPU index")

    _require(payload.get("real_dataset_loader_executed") is True, "real THUMOS loader was not executed")
    _require(payload.get("synthetic_inputs_used") is False, "real-loader gate used synthetic inputs")
    dataset = payload.get("dataset")
    _require(isinstance(dataset, Mapping), "real-loader gate lacks dataset evidence")
    _require_hashed_file(dataset, "annotation_path", "annotation_sha256", "gate annotation")
    _require_hashed_file(dataset, "class_map_path", "class_map_sha256", "gate class map")
    _require(dataset.get("loader_builder") == "opentad.datasets.build_dataloader", "gate used another data loader")
    _require(dataset.get("train_pipeline_from_main_config") is True, "gate data pipeline drifted")

    coverage = payload.get("validity_window_coverage")
    _require(isinstance(coverage, Mapping), "real-loader gate lacks validity coverage")
    _require(coverage.get("all_obtainable_patterns_executed") is True, "gate skipped an obtainable validity pattern")
    _require(set(coverage.get("obtainable_patterns", ())) == set(coverage.get("executed_patterns", ())), "validity coverage evidence is inconsistent")

    teacher = payload.get("local_signed_counterfactual_teacher")
    _require(isinstance(teacher, Mapping), "real-loader gate lacks local counterfactual evidence")
    _require(teacher.get("local_signed_counterfactual_teacher_verified") is True, "local counterfactual teacher was not verified")
    _require(teacher.get("direct_detector_gradient") is False, "gate unexpectedly used direct detector gradients")
    _require(int(teacher.get("candidate_count", 0)) > 0 and teacher.get("at_least_one_nonzero_utility") is True, "gate utility evidence is uninformative")

    update = payload.get("amp_replay_and_successful_update")
    _require(isinstance(update, Mapping), "real-loader gate lacks update evidence")
    for key, expected in {
        "forced_overflow_attempts": 1,
        "overflow_attempt_skipped": True,
        "same_batch_replayed": True,
        "replay_state_restored": True,
        "successful_optimizer_updates": 1,
        "ema_updates": 1,
        "scheduler_updates": 1,
        "duca_schedule_updates": 1,
        "optimizer_update_verified": True,
        "ema_update_verified": True,
        "schedule_update_verified": True,
    }.items():
        _require(update.get(key) == expected, f"real-loader update proof failed: {key}")

    claims = payload.get("claims")
    _require(isinstance(claims, Mapping), "real-loader gate lacks scoped claims")
    for key in (
        "offline_tad",
        "fixed_k384",
        "official_adatad_actionformer_semantics_preserved",
        "training_profile_preserved",
        "real_loader_cuda_gate_passed",
        "real_gt_remap_verified",
        "actual_acquisition_separate_from_fixed_detector_grid",
        "local_signed_counterfactual_teacher_verified",
        "forced_amp_overflow_replay_verified",
        "one_successful_optimizer_ema_schedule_update_verified",
    ):
        _require(claims.get(key) is True, f"real-loader gate claim failed: {key}")
    _require(claims.get("online_tad") is False, "real-loader gate mislabeled the task as online")
    _require(claims.get("metric_claim_allowed") is False and claims.get("paper_ready") is False, "gate overclaimed evidence")
    return {
        "path": str(artifact_path),
        "sha256": artifact_sha256,
        "git_commit": expected_commit,
        "slurm_job_id": str(slurm["slurm_job_id"]),
        "synthetic_gate_sha256": str(payload["synthetic_gate_sha256"]),
        "dataset_annotation_sha256": str(dataset["annotation_sha256"]),
        "dataset_class_map_sha256": str(dataset["class_map_sha256"]),
        "training_profile": str(training_profile),
    }
