from __future__ import annotations

import copy
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_contract import canonical_sha256  # noqa: E402
from tools.bata.continuous_roi_s2_training import (  # noqa: E402
    S2_CANONICAL_EXPERIMENTS_ROOT,
    S2_FAMILIES,
    S2_SOURCE_CONFIGS,
    S2_TRAINING_SEEDS,
    S2_UPDATES_PER_EPOCH,
    bind_training_config,
    checkpoint_sidecar_path,
    current_git_commit,
    validate_checkpoint_sidecar,
    validate_training_runtime_precheck,
)
from tools.bata.spatial_zoom_s1_contract import sha256_file  # noqa: E402


S2_RUNTIME_GATE_BINDING_SCHEMA = "continuous_roi_s2_runtime_gate_binding_v1"
S2_RUNTIME_GATE_METADATA_SCHEMA = "continuous_roi_s2_runtime_gate_metadata_v1"
S2_RUNTIME_GATE_SIDECAR_SCHEMA = "continuous_roi_s2_runtime_gate_sidecar_v1"
S2_RUNTIME_GATE_COMPLETION_SCHEMA = (
    "continuous_roi_s2_runtime_gate_completion_v1"
)
S2_RUNTIME_AUTHORIZATION_SCHEMA = (
    "continuous_roi_s2_training_runtime_authorization_v1"
)
S2_RUNTIME_GATE_SEED = S2_TRAINING_SEEDS[0]
S2_RUNTIME_GATE_SUCCESSFUL_UPDATES = 2


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _relative_repository_root(source_config_path: Path, family: str) -> Path:
    relative = S2_SOURCE_CONFIGS[family]
    root = source_config_path
    for _ in relative.parts:
        root = root.parent
    return root


def bind_runtime_gate_config(
    *,
    source_config_path: str | Path,
    family: str,
    work_dir: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
    full_model_gate_path: str | Path,
    training_runtime_precheck_path: str | Path,
    repository_root: str | Path = ROOT,
    code_commit: str | None = None,
) -> Config:
    repository_root = Path(repository_root).resolve()
    family = str(family).upper()
    if family not in S2_FAMILIES:
        raise ValueError(f"unsupported Continuous-RoI S2 family {family}")
    code_commit = (
        current_git_commit(repository_root)
        if code_commit is None
        else str(code_commit).lower()
    )
    precheck = validate_training_runtime_precheck(
        training_runtime_precheck_path,
        expected_commit=code_commit,
        expected_full_model_gate_sha256=_load_json(full_model_gate_path)[
            "gate_sha256"
        ],
    )
    base_work_dir = (
        Path(precheck["canonical_experiment_root"])
        / family.lower()
        / f"seed{S2_RUNTIME_GATE_SEED}"
    )
    cfg = bind_training_config(
        source_config_path=source_config_path,
        family=family,
        seed=S2_RUNTIME_GATE_SEED,
        work_dir=base_work_dir,
        manifest_path=manifest_path,
        development_annotation_path=development_annotation_path,
        class_map_path=class_map_path,
        development_video_root=development_video_root,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        full_model_gate_path=full_model_gate_path,
        training_runtime_precheck_path=training_runtime_precheck_path,
        runtime_authorization_path=None,
        repository_root=repository_root,
        code_commit=code_commit,
        require_runtime_precheck=True,
        require_runtime_authorization=False,
    )
    base_binding = copy.deepcopy(cfg.continuous_roi_s2_runtime_binding)
    del cfg["continuous_roi_s2_runtime_binding"]
    work_dir = Path(work_dir).resolve()
    expected_suffix = Path("training_runtime_gate") / family.lower()
    if work_dir.parts[-2:] != expected_suffix.parts:
        raise ValueError(
            "Continuous-RoI S2 runtime Gate workdir must end with "
            f"{expected_suffix.as_posix()}"
        )
    cfg.work_dir = str(work_dir)
    cfg.workflow.checkpoint_interval = 1
    cfg.workflow.val_loss_interval = -1
    cfg.workflow.val_eval_interval = -1
    cfg.workflow.val_start_epoch = 1
    cfg.workflow.end_epoch = 1
    cfg.workflow.max_train_iters = S2_RUNTIME_GATE_SUCCESSFUL_UPDATES
    cfg.workflow.disable_checkpoint = False
    cfg.workflow.schedule_and_ema_on_success_only = True
    cfg.workflow.max_amp_retries_per_batch = 8
    cfg.workflow.fail_on_skipped_update = True
    cfg.continuous_roi_s2_gate.update(
        stage="training-runtime-one-step-gate",
        precheck_only=False,
        smoke_only=True,
        max_epochs=1,
        max_train_iters=S2_RUNTIME_GATE_SUCCESSFUL_UPDATES,
        requires_launch_gate=True,
        launch_gate_passed=True,
        allow_detector_training=True,
        allow_tools_train=True,
        allow_tools_test=False,
        allow_detector_map=False,
        allow_long_training=False,
        allowed_entrypoints=["tools/train.py"],
        official_test_open_allowed=False,
        learned_crop_policy_allowed=False,
        selector_parameters=0,
        paper_claim_allowed=False,
    )
    binding = {
        "schema_version": S2_RUNTIME_GATE_BINDING_SCHEMA,
        "runtime_bound": True,
        "family": family,
        "seed": S2_RUNTIME_GATE_SEED,
        "code_commit": code_commit,
        "source_config_path": str(Path(source_config_path).resolve()),
        "work_dir": str(work_dir),
        "base_binding": base_binding,
        "full_model_gate_path": str(Path(full_model_gate_path).resolve()),
        "full_model_gate_sha256": base_binding["full_model_gate_sha256"],
        "training_runtime_precheck_path": str(
            Path(training_runtime_precheck_path).resolve()
        ),
        "training_runtime_precheck_sha256": precheck["precheck_sha256"],
        "slurm_runtime": {
            "job_id": precheck["slurm_job_id"],
            "step_id": precheck["slurm_step_id"],
            "step_gpu_identity": precheck["slurm_step_gpu_identity"],
            "cpus_per_task": precheck["slurm_cpus_per_task"],
            "effective_memory_limit_mb": precheck[
                "effective_memory_limit_mb"
            ],
            "cuda_visible_devices": precheck["cuda_visible_devices"],
        },
        "base_experiment_namespace": precheck["experiment_namespace"],
        "expected_successful_updates": S2_RUNTIME_GATE_SUCCESSFUL_UPDATES,
        "train_batches_per_epoch": S2_UPDATES_PER_EPOCH,
        "checkpoint_selection": "runtime_gate_final_ema_only",
        "checkpoint_consumer_state_key": "state_dict_ema",
        "official_test_opened": False,
        "formal_training_authorized": False,
        "paper_claim_allowed": False,
    }
    cfg.continuous_roi_s2_runtime_gate_binding = binding
    return cfg


def validate_runtime_gate_config(cfg: Config, *, seed: int) -> dict[str, Any]:
    binding = cfg.get("continuous_roi_s2_runtime_gate_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("missing Continuous-RoI S2 runtime Gate binding")
    if (
        binding.get("schema_version") != S2_RUNTIME_GATE_BINDING_SCHEMA
        or int(seed) != S2_RUNTIME_GATE_SEED
        or int(binding.get("seed", -1)) != S2_RUNTIME_GATE_SEED
    ):
        raise ValueError("unsupported Continuous-RoI S2 runtime Gate binding")
    source_config_path = Path(str(binding["source_config_path"])).resolve()
    family = str(binding["family"]).upper()
    repository_root = _relative_repository_root(source_config_path, family)
    base = binding["base_binding"]
    expected = bind_runtime_gate_config(
        source_config_path=source_config_path,
        family=family,
        work_dir=binding["work_dir"],
        manifest_path=base["manifest_path"],
        development_annotation_path=base["development_annotation_path"],
        class_map_path=base["class_map_path"],
        development_video_root=base["development_video_root"],
        pretrained_checkpoint_path=base["pretrained_checkpoint_path"],
        full_model_gate_path=binding["full_model_gate_path"],
        training_runtime_precheck_path=binding[
            "training_runtime_precheck_path"
        ],
        repository_root=repository_root,
        code_commit=binding["code_commit"],
    )
    if canonical_sha256(cfg.to_dict()) != canonical_sha256(expected.to_dict()):
        raise ValueError(
            "Continuous-RoI S2 runtime Gate config changed after materialization"
        )
    return dict(binding)


def build_runtime_gate_checkpoint_metadata(
    cfg: Config,
    *,
    seed: int,
    epoch: int,
    successful_updates: int,
    train_batches_per_epoch: int,
    amp_skipped_attempts: int,
    max_amp_retries_observed: int,
    world_size: int,
) -> dict[str, Any]:
    binding = validate_runtime_gate_config(cfg, seed=seed)
    slurm_runtime = binding["slurm_runtime"]
    if (
        int(epoch) != 0
        or int(successful_updates) != S2_RUNTIME_GATE_SUCCESSFUL_UPDATES
        or int(train_batches_per_epoch) != S2_UPDATES_PER_EPOCH
        or int(world_size) != 1
        or os.environ.get("SLURM_JOB_ID") != slurm_runtime["job_id"]
        or os.environ.get("SLURM_STEP_ID") != slurm_runtime["step_id"]
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        != slurm_runtime["cuda_visible_devices"]
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        != slurm_runtime["cpus_per_task"]
    ):
        raise RuntimeError("runtime Gate did not complete two real DDP updates")
    metadata = {
        "schema_version": S2_RUNTIME_GATE_METADATA_SCHEMA,
        "family": binding["family"],
        "seed": int(seed),
        "epoch": int(epoch),
        "successful_updates": int(successful_updates),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "optimizer_attempts": int(successful_updates)
        + int(amp_skipped_attempts),
        "amp_skipped_attempts": int(amp_skipped_attempts),
        "max_amp_retries_observed": int(max_amp_retries_observed),
        "world_size": int(world_size),
        "distributed_backend": "nccl",
        "model_wrapped_with_ddp": True,
        "scheduler_advanced_on_success": True,
        "ema_advanced_on_success": True,
        "slurm_runtime": slurm_runtime,
        "checkpoint_selection": "runtime_gate_final_ema_only",
        "checkpoint_consumer_state_key": "state_dict_ema",
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "code_commit": binding["code_commit"],
        "full_model_gate_sha256": binding["full_model_gate_sha256"],
        "training_runtime_precheck_sha256": binding[
            "training_runtime_precheck_sha256"
        ],
        "base_experiment_namespace": binding["base_experiment_namespace"],
        "official_test_opened": False,
        "formal_training_authorized": False,
        "paper_claim_allowed": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return metadata


def _validate_ema_state(checkpoint: Mapping[str, Any]) -> None:
    import torch

    raw_state = checkpoint.get("state_dict")
    ema_state = checkpoint.get("state_dict_ema")
    if (
        not isinstance(raw_state, Mapping)
        or not isinstance(ema_state, Mapping)
        or not raw_state
        or not ema_state
        or set(raw_state) != set(ema_state)
    ):
        raise ValueError("runtime Gate checkpoint has an invalid EMA state")
    if not any(
        torch.is_tensor(raw_state[key])
        and torch.is_floating_point(raw_state[key])
        and not torch.equal(raw_state[key], ema_state[key])
        for key in raw_state
    ):
        raise ValueError("runtime Gate EMA did not advance after the update")


def build_runtime_gate_completion(
    *,
    cfg: Config,
    seed: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    import torch

    binding = validate_runtime_gate_config(cfg, seed=seed)
    checkpoint_path = Path(checkpoint_path).resolve()
    sidecar = validate_checkpoint_sidecar(
        checkpoint_path,
        expected_sidecar_schema=S2_RUNTIME_GATE_SIDECAR_SCHEMA,
        expected_metadata_schema=S2_RUNTIME_GATE_METADATA_SCHEMA,
        expected_successful_updates=S2_RUNTIME_GATE_SUCCESSFUL_UPDATES,
        expected_checkpoint_selection="runtime_gate_final_ema_only",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("runtime Gate checkpoint is not a mapping")
    _validate_ema_state(checkpoint)
    optimizer_state = checkpoint.get("optimizer")
    scheduler_state = checkpoint.get("scheduler")
    if (
        not isinstance(optimizer_state, Mapping)
        or not optimizer_state.get("state")
        or not isinstance(scheduler_state, Mapping)
        or int(scheduler_state.get("last_epoch", 0)) <= 0
    ):
        raise ValueError(
            "runtime Gate optimizer or scheduler did not advance"
        )
    metadata = sidecar["experiment_metadata"]
    if checkpoint.get("experiment_metadata") != metadata:
        raise ValueError("runtime Gate checkpoint/sidecar metadata differs")
    report = {
        "schema_version": S2_RUNTIME_GATE_COMPLETION_SCHEMA,
        "status": "PASS",
        "family": binding["family"],
        "seed": int(seed),
        "work_dir": binding["work_dir"],
        "bound_config_sha256": metadata["bound_config_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "checkpoint_sidecar_path": str(checkpoint_sidecar_path(checkpoint_path)),
        "checkpoint_sidecar_file_sha256": sha256_file(
            checkpoint_sidecar_path(checkpoint_path)
        ),
        "checkpoint_metadata_sha256": metadata["metadata_sha256"],
        "successful_updates": metadata["successful_updates"],
        "train_batches_per_epoch": metadata["train_batches_per_epoch"],
        "optimizer_attempts": metadata["optimizer_attempts"],
        "amp_skipped_attempts": metadata["amp_skipped_attempts"],
        "max_amp_retries_observed": metadata["max_amp_retries_observed"],
        "world_size": metadata["world_size"],
        "model_wrapped_with_ddp": metadata["model_wrapped_with_ddp"],
        "slurm_runtime": metadata["slurm_runtime"],
        "scheduler_advanced_on_success": True,
        "ema_advanced_on_success": True,
        "ema_state_nonempty": True,
        "ema_keys_match_model": True,
        "ema_differs_from_model_after_update": True,
        "optimizer_state_nonempty": True,
        "scheduler_last_epoch": int(scheduler_state["last_epoch"]),
        "checkpoint_consumer_state_key": "state_dict_ema",
        "code_commit": binding["code_commit"],
        "full_model_gate_sha256": binding["full_model_gate_sha256"],
        "training_runtime_precheck_sha256": binding[
            "training_runtime_precheck_sha256"
        ],
        "base_experiment_namespace": binding["base_experiment_namespace"],
        "official_test_opened": False,
        "formal_training_authorized": False,
        "paper_claim_allowed": False,
    }
    report["completion_sha256"] = canonical_sha256(report)
    return report


def _validate_runtime_gate_completion(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    expected = report.pop("completion_sha256", None)
    if not expected or canonical_sha256(report) != expected:
        raise ValueError(f"runtime Gate completion self-hash mismatch: {path}")
    report["completion_sha256"] = expected
    if (
        report.get("schema_version") != S2_RUNTIME_GATE_COMPLETION_SCHEMA
        or report.get("status") != "PASS"
        or report.get("seed") != S2_RUNTIME_GATE_SEED
        or report.get("successful_updates")
        != S2_RUNTIME_GATE_SUCCESSFUL_UPDATES
        or report.get("train_batches_per_epoch") != S2_UPDATES_PER_EPOCH
        or report.get("world_size") != 1
        or report.get("model_wrapped_with_ddp") is not True
        or not isinstance(report.get("slurm_runtime"), Mapping)
        or not report["slurm_runtime"].get("job_id")
        or not report["slurm_runtime"].get("step_id")
        or report["slurm_runtime"].get("cpus_per_task") != 5
        or int(
            report["slurm_runtime"].get("effective_memory_limit_mb", 0)
        )
        < 90000
        or report.get("scheduler_advanced_on_success") is not True
        or report.get("ema_advanced_on_success") is not True
        or report.get("ema_state_nonempty") is not True
        or report.get("ema_keys_match_model") is not True
        or report.get("ema_differs_from_model_after_update") is not True
        or report.get("optimizer_state_nonempty") is not True
        or int(report.get("scheduler_last_epoch", 0)) <= 0
        or report.get("checkpoint_consumer_state_key") != "state_dict_ema"
        or report.get("official_test_opened") is not False
        or report.get("formal_training_authorized") is not False
    ):
        raise ValueError(f"runtime Gate completion is incomplete: {path}")
    checkpoint_path = Path(str(report["checkpoint_path"])).resolve()
    sidecar_path = Path(str(report["checkpoint_sidecar_path"])).resolve()
    work_dir = Path(str(report["work_dir"])).resolve()
    family = str(report["family"]).lower()
    if (
        checkpoint_path.parent.name != "checkpoint"
        or checkpoint_path.parent.parent != work_dir
        or work_dir.name != family
        or work_dir.parent.name != "training_runtime_gate"
        or sidecar_path != checkpoint_sidecar_path(checkpoint_path).resolve()
    ):
        raise ValueError("runtime Gate checkpoint path is outside its workdir")
    sidecar = validate_checkpoint_sidecar(
        checkpoint_path,
        expected_sidecar_schema=S2_RUNTIME_GATE_SIDECAR_SCHEMA,
        expected_metadata_schema=S2_RUNTIME_GATE_METADATA_SCHEMA,
        expected_successful_updates=S2_RUNTIME_GATE_SUCCESSFUL_UPDATES,
        expected_checkpoint_selection="runtime_gate_final_ema_only",
    )
    metadata = sidecar["experiment_metadata"]
    if (
        sha256_file(checkpoint_path) != report["checkpoint_sha256"]
        or sha256_file(sidecar_path)
        != report["checkpoint_sidecar_file_sha256"]
        or metadata.get("metadata_sha256")
        != report["checkpoint_metadata_sha256"]
        or metadata.get("bound_config_sha256")
        != report["bound_config_sha256"]
        or metadata.get("family") != report["family"]
        or metadata.get("slurm_runtime") != report["slurm_runtime"]
    ):
        raise ValueError("runtime Gate checkpoint evidence changed")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("runtime Gate checkpoint is not a mapping")
    _validate_ema_state(checkpoint)
    optimizer_state = checkpoint.get("optimizer")
    scheduler_state = checkpoint.get("scheduler")
    if (
        checkpoint.get("experiment_metadata") != metadata
        or not isinstance(optimizer_state, Mapping)
        or not optimizer_state.get("state")
        or not isinstance(scheduler_state, Mapping)
        or int(scheduler_state.get("last_epoch", 0))
        != int(report["scheduler_last_epoch"])
    ):
        raise ValueError("runtime Gate optimizer/scheduler evidence changed")
    return report


def build_runtime_authorization(
    *,
    expected_commit: str,
    full_model_gate_path: str | Path,
    training_runtime_precheck_path: str | Path,
    completion_paths: list[str | Path],
) -> dict[str, Any]:
    precheck_payload = _load_json(training_runtime_precheck_path)
    precheck = validate_training_runtime_precheck(
        training_runtime_precheck_path,
        expected_commit=expected_commit,
        expected_full_model_gate_sha256=precheck_payload[
            "full_model_gate_sha256"
        ],
    )
    completions = []
    slurm_runtime_hashes = set()
    for path in map(Path, completion_paths):
        completion = _validate_runtime_gate_completion(path.resolve())
        completions.append(
            {
                "family": completion["family"],
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "completion_sha256": completion["completion_sha256"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
                "checkpoint_sidecar_file_sha256": completion[
                    "checkpoint_sidecar_file_sha256"
                ],
            }
        )
        slurm_runtime_hashes.add(
            canonical_sha256(completion["slurm_runtime"])
        )
        if (
            completion["code_commit"] != str(expected_commit).lower()
            or completion["full_model_gate_sha256"]
            != precheck["full_model_gate_sha256"]
            or completion["training_runtime_precheck_sha256"]
            != precheck["precheck_sha256"]
            or completion["base_experiment_namespace"]
            != precheck["experiment_namespace"]
        ):
            raise ValueError("runtime Gate completion provenance differs")
    if (
        {item["family"] for item in completions} != set(S2_FAMILIES)
        or len(slurm_runtime_hashes) != 1
    ):
        raise ValueError("runtime authorization requires D160, G96, and U128")
    completions.sort(key=lambda item: S2_FAMILIES.index(item["family"]))
    evidence = {
        "code_commit": str(expected_commit).lower(),
        "base_experiment_namespace": precheck["experiment_namespace"],
        "protocol_sha256": precheck["protocol_sha256"],
        "full_model_gate_path": str(Path(full_model_gate_path).resolve()),
        "full_model_gate_file_sha256": sha256_file(full_model_gate_path),
        "full_model_gate_sha256": precheck["full_model_gate_sha256"],
        "training_runtime_precheck_path": str(
            Path(training_runtime_precheck_path).resolve()
        ),
        "training_runtime_precheck_file_sha256": sha256_file(
            training_runtime_precheck_path
        ),
        "training_runtime_precheck_sha256": precheck["precheck_sha256"],
        "runtime_gate_completions": completions,
        "slurm_runtime_sha256": next(iter(slurm_runtime_hashes)),
    }
    evidence_sha256 = canonical_sha256(evidence)
    campaign_namespace = canonical_sha256(
        {
            "base_experiment_namespace": precheck["experiment_namespace"],
            "runtime_authorization_evidence_sha256": evidence_sha256,
        }
    )
    report = {
        "schema_version": S2_RUNTIME_AUTHORIZATION_SCHEMA,
        "status": "PASS",
        **evidence,
        "runtime_authorization_evidence_sha256": evidence_sha256,
        "campaign_namespace": campaign_namespace,
        "canonical_experiment_root": (
            f"{S2_CANONICAL_EXPERIMENTS_ROOT}/"
            f"{precheck['experiment_namespace']}/campaigns/{campaign_namespace}"
        ),
        "families_authorized": list(S2_FAMILIES),
        "training_cells_authorized": [
            {"family": family, "seed": seed}
            for family in S2_FAMILIES
            for seed in S2_TRAINING_SEEDS
        ],
        "official_test_opened": False,
        "formal_training_authorized": True,
        "learned_roi_policy_authorized": False,
        "paper_claim_allowed": False,
    }
    report["authorization_sha256"] = canonical_sha256(report)
    return report


def validate_runtime_authorization(
    path: str | Path,
    *,
    expected_commit: str,
    expected_full_model_gate_sha256: str,
    expected_precheck_sha256: str,
) -> dict[str, Any]:
    path = Path(path).resolve()
    report = _load_json(path)
    expected = report.pop("authorization_sha256", None)
    if not expected or canonical_sha256(report) != expected:
        raise ValueError("runtime authorization self-hash mismatch")
    report["authorization_sha256"] = expected
    completion_cells = {
        (str(cell.get("family", "")), int(cell.get("seed", -1)))
        for cell in report.get("training_cells_authorized", [])
    }
    expected_cells = {
        (family, seed) for family in S2_FAMILIES for seed in S2_TRAINING_SEEDS
    }
    if (
        report.get("schema_version") != S2_RUNTIME_AUTHORIZATION_SCHEMA
        or report.get("status") != "PASS"
        or report.get("code_commit") != str(expected_commit).lower()
        or report.get("full_model_gate_sha256")
        != expected_full_model_gate_sha256
        or report.get("training_runtime_precheck_sha256")
        != expected_precheck_sha256
        or report.get("families_authorized") != list(S2_FAMILIES)
        or completion_cells != expected_cells
        or report.get("official_test_opened") is not False
        or report.get("formal_training_authorized") is not True
        or report.get("learned_roi_policy_authorized") is not False
        or report.get("paper_claim_allowed") is not False
    ):
        raise ValueError("runtime authorization lacks a required invariant")
    evidence = {
        key: report[key]
        for key in (
            "code_commit",
            "base_experiment_namespace",
            "protocol_sha256",
            "full_model_gate_path",
            "full_model_gate_file_sha256",
            "full_model_gate_sha256",
            "training_runtime_precheck_path",
            "training_runtime_precheck_file_sha256",
            "training_runtime_precheck_sha256",
            "runtime_gate_completions",
            "slurm_runtime_sha256",
        )
    }
    if canonical_sha256(evidence) != report.get(
        "runtime_authorization_evidence_sha256"
    ):
        raise ValueError("runtime authorization evidence hash mismatch")
    expected_campaign_namespace = canonical_sha256(
        {
            "base_experiment_namespace": report[
                "base_experiment_namespace"
            ],
            "runtime_authorization_evidence_sha256": report[
                "runtime_authorization_evidence_sha256"
            ],
        }
    )
    expected_root = (
        f"{S2_CANONICAL_EXPERIMENTS_ROOT}/"
        f"{report['base_experiment_namespace']}/campaigns/"
        f"{expected_campaign_namespace}"
    )
    if (
        report.get("campaign_namespace") != expected_campaign_namespace
        or report.get("canonical_experiment_root") != expected_root
        or sha256_file(report["full_model_gate_path"])
        != report["full_model_gate_file_sha256"]
        or sha256_file(report["training_runtime_precheck_path"])
        != report["training_runtime_precheck_file_sha256"]
    ):
        raise ValueError("runtime authorization campaign binding changed")
    for item in report["runtime_gate_completions"]:
        completion_path = Path(item["path"]).resolve()
        completion = _validate_runtime_gate_completion(completion_path)
        if (
            sha256_file(completion_path) != item["file_sha256"]
            or completion["completion_sha256"] != item["completion_sha256"]
            or completion["checkpoint_sha256"] != item["checkpoint_sha256"]
            or completion["checkpoint_sidecar_file_sha256"]
            != item["checkpoint_sidecar_file_sha256"]
        ):
            raise ValueError("runtime authorization completion binding changed")
    return report


__all__ = [
    "S2_RUNTIME_AUTHORIZATION_SCHEMA",
    "S2_RUNTIME_GATE_BINDING_SCHEMA",
    "S2_RUNTIME_GATE_COMPLETION_SCHEMA",
    "S2_RUNTIME_GATE_METADATA_SCHEMA",
    "S2_RUNTIME_GATE_SIDECAR_SCHEMA",
    "bind_runtime_gate_config",
    "build_runtime_authorization",
    "build_runtime_gate_checkpoint_metadata",
    "build_runtime_gate_completion",
    "validate_runtime_authorization",
    "validate_runtime_gate_config",
]
