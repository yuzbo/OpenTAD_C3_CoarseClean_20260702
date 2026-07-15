from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    S1_PRETRAINED_CHECKPOINT_SHA256,
    S1_TRAINING_SEEDS,
    canonical_sha256,
    sha256_file,
    validate_s1_manifest,
)
from tools.bata.validate_spatial_zoom_s1 import (  # noqa: E402
    CONFIG_PATHS,
    validate_config_matrix,
)

S1_TRAINING_BINDING_SCHEMA = "spatial_zoom_s1_training_binding_v4"
S1_CHECKPOINT_METADATA_SCHEMA = "spatial_zoom_s1_checkpoint_metadata_v4"
S1_CHECKPOINT_SIDECAR_SCHEMA = "spatial_zoom_s1_checkpoint_sidecar_v3"
S1_EXPERIMENT_NAMESPACE_SCHEMA = "spatial_zoom_s1_experiment_namespace_v1"
S1_CANONICAL_EXPERIMENTS_ROOT = (
    "/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/"
    "spatial_zoom_s1_canonical"
)
S1_CANONICAL_STUDY_ROOT = f"{S1_CANONICAL_EXPERIMENTS_ROOT}/sealed_study_v1"


def build_s1_experiment_identity(
    *,
    manifest_sha256: str,
    code_commit: str,
    protocol_fingerprint: str,
    precheck_file_sha256: str,
    precheck_sha256: str,
    pretrained_checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": S1_EXPERIMENT_NAMESPACE_SCHEMA,
        "manifest_sha256": str(manifest_sha256),
        "code_commit": str(code_commit),
        "protocol_fingerprint": str(protocol_fingerprint),
        "precheck_file_sha256": str(precheck_file_sha256),
        "precheck_sha256": str(precheck_sha256),
        "pretrained_checkpoint_sha256": str(pretrained_checkpoint_sha256),
    }
    if any(not value for key, value in payload.items() if key != "schema_version"):
        raise ValueError("formal S1 experiment identity is incomplete")
    namespace = canonical_sha256(payload)
    return {
        **payload,
        "experiment_namespace": namespace,
        "canonical_experiment_root": f"{S1_CANONICAL_EXPERIMENTS_ROOT}/{namespace}",
    }


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip().lower()
    if (
        completed.returncode != 0
        or len(commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError("formal S1 requires a concrete Git commit")
    return commit


def resolve_s1_formal_experiment_identity(
    *,
    manifest_path: str | Path,
    annotation_path: str | Path,
    precheck_path: str | Path,
) -> dict[str, Any]:
    from tools.bata.run_spatial_zoom_s1_precheck import (
        validate_precheck_certificate,
    )

    manifest_path = Path(manifest_path).resolve()
    annotation_path = Path(annotation_path).resolve()
    precheck_path = Path(precheck_path).resolve()
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=annotation_path,
    )
    precheck = validate_precheck_certificate(
        json.loads(precheck_path.read_text(encoding="utf-8")), require_full=True
    )
    if (
        precheck["expected_pretrained_checkpoint_sha256"]
        != manifest["pretrained_checkpoint"]["sha256"]
    ):
        raise ValueError(
            "S1 manifest and full precheck disagree on the pretrained checkpoint"
        )
    matrix = validate_config_matrix()
    return build_s1_experiment_identity(
        manifest_sha256=manifest["manifest_sha256"],
        code_commit=current_git_commit(),
        protocol_fingerprint=matrix["protocol_fingerprint"],
        precheck_file_sha256=sha256_file(precheck_path),
        precheck_sha256=precheck["precheck_sha256"],
        pretrained_checkpoint_sha256=manifest["pretrained_checkpoint"]["sha256"],
    )


def require_clean_git_checkout(*, expected_commit: str) -> None:
    if current_git_commit() != str(expected_commit).lower():
        raise RuntimeError("formal S1 source checkout differs from the bound commit")
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip():
        raise RuntimeError("formal S1 execution requires a clean Git checkout")


def eligible_s1_eval_epochs(workflow: Mapping[str, Any]) -> tuple[int, ...]:
    start = int(workflow.get("val_start_epoch", 0))
    end = int(workflow["end_epoch"])
    interval = int(workflow["val_eval_interval"])
    if interval <= 0 or end <= start:
        raise ValueError(
            "S1 workflow must expose gate evaluations before training ends"
        )
    explicit = set(int(value) for value in workflow.get("val_eval_epochs", ()))
    anchor = workflow.get("val_eval_interval_anchor_epoch")
    eligible = []
    for epoch in range(start, end):
        one_based = epoch + 1
        if anchor is None and not explicit:
            should_eval = one_based % interval == 0
        else:
            anchor_value = int(anchor if anchor is not None else start)
            should_eval = one_based in explicit or (
                one_based >= anchor_value and (one_based - anchor_value) % interval == 0
            )
        if should_eval:
            eligible.append(epoch)
    checkpoint_interval = int(workflow["checkpoint_interval"])
    if any(
        not (epoch == end - 1 or (epoch + 1) % checkpoint_interval == 0)
        for epoch in eligible
    ):
        raise ValueError("every S1 gate-evaluation epoch must have a saved checkpoint")
    if not eligible:
        raise ValueError("S1 workflow has no eligible gate checkpoints")
    return tuple(eligible)


def _source_config_path_for_resolution(resolution: int) -> Path:
    try:
        return (ROOT / CONFIG_PATHS[int(resolution)]).resolve()
    except KeyError as exc:
        raise ValueError(f"unsupported S1 resolution {resolution}") from exc


def bind_s1_training_config(
    *,
    source_config_path: str | Path,
    manifest_path: str | Path,
    annotation_path: str | Path,
    seed: int,
    work_dir: str | Path,
    precheck_path: str | Path | None = None,
) -> Config:
    matrix = validate_config_matrix()
    code_commit = current_git_commit()
    source_config_path = Path(source_config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    annotation_path = Path(annotation_path).resolve()
    seed = int(seed)
    if seed not in S1_TRAINING_SEEDS:
        raise ValueError("S1 training seed is outside the frozen schema")
    source = Config.fromfile(str(source_config_path))
    resolution = int(source.spatial_zoom_s1_contract.runtime_resolution)
    if source_config_path != _source_config_path_for_resolution(resolution):
        raise ValueError("S1 training must bind one of the audited source configs")
    if source.spatial_zoom_s1_contract.get("runtime_bound", False):
        raise ValueError("S1 source config must be unbound")
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=annotation_path,
    )
    precheck = None
    resolved_precheck_path = None
    if precheck_path is not None:
        from tools.bata.run_spatial_zoom_s1_precheck import (
            validate_precheck_certificate,
        )

        resolved_precheck_path = Path(precheck_path).resolve()
        if not resolved_precheck_path.is_file():
            raise FileNotFoundError(resolved_precheck_path)
        precheck = validate_precheck_certificate(
            json.loads(resolved_precheck_path.read_text(encoding="utf-8")),
            require_full=True,
        )
        if (
            precheck["expected_pretrained_checkpoint_sha256"]
            != manifest["pretrained_checkpoint"]["sha256"]
        ):
            raise ValueError(
                "S1 manifest and full precheck disagree on the pretrained checkpoint"
            )
    if manifest["pretrained_checkpoint"]["sha256"] != S1_PRETRAINED_CHECKPOINT_SHA256:
        raise ValueError(
            "S1 manifest does not use the repository-frozen pretrained checkpoint"
        )
    experiment_identity = None
    resolved_work_dir = Path(work_dir).resolve()
    if precheck is not None:
        experiment_identity = build_s1_experiment_identity(
            manifest_sha256=manifest["manifest_sha256"],
            code_commit=code_commit,
            protocol_fingerprint=matrix["protocol_fingerprint"],
            precheck_file_sha256=sha256_file(resolved_precheck_path),
            precheck_sha256=precheck["precheck_sha256"],
            pretrained_checkpoint_sha256=manifest["pretrained_checkpoint"]["sha256"],
        )
        expected_work_dir = (
            Path(experiment_identity["canonical_experiment_root"])
            / f"dense{resolution}"
            / f"seed{seed}"
        ).resolve()
        if resolved_work_dir != expected_work_dir:
            raise ValueError(
                "formal S1 work_dir must use the unique canonical experiment "
                f"namespace: {expected_work_dir}"
            )
    cfg = copy.deepcopy(source)
    cfg.dataset.train.subset_name = manifest["annotation_subsets"]["development"]
    cfg.dataset.train.block_list = list(manifest["splits"]["gate"])
    for split_name in ("val", "test"):
        split_cfg = cfg.dataset[split_name]
        split_cfg.subset_name = manifest["annotation_subsets"]["development"]
        split_cfg.block_list = list(manifest["splits"]["fit"])
    cfg.evaluation.subset = manifest["annotation_subsets"]["development"]
    cfg.post_processing.save_dict = False
    cfg.work_dir = str(resolved_work_dir)
    binding = {
        "schema_version": S1_TRAINING_BINDING_SCHEMA,
        "runtime_bound": True,
        "resolution": resolution,
        "seed": seed,
        "source_config_path": str(source_config_path),
        "source_config_sha256": canonical_sha256(source.to_dict()),
        "code_commit": code_commit,
        "protocol_fingerprint": matrix["protocol_fingerprint"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "annotation_path": str(annotation_path),
        "annotation_sha256": manifest["annotation_sha256"],
        "fit_split_hash": manifest["split_hashes"]["fit"],
        "gate_split_hash": manifest["split_hashes"]["gate"],
        "fit_video_ids": list(manifest["splits"]["fit"]),
        "gate_video_ids": list(manifest["splits"]["gate"]),
        "eligible_checkpoint_epochs": list(eligible_s1_eval_epochs(cfg.workflow)),
        "formal_precheck_verified": precheck is not None,
        "precheck_path": None
        if resolved_precheck_path is None
        else str(resolved_precheck_path),
        "precheck_file_sha256": None
        if resolved_precheck_path is None
        else sha256_file(resolved_precheck_path),
        "precheck_sha256": None if precheck is None else precheck["precheck_sha256"],
        "pretrained_checkpoint_sha256": manifest["pretrained_checkpoint"]["sha256"],
        "experiment_namespace": None
        if experiment_identity is None
        else experiment_identity["experiment_namespace"],
        "canonical_experiment_root": None
        if experiment_identity is None
        else experiment_identity["canonical_experiment_root"],
        "work_dir": cfg.work_dir,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    cfg.spatial_zoom_s1_runtime_binding = binding
    cfg.spatial_zoom_s1_contract.runtime_bound = True
    return cfg


def validate_bound_s1_training_config(cfg: Config, *, seed: int) -> dict[str, Any]:
    if "spatial_zoom_s1_contract" not in cfg:
        raise ValueError("not an S1 config")
    binding = cfg.get("spatial_zoom_s1_runtime_binding")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "S1 source configs are not directly trainable; materialize a frozen "
            "manifest-bound config first"
        )
    if binding.get("schema_version") != S1_TRAINING_BINDING_SCHEMA:
        raise ValueError("unsupported S1 training binding schema")
    if not bool(cfg.spatial_zoom_s1_contract.get("runtime_bound", False)):
        raise ValueError("S1 config is not runtime-bound")
    expected = bind_s1_training_config(
        source_config_path=binding["source_config_path"],
        manifest_path=binding["manifest_path"],
        annotation_path=binding["annotation_path"],
        seed=int(seed),
        work_dir=binding["work_dir"],
        precheck_path=binding.get("precheck_path"),
    )
    if canonical_sha256(cfg.to_dict()) != canonical_sha256(expected.to_dict()):
        raise ValueError("S1 materialized config was modified after manifest binding")
    return dict(binding)


def build_s1_checkpoint_metadata(
    cfg: Config,
    *,
    seed: int,
    epoch: int,
    successful_updates: int,
    train_batches_per_epoch: int,
) -> dict[str, Any]:
    binding = validate_bound_s1_training_config(cfg, seed=int(seed))
    expected_updates = (int(epoch) + 1) * int(train_batches_per_epoch)
    if int(successful_updates) != expected_updates:
        raise RuntimeError(
            f"S1 checkpoint epoch {epoch} has {successful_updates} successful "
            f"updates, expected {expected_updates}"
        )
    metadata = {
        "schema_version": S1_CHECKPOINT_METADATA_SCHEMA,
        "resolution": int(binding["resolution"]),
        "seed": int(seed),
        "epoch": int(epoch),
        "successful_updates": int(successful_updates),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "source_config_sha256": binding["source_config_sha256"],
        "code_commit": binding["code_commit"],
        "formal_precheck_verified": bool(binding["formal_precheck_verified"]),
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "manifest_sha256": binding["manifest_sha256"],
        "annotation_sha256": binding["annotation_sha256"],
        "fit_split_hash": binding["fit_split_hash"],
        "gate_split_hash": binding["gate_split_hash"],
        "official_test_opened": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return metadata


def checkpoint_sidecar_path(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".metadata.json")


def validate_s1_checkpoint_sidecar(
    checkpoint_path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    sidecar_path = checkpoint_sidecar_path(checkpoint_path)
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            sidecar_path if checkpoint_path.is_file() else checkpoint_path
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar_hash = sidecar.pop("sidecar_sha256", None)
    if not sidecar_hash or canonical_sha256(sidecar) != sidecar_hash:
        raise ValueError("S1 checkpoint sidecar hash mismatch")
    sidecar["sidecar_sha256"] = sidecar_hash
    if sidecar.get("schema_version") != S1_CHECKPOINT_SIDECAR_SCHEMA:
        raise ValueError("unsupported S1 checkpoint sidecar schema")
    if sidecar.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("S1 checkpoint file hash does not match its sidecar")
    metadata = sidecar.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("S1 checkpoint sidecar has no experiment metadata")
    if metadata.get("schema_version") != S1_CHECKPOINT_METADATA_SCHEMA:
        raise ValueError("unsupported S1 checkpoint experiment metadata schema")
    if metadata.get("pretrained_checkpoint_sha256") != S1_PRETRAINED_CHECKPOINT_SHA256:
        raise ValueError(
            "S1 checkpoint metadata does not use the repository-frozen "
            "pretrained checkpoint"
        )
    metadata_copy = dict(metadata)
    metadata_hash = metadata_copy.pop("metadata_sha256", None)
    if not metadata_hash or canonical_sha256(metadata_copy) != metadata_hash:
        raise ValueError("S1 checkpoint experiment metadata hash mismatch")
    if expected_metadata is not None and dict(metadata) != dict(expected_metadata):
        raise ValueError("S1 checkpoint metadata does not match the active run")
    return sidecar


def require_slurm_single_gpu_allocation() -> str:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal S1 execution requires an allocated Slurm job")
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible) != 1:
        raise RuntimeError(
            "formal S1 execution requires exactly one Slurm-visible CUDA device"
        )
    gpu_count = os.environ.get("SLURM_GPUS_ON_NODE", "").strip()
    if gpu_count and gpu_count != "1":
        raise RuntimeError("formal S1 execution requires a one-GPU Slurm allocation")
    physical_ids = [
        value.strip()
        for value in os.environ.get("SLURM_JOB_GPUS", "").split(",")
        if value.strip()
    ]
    if len(physical_ids) != 1:
        raise RuntimeError(
            "formal S1 execution requires one auditable SLURM_JOB_GPUS identity"
        )
    return physical_ids[0]


__all__ = [
    "S1_CANONICAL_EXPERIMENTS_ROOT",
    "S1_CANONICAL_STUDY_ROOT",
    "S1_CHECKPOINT_METADATA_SCHEMA",
    "S1_CHECKPOINT_SIDECAR_SCHEMA",
    "S1_TRAINING_BINDING_SCHEMA",
    "bind_s1_training_config",
    "build_s1_experiment_identity",
    "build_s1_checkpoint_metadata",
    "checkpoint_sidecar_path",
    "current_git_commit",
    "eligible_s1_eval_epochs",
    "require_slurm_single_gpu_allocation",
    "require_clean_git_checkout",
    "resolve_s1_formal_experiment_identity",
    "validate_bound_s1_training_config",
    "validate_s1_checkpoint_sidecar",
]
