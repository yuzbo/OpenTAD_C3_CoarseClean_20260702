from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from tools.bata import duca_p0_training


FORMAL_PROTOCOLS = {
    "duca_rime_phase2_mixed_k_baseline_v1",
    "duca_rime_uniform_control_v1",
    "duca_rime_physical_dynamic_k_v1",
}
TRAIN_ARMS = {
    "U-mixed-K",
    "U-fixed",
    "F-bound",
    "D-shuffle",
    "D-no-risk",
    "AdapTok-TAD",
    "RIME-full",
    "RIME-full-TriDet",
    "U-fixed-TriDet",
}
DUCA_TRAINING_AUDIT_FILENAME = "duca_rime_training_audit.json"
DUCA_P0_CHECKPOINT_METADATA_SCHEMA = duca_p0_training.DUCA_P0_CHECKPOINT_METADATA_SCHEMA
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = duca_p0_training.DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA
PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE = (
    "historical_uniform_score_net_unused_exact_whitelist_v1"
)
PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS = (
    "module.frame_selector.score_net.0.bias",
    "module.frame_selector.score_net.0.weight",
    "module.frame_selector.score_net.2.bias",
    "module.frame_selector.score_net.2.weight",
    "module.frame_selector.score_net.4.bias",
    "module.frame_selector.score_net.4.weight",
)

atomic_write_json = duca_p0_training.atomic_write_json
build_checkpoint_metadata = duca_p0_training.build_checkpoint_metadata
build_training_audit = duca_p0_training.build_training_audit
capture_global_rng_state = duca_p0_training.capture_global_rng_state
new_update_audit = duca_p0_training.new_update_audit
restore_global_rng_state = duca_p0_training.restore_global_rng_state
restore_training_state = duca_p0_training.restore_training_state
selector_schedule_step = duca_p0_training.selector_schedule_step
validate_update_state = duca_p0_training.validate_update_state


def is_formal_protocol(value: str) -> bool:
    return str(value) in FORMAL_PROTOCOLS


def validate_phase2_baseline_checkpoint_compatibility(
    *,
    missing_keys,
    unexpected_keys,
) -> dict[str, Any]:
    missing = sorted(str(key) for key in missing_keys)
    unexpected = sorted(str(key) for key in unexpected_keys)
    expected_unexpected = sorted(PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS)
    if missing:
        raise RuntimeError(
            "Phase-2 baseline checkpoint is missing current-model parameters: "
            + ", ".join(missing)
        )
    if unexpected != expected_unexpected:
        raise RuntimeError(
            "Phase-2 baseline checkpoint compatibility differs from the exact "
            "historical uniform-selector whitelist"
        )
    return {
        "mode": PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE,
        "missing_keys": [],
        "ignored_unexpected_keys": expected_unexpected,
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _bound_file(path: str | Path | None, expected: str | None, label: str) -> tuple[str, str]:
    if not path:
        raise ValueError(f"RIME {label} path is missing")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = _sha256_file(resolved)
    if not expected or actual != str(expected).lower():
        raise ValueError(f"RIME {label} SHA-256 drift")
    return str(resolved), actual


def _optional_bound_file(
    path: str | Path | None,
    expected: str | None,
    label: str,
) -> tuple[str | None, str | None]:
    if not path and not expected:
        return None, None
    return _bound_file(path, expected, label)


def _ordered_training_video_ids(train_dataset) -> tuple[str, ...]:
    rows = getattr(train_dataset, "data_list", None)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("RIME training dataset has no auditable data_list")
    video_ids = tuple(str(row[0]) for row in rows)
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("RIME training dataset contains duplicate video identities")
    return video_ids


def derive_train_loader_contract(
    *,
    cfg,
    train_dataset,
    train_loader,
    world_size: int,
) -> dict[str, Any]:
    if train_dataset.__class__.__name__ != "DucaStatelessThumosPaddingDataset":
        raise RuntimeError("RIME formal training requires the stateless THUMOS dataset")
    if int(train_dataset.stateless_seed) != 3407 or int(world_size) != 1:
        raise RuntimeError("RIME exposure is frozen to stateless seed 3407 and one process")
    video_ids = _ordered_training_video_ids(train_dataset)
    if (
        len(video_ids) != 100
        or int(len(train_loader)) != 100
        or int(cfg.solver.train.batch_size) != 1
    ):
        raise RuntimeError("RIME exposure requires 100 unique videos and 100 batches/epoch")
    sampler = getattr(train_loader, "sampler", None)
    if (
        sampler is None
        or sampler.__class__.__name__ != "DistributedSampler"
        or not hasattr(sampler, "seed")
        or not callable(getattr(sampler, "set_epoch", None))
        or int(getattr(sampler, "num_replicas", -1)) != 1
        or int(getattr(sampler, "rank", -1)) != 0
        or getattr(sampler, "shuffle", None) is not True
        or getattr(sampler, "drop_last", None) is not True
    ):
        raise RuntimeError(
            "RIME exposure requires a seeded rank-0 one-replica shuffled "
            "drop-last DistributedSampler"
        )
    sampler_seed = int(sampler.seed)
    original_epoch = int(getattr(sampler, "epoch", 0))
    scheduled_video_ids = []
    epoch_index_sha256 = []
    try:
        for epoch in range(60):
            sampler.set_epoch(epoch)
            indices = [int(index) for index in iter(sampler)]
            if (
                len(indices) != len(video_ids)
                or len(set(indices)) != len(video_ids)
                or any(index < 0 or index >= len(video_ids) for index in indices)
            ):
                raise RuntimeError(
                    "RIME one-process sampler must expose every training video "
                    "exactly once per epoch"
                )
            epoch_index_sha256.append(_canonical_sha256(indices))
            scheduled_video_ids.extend(video_ids[index] for index in indices)
    finally:
        sampler.set_epoch(original_epoch)
    if len(scheduled_video_ids) != 6000:
        raise RuntimeError("RIME exposure schedule must contain exactly 6000 updates")
    counts = {video_id: scheduled_video_ids.count(video_id) for video_id in video_ids}
    if set(counts.values()) != {60}:
        raise RuntimeError("RIME exposure must visit every training video exactly 60 times")
    payload = {
        "schema_version": "duca_rime_train_loader_contract_v1",
        "dataset_class": train_dataset.__class__.__name__,
        "ordered_video_ids": list(video_ids),
        "ordered_video_ids_sha256": _canonical_sha256(video_ids),
        "video_count": len(video_ids),
        "batches_per_epoch": int(len(train_loader)),
        "epoch_count": 60,
        "successful_detector_updates": len(scheduled_video_ids),
        "per_video_exposure_count": 60,
        "sampler_class": train_loader.sampler.__class__.__name__,
        "sampler_seed": sampler_seed,
        "sampler_rank": int(sampler.rank),
        "sampler_num_replicas": int(sampler.num_replicas),
        "sampler_drop_last": bool(sampler.drop_last),
        "shuffle": bool(sampler.shuffle),
        "drop_last": bool(sampler.drop_last),
        "world_size": int(world_size),
        "batch_size": int(cfg.solver.train.batch_size),
        "stateless_augmentation_seed": int(train_dataset.stateless_seed),
        "epoch_index_sha256": epoch_index_sha256,
        "scheduled_video_ids_sha256": _canonical_sha256(scheduled_video_ids),
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def bind_train_loader_contract(
    contract: Mapping[str, Any],
    *,
    cfg,
    train_dataset,
    train_loader,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exposure_path, _exposure_sha = _bound_file(
        os.environ.get("DUCA_RIME_TRAINING_EXPOSURE_JSON"),
        os.environ.get("DUCA_RIME_TRAINING_EXPOSURE_SHA256"),
        "training exposure",
    )
    exposure = json.loads(Path(exposure_path).read_text(encoding="utf-8"))
    actual = derive_train_loader_contract(
        cfg=cfg,
        train_dataset=train_dataset,
        train_loader=train_loader,
        world_size=world_size,
    )
    if exposure.get("train_loader_contract") != actual:
        raise RuntimeError("runtime RIME video exposure differs from the sealed schedule")
    bound = dict(contract)
    bound["train_loader_contract"] = actual
    return bound, actual


def formal_training_contract(cfg) -> dict[str, Any] | None:
    formal_protocol = str(cfg.workflow.get("formal_protocol", ""))
    if not is_formal_protocol(formal_protocol):
        return None
    contract = duca_p0_training.formal_training_contract(cfg)
    if contract is None:
        raise ValueError("RIME formal training contract was not activated")
    arm = str(cfg.duca_rime_variant.arm)
    if arm not in TRAIN_ARMS:
        raise ValueError(f"RIME config has an unregistered train arm: {arm}")
    if (
        int(cfg.solver.train.batch_size) != 1
        or int(contract["expected_successful_optimizer_updates"]) != 6000
        or int(contract["end_epoch"]) * int(contract["expected_train_batches_per_epoch"])
        != 6000
    ):
        raise ValueError("RIME formal training requires batch_size=1 and exactly 6000 updates")
    if arm == "U-mixed-K":
        variant = cfg.duca_rime_variant
        if (
            formal_protocol != "duca_rime_phase2_mixed_k_baseline_v1"
            or tuple(int(value) for value in variant.candidate_budgets)
            != (192, 256, 384, 512)
            or tuple(int(value) for value in variant.training_schedule_counts)
            != (8, 12, 16, 24)
            or float(variant.training_target_mean_cost) != 384.0
            or variant.position_policy != "exact_uniform"
            or variant.coarse_probe_executed is not False
            or cfg.model.frame_selector.rime_arm != "uniform_mixed_k"
            or cfg.model.frame_selector.actionness_source_cfg is not None
        ):
            raise ValueError("U-mixed-K formal schedule/config contract drift")
    if arm not in {"U-fixed", "U-fixed-TriDet"}:
        rime_contract = cfg.duca_rime_contract
        if (
            rime_contract.task != "offline_temporal_action_detection"
            or rime_contract.online_tad is not False
            or rime_contract.pad_to_kmax is not False
            or int(rime_contract.execution_quantum) != 16
            or rime_contract.official_final_subset_consumed is not False
        ):
            raise ValueError("RIME dynamic exact-K/physical-time contract drift")
    contract = dict(contract)
    contract["formal_protocol"] = formal_protocol
    contract["rime_arm"] = arm
    contract["checkpoint_retention"] = int(
        cfg.workflow.get("checkpoint_retention", 0)
    )
    if contract["checkpoint_retention"] != 1:
        raise ValueError(
            "RIME formal training must retain exactly one resumable checkpoint"
        )
    return contract


def after_checkpoint_saved(
    *,
    checkpoint_path: str | Path,
    work_dir: str | Path,
    epoch: int,
    contract: Mapping[str, Any],
) -> list[str]:
    if int(contract.get("checkpoint_retention", 0)) != 1:
        raise RuntimeError("RIME checkpoint retention contract is not active")
    checkpoint_dir = (Path(work_dir).expanduser().resolve() / "checkpoint").resolve()
    current = Path(checkpoint_path).expanduser().resolve()
    if (
        current.parent != checkpoint_dir
        or current.name != f"epoch_{int(epoch)}.pth"
        or not current.is_file()
        or current.is_symlink()
    ):
        raise RuntimeError("RIME checkpoint pruning received an unsafe current path")
    removed = []
    for candidate in sorted(checkpoint_dir.glob("epoch_*.pth")):
        resolved = candidate.resolve()
        match = re.fullmatch(r"epoch_([0-9]+)\.pth", candidate.name)
        if (
            resolved == current
            or resolved.parent != checkpoint_dir
            or candidate.is_symlink()
            or match is None
        ):
            continue
        candidate_epoch = int(match.group(1))
        if candidate_epoch >= int(epoch):
            raise RuntimeError("RIME checkpoint directory contains a future checkpoint")
        candidate.unlink()
        removed.append(str(resolved))
    remaining = [
        path.resolve()
        for path in checkpoint_dir.glob("epoch_*.pth")
        if path.is_file()
    ]
    if remaining != [current]:
        raise RuntimeError("RIME rolling checkpoint retention failed closed")
    return removed


def _training_stage_authorization(
    *,
    git_commit: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    authorization_value = os.environ.get("DUCA_RIME_PHASE4_AUTHORIZATION")
    authorization_sha_value = os.environ.get("DUCA_RIME_PHASE4_AUTHORIZATION_SHA256")
    if not authorization_value and not authorization_sha_value:
        if int(seed) != 3407:
            raise ValueError("Phase-3 development training seed is frozen to 3407")
        if variant in {"RIME-full-TriDet", "U-fixed-TriDet"}:
            raise ValueError("TriDet training is reserved for authorized Phase-4 replication")
        return {
            "research_phase": 3,
            "phase4_authorization_path": None,
            "phase4_authorization_sha256": None,
            "formal_budget_panel": None,
            "detector_backend": "ActionFormer",
        }

    authorization_path, authorization_sha = _bound_file(
        authorization_value,
        authorization_sha_value,
        "Phase-4 authorization",
    )
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    detector_backend = (
        "TriDet"
        if variant in {"RIME-full-TriDet", "U-fixed-TriDet"}
        else "ActionFormer"
    )
    target_mean_cost = float(os.environ.get("DUCA_RIME_TARGET_MEAN_COST", "nan"))
    if (
        authorization.get("schema_version") != "duca_rime_stage_receipt_v1"
        or authorization.get("phase") != "phase4_authorization"
        or authorization.get("status") != "authorized"
        or authorization.get("gate_pass") is not True
        or authorization.get("git_commit") != str(git_commit)
        or authorization.get("official_final_subset_consumed") is not False
        or int(seed) not in {int(value) for value in authorization.get("formal_seeds", ())}
        or detector_backend not in authorization.get("required_detectors", ())
        or target_mean_cost not in {
            float(value) for value in authorization.get("required_budget_panels", ())
        }
    ):
        raise ValueError("Phase-4 authorization does not cover this training cell")
    return {
        "research_phase": 4,
        "phase4_authorization_path": authorization_path,
        "phase4_authorization_sha256": authorization_sha,
        "formal_budget_panel": target_mean_cost,
        "detector_backend": detector_backend,
    }


def build_runtime_bindings(
    *,
    git_commit: str,
    variant: str,
    seed: int,
    slurm_job_id: str | None,
    source_config_path: str | Path,
    source_config_sha256: str,
    resolved_config_sha256: str,
    runtime_config_sha256: str,
    evaluation_annotation_path: str | Path,
    evaluation_class_map_path: str | Path,
    evaluation_config: Mapping[str, Any],
    runtime_pretrain_path: str | Path,
) -> dict[str, Any]:
    from tools.bata.duca_p0_evaluation import evaluation_config_sha256

    if variant not in TRAIN_ARMS:
        raise ValueError(f"invalid RIME train arm: {variant}")
    if re.fullmatch(r"[0-9a-f]{40}", str(git_commit)) is None:
        raise ValueError("RIME runtime binding requires an exact Git commit")
    phase2_mixed_k_baseline = variant == "U-mixed-K"
    if phase2_mixed_k_baseline:
        if int(seed) != 3407 or any(
            value
            for value in (
                os.environ.get("DUCA_RIME_PHASE4_AUTHORIZATION"),
                os.environ.get("DUCA_RIME_PHASE4_AUTHORIZATION_SHA256"),
            )
        ):
            raise ValueError(
                "U-mixed-K Phase-2 training is frozen to seed 3407 and no "
                "Phase-4 authorization"
            )
        stage = {
            "research_phase": 2,
            "phase4_authorization_path": None,
            "phase4_authorization_sha256": None,
            "formal_budget_panel": 384.0,
            "detector_backend": "ActionFormer",
        }
    else:
        stage = _training_stage_authorization(
            git_commit=str(git_commit),
            variant=str(variant),
            seed=int(seed),
        )
    expected_resolved = os.environ.get("DUCA_RESOLVED_CONFIG_SHA256")
    if expected_resolved != str(resolved_config_sha256):
        raise ValueError("RIME resolved config differs from the sealed launch")

    if phase2_mixed_k_baseline:
        phase1_path, phase1_sha = _bound_file(
            os.environ.get("DUCA_RIME_PHASE1_RECEIPT"),
            os.environ.get("DUCA_RIME_PHASE1_RECEIPT_SHA256"),
            "Phase-1 receipt",
        )
        phase1 = json.loads(Path(phase1_path).read_text(encoding="utf-8"))
        if (
            phase1.get("schema_version") != "duca_rime_stage_receipt_v1"
            or phase1.get("phase") != "phase1"
            or phase1.get("gate_pass") is not True
            or phase1.get("official_final_subset_consumed") is not False
            or phase1.get("git_commit") != str(git_commit)
            or len(str(phase1.get("split_assignment_sha256", ""))) != 64
        ):
            raise ValueError(
                "Phase-1 receipt does not authorize mixed-K baseline training"
            )
        authorization_receipt = phase1
        phase2 = {}
        phase2_path = None
        phase2_sha = None
    else:
        phase2_path, phase2_sha = _bound_file(
            os.environ.get("DUCA_RIME_PHASE2_RECEIPT"),
            os.environ.get("DUCA_RIME_PHASE2_RECEIPT_SHA256"),
            "Phase-2 receipt",
        )
        phase2 = json.loads(Path(phase2_path).read_text(encoding="utf-8"))
        if (
            phase2.get("schema_version") != "duca_rime_stage_receipt_v1"
            or phase2.get("phase") != "phase2"
            or phase2.get("gate_pass") is not True
            or phase2.get("phase3_training_authorized") is not True
            or phase2.get("official_final_subset_consumed") is not False
            or phase2.get("git_commit") != str(git_commit)
        ):
            raise ValueError("Phase-2 receipt does not authorize RIME training")
        authorization_receipt = phase2
        phase1_path = None
        phase1_sha = None
    exposure_path, exposure_sha = _bound_file(
        os.environ.get("DUCA_RIME_TRAINING_EXPOSURE_JSON"),
        os.environ.get("DUCA_RIME_TRAINING_EXPOSURE_SHA256"),
        "training exposure",
    )
    exposure = json.loads(Path(exposure_path).read_text(encoding="utf-8"))
    expected_exposure_schema = {
        2: "duca_rime_phase2_mixed_k_training_exposure_v1",
        3: "duca_rime_phase3_training_exposure_v1",
        4: "duca_rime_phase4_training_exposure_v1",
    }[int(stage["research_phase"])]
    if (
        exposure.get("schema_version") != expected_exposure_schema
        or int(exposure.get("successful_detector_updates", -1)) != 6000
        or exposure.get("split_assignment_sha256")
        != authorization_receipt.get("split_assignment_sha256")
        or exposure.get("official_final_subset_consumed") is not False
    ):
        raise ValueError("RIME training exposure artifact is invalid")
    if stage["research_phase"] == 2 and (
        int(exposure.get("seed", -1)) != 3407
        or float(exposure.get("target_mean_cost", math.nan)) != 384.0
        or exposure.get("detector_backend") != "ActionFormer"
    ):
        raise ValueError(
            "Phase-2 mixed-K training exposure does not match its frozen cell"
        )
    if stage["research_phase"] == 4 and (
        int(exposure.get("seed", -1)) != int(seed)
        or float(exposure.get("target_mean_cost", math.nan))
        != float(stage["formal_budget_panel"])
        or exposure.get("detector_backend") != stage["detector_backend"]
    ):
        raise ValueError("Phase-4 training exposure does not match the formal cell")
    pretrain_path, pretrain_sha = _bound_file(
        runtime_pretrain_path,
        os.environ.get("DUCA_RIME_PRETRAIN_SHA256"),
        "VideoMAE initialization",
    )
    annotation_path, annotation_sha = _bound_file(
        evaluation_annotation_path,
        os.environ.get("DUCA_RIME_EVALUATION_ANNOTATION_SHA256"),
        "evaluation annotation",
    )
    class_map_path, class_map_sha = _bound_file(
        evaluation_class_map_path,
        os.environ.get("DUCA_RIME_EVALUATION_CLASS_MAP_SHA256"),
        "evaluation class map",
    )
    targets_path, targets_sha = _optional_bound_file(
        os.environ.get("DUCA_RIME_TARGETS_JSONL"),
        os.environ.get("DUCA_RIME_TARGETS_SHA256"),
        "cross-fitted targets",
    )
    protocol_path, protocol_sha = _optional_bound_file(
        os.environ.get("DUCA_RIME_BUDGET_PROTOCOL_JSON"),
        os.environ.get("DUCA_RIME_BUDGET_PROTOCOL_SHA256"),
        "budget protocol",
    )
    replay_path, replay_sha = _optional_bound_file(
        os.environ.get("DUCA_RIME_REPLAY_JSONL"),
        os.environ.get("DUCA_RIME_REPLAY_SHA256"),
        "budget replay",
    )
    fixed_control = variant in {"U-mixed-K", "U-fixed", "U-fixed-TriDet"}
    if not fixed_control and (
        targets_path is None or protocol_path is None
    ):
        raise ValueError("trainable RIME selector arms require targets and frozen protocol")
    if fixed_control and any(
        value is not None
        for value in (
            targets_path,
            protocol_path,
            replay_path,
        )
    ):
        raise ValueError(
            "uniform controls must not consume RIME targets/protocol/replay"
        )
    if variant in {"D-shuffle", "AdapTok-TAD"} and replay_path is None:
        raise ValueError(f"{variant} requires its immutable budget replay")
    if variant not in {"D-shuffle", "AdapTok-TAD"} and replay_path is not None:
        raise ValueError(f"{variant} must not consume a budget replay during training")
    if protocol_path is not None:
        expected_target = (
            384.0
            if int(stage["research_phase"]) == 3
            else float(stage["formal_budget_panel"])
        )
        registered_protocols = phase2.get("formal_budget_protocols")
        if not isinstance(registered_protocols, list):
            raise ValueError("Phase-2 receipt lacks formal budget protocols")
        matches = [
            row
            for row in registered_protocols
            if isinstance(row, Mapping)
            and float(row.get("target_mean_cost", math.nan)) == expected_target
        ]
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        if (
            len(matches) != 1
            or str(Path(matches[0].get("path", "")).resolve()) != protocol_path
            or matches[0].get("sha256") != protocol_sha
            or protocol.get("schema_version") != "duca_rime_budget_protocol_v1"
            or protocol.get("fit_split") != "train_only"
            or protocol.get("uses_validation_or_test_labels") is not False
            or float(protocol.get("target_mean_cost", math.nan)) != expected_target
        ):
            raise ValueError(
                "RIME training cell is not bound to its registered budget protocol"
            )
    actual_eval_hash = evaluation_config_sha256(
        evaluation_config,
        expected_subset=str(evaluation_config["subset"]),
    )
    if actual_eval_hash != os.environ.get("DUCA_RIME_EVALUATION_CONFIG_SHA256"):
        raise ValueError("RIME evaluation config differs from the sealed launch")

    bindings = {
        "git_commit": str(git_commit),
        "variant": str(variant),
        "seed": int(seed),
        "slurm_job_id": None if slurm_job_id is None else str(slurm_job_id),
        "source_config_path": str(Path(source_config_path).resolve()),
        "source_config_sha256": str(source_config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "runtime_config_sha256": str(runtime_config_sha256),
        "phase1_receipt_path": phase1_path,
        "phase1_receipt_sha256": phase1_sha,
        "phase2_receipt_path": phase2_path,
        "phase2_receipt_sha256": phase2_sha,
        "split_assignment_sha256": str(
            authorization_receipt["split_assignment_sha256"]
        ),
        "training_exposure_path": exposure_path,
        "training_exposure_sha256": exposure_sha,
        "initialization_path": pretrain_path,
        "initialization_sha256": pretrain_sha,
        "targets_path": targets_path,
        "targets_sha256": targets_sha,
        "budget_protocol_path": protocol_path,
        "budget_protocol_sha256": protocol_sha,
        "budget_replay_path": replay_path,
        "budget_replay_sha256": replay_sha,
        "evaluation_annotation_path": annotation_path,
        "evaluation_annotation_sha256": annotation_sha,
        "evaluation_class_map_path": class_map_path,
        "evaluation_class_map_sha256": class_map_sha,
        "evaluation_config_sha256": actual_eval_hash,
        "official_final_subset_consumed": False,
        **stage,
    }
    bindings["binding_sha256"] = _canonical_sha256(bindings)
    return bindings


def validate_terminal_checkpoint_binding(
    *,
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    git_commit: str,
    evaluation_arm: str,
    seed: int,
    training_receipt_path: str | Path | None = None,
    training_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    receipt_path, _receipt_sha = _bound_file(
        (
            training_receipt_path
            if training_receipt_path is not None
            else os.environ.get("DUCA_RIME_TRAINING_RECEIPT")
        ),
        (
            training_receipt_sha256
            if training_receipt_sha256 is not None
            else os.environ.get("DUCA_RIME_TRAINING_RECEIPT_SHA256")
        ),
        "RIME training receipt",
    )
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    checkpoint_resolved = str(Path(checkpoint_path).expanduser().resolve())
    checkpoint_sha = _sha256_file(checkpoint_resolved)
    source_arm = str(receipt.get("arm"))
    receipt_schema = receipt.get("schema_version")
    if (
        receipt_schema
        not in {
            "duca_rime_phase2_mixed_k_training_receipt_v1",
            "duca_rime_phase3_training_receipt_v1",
            "duca_rime_phase4_training_receipt_v1",
        }
        or receipt.get("status") != "passed"
        or receipt.get("git_commit") != str(git_commit)
        or int(receipt.get("seed", -1)) != int(seed)
        or int(receipt.get("successful_detector_updates", -1)) != 6000
        or receipt.get("formal_update_audit_passed") is not True
        or receipt.get("uses_official_final") is not False
        or str(Path(receipt.get("checkpoint_path", "")).resolve()) != checkpoint_resolved
        or receipt.get("checkpoint_sha256") != checkpoint_sha
        or int(receipt.get("checkpoint_epoch", -1)) != 59
        or receipt.get("checkpoint_state_key") != "state_dict_ema"
        or (
            str(evaluation_arm) != source_arm
            and not (
                str(evaluation_arm) == "U-same-K"
                and source_arm == "RIME-full"
            )
            and not (
                str(evaluation_arm) == "U-same-K-TriDet"
                and source_arm == "RIME-full-TriDet"
            )
        )
    ):
        raise ValueError("RIME terminal checkpoint/training receipt binding mismatch")
    compaction_path, compaction_sha = _bound_file(
        receipt.get("checkpoint_compaction_receipt_path"),
        receipt.get("checkpoint_compaction_receipt_sha256"),
        "checkpoint compaction receipt",
    )
    compaction_receipt = json.loads(
        Path(compaction_path).read_text(encoding="utf-8")
    )
    compaction = checkpoint.get("duca_rime_compaction")
    if (
        not isinstance(compaction, Mapping)
        or compaction.get("schema_version")
        != "duca_rime_compact_checkpoint_receipt_v1"
        or compaction.get("git_commit") != str(git_commit)
        or compaction.get("variant") != source_arm
        or int(compaction.get("seed", -1)) != int(seed)
        or compaction.get("evaluation_equivalent") is not True
        or compaction.get("optimizer_state_retained") is not False
        or compaction.get("training_resume_supported") is not False
        or compaction_receipt.get("schema_version")
        != "duca_rime_compact_checkpoint_receipt_v1"
        or compaction_receipt.get("status") != "passed"
        or compaction_receipt.get("git_commit") != str(git_commit)
        or compaction_receipt.get("variant") != source_arm
        or int(compaction_receipt.get("seed", -1)) != int(seed)
        or Path(
            str(compaction_receipt.get("compact_checkpoint_path", ""))
        ).resolve()
        != Path(checkpoint_resolved)
        or compaction_receipt.get("compact_checkpoint_sha256") != checkpoint_sha
        or compaction_receipt.get("evaluation_equivalent") is not True
        or compaction_receipt.get("training_resume_supported") is not False
    ):
        raise ValueError("RIME terminal checkpoint compaction binding is invalid")
    metadata = checkpoint.get("experiment_metadata")
    audit = None if not isinstance(metadata, Mapping) else metadata.get("training_audit")
    if not isinstance(metadata, Mapping):
        raise ValueError("RIME terminal checkpoint lacks experiment metadata")
    unsigned_metadata = dict(metadata)
    metadata_sha256 = unsigned_metadata.pop("metadata_sha256", None)
    if (
        metadata.get("schema_version") != DUCA_P0_CHECKPOINT_METADATA_SCHEMA
        or metadata_sha256 != duca_p0_training.canonical_sha256(unsigned_metadata)
    ):
        raise ValueError("RIME terminal checkpoint metadata hash is invalid")
    if isinstance(audit, Mapping):
        unsigned_audit = dict(audit)
        audit_sha256 = unsigned_audit.pop("audit_sha256", None)
    else:
        unsigned_audit = {}
        audit_sha256 = None
    if (
        not isinstance(audit, Mapping)
        or audit_sha256 != duca_p0_training.canonical_sha256(unsigned_audit)
        or audit.get("status") != "complete"
        or audit.get("git_commit") != str(git_commit)
        or audit.get("variant") != source_arm
        or int(audit.get("seed", -1)) != int(seed)
        or int(audit.get("expected_successful_optimizer_updates", -1)) != 6000
        or int(audit.get("update_audit", {}).get("successful_optimizer_updates", -1))
        != 6000
    ):
        raise ValueError("RIME terminal checkpoint contains an invalid training audit")
    if receipt_schema == "duca_rime_phase2_mixed_k_training_receipt_v1":
        if (
            int(receipt.get("research_phase", -1)) != 2
            or float(receipt.get("target_mean_cost", math.nan)) != 384.0
            or receipt.get("detector_backend") != "ActionFormer"
            or int(audit.get("research_phase", -1)) != 2
            or audit.get("phase4_authorization_path") is not None
            or audit.get("phase4_authorization_sha256") is not None
            or float(audit.get("formal_budget_panel", math.nan)) != 384.0
            or audit.get("detector_backend") != "ActionFormer"
            or source_arm != "U-mixed-K"
        ):
            raise ValueError(
                "RIME Phase-2 mixed-K checkpoint binding is invalid"
            )
    elif receipt_schema == "duca_rime_phase3_training_receipt_v1":
        if (
            int(audit.get("research_phase", -1)) != 3
            or audit.get("phase4_authorization_path") is not None
            or audit.get("phase4_authorization_sha256") is not None
            or audit.get("formal_budget_panel") is not None
            or audit.get("detector_backend") != "ActionFormer"
        ):
            raise ValueError("RIME Phase-3 checkpoint contains a Phase-4 binding")
    else:
        authorization_path, authorization_sha = _bound_file(
            receipt.get("phase4_authorization_path"),
            receipt.get("phase4_authorization_sha256"),
            "Phase-4 authorization",
        )
        authorization = json.loads(
            Path(authorization_path).read_text(encoding="utf-8")
        )
        target_mean_cost = float(receipt.get("target_mean_cost", math.nan))
        detector_backend = str(receipt.get("detector_backend", ""))
        if (
            int(receipt.get("research_phase", -1)) != 4
            or int(audit.get("research_phase", -1)) != 4
            or audit.get("phase4_authorization_path") != authorization_path
            or audit.get("phase4_authorization_sha256") != authorization_sha
            or float(audit.get("formal_budget_panel", math.nan))
            != target_mean_cost
            or audit.get("detector_backend") != detector_backend
            or target_mean_cost not in {192.0, 384.0}
            or detector_backend not in {"ActionFormer", "TriDet"}
            or authorization.get("schema_version")
            != "duca_rime_stage_receipt_v1"
            or authorization.get("phase") != "phase4_authorization"
            or authorization.get("status") != "authorized"
            or authorization.get("gate_pass") is not True
            or authorization.get("git_commit") != str(git_commit)
            or int(seed)
            not in {int(value) for value in authorization.get("formal_seeds", ())}
            or detector_backend
            not in authorization.get("required_detectors", ())
            or target_mean_cost
            not in {
                float(value)
                for value in authorization.get("required_budget_panels", ())
            }
            or authorization.get("official_final_subset_consumed") is not False
        ):
            raise ValueError(
                "RIME Phase-4 receipt, checkpoint, and authorization are not bound"
            )
    return {
        "training_receipt_path": receipt_path,
        "training_receipt_sha256": _sha256_file(receipt_path),
        "checkpoint_compaction_receipt_sha256": compaction_sha,
        "source_arm": source_arm,
        "evaluation_arm": str(evaluation_arm),
        "checkpoint_path": checkpoint_resolved,
        "checkpoint_sha256": checkpoint_sha,
        "successful_detector_updates": 6000,
        "split_assignment_sha256": audit["split_assignment_sha256"],
        "training_exposure_sha256": audit["training_exposure_sha256"],
        "initialization_sha256": audit["initialization_sha256"],
        "official_final_subset_consumed_during_training": False,
        "research_phase": (
            4 if receipt_schema == "duca_rime_phase4_training_receipt_v1" else 3
        ),
        "phase4_authorization_sha256": (
            audit.get("phase4_authorization_sha256")
            if receipt_schema == "duca_rime_phase4_training_receipt_v1"
            else None
        ),
        "target_mean_cost": (
            audit.get("formal_budget_panel")
            if receipt_schema == "duca_rime_phase4_training_receipt_v1"
            else None
        ),
        "detector_backend": audit.get("detector_backend"),
    }


__all__ = [
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "FORMAL_PROTOCOLS",
    "TRAIN_ARMS",
    "atomic_write_json",
    "after_checkpoint_saved",
    "bind_train_loader_contract",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "capture_global_rng_state",
    "derive_train_loader_contract",
    "formal_training_contract",
    "is_formal_protocol",
    "new_update_audit",
    "restore_global_rng_state",
    "restore_training_state",
    "selector_schedule_step",
    "validate_terminal_checkpoint_binding",
    "validate_update_state",
]
