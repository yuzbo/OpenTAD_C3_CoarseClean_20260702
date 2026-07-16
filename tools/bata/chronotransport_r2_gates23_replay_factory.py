#!/usr/bin/env python3
"""Fixed repository-owned construction and execution for formal Gate-2/3 replay."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from mmengine.config import Config

from opentad.models.chronotransport.filesystem import (
    audit_formal_python_runtime,
    load_bound_torch,
    load_registered_python_config,
)
from opentad.models.chronotransport.protocol import canonical_sha256
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
from opentad.utils import set_seed
from tools.bata.chronotransport_r2_stage_b_factory import (
    ManifestFitBatchSequence,
    R2_STAGE_B_CONFIG,
    R2_STAGE_B_CONFIG_RELATIVE,
    RegisteredManifestFitDataset,
    _runtime,
    build_repository_stage_b_components,
    sealed_stage_b_replay,
)


ROOT = Path(os.path.abspath(__file__)).parents[2]
_DETERMINISTIC_EVAL_SEED = 20260711
_NO_LEAK = {
    "gt_used_for_scheduler": False,
    "teacher_used_for_scheduler": False,
    "dense_reference_used_for_scheduler": False,
    "raw_prediction_cache_used_for_scheduler": False,
    "counterfactual_ledger_used_for_scheduler": False,
    "evaluation_oracle_used_for_scheduler": False,
    "scheduler_target_access": False,
    "targets_evaluation_only": True,
}
_EXECUTION = {
    "repair_count": 0,
    "nan_fallback": False,
    "whole_window_dense_fallback": False,
    "safety_override_budget_violation": False,
    "window_cache_reset": True,
}


def build_repository_gates23_seed_context(
    *,
    registration: Mapping[str, Any],
    seed: int,
    registration_commit: str,
    registration_relpath: str,
):
    """Build the fixed OpenTAD model and canonical fit batches for one seed."""

    from opentad.models.chronotransport.gates23 import _require_regular_input

    manifest_identity = registration["window_manifest"]
    return build_repository_stage_b_components(
        registration=registration,
        manifest_path=_require_regular_input(
            manifest_identity["source_path"], label="formal Gate2/3 window manifest"
        ),
        media_registry_path=_require_regular_input(
            manifest_identity["registry_path"], label="formal Gate2/3 media registry"
        ),
        config_identity_path=_require_regular_input(
            manifest_identity["config_identity_path"],
            label="formal Gate2/3 config identity",
        ),
        exposure_artifact=registration["exposures"]["stage_b"],
        seed=seed,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def _deterministic_split_dataset(
    cfg: Config,
    manifest: Mapping[str, Any],
    split: str,
    registration: Mapping[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    if split not in ("calibration", "evaluation"):
        raise ValueError("formal Gate2/3 replay split must be calibration or evaluation")
    manifested = {str(row["window_id"]): dict(row) for row in manifest["windows"]}
    windows = [manifested[window] for window in manifest["splits"][split]]
    dataset = RegisteredManifestFitDataset(
        cfg,
        manifest,
        registration,
        split=split,
        augment=False,
    )
    return dataset, windows


def _scalar(value: torch.Tensor, *, label: str) -> float:
    flat = value.detach().float().cpu().reshape(-1)
    if flat.numel() != 1:
        raise ValueError(f"formal Gate2/3 {label} must be one scalar")
    result = float(flat.item())
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"formal Gate2/3 {label} must be finite and non-negative")
    return result


def _run_window_vector(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    registered_actions: Mapping[str, str],
) -> dict[str, Any]:
    runtime = _runtime(model)
    q_hat: list[float] = []
    regret: list[float] = []
    feature_mse: list[float] = []
    requested: list[str] = []
    executed: list[str] = []
    materialized = None
    augmentation = None
    for schedule in R2_NON_DENSE_NAMES:
        replay = sealed_stage_b_replay(
            model,
            batch,
            schedule,
            registration=registration,
            track_grad=False,
        )
        summary = runtime.latest_summary
        if not isinstance(summary, Mapping):
            raise RuntimeError("formal Gate2/3 replay is missing runtime evidence")
        if (
            summary.get("evidence_valid") is not True
            or summary.get("schedule_repair_count") != 0
            or summary.get("whole_window_dense_fallback") is not False
            or summary.get("cache_reset_per_window") is not True
            or summary.get("requested_action_sha256")
            != summary.get("executed_action_sha256")
        ):
            raise RuntimeError("formal Gate2/3 replay repaired or fell back")
        if replay.requested_action_sha256 != registered_actions[schedule]:
            raise RuntimeError("formal Gate2/3 requested action differs from registration")
        if replay.executed_action_sha256 != registered_actions[schedule]:
            raise RuntimeError("formal Gate2/3 executed action differs from registration")
        current_materialized = str(replay.materialized_window_sha256)
        current_augmentation = str(replay.augmentation_sha256)
        if materialized is None:
            materialized = current_materialized
            augmentation = current_augmentation
        elif (
            materialized != current_materialized
            or augmentation != current_augmentation
        ):
            raise RuntimeError("formal Gate2/3 candidates did not share one materialized batch")
        if replay.dense_features.shape != replay.counterfactual_features.shape:
            raise RuntimeError("formal Gate2/3 paired feature shapes differ")
        mse = torch.mean(
            (replay.counterfactual_features.float() - replay.dense_features.float()) ** 2
        )
        q_hat.append(_scalar(replay.predicted_quantile, label="predicted quantile"))
        regret.append(_scalar(replay.regret_target, label="detector regret"))
        feature_mse.append(_scalar(mse, label="feature MSE"))
        requested.append(str(replay.requested_action_sha256))
        executed.append(str(replay.executed_action_sha256))
    if materialized is None or augmentation is None:
        raise RuntimeError("formal Gate2/3 replay produced no candidate vector")
    return {
        "materialized_window_sha256": materialized,
        "augmentation_sha256": augmentation,
        "q_hat": q_hat,
        "detector_regret": regret,
        "feature_mse": feature_mse,
        "requested_action_sha256": requested,
        "executed_action_sha256": executed,
    }


def build_registered_gates23_replay_artifact(
    *,
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path,
    repository_root: Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """Execute registered checkpoints on canonical cal/eval batches and freeze rows."""

    from opentad.models.chronotransport.gates23 import (
        GATES23_REPLAY_FORMAL_SCHEMA,
        _normalize_row,
        _path_without_symlink_components,
        _require_regular_input,
        _validate_candidate_actions,
        _validate_phase_bindings,
        _validate_split_windows,
        _validate_stage_b_phase_markers_full,
        load_exact_canonical_json,
    )

    repository_root = _path_without_symlink_components(
        repository_root, label="formal Gate2/3 repository root"
    )
    fixed_root = _path_without_symlink_components(
        ROOT, label="fixed Gate2/3 repository root"
    )
    if repository_root != fixed_root:
        raise ValueError("formal Gate2/3 replay factory is bound to its repository root")
    phases, _, contexts = _validate_stage_b_phase_markers_full(
        phase_marker_paths,
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    manifest = registration["window_manifest"]["artifact"]
    window_rows = {str(row["window_id"]): row for row in manifest["windows"]}
    split_windows = {
        split: list(map(str, manifest["splits"][split]))
        for split in ("calibration", "evaluation")
    }
    video_by_window = {
        window: str(window_rows[window]["video_id"])
        for split in ("calibration", "evaluation")
        for window in split_windows[split]
    }
    library = {
        str(row["name"]): row for row in registration["candidate_library"]["candidates"]
    }
    actions = {
        name: str(library[name]["action_sha256"]) for name in R2_NON_DENSE_NAMES
    }
    rows: list[dict[str, Any]] = []
    _require_regular_input(R2_STAGE_B_CONFIG, label="formal Gate2/3 Stage-B config")
    cfg, _ = load_registered_python_config(
        repository_root=ROOT,
        config_relative=R2_STAGE_B_CONFIG_RELATIVE,
        registered_sources=registration["source_files"],
    )
    audit_formal_python_runtime(
        repository_root=repository_root,
        registered_sources=registration["source_files"],
        entrypoint_relative="tools/bata/run_chronotransport_r2_gates23.py",
    )
    for split_index, split in enumerate(("calibration", "evaluation")):
        dataset, windows = _deterministic_split_dataset(
            cfg, manifest, split, registration
        )
        for seed in (3407, 3408, 3409):
            context = contexts[str(seed)]
            marker_path = Path(phase_marker_paths[seed])
            marker = load_exact_canonical_json(
                marker_path, label=f"Stage-B phase marker {seed}"
            )
            checkpoint_path = _require_regular_input(
                marker["trained_checkpoint"]["path"],
                label=f"formal Gate2/3 Stage-B checkpoint {seed}",
            )
            _, checkpoint, _, _ = load_bound_torch(
                checkpoint_path,
                label=f"formal Gate2/3 Stage-B checkpoint {seed}",
            )
            context.model.load_state_dict(checkpoint["state_dict_ema"], strict=True)
            context.model.eval()
            runtime = _runtime(context.model)
            runtime.capture_replay_signals = True
            runtime.set_checkpoint_loaded(True)
            batches: Sequence[Mapping[str, Any]] = ManifestFitBatchSequence(
                dataset, windows, torch.device("cuda:0")
            )
            for window_index in range(len(batches)):
                set_seed(_DETERMINISTIC_EVAL_SEED + split_index * 30 + window_index)
                batch = batches[window_index]
                vector = _run_window_vector(
                    context.model,
                    batch,
                    registration=context.registration,
                    registered_actions=actions,
                )
                window = windows[window_index]
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "window_id": str(window["window_id"]),
                        "video_id": str(window["video_id"]),
                        "trained_checkpoint_sha256": phases[str(seed)][
                            "trained_checkpoint_sha256"
                        ],
                        "predictor_canonical_sha256": phases[str(seed)][
                            "predictor_canonical_sha256"
                        ],
                        "candidate_order": list(R2_NON_DENSE_NAMES),
                        "execution": dict(_EXECUTION),
                        "no_leak": dict(_NO_LEAK),
                        **vector,
                    }
                )
    validated_splits = _validate_split_windows(split_windows)
    validated_actions = _validate_candidate_actions(actions)
    validated_phases = _validate_phase_bindings(phases)
    normalized_rows = []
    ordinal = 0
    for split in ("calibration", "evaluation"):
        for seed in (3407, 3408, 3409):
            for window in validated_splits[split]:
                normalized_rows.append(
                    _normalize_row(
                        rows[ordinal],
                        expected_seed=seed,
                        expected_split=split,
                        expected_window=window,
                        expected_video=video_by_window[window],
                        actions=validated_actions,
                        phase_bindings=validated_phases,
                    )
                )
                ordinal += 1
    artifact: dict[str, Any] = {
        "schema": GATES23_REPLAY_FORMAL_SCHEMA,
        "protocol": "CT-P3R-3S-r2",
        "registration_sha256": registration["registration_sha256"],
        "registration_commit": registration_commit,
        "gate1_unlock_artifact_sha256": gate1_unlock["artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "library_sha256": registration["candidate_library"]["library_sha256"],
        "seed_order": [3407, 3408, 3409],
        "candidate_order": list(R2_NON_DENSE_NAMES),
        "candidate_action_sha256_by_name": validated_actions,
        "split_window_ids": validated_splits,
        "video_id_by_window": video_by_window,
        "phase_bindings": validated_phases,
        "row_count": len(normalized_rows),
        "rows": normalized_rows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


__all__ = [
    "build_repository_gates23_seed_context",
    "build_registered_gates23_replay_artifact",
]
