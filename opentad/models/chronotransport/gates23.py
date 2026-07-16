"""Formal CPU-only Gate-2/Gate-3 adjudication for CT-P3R-3S-r2.

This module consumes complete, hash-bound replay vectors.  It never executes
the detector, opens CUDA, or permits evaluation targets to enter scheduler
selection.  Formal replay construction deliberately remains outside the raw
row API; the only raw-row builder below emits a disjoint test-fixture schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
from statistics import median
from typing import Any, Mapping, Sequence

from .formal_stage_b import (
    _STAGE_B_CHECKPOINT_KEYS,
    _validate_fit_schedule_constant_artifact,
    validate_r2_stage_b_phase_completion_marker,
)
from .protocol import R2_PROTOCOL_ID, R2_SEEDS, canonical_json_bytes, canonical_sha256
from .registration import FORMAL_OUTPUT_BASE, validate_pre_gate1_registration
from .scheduler import R2_NON_DENSE_NAMES


BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260711
CALIBRATION_WINDOWS = 30
EVALUATION_WINDOWS = 30
CALIBRATION_RANK = 28
FIT_BASELINE_WINDOWS = 140
FIT_BASELINE_RANK = 127
QUANTILE = 0.9
SCHEDULER_EPSILON = 1.0

GATES23_REPLAY_FORMAL_SCHEMA = "chronotransport-r2-gates23-replay-formal-v1"
GATES23_REPLAY_FIXTURE_SCHEMA = "chronotransport-r2-gates23-replay-test-fixture-v1"
GATES23_REPORT_SCHEMA = "chronotransport-r2-gates23-report-v1"
GATES23_TERMINAL_SCHEMA = "chronotransport-r2-gates23-terminal-formal-v2"
STAGE_B_PHASE_SCHEMA = "chronotransport-r2-stage-b-phase-completion-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SPLITS = ("calibration", "evaluation")
_PERIODS = (2, 4, 8)
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
_ROW_FIELDS = {
    "seed",
    "split",
    "window_id",
    "video_id",
    "trained_checkpoint_sha256",
    "predictor_canonical_sha256",
    "materialized_window_sha256",
    "augmentation_sha256",
    "candidate_order",
    "q_hat",
    "detector_regret",
    "feature_mse",
    "requested_action_sha256",
    "executed_action_sha256",
    "execution",
    "no_leak",
}
_ARTIFACT_FIELDS = {
    "schema",
    "protocol",
    "registration_sha256",
    "registration_commit",
    "gate1_unlock_artifact_sha256",
    "manifest_sha256",
    "library_sha256",
    "seed_order",
    "candidate_order",
    "candidate_action_sha256_by_name",
    "split_window_ids",
    "video_id_by_window",
    "phase_bindings",
    "row_count",
    "rows",
    "artifact_sha256",
}
_PHASE_BINDING_FIELDS = {
    "phase_marker_sha256",
    "trained_checkpoint_sha256",
    "predictor_canonical_sha256",
    "fit_baseline_payload_sha256",
}


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ValueError(f"{label} must be one full Git commit")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires non-empty values")
    return float(sum(values) / len(values))


def _percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(map(float, values))
    if len(ordered) != BOOTSTRAP_SAMPLES:
        raise ValueError("formal percentile CI requires exactly 5000 replicates")
    last = len(ordered) - 1
    return [
        ordered[int(math.floor(0.025 * last))],
        ordered[int(math.ceil(0.975 * last))],
    ]


def _one_sided_lower(values: Sequence[float]) -> float:
    ordered = sorted(map(float, values))
    if len(ordered) != BOOTSTRAP_SAMPLES:
        raise ValueError("formal one-sided LCB requires exactly 5000 replicates")
    return float(ordered[int(math.floor(0.05 * (len(ordered) - 1)))])


def _decode_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error


def _path_without_symlink_components(
    path: Path | str, *, label: str, allow_missing: bool = False
) -> Path:
    """Return an absolute lexical path after lstat-checking every existing component."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                break
            raise FileNotFoundError(current) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}")
    return absolute


def load_exact_canonical_json(path: Path | str, *, label: str) -> dict[str, Any]:
    """Load exact canonical JSON plus one LF, rejecting duplicate keys."""

    path = _path_without_symlink_components(path, label=label)
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    value = _decode_json(raw, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{label} bytes must be exact canonical JSON plus one newline")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_input(path: Path | str, *, label: str) -> Path:
    absolute = _path_without_symlink_components(path, label=label)
    metadata = os.lstat(absolute)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return absolute


def _registered_stage_b_provenance(
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    gate1_unlock_path: Path,
    registration_commit: str,
) -> dict[str, str]:
    return {
        "registration_sha256": str(registration["registration_sha256"]),
        "registration_commit": registration_commit,
        "spec_commit": str(registration["spec"]["commit"]),
        "spec_sha256": str(registration["spec"]["sha256"]),
        "implementation_commit": str(registration["implementation_commit"]),
        "source_files_sha256": canonical_sha256(registration["source_files"]),
        "upstream_commits_sha256": canonical_sha256(registration["upstream_commits"]),
        "split_hashes_sha256": canonical_sha256(
            registration["window_manifest"]["split_hashes"]
        ),
        "action_library_sha256": str(
            registration["candidate_library"]["library_sha256"]
        ),
        "environment_sha256": str(
            registration["environment"]["environment_sha256"]
        ),
        "cost_plan_sha256": canonical_sha256(registration["profiler"]),
        "gate1_unlock_payload_sha256": canonical_sha256(gate1_unlock),
        "gate1_unlock_file_sha256": _file_sha256(gate1_unlock_path),
        "gate1_status": "PASS",
    }


def _validate_formal_gate_context(
    *,
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently bind every formal mint to clean detached R and Gate-1 PASS."""

    root = _path_without_symlink_components(
        repository_root, label="formal Gate2/3 repository root"
    )
    registration_commit = _require_commit(
        registration_commit, "formal Gate2/3 registration commit R"
    )
    registration_file = root / registration_relpath
    _require_regular_input(registration_file, label="formal registration artifact")
    registered = validate_pre_gate1_registration(
        registration,
        repository_root=root,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    gate1_path = _require_regular_input(
        gate1_unlock_path, label="formal Gate1 unlock artifact"
    )
    serialized = load_exact_canonical_json(gate1_path, label="formal Gate1 unlock artifact")
    if serialized != dict(gate1_unlock):
        raise ValueError("formal Gate1 unlock mapping differs from its exact file bytes")
    from .gate1_unlock import (
        build_gate1_unlock_artifact,
        validate_gate1_unlock_artifact,
    )

    rebuilt = build_gate1_unlock_artifact(
        serialized.get("gate1_input", {}),
        repository_root=str(root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if rebuilt != serialized:
        raise ValueError("formal Gate1 unlock is not an exact full recomputation")
    validated_gate1 = validate_gate1_unlock_artifact(
        rebuilt,
        registration=registered,
        repository_root=str(root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    return registered, validated_gate1


def _load_exact_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], bytes]:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular file, not a symlink")
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty and LF terminated")
    rows: list[dict[str, Any]] = []
    rebuilt = bytearray()
    for ordinal, line in enumerate(raw.splitlines()):
        row = _decode_json(line, label=f"{label} row {ordinal}")
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {ordinal} must be a JSON object")
        encoded = canonical_json_bytes(row)
        if encoded != line:
            raise ValueError(f"{label} row {ordinal} is not exact canonical JSON")
        rows.append(row)
        rebuilt.extend(encoded + b"\n")
    if bytes(rebuilt) != raw:
        raise ValueError(f"{label} exact bytes mismatch")
    return rows, raw


def _validate_seed_order(value: Any) -> list[int]:
    if value != list(R2_SEEDS):
        raise ValueError("Gate2/3 seed order must be exactly 3407/3408/3409")
    return list(R2_SEEDS)


def _validate_candidate_actions(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(R2_NON_DENSE_NAMES):
        raise ValueError("Gate2/3 candidate action mapping fields mismatch")
    return {
        name: _require_sha256(value[name], f"registered action for {name}")
        for name in R2_NON_DENSE_NAMES
    }


def _validate_phase_bindings(value: Any) -> dict[str, dict[str, str]]:
    expected_seeds = {str(seed) for seed in R2_SEEDS}
    if not isinstance(value, Mapping) or set(value) != expected_seeds:
        raise ValueError("Gate2/3 requires exactly three Stage-B phase bindings")
    normalized = {}
    for seed in R2_SEEDS:
        raw = value[str(seed)]
        _require_exact_fields(raw, _PHASE_BINDING_FIELDS, f"phase binding {seed}")
        normalized[str(seed)] = {
            field: _require_sha256(raw[field], f"phase binding {seed} {field}")
            for field in sorted(_PHASE_BINDING_FIELDS)
        }
    return normalized


def _validate_split_windows(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or set(value) != set(_SPLITS):
        raise ValueError("Gate2/3 split-window fields mismatch")
    result = {}
    for split in _SPLITS:
        windows = value[split]
        if (
            not isinstance(windows, list)
            or len(windows) != 30
            or len(set(windows)) != 30
            or any(not isinstance(item, str) or not item for item in windows)
        ):
            raise ValueError(f"Gate2/3 {split} requires exactly 30 unique windows")
        result[split] = list(windows)
    if set(result["calibration"]) & set(result["evaluation"]):
        raise ValueError("Gate2/3 calibration/evaluation windows must be disjoint")
    return result


def _normalize_vector(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != len(R2_NON_DENSE_NAMES):
        raise ValueError(f"Gate2/3 {field} requires a complete 16-candidate vector")
    return [_finite_nonnegative(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _normalize_row(
    raw: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_split: str,
    expected_window: str,
    expected_video: str,
    actions: Mapping[str, str],
    phase_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    _require_exact_fields(raw, _ROW_FIELDS, "Gate2/3 replay row")
    if type(raw["seed"]) is not int or raw["seed"] != expected_seed:
        raise ValueError("Gate2/3 replay rows are not in canonical seed order")
    if raw["split"] != expected_split or raw["window_id"] != expected_window:
        raise ValueError("Gate2/3 replay rows are not in canonical order")
    if raw["video_id"] != expected_video:
        raise ValueError("Gate2/3 replay row video/window binding mismatch")
    phase = phase_bindings[str(expected_seed)]
    if (
        raw["trained_checkpoint_sha256"] != phase["trained_checkpoint_sha256"]
        or raw["predictor_canonical_sha256"] != phase["predictor_canonical_sha256"]
    ):
        raise ValueError("Gate2/3 replay row differs from its Stage-B phase binding")
    for field in ("materialized_window_sha256", "augmentation_sha256"):
        _require_sha256(raw[field], f"Gate2/3 row {field}")
    if raw["candidate_order"] != list(R2_NON_DENSE_NAMES):
        raise ValueError("Gate2/3 replay row candidate order mismatch")
    expected_actions = [actions[name] for name in R2_NON_DENSE_NAMES]
    if raw["requested_action_sha256"] != expected_actions:
        raise ValueError("Gate2/3 requested actions differ from registration")
    if raw["executed_action_sha256"] != expected_actions:
        raise ValueError("Gate2/3 executed actions differ from registration")
    if raw["execution"] != _EXECUTION:
        raise ValueError("Gate2/3 execution requires exact no-repair/no-fallback evidence")
    if raw["no_leak"] != _NO_LEAK:
        raise ValueError("Gate2/3 replay row violates the complete no-leak contract")
    row = dict(raw)
    row["q_hat"] = _normalize_vector(raw["q_hat"], field="q_hat")
    row["detector_regret"] = _normalize_vector(
        raw["detector_regret"], field="detector_regret"
    )
    row["feature_mse"] = _normalize_vector(raw["feature_mse"], field="feature_mse")
    row["row_sha256"] = canonical_sha256(row)
    return row


def _build_replay_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: str,
    registration_sha256: str,
    registration_commit: str,
    gate1_unlock_artifact_sha256: str,
    manifest_sha256: str,
    library_sha256: str,
    split_window_ids: Mapping[str, Sequence[str]],
    video_id_by_window: Mapping[str, str],
    candidate_action_sha256_by_name: Mapping[str, str],
    phase_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if schema != GATES23_REPLAY_FIXTURE_SCHEMA:
        raise ValueError(
            "raw Gate2/3 rows can build only the disjoint test-fixture schema"
        )
    registration_sha256 = _require_sha256(registration_sha256, "registration SHA-256")
    registration_commit = _require_commit(registration_commit, "registration commit R")
    gate1_sha = _require_sha256(gate1_unlock_artifact_sha256, "Gate1 unlock SHA-256")
    manifest_sha256 = _require_sha256(manifest_sha256, "manifest SHA-256")
    library_sha256 = _require_sha256(library_sha256, "library SHA-256")
    splits = _validate_split_windows(
        {split: list(split_window_ids[split]) for split in _SPLITS}
        if isinstance(split_window_ids, Mapping) and set(split_window_ids) == set(_SPLITS)
        else split_window_ids
    )
    all_windows = [window for split in _SPLITS for window in splits[split]]
    if not isinstance(video_id_by_window, Mapping) or set(video_id_by_window) != set(all_windows):
        raise ValueError("Gate2/3 video-by-window mapping fields mismatch")
    videos = {}
    for window in all_windows:
        video = video_id_by_window[window]
        if not isinstance(video, str) or not video:
            raise ValueError("Gate2/3 video IDs must be non-empty strings")
        videos[window] = video
    if len(set(videos.values())) != len(videos):
        raise ValueError("Gate2/3 requires one unique manifested video per window")
    actions = _validate_candidate_actions(candidate_action_sha256_by_name)
    phases = _validate_phase_bindings(phase_bindings)
    if not isinstance(rows, Sequence) or len(rows) != len(_SPLITS) * len(R2_SEEDS) * 30:
        raise ValueError("Gate2/3 replay requires exactly 180 seed-window vectors")
    normalized_rows = []
    ordinal = 0
    for split in _SPLITS:
        for seed in R2_SEEDS:
            for window in splits[split]:
                normalized_rows.append(
                    _normalize_row(
                        rows[ordinal],
                        expected_seed=seed,
                        expected_split=split,
                        expected_window=window,
                        expected_video=videos[window],
                        actions=actions,
                        phase_bindings=phases,
                    )
                )
                ordinal += 1
    artifact: dict[str, Any] = {
        "schema": schema,
        "protocol": R2_PROTOCOL_ID,
        "registration_sha256": registration_sha256,
        "registration_commit": registration_commit,
        "gate1_unlock_artifact_sha256": gate1_sha,
        "manifest_sha256": manifest_sha256,
        "library_sha256": library_sha256,
        "seed_order": list(R2_SEEDS),
        "candidate_order": list(R2_NON_DENSE_NAMES),
        "candidate_action_sha256_by_name": actions,
        "split_window_ids": splits,
        "video_id_by_window": videos,
        "phase_bindings": phases,
        "row_count": len(normalized_rows),
        "rows": normalized_rows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def build_gates23_replay_artifact_for_test_only(
    rows: Sequence[Mapping[str, Any]], **identity: Any
) -> dict[str, Any]:
    """Build only a disjoint fixture schema; formal CLI rejects it."""

    return _build_replay_artifact(
        rows, schema=GATES23_REPLAY_FIXTURE_SCHEMA, **identity
    )


def _validate_replay_artifact(
    artifact: Mapping[str, Any], *, expected_schema: str
) -> dict[str, Any]:
    _require_exact_fields(artifact, _ARTIFACT_FIELDS, "Gate2/3 replay artifact")
    if artifact["schema"] != expected_schema or artifact["protocol"] != R2_PROTOCOL_ID:
        raise ValueError("unsupported Gate2/3 replay artifact schema/protocol")
    unsigned = dict(artifact)
    claimed = unsigned.pop("artifact_sha256")
    if claimed != canonical_sha256(unsigned):
        raise ValueError("Gate2/3 replay artifact SHA-256 mismatch")
    _validate_seed_order(artifact["seed_order"])
    if artifact["candidate_order"] != list(R2_NON_DENSE_NAMES):
        raise ValueError("Gate2/3 replay artifact candidate order mismatch")
    if type(artifact["row_count"]) is not int or artifact["row_count"] != 180:
        raise ValueError("Gate2/3 replay artifact row_count must equal 180")
    raw_rows = []
    for row in artifact["rows"]:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS | {"row_sha256"}:
            raise ValueError("Gate2/3 replay row fields mismatch")
        raw = dict(row)
        row_sha = raw.pop("row_sha256")
        if row_sha != canonical_sha256(raw):
            raise ValueError("Gate2/3 replay row SHA-256 mismatch")
        raw_rows.append(raw)
    if expected_schema == GATES23_REPLAY_FIXTURE_SCHEMA:
        rebuilt = _build_replay_artifact(
            raw_rows,
            schema=expected_schema,
            registration_sha256=artifact["registration_sha256"],
            registration_commit=artifact["registration_commit"],
            gate1_unlock_artifact_sha256=artifact["gate1_unlock_artifact_sha256"],
            manifest_sha256=artifact["manifest_sha256"],
            library_sha256=artifact["library_sha256"],
            split_window_ids=artifact["split_window_ids"],
            video_id_by_window=artifact["video_id_by_window"],
            candidate_action_sha256_by_name=artifact[
                "candidate_action_sha256_by_name"
            ],
            phase_bindings=artifact["phase_bindings"],
        )
        if rebuilt != dict(artifact):
            raise ValueError("Gate2/3 replay artifact differs from its canonical rebuild")
        return rebuilt

    _require_sha256(artifact["registration_sha256"], "registration SHA-256")
    _require_commit(artifact["registration_commit"], "registration commit R")
    _require_sha256(
        artifact["gate1_unlock_artifact_sha256"], "Gate1 unlock SHA-256"
    )
    _require_sha256(artifact["manifest_sha256"], "manifest SHA-256")
    _require_sha256(artifact["library_sha256"], "library SHA-256")
    splits = _validate_split_windows(artifact["split_window_ids"])
    all_windows = [window for split in _SPLITS for window in splits[split]]
    videos = artifact["video_id_by_window"]
    if not isinstance(videos, Mapping) or set(videos) != set(all_windows):
        raise ValueError("Gate2/3 formal replay video-by-window fields mismatch")
    if any(not isinstance(videos[window], str) or not videos[window] for window in all_windows):
        raise ValueError("Gate2/3 formal replay video IDs must be non-empty strings")
    if len(set(videos.values())) != len(videos):
        raise ValueError("Gate2/3 formal replay requires one unique video per window")
    actions = _validate_candidate_actions(artifact["candidate_action_sha256_by_name"])
    phases = _validate_phase_bindings(artifact["phase_bindings"])
    normalized_rows = []
    ordinal = 0
    for split in _SPLITS:
        for seed in R2_SEEDS:
            for window in splits[split]:
                normalized_rows.append(
                    _normalize_row(
                        raw_rows[ordinal],
                        expected_seed=seed,
                        expected_split=split,
                        expected_window=window,
                        expected_video=videos[window],
                        actions=actions,
                        phase_bindings=phases,
                    )
                )
                ordinal += 1
    if normalized_rows != artifact["rows"]:
        raise ValueError("formal Gate2/3 replay rows differ from canonical structure")
    return dict(artifact)


def validate_gates23_replay_artifact_for_test_only(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return _validate_replay_artifact(
        artifact, expected_schema=GATES23_REPLAY_FIXTURE_SCHEMA
    )


def _assert_matched_recompute_masks(registration: Mapping[str, Any]) -> None:
    candidates = {
        row["name"]: row["actions"]
        for row in registration["candidate_library"]["candidates"]
    }
    for period in _PERIODS:
        transport = candidates[f"periodic{period}_transport"]
        hold = candidates[f"periodic{period}_hold"]
        transport_mask = [[cell == 0 for cell in chunk] for chunk in transport]
        hold_mask = [[cell == 0 for cell in chunk] for chunk in hold]
        if transport_mask != hold_mask:
            raise ValueError(f"Gate2 P{period} TRANSPORT/HOLD recompute masks differ")


def run_registered_gates23_replay(
    *,
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """Execute the sole fixed repository-owned Gate2/3 replay implementation."""

    registered, validated_gate1 = _validate_formal_gate_context(
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    from tools.bata.chronotransport_r2_gates23_replay_factory import (
        build_registered_gates23_replay_artifact,
    )

    return build_registered_gates23_replay_artifact(
        registration=registered,
        gate1_unlock=validated_gate1,
        phase_marker_paths=phase_marker_paths,
        gate1_unlock_path=Path(gate1_unlock_path),
        repository_root=Path(repository_root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def validate_gates23_replay_artifact(
    artifact: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """Re-execute formal replay; serialized raw rows can never mint evidence."""

    _validate_formal_gate_context(
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    expected = run_registered_gates23_replay(
        registration=registration,
        gate1_unlock=gate1_unlock,
        phase_marker_paths=phase_marker_paths,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if not isinstance(artifact, Mapping) or dict(artifact) != expected:
        raise ValueError(
            "formal Gate2/3 artifact differs from repository-owned replay exact recomputation"
        )
    validated = _validate_replay_artifact(
        expected, expected_schema=GATES23_REPLAY_FORMAL_SCHEMA
    )
    if validated["registration_commit"] != registration_commit:
        raise ValueError("formal Gate2/3 replay registration_commit differs from actual R")
    _assert_matched_recompute_masks(registration)
    return validated


def _static_phase_marker(marker: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "protocol",
        "status",
        "registration_sha256",
        "registration_commit",
        "seed",
        "manifest_sha256",
        "library_sha256",
        "config_sha256",
        "candidate_order",
        "trained_checkpoint",
        "ledger",
        "fit_baseline",
        "artifact_sha256",
    }
    _require_exact_fields(marker, fields, "Stage-B phase-completion marker")
    if (
        marker["schema"] != STAGE_B_PHASE_SCHEMA
        or marker["protocol"] != R2_PROTOCOL_ID
        or marker["status"] != "PHASE_COMPLETE"
    ):
        raise ValueError("Stage-B phase-completion marker status/schema mismatch")
    unsigned = dict(marker)
    digest = unsigned.pop("artifact_sha256")
    if digest != canonical_sha256(unsigned):
        raise ValueError("Stage-B phase-completion marker SHA-256 mismatch")
    if type(marker["seed"]) is not int or marker["seed"] not in R2_SEEDS:
        raise ValueError("Stage-B phase-completion marker seed mismatch")
    for field in ("registration_sha256", "manifest_sha256", "library_sha256", "config_sha256"):
        _require_sha256(marker[field], f"Stage-B marker {field}")
    _require_commit(marker["registration_commit"], "Stage-B marker registration commit")
    if marker["candidate_order"] != list(R2_NON_DENSE_NAMES):
        raise ValueError("Stage-B phase marker candidate order mismatch")
    _require_exact_fields(
        marker["trained_checkpoint"],
        {
            "path",
            "bytes",
            "exact_bytes_sha256",
            "state_dict_ema_sha256",
            "predictor_canonical_sha256",
        },
        "Stage-B trained checkpoint binding",
    )
    _require_exact_fields(
        marker["ledger"],
        {"path", "bytes", "exact_bytes_sha256", "canonical_rows_sha256", "row_count"},
        "Stage-B ledger binding",
    )
    _require_exact_fields(
        marker["fit_baseline"],
        {
            "path",
            "bytes",
            "exact_bytes_sha256",
            "payload_sha256",
            "row_count",
            "fit_window_order_sha256",
            "fit_replay_key_sha256",
        },
        "Stage-B fit-baseline binding",
    )
    return dict(marker)


def _validate_stage_b_phase_markers_full(
    marker_paths: Mapping[int, Path | str],
    *,
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    """Run the complete Stage-B validator for all three canonical seed phases."""

    registered, validated_gate1 = _validate_formal_gate_context(
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    registration_commit = _require_commit(registration_commit, "registration commit R")
    if not isinstance(marker_paths, Mapping) or set(marker_paths) != set(R2_SEEDS):
        raise ValueError("Gate2/3 requires all three Stage-B phase-marker paths")
    base = _path_without_symlink_components(
        FORMAL_OUTPUT_BASE, label="formal Gate2/3 output base"
    )
    registered_manifest = registered["window_manifest"]["artifact"]
    registered_fit_windows = list(registered_manifest["splits"]["fit"])
    registered_library_rows = {
        row["name"]: row for row in registered["candidate_library"]["candidates"]
    }
    registered_actions = {
        name: registered_library_rows[name]["action_sha256"]
        for name in R2_NON_DENSE_NAMES
    }
    registered_config_sha256 = registered["source_files"][
        "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py"
    ]
    phases: dict[str, dict[str, str]] = {}
    constants: dict[str, dict[str, float]] = {}
    seed_contexts: dict[str, Any] = {}
    common_identity = None
    gate1_path = _require_regular_input(
        gate1_unlock_path, label="formal Gate1 unlock artifact"
    )
    registered_provenance = None
    for seed in R2_SEEDS:
        seed_root = _path_without_symlink_components(
            base / registration_commit / str(seed),
            label=f"formal Gate2/3 seed output root {seed}",
        )
        marker_path = _require_regular_input(
            marker_paths[seed], label=f"Stage-B phase marker {seed}"
        )
        if marker_path.parent != seed_root:
            raise ValueError("Stage-B phase marker must be a direct child of canonical R/seed")
        marker = _static_phase_marker(
            load_exact_canonical_json(marker_path, label=f"Stage-B phase marker {seed}")
        )
        if (
            marker["seed"] != seed
            or marker["registration_sha256"] != registered["registration_sha256"]
            or marker["registration_commit"] != registration_commit
            or marker["config_sha256"] != registered_config_sha256
        ):
            raise ValueError("Stage-B phase marker registration/seed/config identity mismatch")
        identity = (
            marker["manifest_sha256"],
            marker["library_sha256"],
            marker["config_sha256"],
        )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError("three Stage-B phase markers do not share manifest/library/config")

        checkpoint_binding = marker["trained_checkpoint"]
        checkpoint_path = _require_regular_input(
            checkpoint_binding["path"], label=f"Stage-B checkpoint {seed}"
        )
        if checkpoint_path.parent != seed_root:
            raise ValueError("Stage-B checkpoint is outside canonical R/seed or missing")
        if (
            type(checkpoint_binding["bytes"]) is not int
            or checkpoint_binding["bytes"] <= 0
            or checkpoint_path.stat().st_size != checkpoint_binding["bytes"]
            or _file_sha256(checkpoint_path) != checkpoint_binding["exact_bytes_sha256"]
        ):
            raise ValueError("Stage-B checkpoint exact bytes differ from phase marker")
        for field in (
            "exact_bytes_sha256",
            "state_dict_ema_sha256",
            "predictor_canonical_sha256",
        ):
            _require_sha256(checkpoint_binding[field], f"Stage-B checkpoint {field}")
        try:
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as error:
            raise ValueError("Stage-B trained checkpoint is not loadable") from error
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != _STAGE_B_CHECKPOINT_KEYS:
            raise ValueError("Stage-B trained checkpoint fields do not match the frozen key set")
        if registered_provenance is None:
            registered_provenance = _registered_stage_b_provenance(
                registered, validated_gate1, gate1_path, registration_commit
            )

        ledger_binding = marker["ledger"]
        ledger_path = _require_regular_input(
            ledger_binding["path"], label=f"Stage-B ledger {seed}"
        )
        if ledger_path.parent != seed_root:
            raise ValueError("Stage-B ledger must be a direct child of canonical R/seed")

        baseline_binding = marker["fit_baseline"]
        baseline_path = _require_regular_input(
            baseline_binding["path"], label=f"Stage-B fit baseline {seed}"
        )
        if baseline_path.parent != seed_root:
            raise ValueError("Stage-B fit baseline must be a direct child of canonical R/seed")

        from tools.bata.chronotransport_r2_gates23_replay_factory import (
            build_repository_gates23_seed_context,
        )

        seed_context = build_repository_gates23_seed_context(
            registration=registered,
            seed=seed,
            registration_commit=registration_commit,
            registration_relpath=registration_relpath,
        )
        validated_marker = validate_r2_stage_b_phase_completion_marker(
            marker_path,
            registration_sha256=registered["registration_sha256"],
            registration_commit=registration_commit,
            seed=seed,
            model=seed_context.model,
            batches=seed_context.batches,
            exposure_artifact=seed_context.exposure_artifact,
            dense_checkpoint_path=registered["dense_checkpoint"][
                "content_addressed_path"
            ],
            dense_checkpoint_sha256=registered["dense_checkpoint"]["sha256"],
            dense_checkpoint_use_ema=seed_context.dense_checkpoint_use_ema,
            registered_provenance=registered_provenance,
            checkpoint_path=checkpoint_path,
            ledger_path=ledger_path,
            fit_baseline_path=baseline_path,
            candidate_action_sha256_by_name=registered_actions,
            manifest_sha256=registered_manifest["manifest_sha256"],
            library_sha256=registered["candidate_library"]["library_sha256"],
            config_sha256=registered_config_sha256,
        )
        if validated_marker != marker:
            raise ValueError("Stage-B phase marker differs from complete Stage-B validation")
        baseline = _validate_fit_schedule_constant_artifact(
            load_exact_canonical_json(
                baseline_path, label=f"Stage-B fit baseline {seed}"
            )
        )
        phases[str(seed)] = {
            "phase_marker_sha256": marker["artifact_sha256"],
            "trained_checkpoint_sha256": checkpoint_binding["exact_bytes_sha256"],
            "predictor_canonical_sha256": checkpoint_binding[
                "predictor_canonical_sha256"
            ],
            "fit_baseline_payload_sha256": baseline["artifact_sha256"],
        }
        constants[str(seed)] = {
            name: _finite_nonnegative(
                baseline["schedule_constants"][name],
                f"Stage-B baseline constant {seed}/{name}",
            )
            for name in R2_NON_DENSE_NAMES
        }
        seed_contexts[str(seed)] = seed_context
    return _validate_phase_bindings(phases), constants, seed_contexts


def validate_stage_b_phase_markers_static(
    marker_paths: Mapping[int, Path | str],
    *,
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, float]]]:
    """Compatibility name for full checkpoint/model/ledger/phase revalidation."""

    phases, constants, _ = _validate_stage_b_phase_markers_full(
        marker_paths,
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    return phases, constants


def _evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = [row for row in rows if row.get("split") == "evaluation"]
    for seed in R2_SEEDS:
        seed_rows = [row for row in result if row.get("seed") == seed]
        windows = [row.get("window_id") for row in seed_rows]
        if len(seed_rows) != EVALUATION_WINDOWS or len(set(windows)) != EVALUATION_WINDOWS:
            raise ValueError("Gate2/3 requires 30 evaluation windows for every seed")
    expected_windows = {
        tuple(row.get("window_id") for row in result if row.get("seed") == seed)
        for seed in R2_SEEDS
    }
    if len(expected_windows) != 1:
        raise ValueError("Gate2/3 seeds must share the same 30 evaluation windows")
    return result


def _calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = [row for row in rows if row.get("split") == "calibration"]
    for seed in R2_SEEDS:
        seed_rows = [row for row in result if row.get("seed") == seed]
        windows = [row.get("window_id") for row in seed_rows]
        if len(seed_rows) != CALIBRATION_WINDOWS or len(set(windows)) != CALIBRATION_WINDOWS:
            raise ValueError("Gate3 requires 30 calibration windows for every seed")
    shared = {
        tuple(row.get("window_id") for row in result if row.get("seed") == seed)
        for seed in R2_SEEDS
    }
    if len(shared) != 1:
        raise ValueError("Gate3 seeds must share the same 30 calibration windows")
    return result


def _hierarchical_bootstrap(
    windows: Sequence[str],
    *,
    value_by_key: Mapping[tuple[str, int], float],
) -> list[float]:
    if len(windows) != 30 or len(set(windows)) != 30:
        raise ValueError("hierarchical bootstrap requires 30 unique windows")
    rng = random.Random(BOOTSTRAP_SEED)
    replicates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_windows = [rng.choice(windows) for _ in windows]
        sampled_seeds = [rng.choice(R2_SEEDS) for _ in R2_SEEDS]
        values = []
        for sampled_window in sampled_windows:
            for sampled_seed in sampled_seeds:
                values.append(value_by_key[(sampled_window, sampled_seed)])
        replicates.append(_mean(values))
    return replicates


def adjudicate_gate2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 2 over the complete 30-window x 3-seed x 3-period vectors."""

    evaluation = _evaluation_rows(rows)
    index = {name: position for position, name in enumerate(R2_NON_DENSE_NAMES)}
    by_key = {(str(row["window_id"]), int(row["seed"])): row for row in evaluation}
    windows = [str(row["window_id"]) for row in evaluation if row["seed"] == R2_SEEDS[0]]
    detector_by_key = {}
    feature_by_key = {}
    hold_by_key = {}
    for key, row in by_key.items():
        detector_improvements = []
        feature_improvements = []
        hold_regrets = []
        for period in _PERIODS:
            transport = index[f"periodic{period}_transport"]
            hold = index[f"periodic{period}_hold"]
            detector_improvements.append(
                float(row["detector_regret"][hold])
                - float(row["detector_regret"][transport])
            )
            feature_improvements.append(
                float(row["feature_mse"][hold])
                - float(row["feature_mse"][transport])
            )
            hold_regrets.append(float(row["detector_regret"][hold]))
        detector_by_key[key] = _mean(detector_improvements)
        feature_by_key[key] = _mean(feature_improvements)
        hold_by_key[key] = _mean(hold_regrets)
    detector_mean = _mean(list(detector_by_key.values()))
    feature_mean = _mean(list(feature_by_key.values()))
    hold_mean = _mean(list(hold_by_key.values()))
    relative = None if hold_mean <= 1e-12 else detector_mean / hold_mean
    detector_ci = _percentile_interval(
        _hierarchical_bootstrap(windows, value_by_key=detector_by_key)
    )
    feature_ci = _percentile_interval(
        _hierarchical_bootstrap(windows, value_by_key=feature_by_key)
    )
    per_seed = {}
    for seed in R2_SEEDS:
        per_seed[str(seed)] = {
            "detector_improvement": _mean(
                [detector_by_key[(window, seed)] for window in windows]
            ),
            "feature_improvement": _mean(
                [feature_by_key[(window, seed)] for window in windows]
            ),
            "hold_detector_regret": _mean(
                [hold_by_key[(window, seed)] for window in windows]
            ),
        }
    hold_only_index = index["hold_only"]
    transport_only_index = index["transport_only"]
    hold_transport_diagnostics = {
        "gate_membership": False,
        "exact_actions_required": True,
        "hold_only_mean_detector_regret": _mean(
            [float(row["detector_regret"][hold_only_index]) for row in evaluation]
        ),
        "transport_only_mean_detector_regret": _mean(
            [float(row["detector_regret"][transport_only_index]) for row in evaluation]
        ),
        "hold_only_mean_feature_mse": _mean(
            [float(row["feature_mse"][hold_only_index]) for row in evaluation]
        ),
        "transport_only_mean_feature_mse": _mean(
            [float(row["feature_mse"][transport_only_index]) for row in evaluation]
        ),
    }
    hard = {
        "pooled_detector_relative_reduction_ge_5pct": relative is not None
        and relative >= 0.05,
        "detector_absolute_ci_lower_gt_0": detector_ci[0] > 0.0,
        "feature_absolute_ci_lower_gt_0": feature_ci[0] > 0.0,
        "each_seed_detector_and_feature_mean_nonnegative": all(
            item["detector_improvement"] >= 0.0
            and item["feature_improvement"] >= 0.0
            for item in per_seed.values()
        ),
    }
    passed = all(hard.values())
    return {
        "schema": "chronotransport-r2-gate2-formal-v1",
        "status": "PASS" if passed else "FAIL",
        "mechanism": passed,
        "evaluation_windows": 30,
        "seed_count": 3,
        "periods": list(_PERIODS),
        "complete_seed_window_period_rows": 30 * 3 * 3,
        "detector_improvement": detector_mean,
        "feature_improvement": feature_mean,
        "hold_detector_regret": hold_mean,
        "detector_relative_reduction": relative,
        "detector_improvement_ci95": detector_ci,
        "feature_improvement_ci95": feature_ci,
        "per_seed": per_seed,
        "hold_only_transport_only_diagnostics": hold_transport_diagnostics,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "outer_unit": "unique_manifested_window",
            "inner_unit": "seed",
            "complete_vector_unit": "three_periods",
        },
        "hard_conditions": hard,
    }


def calibrate_simultaneous_window_offset(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Window-max residual, then exact rank 28 of 30, independently per seed."""

    calibration = _calibration_rows(rows)
    q_conf = {}
    scores_by_seed = {}
    for seed in R2_SEEDS:
        seed_rows = [row for row in calibration if row["seed"] == seed]
        scores = []
        for row in seed_rows:
            residuals = [
                max(float(target) - float(prediction), 0.0)
                for prediction, target in zip(row["q_hat"], row["detector_regret"])
            ]
            scores.append(
                {"window_id": str(row["window_id"]), "window_max_residual": max(residuals)}
            )
        ordered = sorted(item["window_max_residual"] for item in scores)
        q_conf[str(seed)] = float(ordered[CALIBRATION_RANK - 1])
        scores_by_seed[str(seed)] = scores
    return {
        "schema": "chronotransport-r2-gate3-calibration-v1",
        "target_quantile": QUANTILE,
        "calibration_windows_per_seed": CALIBRATION_WINDOWS,
        "candidate_count": len(R2_NON_DENSE_NAMES),
        "order_statistic_rank": CALIBRATION_RANK,
        "residual_reduction": "candidate_max_per_window_before_rank",
        "q_conf_by_seed": q_conf,
        "window_scores_by_seed": scores_by_seed,
        "guarantee": "window_level_simultaneous_marginal_only",
        "selected_conditional_guarantee": False,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            result[order[position]] = rank
        cursor = end
    return result


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    if len(set(first)) < 3 or len(set(second)) < 3:
        raise ValueError("fewer_than_3_distinct_ranks")
    left = _average_ranks(first)
    right = _average_ranks(second)
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    if denominator <= 0.0:
        raise ValueError("fewer_than_3_distinct_ranks")
    return float(numerator / denominator)


def _validate_costs(value: Mapping[str, Any]) -> dict[str, float]:
    expected = {*R2_NON_DENSE_NAMES, "dense"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Gate3 measured candidate costs must contain exact 16+dense names")
    result = {name: _finite_nonnegative(value[name], f"measured p50 cost {name}") for name in expected}
    if any(cost <= 0.0 for cost in result.values()):
        raise ValueError("Gate3 measured p50 costs must be positive")
    return result


def select_risk_constrained_schedule(
    *,
    q_hat: Sequence[float],
    q_conf: float,
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    metadata_valid: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Exact r2 non-dense scheduler with dense safety fallback."""

    if not isinstance(q_hat, Sequence) or len(q_hat) != len(R2_NON_DENSE_NAMES):
        raise ValueError("Gate3 scheduler requires exactly 16 q_hat values")
    costs = _validate_costs(candidate_cost_p50)
    q_conf = _finite_nonnegative(q_conf, "q_conf")
    budget = _finite_nonnegative(budget, "B*")
    if budget <= 0.0:
        raise ValueError("Gate3 B* must be positive")
    if metadata_valid is None:
        metadata_valid = [True] * len(R2_NON_DENSE_NAMES)
    if (
        not isinstance(metadata_valid, Sequence)
        or len(metadata_valid) != len(R2_NON_DENSE_NAMES)
        or any(type(item) is not bool for item in metadata_valid)
    ):
        raise ValueError("Gate3 scheduler metadata-valid vector must contain 16 booleans")
    feasibility = []
    feasible_indices = []
    for index, name in enumerate(R2_NON_DENSE_NAMES):
        prediction = q_hat[index]
        finite_prediction = (
            not isinstance(prediction, bool)
            and isinstance(prediction, (int, float))
            and math.isfinite(float(prediction))
        )
        upper = float(prediction) + q_conf if finite_prediction else None
        feasible = bool(
            finite_prediction
            and metadata_valid[index]
            and costs[name] <= budget
            and upper is not None
            and math.isfinite(upper)
            and upper <= SCHEDULER_EPSILON
        )
        if feasible:
            feasible_indices.append(index)
        feasibility.append(
            {
                "candidate_index": index,
                "schedule": name,
                "requested_p50": costs[name],
                "upper": upper,
                "metadata_valid": bool(metadata_valid[index]),
                "feasible": feasible,
            }
        )
    if feasible_indices:
        selected_index = min(
            feasible_indices,
            key=lambda index: (costs[R2_NON_DENSE_NAMES[index]], index),
        )
        selected = R2_NON_DENSE_NAMES[selected_index]
        return {
            "selected_schedule": selected,
            "selected_candidate_index": selected_index,
            "selected_upper": feasibility[selected_index]["upper"],
            "selected_requested_p50": costs[selected],
            "dense_safety_fallback": False,
            "safety_override_budget_violation": False,
            "candidate_feasibility": feasibility,
        }
    return {
        "selected_schedule": "dense",
        "selected_candidate_index": None,
        "selected_upper": 0.0,
        "selected_requested_p50": costs["dense"],
        "dense_safety_fallback": True,
        "safety_override_budget_violation": costs["dense"] > budget,
        "candidate_feasibility": feasibility,
    }


def _pinball(prediction: float, target: float) -> float:
    residual = target - prediction
    return float(QUANTILE * residual if residual >= 0.0 else (QUANTILE - 1.0) * residual)


def _validate_baseline_constants(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != {str(seed) for seed in R2_SEEDS}:
        raise ValueError("Gate3 requires all three fit-only baseline constant vectors")
    result = {}
    for seed in R2_SEEDS:
        raw = value[str(seed)]
        if not isinstance(raw, Mapping) or set(raw) != set(R2_NON_DENSE_NAMES):
            raise ValueError("Gate3 fit baseline constants must contain exact 16 schedules")
        result[str(seed)] = {
            name: _finite_nonnegative(raw[name], f"fit baseline {seed}/{name}")
            for name in R2_NON_DENSE_NAMES
        }
    return result


def _selection_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    q_conf_by_seed: Mapping[str, float],
    costs: Mapping[str, float],
    budget: float,
    mode: str,
) -> dict[str, Any]:
    selections = []
    prediction_losses = []
    for row in rows:
        offset = q_conf_by_seed[str(row["seed"])] if mode == "calibrated" else 0.0
        selection = select_risk_constrained_schedule(
            q_hat=row["q_hat"],
            q_conf=offset,
            candidate_cost_p50=costs,
            budget=budget,
        )
        if selection["selected_candidate_index"] is None:
            regret = 0.0
            upper = 0.0
        else:
            index = selection["selected_candidate_index"]
            regret = float(row["detector_regret"][index])
            upper = float(row["q_hat"][index]) + offset
        selections.append(
            {
                "seed": int(row["seed"]),
                "window_id": str(row["window_id"]),
                "schedule": selection["selected_schedule"],
                "candidate_index": selection["selected_candidate_index"],
                "regret": regret,
                "upper": upper,
                "cost_p50": selection["selected_requested_p50"],
            }
        )
        for prediction, target in zip(row["q_hat"], row["detector_regret"]):
            prediction_losses.append(_pinball(float(prediction) + offset, float(target)))
    non_dense = [row for row in selections if row["candidate_index"] is not None]
    return {
        "non_dense_selection_rate": len(non_dense) / len(selections),
        "dense_fallback_rate": 1.0 - len(non_dense) / len(selections),
        "mean_selected_regret": _mean([row["regret"] for row in selections]),
        "mean_selected_upper_sharpness": _mean([row["upper"] for row in selections]),
        "mean_pinball_loss_all_candidates": _mean(prediction_losses),
        "mean_selected_cost_p50": _mean([row["cost_p50"] for row in selections]),
        "selections": selections,
    }


def _coverage_lcb(
    windows: Sequence[str], selections: Sequence[Mapping[str, Any]]
) -> float:
    selected_by_window = {
        window: [row for row in selections if row["window_id"] == window]
        for window in windows
    }
    rng = random.Random(BOOTSTRAP_SEED)
    replicates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [rng.choice(windows) for _window in windows]
        rows = [row for window in sampled for row in selected_by_window[window]]
        if not rows:
            replicates.append(0.0)
        else:
            replicates.append(
                sum(row["coverage_margin"] >= 0.0 for row in rows) / len(rows)
            )
    return _one_sided_lower(replicates)


def adjudicate_gate3(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    fit_baseline_constants_by_seed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evaluation = _evaluation_rows(rows)
    calibration = calibrate_simultaneous_window_offset(rows)
    costs = _validate_costs(candidate_cost_p50)
    budget = _finite_nonnegative(budget, "B*")
    baselines = _validate_baseline_constants(fit_baseline_constants_by_seed)
    windows = [str(row["window_id"]) for row in evaluation if row["seed"] == R2_SEEDS[0]]

    rho_by_key: dict[tuple[str, int], float] = {}
    invalid = []
    for row in evaluation:
        try:
            rho_by_key[(str(row["window_id"]), int(row["seed"]))] = _spearman(
                list(map(float, row["q_hat"])),
                list(map(float, row["detector_regret"])),
            )
        except ValueError:
            invalid.append(
                {
                    "seed": int(row["seed"]),
                    "window_id": str(row["window_id"]),
                    "reason": "fewer_than_3_distinct_ranks",
                }
            )
    per_seed_rho = {}
    if invalid:
        pooled_rho = None
        pooled_ci = None
        median_seed = None
    else:
        for seed in R2_SEEDS:
            per_seed_rho[str(seed)] = _mean(
                [rho_by_key[(window, seed)] for window in windows]
            )
        pooled_rho = _mean(list(rho_by_key.values()))
        pooled_ci = _percentile_interval(
            _hierarchical_bootstrap(windows, value_by_key=rho_by_key)
        )
        median_seed = float(median(per_seed_rho.values()))

    calibrated = _selection_diagnostics(
        evaluation,
        q_conf_by_seed=calibration["q_conf_by_seed"],
        costs=costs,
        budget=budget,
        mode="calibrated",
    )
    uncalibrated = _selection_diagnostics(
        evaluation,
        q_conf_by_seed={str(seed): 0.0 for seed in R2_SEEDS},
        costs=costs,
        budget=budget,
        mode="uncalibrated",
    )
    dense = {
        "non_dense_selection_rate": 0.0,
        "dense_fallback_rate": 1.0,
        "mean_selected_regret": 0.0,
        "mean_selected_upper_sharpness": 0.0,
        "mean_pinball_loss_all_candidates": 0.0,
        "mean_selected_cost_p50": costs["dense"],
        "selections": [
            {
                "seed": int(row["seed"]),
                "window_id": str(row["window_id"]),
                "schedule": "dense",
                "candidate_index": None,
                "regret": 0.0,
                "upper": 0.0,
                "cost_p50": costs["dense"],
            }
            for row in evaluation
        ],
    }

    by_row = {(int(row["seed"]), str(row["window_id"])): row for row in evaluation}
    selected = []
    for selection in calibrated["selections"]:
        if selection["candidate_index"] is None:
            continue
        source = by_row[(selection["seed"], selection["window_id"])]
        index = int(selection["candidate_index"])
        margin = float(selection["upper"]) - float(source["detector_regret"][index])
        selected.append({**selection, "coverage_margin": margin})
    support_per_seed = {
        str(seed): sum(row["seed"] == seed for row in selected) for seed in R2_SEEDS
    }
    selected_windows = sorted({row["window_id"] for row in selected})
    pooled_coverage = (
        sum(row["coverage_margin"] >= 0.0 for row in selected) / len(selected)
        if selected
        else 0.0
    )
    per_seed_coverage = {}
    for seed in R2_SEEDS:
        subset = [row for row in selected if row["seed"] == seed]
        per_seed_coverage[str(seed)] = (
            sum(row["coverage_margin"] >= 0.0 for row in subset) / len(subset)
            if subset
            else None
        )
    all_candidate = {}
    for row in evaluation:
        offset = calibration["q_conf_by_seed"][str(row["seed"])]
        all_candidate[(int(row["seed"]), str(row["window_id"]))] = all(
            float(prediction) + offset - float(target) >= 0.0
            for prediction, target in zip(row["q_hat"], row["detector_regret"])
        )
    all_candidate_per_seed = {
        str(seed): sum(all_candidate[(seed, window)] for window in windows) / len(windows)
        for seed in R2_SEEDS
    }
    all_selected_window_values = []
    for window in selected_windows:
        margins = [row["coverage_margin"] for row in selected if row["window_id"] == window]
        all_selected_window_values.append(min(margins) >= 0.0)
    all_selected_rate = (
        sum(all_selected_window_values) / len(all_selected_window_values)
        if all_selected_window_values
        else 0.0
    )
    coverage_lcb = _coverage_lcb(windows, selected)

    predictor_losses = []
    baseline_losses = []
    for row in evaluation:
        seed_constants = baselines[str(row["seed"])]
        for index, name in enumerate(R2_NON_DENSE_NAMES):
            target = float(row["detector_regret"][index])
            predictor_losses.append(_pinball(float(row["q_hat"][index]), target))
            baseline_losses.append(_pinball(seed_constants[name], target))
    predictor_pinball = _mean(predictor_losses)
    baseline_pinball = _mean(baseline_losses)
    pinball_improvement = (
        None
        if baseline_pinball <= 1e-12
        else (baseline_pinball - predictor_pinball) / baseline_pinball
    )

    hard = {
        "all_seed_window_rank_vectors_valid": not invalid,
        "each_seed_non_dense_support_ge_6": all(value >= 6 for value in support_per_seed.values()),
        "pooled_non_dense_support_ge_18": len(selected) >= 18,
        "distinct_selected_windows_ge_10": len(selected_windows) >= 10,
        "pooled_selected_coverage_ge_0_85": pooled_coverage >= 0.85,
        "each_seed_mean_rho_ge_0": not invalid
        and all(value >= 0.0 for value in per_seed_rho.values()),
        "median_seed_mean_rho_ge_0_2": median_seed is not None and median_seed >= 0.2,
        "pooled_rho_hierarchical_ci_lower_gt_0": pooled_ci is not None
        and pooled_ci[0] > 0.0,
        "baseline_pinball_defined": baseline_pinball > 1e-12,
        "pinball_relative_improvement_ge_10pct": pinball_improvement is not None
        and pinball_improvement >= 0.10,
    }
    passed = all(hard.values())
    return {
        "schema": "chronotransport-r2-gate3-formal-v1",
        "status": "PASS" if passed else "FAIL",
        "calibrated_risk_on_frozen_window_protocol": passed,
        "calibration": calibration,
        "scheduler": {
            "candidate_domain": "16_non_dense_only",
            "epsilon": SCHEDULER_EPSILON,
            "budget": budget,
            "budget_source": "measured_full_stack_p50_periodic4_transport",
            "selection": "lowest_requested_measured_p50_then_canonical_order",
            "fallback": "dense_safety",
            "dense_budget_feasible_success": False,
        },
        "selected_support": {
            "per_seed": support_per_seed,
            "pooled": len(selected),
            "distinct_windows": len(selected_windows),
            "windows_without_non_dense_selection": len(windows) - len(selected_windows),
        },
        "coverage": {
            "per_seed_selected": per_seed_coverage,
            "pooled_selected": pooled_coverage,
            "selected_unique_windows": len(selected_windows),
            "window_clustered_one_sided_95_lcb": coverage_lcb,
            "lcb_is_hard_gate": False,
            "status": "OVERCOVERED" if pooled_coverage > 0.95 else (
                "UNDER_TARGET" if pooled_coverage < 0.85 else "IN_RANGE"
            ),
            "all_candidate_simultaneous": {
                "pooled": sum(all_candidate.values()) / len(all_candidate),
                "per_seed": all_candidate_per_seed,
                "seed_window_count": len(all_candidate),
            },
            "all_selected_window": {
                "rate": all_selected_rate,
                "denominator": len(selected_windows),
                "excluded_no_selection_windows": len(windows) - len(selected_windows),
            },
            "selected_conditional_theoretical_guarantee": False,
        },
        "ranking": {
            "unit": "seed_x_evaluation_window_complete_16_candidate_vector",
            "invalid_seed_windows": invalid,
            "per_seed_mean_rho": per_seed_rho,
            "median_seed_mean_rho": median_seed,
            "pooled_rho": pooled_rho,
            "pooled_hierarchical_ci95": pooled_ci,
            "bootstrap": {
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "outer_unit": "unique_manifested_window",
                "inner_unit": "seed",
                "complete_vector_unit": "sixteen_candidates",
            },
        },
        "pinball": {
            "tau": QUANTILE,
            "predictor_mean": predictor_pinball,
            "baseline_mean": baseline_pinball,
            "relative_improvement": pinball_improvement,
            "baseline_source": "fit_only_140x16_schedule_conditioned_rank127",
            "baseline_windows": FIT_BASELINE_WINDOWS,
            "baseline_order_statistic_rank": FIT_BASELINE_RANK,
        },
        "diagnostics": {
            "calibrated": calibrated,
            "uncalibrated": uncalibrated,
            "dense": dense,
        },
        "hard_conditions": hard,
        "intersection_union_gate": True,
        "multiplicity_correction": "none_additional",
    }


def _gate23_report(
    artifact: Mapping[str, Any],
    *,
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    fit_baseline_constants_by_seed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gate2 = adjudicate_gate2(artifact["rows"])
    if gate2["status"] == "PASS":
        gate3 = adjudicate_gate3(
            artifact["rows"],
            candidate_cost_p50=candidate_cost_p50,
            budget=budget,
            fit_baseline_constants_by_seed=fit_baseline_constants_by_seed,
        )
    else:
        gate3 = {
            "schema": "chronotransport-r2-gate3-not-run-v1",
            "status": "NOT_RUN",
            "reason": "Gate2 FAIL stop-chain",
        }
    overall = "PASS" if gate2["status"] == "PASS" and gate3["status"] == "PASS" else "FAIL"
    report: dict[str, Any] = {
        "schema": GATES23_REPORT_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "status": overall,
        "registration_sha256": artifact["registration_sha256"],
        "registration_commit": artifact["registration_commit"],
        "gate1_unlock_artifact_sha256": artifact["gate1_unlock_artifact_sha256"],
        "gates23_replay_artifact_sha256": artifact["artifact_sha256"],
        "phase_bindings": artifact["phase_bindings"],
        "gate2": gate2,
        "gate3": gate3,
        "claim_flags": {
            "oracle_headroom": True,
            "mechanism": gate2["status"] == "PASS",
            "calibrated_risk_on_frozen_window_protocol": gate3["status"] == "PASS",
            "metric_adatad_thumos14_official_full_video": False,
            "latency_slurm_single_device_fixed_stack": False,
            "deploy": False,
            "paper": False,
        },
    }
    report["artifact_sha256"] = canonical_sha256(report)
    return report


def adjudicate_gates23_for_test_only(
    artifact: Mapping[str, Any],
    *,
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    fit_baseline_constants_by_seed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validated = validate_gates23_replay_artifact_for_test_only(artifact)
    return _gate23_report(
        validated,
        candidate_cost_p50=candidate_cost_p50,
        budget=budget,
        fit_baseline_constants_by_seed=fit_baseline_constants_by_seed,
    )


def adjudicate_gates23(
    artifact: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """Formal Gate2/3 report; costs and B* are derived only from Gate1 evidence."""

    registered, validated_gate1 = _validate_formal_gate_context(
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    _, fit_baseline_constants_by_seed = validate_stage_b_phase_markers_static(
        phase_marker_paths,
        registration=registered,
        gate1_unlock=validated_gate1,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    validated = validate_gates23_replay_artifact(
        artifact,
        registration=registered,
        gate1_unlock=validated_gate1,
        phase_marker_paths=phase_marker_paths,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    gate1_result = validated_gate1.get("gate1_result")
    if not isinstance(gate1_result, Mapping):
        raise ValueError("formal Gate2/3 requires the exact Gate1 result")
    costs = gate1_result.get("candidate_cost_p50")
    if not isinstance(costs, Mapping):
        raise ValueError("Gate1 unlock is missing measured candidate p50 costs")
    selected_costs = {
        name: costs[name] for name in (*R2_NON_DENSE_NAMES, "dense")
        if name in costs
    }
    selected_costs = _validate_costs(selected_costs)
    budget = _finite_nonnegative(gate1_result.get("budget"), "Gate1 B*")
    if (
        gate1_result.get("budget_source") != "measured_p50:periodic4_transport"
        or budget != selected_costs["periodic4_transport"]
    ):
        raise ValueError("Gate2/3 B* must be Gate1 measured periodic4_transport p50")
    return _gate23_report(
        validated,
        candidate_cost_p50=selected_costs,
        budget=budget,
        fit_baseline_constants_by_seed=fit_baseline_constants_by_seed,
    )


def validate_gates23_report_for_test_only(
    report: Mapping[str, Any],
    *,
    replay_artifact: Mapping[str, Any],
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    fit_baseline_constants_by_seed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = adjudicate_gates23_for_test_only(
        replay_artifact,
        candidate_cost_p50=candidate_cost_p50,
        budget=budget,
        fit_baseline_constants_by_seed=fit_baseline_constants_by_seed,
    )
    if not isinstance(report, Mapping) or dict(report) != expected:
        raise ValueError("Gate2/3 report differs from its exact recomputation")
    return expected


def validate_gates23_report(
    report: Mapping[str, Any],
    *,
    replay_artifact: Mapping[str, Any],
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """Recompute the formal report from all exact upstream artifacts."""

    _validate_formal_gate_context(
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    expected = adjudicate_gates23(
        replay_artifact,
        registration=registration,
        gate1_unlock=gate1_unlock,
        phase_marker_paths=phase_marker_paths,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if not isinstance(report, Mapping) or dict(report) != expected:
        raise ValueError("Gate2/3 report differs from its exact recomputation")
    return expected


def validate_gate3_unlock_artifact(
    report: Mapping[str, Any],
    *,
    replay_artifact: Mapping[str, Any],
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """The only public Gate-3 unlock contract for Stage C."""

    _validate_formal_gate_context(
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    validated = validate_gates23_report(
        report,
        replay_artifact=replay_artifact,
        registration=registration,
        gate1_unlock=gate1_unlock,
        phase_marker_paths=phase_marker_paths,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if (
        validated["status"] != "PASS"
        or validated["gate2"].get("status") != "PASS"
        or validated["gate2"].get("mechanism") is not True
        or validated["gate3"].get("status") != "PASS"
        or validated["gate3"].get(
            "calibrated_risk_on_frozen_window_protocol"
        )
        is not True
        or validated["claim_flags"]
        .get("calibrated_risk_on_frozen_window_protocol")
        is not True
    ):
        raise ValueError("Stage C requires an exact PASS Gate-3 unlock artifact")
    return validated


def build_gates23_terminal_marker(
    *,
    report: Mapping[str, Any],
    replay_artifact: Mapping[str, Any],
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path | str,
    repository_root: Path | str,
    registration_commit: str,
    registration_relpath: str,
    report_path: Path | str,
) -> dict[str, Any]:
    """Derive the authoritative terminal solely from an exact recomputed report."""

    registration_commit = _require_commit(registration_commit, "registration commit R")
    _validate_formal_gate_context(
        registration=registration,
        gate1_unlock=gate1_unlock,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    base = _path_without_symlink_components(
        FORMAL_OUTPUT_BASE, label="formal Gate2/3 output base"
    )
    expected_report_path = (
        base / registration_commit / "shared" / "gates23" / "gates23_report.json"
    )
    exact_report_path = _require_regular_input(
        report_path, label="formal Gate2/3 report"
    )
    if exact_report_path != expected_report_path:
        raise ValueError("formal Gate2/3 terminal requires the canonical R report path")
    serialized_report = load_exact_canonical_json(
        exact_report_path, label="formal Gate2/3 report"
    )
    if not isinstance(report, Mapping) or serialized_report != dict(report):
        raise ValueError("formal Gate2/3 report mapping differs from its exact file bytes")
    validated = validate_gates23_report(
        serialized_report,
        replay_artifact=replay_artifact,
        registration=registration,
        gate1_unlock=gate1_unlock,
        phase_marker_paths=phase_marker_paths,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if validated.get("status") == "PASS":
        validate_gate3_unlock_artifact(
            validated,
            replay_artifact=replay_artifact,
            registration=registration,
            gate1_unlock=gate1_unlock,
            phase_marker_paths=phase_marker_paths,
            gate1_unlock_path=gate1_unlock_path,
            repository_root=repository_root,
            registration_commit=registration_commit,
            registration_relpath=registration_relpath,
        )
        terminal_state = "SUCCESS"
        reason = "Gate2 and Gate3 passed"
    elif validated.get("status") == "FAIL":
        terminal_state = "FAIL"
        reason = "science gate failed; downstream Stage C remains stopped"
    else:
        raise ValueError("formal Gate2/3 report must terminate in PASS or FAIL")
    report_sha256 = _require_sha256(
        validated.get("artifact_sha256"), "Gate2/3 report SHA-256"
    )
    marker: dict[str, Any] = {
        "schema": GATES23_TERMINAL_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "terminal_state": terminal_state,
        "registration_commit": registration_commit,
        "report_path": str(exact_report_path),
        "report_sha256": report_sha256,
        "report_file_sha256": _file_sha256(exact_report_path),
        "reason": reason,
    }
    marker["artifact_sha256"] = canonical_sha256(marker)
    return marker


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "R2_SEEDS",
    "GATES23_REPLAY_FORMAL_SCHEMA",
    "GATES23_REPORT_SCHEMA",
    "build_gates23_replay_artifact_for_test_only",
    "validate_gates23_replay_artifact_for_test_only",
    "run_registered_gates23_replay",
    "validate_gates23_replay_artifact",
    "validate_stage_b_phase_markers_static",
    "calibrate_simultaneous_window_offset",
    "select_risk_constrained_schedule",
    "adjudicate_gate2",
    "adjudicate_gate3",
    "adjudicate_gates23_for_test_only",
    "adjudicate_gates23",
    "validate_gates23_report_for_test_only",
    "validate_gates23_report",
    "validate_gate3_unlock_artifact",
    "build_gates23_terminal_marker",
    "load_exact_canonical_json",
]
