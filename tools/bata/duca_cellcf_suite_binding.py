from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from tools.bata.duca_cellcf_protocol import (
    LEGACY_EXPOSURE132_COMMITS,
    protocol_for_name,
)
from tools.bata.duca_cellcf_training import (
    DUCA_P0_TRAINING_AUDIT_SCHEMA,
    VARIANTS,
    canonical_sha256,
    sha256_file,
)


SCHEMA = "duca_cellcf_suite_manifest_v1"
POST_RUN_SCHEMA = "duca_cellcf_post_run_evidence_v1"
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", normalized) is not None,
        f"{label} is not a SHA256",
    )
    return normalized


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _validate_embedded_hash(
    payload: Mapping[str, Any], key: str, label: str
) -> None:
    observed = payload.get(key)
    unsigned = dict(payload)
    unsigned.pop(key, None)
    _require(
        observed == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _resolve_profile(
    payload: Mapping[str, Any],
    *,
    expected_profile: str,
    expected_commit: str,
    label: str,
) -> str:
    observed = payload.get("training_profile")
    if observed is None:
        _require(
            expected_profile == "exposure132"
            and expected_commit in LEGACY_EXPOSURE132_COMMITS,
            f"{label} lacks a training profile outside the one audited legacy exposure132 commit",
        )
        return "exposure132"
    _require(observed == expected_profile, f"{label} training profile mismatch")
    return str(observed)


def _reopen_hashed_record(
    record: Mapping[str, Any], label: str
) -> tuple[Path, str]:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    _require(path.is_file(), f"{label} is missing: {path}")
    expected = _require_sha256(record.get("sha256"), f"{label} hash")
    observed = sha256_file(path)
    _require(observed == expected, f"{label} hash mismatch")
    return path, observed


def _validate_terminal_audit(
    post_run: Mapping[str, Any],
    *,
    variant: str,
    expected_commit: str,
    expected_profile: str,
    expected_seed: int,
    protocol_sha256: str,
    order_sha256: str,
    gate_sha256: str,
    pilot_sha256: str,
) -> dict[str, Any]:
    audit_path, audit = _load_json(
        post_run.get("training_audit_path", ""),
        f"{variant} terminal training audit",
    )
    _require(
        post_run.get("training_audit_sha256") == sha256_file(audit_path),
        f"{variant} terminal training audit hash mismatch",
    )
    _require(
        audit.get("schema_version") == DUCA_P0_TRAINING_AUDIT_SCHEMA,
        f"{variant} terminal training audit schema mismatch",
    )
    _validate_embedded_hash(
        audit,
        "audit_sha256",
        f"{variant} terminal training audit",
    )
    _require(
        audit.get("status") == "complete",
        f"{variant} terminal training audit is not complete",
    )
    _resolve_profile(
        audit,
        expected_profile=expected_profile,
        expected_commit=expected_commit,
        label=f"{variant} terminal training audit",
    )
    protocol = protocol_for_name(expected_profile)
    expected_updates = protocol.expected_successful_optimizer_updates
    expected = {
        "variant": variant,
        "git_commit": expected_commit,
        "seed": expected_seed,
        "protocol_sha256": protocol_sha256,
        "ordered_exposure_sha256": order_sha256,
        "real_loader_gate_sha256": gate_sha256,
        "ddp_pilot_sha256": pilot_sha256,
        "expected_successful_optimizer_updates": expected_updates,
        "last_completed_epoch": protocol.terminal_epoch,
        "epochs_completed": protocol.end_epoch,
        "scheduler_last_epoch": expected_updates,
        "selector_schedule_step": expected_updates,
    }
    for key, value in expected.items():
        _require(
            audit.get(key) == value,
            f"{variant} terminal training audit mismatch: {key}",
        )
    records = audit.get("epoch_records")
    _require(
        isinstance(records, list)
        and [int(record.get("epoch", -1)) for record in records]
        == list(range(protocol.end_epoch)),
        f"{variant} terminal epoch records are incomplete",
    )
    counters = audit.get("update_audit")
    _require(
        isinstance(counters, Mapping),
        f"{variant} terminal update audit is missing",
    )
    for key in (
        "attempted_batches",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
        "duca_schedule_updates",
    ):
        _require(
            int(counters.get(key, -1)) == expected_updates,
            f"{variant} terminal update counter mismatch: {key}",
        )
    skipped = int(counters.get("amp_skipped_attempts", -1))
    _require(skipped >= 0, f"{variant} AMP skip counter is invalid")
    _require(
        int(counters.get("optimizer_attempts", -1))
        == expected_updates + skipped,
        f"{variant} optimizer attempt accounting mismatch",
    )
    _require(
        int(counters.get("replay_exhaustions", -1)) == 0,
        f"{variant} exhausted AMP replay",
    )
    _require(
        int(counters.get("forced_amp_overflow_attempts", -1)) == 0,
        f"{variant} formal training injected AMP overflow",
    )
    slurm_job_id = str(audit.get("slurm_job_id", ""))
    _require(
        slurm_job_id.isdigit() and int(slurm_job_id) > 0,
        f"{variant} terminal audit lacks a valid Slurm job id",
    )
    return {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "slurm_job_id": int(slurm_job_id),
        "amp_skipped_attempts": skipped,
        "successful_optimizer_updates": expected_updates,
    }


def load_suite_aggregate_binding(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_commit: str,
    expected_profile: str,
    post_run_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
        "expected commit is invalid",
    )
    _require(
        set(post_run_paths) == set(VARIANTS),
        "suite binding requires exactly the three frozen post-run paths",
    )
    aggregate_path, aggregate = _load_json(path, "CellCF aggregate evidence")
    aggregate_sha = sha256_file(aggregate_path)
    _require(
        aggregate_sha
        == _require_sha256(expected_sha256, "CellCF aggregate evidence hash"),
        "CellCF aggregate evidence SHA256 mismatch",
    )
    _require(
        aggregate.get("schema") == SCHEMA and aggregate.get("ok") is True,
        "CellCF aggregate evidence schema/status mismatch",
    )
    _require(
        aggregate.get("status") in {"runs_complete_cost_pending", "complete"},
        "CellCF aggregate evidence does not contain three completed runs",
    )
    _require(
        aggregate.get("task") == "offline_temporal_action_detection",
        "CellCF aggregate task mismatch",
    )
    _require(
        aggregate.get("git_commit") == expected_commit,
        "CellCF aggregate commit mismatch",
    )
    _require(
        aggregate.get("git_tree_clean") is True,
        "CellCF aggregate was not produced from a clean tracked tree",
    )
    _resolve_profile(
        aggregate,
        expected_profile=expected_profile,
        expected_commit=expected_commit,
        label="CellCF aggregate evidence",
    )
    seed = aggregate.get("seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "CellCF aggregate seed is invalid",
    )
    _require(
        tuple(aggregate.get("variant_order", ())) == tuple(VARIANTS),
        "CellCF aggregate variant order mismatch",
    )
    protocol_sha = _require_sha256(
        aggregate.get("shared_protocol_sha256"),
        "CellCF aggregate shared protocol hash",
    )
    _require(
        canonical_sha256(aggregate.get("shared_protocol")) == protocol_sha,
        "CellCF aggregate shared protocol payload mismatch",
    )
    order_sha = _require_sha256(
        aggregate.get("ordered_exposure_sha256"),
        "CellCF aggregate exposure-order hash",
    )
    gate_path, gate_sha = _reopen_hashed_record(
        aggregate.get("real_loader_gate", {}),
        "CellCF aggregate real-loader gate",
    )
    pilot_path, pilot_sha = _reopen_hashed_record(
        aggregate.get("ddp_pilot", {}),
        "CellCF aggregate DDP pilot",
    )

    completed = aggregate.get("completed_runs")
    _require(
        isinstance(completed, Mapping) and set(completed) == set(VARIANTS),
        "CellCF aggregate completed-run set mismatch",
    )
    variant_configs = {
        str(item.get("name")): item
        for item in aggregate.get("variants", ())
        if isinstance(item, Mapping)
    }
    _require(
        set(variant_configs) == set(VARIANTS),
        "CellCF aggregate config set mismatch",
    )

    post_run_records: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        expected_path = Path(post_run_paths[variant]).expanduser().resolve()
        record = completed[variant]
        _require(
            isinstance(record, Mapping),
            f"{variant} aggregate completed-run record is invalid",
        )
        record_path, record_sha = _reopen_hashed_record(
            record,
            f"{variant} post-run evidence",
        )
        _require(
            record_path == expected_path,
            f"{variant} aggregate binds another post-run path",
        )
        _, post_run = _load_json(record_path, f"{variant} post-run evidence")
        _require(
            post_run.get("schema") == POST_RUN_SCHEMA
            and post_run.get("ok") is True,
            f"{variant} post-run schema/status mismatch",
        )
        _validate_embedded_hash(
            post_run,
            "artifact_chain_sha256",
            f"{variant} post-run evidence",
        )
        _resolve_profile(
            post_run,
            expected_profile=expected_profile,
            expected_commit=expected_commit,
            label=f"{variant} post-run evidence",
        )
        expected_post = {
            "variant": variant,
            "git_commit": expected_commit,
            "seed": seed,
            "protocol_sha256": protocol_sha,
            "ordered_exposure_sha256": order_sha,
            "real_loader_gate_sha256": gate_sha,
            "ddp_pilot_sha256": pilot_sha,
            "config_sha256": variant_configs[variant].get("config_sha256"),
            "resolved_config_sha256": variant_configs[variant].get(
                "resolved_config_sha256"
            ),
        }
        for key, value in expected_post.items():
            _require(
                post_run.get(key) == value,
                f"{variant} post-run suite binding mismatch: {key}",
            )
        metrics = post_run.get("metrics")
        _require(
            isinstance(metrics, Mapping)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in metrics.values()
            ),
            f"{variant} post-run metrics are invalid",
        )
        _require(
            record.get("metrics") == metrics,
            f"{variant} aggregate metrics differ from post-run evidence",
        )
        _require(
            record.get("checkpoint_path") == post_run.get("checkpoint_path")
            and record.get("checkpoint_sha256")
            == post_run.get("checkpoint_sha256"),
            f"{variant} aggregate checkpoint differs from post-run evidence",
        )
        audit = _validate_terminal_audit(
            post_run,
            variant=variant,
            expected_commit=expected_commit,
            expected_profile=expected_profile,
            expected_seed=int(seed),
            protocol_sha256=protocol_sha,
            order_sha256=order_sha,
            gate_sha256=gate_sha,
            pilot_sha256=pilot_sha,
        )
        post_run_records[variant] = {
            "path": str(record_path),
            "sha256": record_sha,
            "payload": post_run,
            "training_audit": audit,
        }

    return {
        "schema": SCHEMA,
        "path": str(aggregate_path),
        "sha256": aggregate_sha,
        "git_commit": expected_commit,
        "training_profile": expected_profile,
        "seed": int(seed),
        "variant_order": list(VARIANTS),
        "shared_protocol_sha256": protocol_sha,
        "ordered_exposure_sha256": order_sha,
        "real_loader_gate": {"path": str(gate_path), "sha256": gate_sha},
        "ddp_pilot": {"path": str(pilot_path), "sha256": pilot_sha},
        "post_runs": post_run_records,
    }


__all__ = [
    "LEGACY_EXPOSURE132_COMMITS",
    "POST_RUN_SCHEMA",
    "SCHEMA",
    "load_suite_aggregate_binding",
]
