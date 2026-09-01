from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.bata.duca_cellcf_protocol import (
    LEGACY_EXPOSURE132_COMMITS,
)
from tools.bata.duca_cellcf_suite_binding import (
    load_suite_aggregate_binding,
)
from tools.bata.duca_cellcf_training import (
    DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
    DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
    DUCA_P0_TRAINING_AUDIT_SCHEMA,
    VARIANTS,
    canonical_sha256,
    sha256_file,
)
from tools.bata.duca_p0_evaluation import (
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
)


SCHEMA = "duca_cellcf_fixed_convergence_trajectory_v1"
EVALUATION_SCHEMA = "duca_cellcf_terminal_evaluation_v1"
POST_RUN_SCHEMA = "duca_cellcf_post_run_evidence_v1"
VARIANT_RECEIPT_SCHEMA = "duca_cellcf_convergence_variant_receipt_v1"
FIXED_EPOCHS = (59, 89, 131)
PRIMARY_EPOCH = 131
PRIMARY_STATE_KEY = "state_dict_ema"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _validate_embedded_hash(
    payload: Mapping[str, Any], hash_key: str, label: str
) -> None:
    observed = payload.get(hash_key)
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    _require(
        isinstance(observed, str) and observed == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _validate_metrics(
    reported: Any, recomputed: Mapping[str, Any], label: str
) -> dict[str, float]:
    _require(isinstance(reported, Mapping), f"{label} metrics are missing")
    expected_keys = [
        "average_mAP",
        *(f"mAP@{threshold}" for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)),
    ]
    output: dict[str, float] = {}
    for key in expected_keys:
        reported_value = reported.get(key)
        recomputed_value = recomputed.get(key)
        _require(
            isinstance(reported_value, (int, float))
            and math.isfinite(float(reported_value)),
            f"{label} metric {key} is invalid",
        )
        _require(
            isinstance(recomputed_value, (int, float))
            and math.isfinite(float(recomputed_value)),
            f"{label} recomputed metric {key} is invalid",
        )
        _require(
            math.isclose(
                float(reported_value),
                float(recomputed_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{label} metric {key} differs from official recomputation",
        )
        output[key] = float(recomputed_value)
    return output


def _expected_checkpoint_path(post_run: Mapping[str, Any], epoch: int) -> Path:
    terminal = Path(str(post_run.get("checkpoint_path", ""))).expanduser().resolve()
    _require(terminal.is_file(), f"terminal checkpoint is missing: {terminal}")
    _require(
        terminal.name == f"epoch_{PRIMARY_EPOCH}.pth",
        "terminal checkpoint path does not match the frozen epoch",
    )
    return terminal.with_name(f"epoch_{epoch}.pth")


def _inspect_checkpoint_payload(
    checkpoint_path: Path,
    expected_metadata: Mapping[str, Any],
    epoch: int,
    expected_updates: int,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    _require(isinstance(checkpoint, Mapping), "checkpoint is not a mapping")
    _require(int(checkpoint.get("epoch", -1)) == epoch, "checkpoint epoch mismatch")
    for key in (
        "state_dict_ema",
        "scheduler",
        "experiment_metadata",
        "rng_state",
    ):
        _require(key in checkpoint, f"checkpoint is missing {key}")
    _require(
        checkpoint["experiment_metadata"] == expected_metadata,
        "checkpoint metadata differs from its sidecar",
    )
    _require(
        int(checkpoint["scheduler"].get("last_epoch", -1)) == expected_updates,
        "checkpoint scheduler update mismatch",
    )
    selector_steps: dict[str, int] = {}
    for state_key in ("state_dict", "state_dict_ema"):
        state = checkpoint.get(state_key)
        _require(isinstance(state, Mapping), f"checkpoint is missing {state_key}")
        matches = [
            value
            for key, value in state.items()
            if key.endswith(
                ".frame_selector._loss_weight_schedule_step"
            )
        ]
        _require(
            len(matches) == 1,
            f"{state_key} selector schedule is missing or ambiguous",
        )
        selector_steps[state_key] = int(matches[0].detach().cpu().item())
        _require(
            selector_steps[state_key] == expected_updates,
            f"{state_key} selector schedule update mismatch",
        )
    del checkpoint
    return {
        "payload_reopened": True,
        "epoch": epoch,
        "scheduler_last_epoch": expected_updates,
        "selector_schedule_steps": selector_steps,
        "embedded_metadata_exact": True,
    }


def _validate_checkpoint_sidecar(
    checkpoint: Path,
    *,
    epoch: int,
    variant: str,
    expected_commit: str,
    suite_binding: Mapping[str, Any],
    post_run: Mapping[str, Any],
    checkpoint_inspector: Callable[
        [Path, Mapping[str, Any], int, int], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    _require(checkpoint.is_file(), f"{variant} epoch {epoch} checkpoint is missing")
    sidecar_path, sidecar = _load_json(
        f"{checkpoint}.metadata.json",
        f"{variant} epoch {epoch} checkpoint sidecar",
    )
    _require(
        sidecar.get("schema_version") == DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
        f"{variant} epoch {epoch} sidecar schema mismatch",
    )
    _validate_embedded_hash(
        sidecar,
        "sidecar_sha256",
        f"{variant} epoch {epoch} checkpoint sidecar",
    )
    checkpoint_sha = sha256_file(checkpoint)
    _require(
        sidecar.get("checkpoint_sha256") == checkpoint_sha,
        f"{variant} epoch {epoch} checkpoint hash differs from sidecar",
    )
    metadata = sidecar.get("experiment_metadata")
    _require(
        isinstance(metadata, Mapping),
        f"{variant} epoch {epoch} checkpoint metadata is missing",
    )
    _require(
        metadata.get("schema_version") == DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
        f"{variant} epoch {epoch} checkpoint metadata schema mismatch",
    )
    _validate_embedded_hash(
        metadata,
        "metadata_sha256",
        f"{variant} epoch {epoch} checkpoint metadata",
    )
    audit = metadata.get("training_audit")
    _require(
        isinstance(audit, Mapping),
        f"{variant} epoch {epoch} training audit is missing",
    )
    _require(
        audit.get("schema_version") == DUCA_P0_TRAINING_AUDIT_SCHEMA,
        f"{variant} epoch {epoch} training audit schema mismatch",
    )
    _validate_embedded_hash(
        audit,
        "audit_sha256",
        f"{variant} epoch {epoch} training audit",
    )
    _require(audit.get("git_commit") == expected_commit, f"{variant} epoch {epoch} audit commit mismatch")
    _require(audit.get("variant") == variant, f"{variant} epoch {epoch} audit variant mismatch")
    observed_profile = audit.get("training_profile")
    if observed_profile is None:
        _require(
            expected_commit in LEGACY_EXPOSURE132_COMMITS
            and suite_binding.get("training_profile") == "exposure132",
            f"{variant} epoch {epoch} audit lacks an admissible training profile",
        )
    else:
        _require(
            observed_profile == suite_binding.get("training_profile"),
            f"{variant} epoch {epoch} audit training profile mismatch",
        )
    terminal_audit = suite_binding["post_runs"][variant]["training_audit"]
    expected_bindings = {
        "seed": suite_binding["seed"],
        "slurm_job_id": str(terminal_audit["slurm_job_id"]),
        "protocol_sha256": suite_binding["shared_protocol_sha256"],
        "ordered_exposure_sha256": suite_binding[
            "ordered_exposure_sha256"
        ],
        "real_loader_gate_sha256": suite_binding["real_loader_gate"][
            "sha256"
        ],
        "ddp_pilot_sha256": suite_binding["ddp_pilot"]["sha256"],
        "source_config_sha256": post_run["config_sha256"],
        "resolved_config_sha256": post_run["resolved_config_sha256"],
        "runtime_config_sha256": post_run["runtime_config_sha256"],
        "evaluation_annotation_sha256": post_run[
            "evaluation_annotation_sha256"
        ],
        "evaluation_class_map_sha256": post_run[
            "evaluation_class_map_sha256"
        ],
        "evaluation_config_sha256": post_run["evaluation_config_sha256"],
    }
    for key, value in expected_bindings.items():
        _require(
            audit.get(key) == value,
            f"{variant} epoch {epoch} audit suite binding mismatch: {key}",
        )
    _require(
        int(audit.get("last_completed_epoch", -1)) == epoch,
        f"{variant} epoch {epoch} audit is not checkpoint-aligned",
    )
    expected_status = "complete" if epoch == PRIMARY_EPOCH else "in_progress"
    _require(
        audit.get("status") == expected_status,
        f"{variant} epoch {epoch} audit status mismatch",
    )
    expected_updates = (epoch + 1) * 100
    _require(
        int(audit.get("scheduler_last_epoch", -1)) == expected_updates
        and int(audit.get("selector_schedule_step", -1)) == expected_updates
        and int(
            audit.get("update_audit", {}).get(
                "successful_optimizer_updates", -1
            )
        )
        == expected_updates,
        f"{variant} epoch {epoch} successful-update state mismatch",
    )
    checkpoint_contract = dict(
        checkpoint_inspector(
            checkpoint,
            sidecar["experiment_metadata"],
            epoch,
            expected_updates,
        )
    )
    _require(
        checkpoint_contract.get("payload_reopened") is True,
        f"{variant} epoch {epoch} checkpoint was not reopened",
    )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sidecar_path": str(sidecar_path),
        "checkpoint_sidecar_sha256": sha256_file(sidecar_path),
        "successful_optimizer_updates": expected_updates,
        "scheduler_last_epoch": int(audit.get("scheduler_last_epoch", -1)),
        "selector_schedule_step": int(audit.get("selector_schedule_step", -1)),
        "checkpoint_payload_contract": checkpoint_contract,
    }


def _validate_evaluation(
    variant: str,
    epoch: int,
    evaluation_path: str | Path,
    *,
    expected_commit: str,
    suite_binding: Mapping[str, Any],
    post_run: Mapping[str, Any],
    recompute=recompute_official_map,
    checkpoint_inspector: Callable[
        [Path, Mapping[str, Any], int, int], Mapping[str, Any]
    ] = _inspect_checkpoint_payload,
) -> dict[str, Any]:
    resolved, evaluation = _load_json(
        evaluation_path, f"{variant} epoch {epoch} evaluation"
    )
    _require(
        evaluation.get("schema_version") == EVALUATION_SCHEMA,
        f"{variant} epoch {epoch} evaluation schema mismatch",
    )
    _validate_embedded_hash(
        evaluation,
        "evaluation_sha256",
        f"{variant} epoch {epoch} evaluation",
    )
    _require(
        evaluation.get("git_commit") == expected_commit,
        f"{variant} epoch {epoch} evaluation commit mismatch",
    )
    for key in (
        "config_sha256",
        "resolved_config_sha256",
        "evaluation_annotation_sha256",
        "evaluation_class_map_sha256",
        "evaluation_config_sha256",
    ):
        _require(
            evaluation.get(key) == post_run.get(key),
            f"{variant} epoch {epoch} evaluation binding mismatch: {key}",
        )
    _require(
        int(evaluation.get("checkpoint_epoch", -1)) == epoch,
        f"{variant} epoch {epoch} evaluation checkpoint epoch mismatch",
    )
    _require(
        evaluation.get("checkpoint_state_key") == PRIMARY_STATE_KEY,
        f"{variant} epoch {epoch} evaluation did not use EMA",
    )
    runtime_config_sha = evaluation.get("runtime_config_sha256")
    _require(
        isinstance(runtime_config_sha, str)
        and len(runtime_config_sha) == 64
        and all(character in "0123456789abcdef" for character in runtime_config_sha),
        f"{variant} epoch {epoch} evaluation runtime hash is invalid",
    )
    _require(
        evaluation.get("evaluator") == post_run.get("evaluator")
        == official_evaluator_identity(),
        f"{variant} epoch {epoch} evaluator identity mismatch",
    )
    for path_key, hash_key, label in (
        (
            "evaluation_annotation_path",
            "evaluation_annotation_sha256",
            "annotation",
        ),
        (
            "evaluation_class_map_path",
            "evaluation_class_map_sha256",
            "class map",
        ),
    ):
        artifact = Path(str(evaluation.get(path_key, ""))).expanduser().resolve()
        _require(
            artifact.is_file(),
            f"{variant} epoch {epoch} evaluation {label} is missing",
        )
        _require(
            evaluation.get(hash_key) == sha256_file(artifact),
            f"{variant} epoch {epoch} evaluation {label} hash mismatch",
        )
    checkpoint = _expected_checkpoint_path(post_run, epoch)
    _require(
        Path(str(evaluation.get("checkpoint_path", ""))).expanduser().resolve()
        == checkpoint,
        f"{variant} epoch {epoch} evaluation used another checkpoint",
    )
    checkpoint_binding = _validate_checkpoint_sidecar(
        checkpoint,
        epoch=epoch,
        variant=variant,
        expected_commit=expected_commit,
        suite_binding=suite_binding,
        post_run=post_run,
        checkpoint_inspector=checkpoint_inspector,
    )
    _require(
        evaluation.get("checkpoint_sha256")
        == checkpoint_binding["checkpoint_sha256"],
        f"{variant} epoch {epoch} evaluation checkpoint hash mismatch",
    )
    prediction = Path(str(evaluation.get("prediction_path", ""))).expanduser().resolve()
    _require(prediction.is_file(), f"{variant} epoch {epoch} prediction is missing")
    _require(
        evaluation.get("prediction_sha256") == sha256_file(prediction),
        f"{variant} epoch {epoch} prediction hash mismatch",
    )
    normalized_evaluation_config = normalize_evaluation_config(
        evaluation.get("evaluation_config")
    )
    _require(
        canonical_sha256(normalized_evaluation_config)
        == evaluation.get("evaluation_config_sha256"),
        f"{variant} epoch {epoch} evaluation config hash mismatch",
    )
    recomputed = recompute(prediction, normalized_evaluation_config)
    metrics = _validate_metrics(
        evaluation.get("metrics"),
        recomputed.get("metrics", {}),
        f"{variant} epoch {epoch}",
    )
    _require(
        int(evaluation.get("result_count", -1))
        == int(recomputed.get("result_count", -2)),
        f"{variant} epoch {epoch} result count mismatch",
    )
    _require(
        int(evaluation.get("video_count", -1))
        == int(recomputed.get("video_count", -2)),
        f"{variant} epoch {epoch} video count mismatch",
    )
    if epoch == PRIMARY_EPOCH:
        terminal_path = Path(
            str(post_run.get("terminal_evaluation_path", ""))
        ).expanduser().resolve()
        _require(
            resolved == terminal_path,
            f"{variant} epoch {epoch} must reuse sealed terminal evaluation",
        )
        _require(
            sha256_file(resolved) == post_run.get("terminal_evaluation_sha256"),
            f"{variant} sealed terminal evaluation hash mismatch",
        )
    return {
        "variant": variant,
        "epoch": epoch,
        "role": "primary_terminal" if epoch == PRIMARY_EPOCH else "diagnostic_fixed",
        "scheduler_horizon_epochs": 132,
        "official_60_epoch_run": False,
        "evaluation_path": str(resolved),
        "evaluation_sha256": sha256_file(resolved),
        "evaluation_runtime_config_sha256": runtime_config_sha,
        "prediction_path": str(prediction),
        "prediction_sha256": sha256_file(prediction),
        **checkpoint_binding,
        "metrics": metrics,
    }


def _validate_variant_receipts(
    receipt_paths: Mapping[str, str | Path],
    *,
    expected_commit: str,
    expected_evidence_commit: str,
    suite_binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    _require(
        set(receipt_paths) == set(VARIANTS),
        "variant receipts must cover exactly the three frozen variants",
    )
    rows_by_variant = {
        variant: {
            int(row["epoch"]): row
            for row in rows
            if row.get("variant") == variant
        }
        for variant in VARIANTS
    }
    records: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        resolved, receipt = _load_json(
            receipt_paths[variant], f"{variant} convergence receipt"
        )
        _require(
            receipt.get("schema") == VARIANT_RECEIPT_SCHEMA
            and receipt.get("ok") is True,
            f"{variant} convergence receipt is invalid",
        )
        _validate_embedded_hash(
            receipt,
            "receipt_sha256",
            f"{variant} convergence receipt",
        )
        expected_fields = {
            "task": "offline_temporal_action_detection",
            "git_commit": expected_commit,
            "training_profile": "exposure132",
            "variant": variant,
            "seed": suite_binding["seed"],
        }
        expected_fields["evidence_git_commit"] = expected_evidence_commit
        for key, value in expected_fields.items():
            _require(
                receipt.get(key) == value,
                f"{variant} convergence receipt mismatch: {key}",
            )
        variant_rows = rows_by_variant[variant]
        _require(
            set(variant_rows) == set(FIXED_EPOCHS),
            f"{variant} convergence rows are incomplete",
        )
        expected_artifacts = [
            (
                Path(suite_binding["post_runs"][variant]["path"]).resolve(),
                suite_binding["post_runs"][variant]["sha256"],
            ),
            *[
                (
                    Path(str(variant_rows[epoch]["evaluation_path"])).resolve(),
                    variant_rows[epoch]["evaluation_sha256"],
                )
                for epoch in FIXED_EPOCHS
            ],
        ]
        artifacts = receipt.get("artifacts")
        _require(
            isinstance(artifacts, list)
            and len(artifacts) == len(expected_artifacts),
            f"{variant} convergence receipt artifact list mismatch",
        )
        for index, (artifact, expected) in enumerate(
            zip(artifacts, expected_artifacts)
        ):
            _require(
                isinstance(artifact, Mapping),
                f"{variant} convergence receipt artifact {index} is invalid",
            )
            expected_path, expected_sha = expected
            observed_path = Path(
                str(artifact.get("path", ""))
            ).expanduser().resolve()
            _require(
                observed_path == expected_path,
                f"{variant} convergence receipt artifact {index} path mismatch",
            )
            _require(
                artifact.get("sha256") == expected_sha
                and sha256_file(observed_path) == expected_sha,
                f"{variant} convergence receipt artifact {index} hash mismatch",
            )
        expected_runtime_hashes = {
            str(epoch): variant_rows[epoch][
                "evaluation_runtime_config_sha256"
            ]
            for epoch in FIXED_EPOCHS
        }
        _require(
            receipt.get("evaluation_runtime_config_sha256")
            == expected_runtime_hashes,
            f"{variant} convergence receipt runtime config hashes mismatch",
        )
        records[variant] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "receipt_sha256": receipt["receipt_sha256"],
            "evaluation_runtime_config_sha256": expected_runtime_hashes,
            "artifacts": artifacts,
        }
    return records


def build_convergence_evidence(
    *,
    expected_commit: str,
    expected_evidence_commit: str,
    suite_aggregate_path: str | Path,
    suite_aggregate_sha256: str,
    post_run_paths: Mapping[str, str | Path],
    variant_receipt_paths: Mapping[str, str | Path],
    evaluation_paths: Mapping[tuple[str, int], str | Path],
    recompute=recompute_official_map,
    checkpoint_inspector: Callable[
        [Path, Mapping[str, Any], int, int], Mapping[str, Any]
    ] = _inspect_checkpoint_payload,
) -> dict[str, Any]:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected_evidence_commit) is not None,
        "expected evidence commit is invalid",
    )
    _require(
        set(post_run_paths) == set(VARIANTS),
        "post-run evidence must cover exactly the three frozen variants",
    )
    expected_evaluations = {
        (variant, epoch) for variant in VARIANTS for epoch in FIXED_EPOCHS
    }
    _require(
        set(evaluation_paths) == expected_evaluations,
        "evaluations must cover exactly variants x fixed epochs 59/89/131",
    )
    suite_binding = load_suite_aggregate_binding(
        suite_aggregate_path,
        suite_aggregate_sha256,
        expected_commit=expected_commit,
        expected_profile="exposure132",
        post_run_paths=post_run_paths,
    )
    post_runs = {
        variant: suite_binding["post_runs"][variant]["payload"]
        for variant in VARIANTS
    }
    for variant, payload in post_runs.items():
        _require(
            int(payload.get("successful_optimizer_updates", -1)) == 13200
            and int(payload.get("checkpoint_epoch", -1)) == PRIMARY_EPOCH
            and payload.get("checkpoint_state_key") == PRIMARY_STATE_KEY,
            f"{variant} trajectory terminal contract mismatch",
        )
    post_run_records = {
        variant: {
            "path": suite_binding["post_runs"][variant]["path"],
            "sha256": suite_binding["post_runs"][variant]["sha256"],
            "training_audit_path": suite_binding["post_runs"][variant][
                "training_audit"
            ]["path"],
            "training_audit_sha256": suite_binding["post_runs"][variant][
                "training_audit"
            ]["sha256"],
        }
        for variant in VARIANTS
    }

    rows = []
    for variant in VARIANTS:
        for epoch in FIXED_EPOCHS:
            rows.append(
                _validate_evaluation(
                    variant,
                    epoch,
                    evaluation_paths[(variant, epoch)],
                    expected_commit=expected_commit,
                    suite_binding=suite_binding,
                    post_run=post_runs[variant],
                    recompute=recompute,
                    checkpoint_inspector=checkpoint_inspector,
                )
            )
    receipt_records = _validate_variant_receipts(
        variant_receipt_paths,
        expected_commit=expected_commit,
        expected_evidence_commit=expected_evidence_commit,
        suite_binding=suite_binding,
        rows=rows,
    )
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "evidence_git_commit": expected_evidence_commit,
        "variants": list(VARIANTS),
        "fixed_epochs": list(FIXED_EPOCHS),
        "primary_epoch": PRIMARY_EPOCH,
        "primary_state_key": PRIMARY_STATE_KEY,
        "suite_aggregate_binding": {
            key: suite_binding[key]
            for key in (
                "path",
                "sha256",
                "git_commit",
                "training_profile",
                "seed",
                "variant_order",
                "shared_protocol_sha256",
                "ordered_exposure_sha256",
                "real_loader_gate",
                "ddp_pilot",
            )
        },
        "checkpoint_selection": {
            "allowed": False,
            "rule": "predeclared_terminal_epoch_131_only",
            "metric_based_selection_forbidden": True,
        },
        "interpretation": {
            "epoch_59": "diagnostic_point_under_132_epoch_scheduler_not_official_60_epoch_run",
            "epoch_89": "diagnostic_point_under_132_epoch_scheduler",
            "epoch_131": "sole_primary_terminal_result",
        },
        "post_run_evidence": post_run_records,
        "variant_receipts": receipt_records,
        "rows": rows,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _exclusive_write_tsv(
    path: str | Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "variant",
        "epoch",
        "role",
        "successful_optimizer_updates",
        "average_mAP",
        "mAP@0.3",
        "mAP@0.4",
        "mAP@0.5",
        "mAP@0.6",
        "mAP@0.7",
    ]
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "variant": row["variant"],
                    "epoch": row["epoch"],
                    "role": row["role"],
                    "successful_optimizer_updates": row[
                        "successful_optimizer_updates"
                    ],
                    **{key: metrics[key] for key in columns[4:]},
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _parse_variant_path(values: Sequence[str], label: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        variant, separator, path = value.partition("=")
        _require(separator == "=" and variant in VARIANTS and path, f"invalid {label}: {value}")
        _require(variant not in output, f"duplicate {label}: {variant}")
        output[variant] = path
    return output


def _parse_evaluation_path(values: Sequence[str]) -> dict[tuple[str, int], str]:
    output: dict[tuple[str, int], str] = {}
    for value in values:
        key, separator, path = value.partition("=")
        variant, colon, epoch_text = key.partition(":")
        _require(
            separator == "="
            and colon == ":"
            and variant in VARIANTS
            and epoch_text.isdigit()
            and int(epoch_text) in FIXED_EPOCHS
            and path,
            f"invalid evaluation binding: {value}",
        )
        pair = (variant, int(epoch_text))
        _require(pair not in output, f"duplicate evaluation binding: {key}")
        output[pair] = path
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-evidence-commit", required=True)
    parser.add_argument("--suite-aggregate", required=True)
    parser.add_argument("--suite-aggregate-sha256", required=True)
    parser.add_argument("--post-run", action="append", required=True)
    parser.add_argument("--variant-receipt", action="append", required=True)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    args = parser.parse_args(argv)
    if Path(args.output_json).expanduser().exists() or Path(
        args.output_tsv
    ).expanduser().exists():
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "error_type": "FileExistsError",
                    "error": "refusing to overwrite convergence outputs",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    try:
        payload = build_convergence_evidence(
            expected_commit=args.expected_commit,
            expected_evidence_commit=args.expected_evidence_commit,
            suite_aggregate_path=args.suite_aggregate,
            suite_aggregate_sha256=args.suite_aggregate_sha256,
            post_run_paths=_parse_variant_path(args.post_run, "post-run binding"),
            variant_receipt_paths=_parse_variant_path(
                args.variant_receipt, "variant receipt binding"
            ),
            evaluation_paths=_parse_evaluation_path(args.evaluation),
        )
        _exclusive_write_tsv(args.output_tsv, payload["rows"])
        _exclusive_write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not Path(args.output_json).expanduser().exists():
            _exclusive_write_json(args.output_json, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
