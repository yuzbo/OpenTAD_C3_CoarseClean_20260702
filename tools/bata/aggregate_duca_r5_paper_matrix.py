from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping
from uuid import uuid4

from tools.bata.duca_full_stack_cost import validate_and_rebuild_profile_summary
from tools.bata.duca_p0_evaluation import (
    canonical_sha256,
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
)
from tools.bata.duca_r5_paper_matrix import (
    DENSE_RECEIPT_SCHEMA,
    EXPECTED_DENSE_TRAINED_COMMIT,
)
from tools.bata.profile_duca_full_stack_cost import (
    load_r5_terminal_cost_binding,
)


SCHEMA = "duca_r5_paper_matrix_results_v1"
IOU_KEYS = tuple(f"mAP@{value:.1f}" for value in (0.3, 0.4, 0.5, 0.6, 0.7))
COST_COMPARABILITY_KEYS = (
    "protocol",
    "hardware_fingerprint",
    "host_fingerprint",
    "software_fingerprint",
    "source_dataset_fingerprint",
    "inference_fingerprint",
    "batch_size",
    "loader_workers",
    "warmup_samples",
    "sample_count",
    "amp",
    "uses_ema",
    "power_sampling_enabled",
    "power_interval_ms",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"R5 aggregation failed: {message}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def _load_sha256_file(path: Path, *, target: Path, label: str) -> str:
    _require(path.is_file(), f"{label} SHA256 file is missing: {path}")
    digest = path.read_text(encoding="utf-8").strip()
    _require(
        len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
        f"{label} SHA256 file is invalid",
    )
    _require(target.is_file() and _sha256(target) == digest, f"{label} content drift")
    return digest


def _finite_metric(metrics: Mapping[str, Any], key: str, label: str) -> float:
    value = metrics.get(key)
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} lacks finite {key}",
    )
    return float(value)


def _require_bound_file(path: Any, digest: Any, *, label: str) -> Path:
    artifact = Path(str(path or "")).expanduser().resolve()
    _require(artifact.is_file(), f"{label} is missing: {artifact}")
    _require(_sha256(artifact) == str(digest or ""), f"{label} content drift")
    return artifact


def _require_same_metrics(
    reported: Mapping[str, Any], recomputed: Mapping[str, Any], *, label: str
) -> None:
    for key in ("average_mAP", *IOU_KEYS):
        expected = _finite_metric(recomputed, key, f"{label} recomputed metrics")
        observed = _finite_metric(reported, key, f"{label} reported metrics")
        _require(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-10),
            f"{label} official metric mismatch: {key}",
        )


def _profile_comparability_payload(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    comparable: dict[str, Any] = {}
    for key in COST_COMPARABILITY_KEYS:
        _require(key in payload and payload.get(key) is not None, f"{label} lacks {key}")
        value = payload[key]
        if isinstance(value, str):
            _require(bool(value.strip()), f"{label} has empty {key}")
        comparable[key] = value
    for key in ("profile_session_id", "profile_pair_id"):
        value = str(payload.get(key, ""))
        _require(bool(value.strip()), f"{label} lacks {key}")
        comparable[key] = value
    repeat_index = payload.get("profile_repeat_index")
    _require(
        isinstance(repeat_index, int)
        and not isinstance(repeat_index, bool)
        and repeat_index > 0,
        f"{label} has invalid profile_repeat_index",
    )
    order_position = payload.get("profile_order_position")
    _require(
        order_position in (1, 2),
        f"{label} has invalid profile_order_position",
    )
    comparable["profile_repeat_index"] = repeat_index
    comparable["profile_order_position"] = order_position
    return comparable


def _require_paired_comparable(
    candidate: Mapping[str, Any], dense: Mapping[str, Any], *, label: str
) -> None:
    for key in COST_COMPARABILITY_KEYS:
        _require(
            candidate.get(key) == dense.get(key),
            f"{label} cost profile mismatch: {key}",
        )
    _require(
        candidate.get("profile_session_id") == dense.get("profile_session_id"),
        f"{label} candidate and dense were not measured in one session",
    )
    _require(
        candidate.get("profile_pair_id") == dense.get("profile_pair_id"),
        f"{label} candidate and dense are not a paired profile",
    )
    _require(
        candidate.get("profile_repeat_index") == dense.get("profile_repeat_index"),
        f"{label} candidate and dense repeat identity mismatch",
    )
    _require(
        {
            candidate.get("profile_order_position"),
            dense.get("profile_order_position"),
        }
        == {1, 2},
        f"{label} paired order must contain positions one and two",
    )


def _validate_dense_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
    receipt = row.get("receipt")
    _require(isinstance(receipt, Mapping), "dense baseline lacks its sealed receipt")
    receipt = dict(receipt)
    observed_self_hash = receipt.pop("receipt_sha256", None)
    _require(
        observed_self_hash == canonical_sha256(receipt),
        "dense baseline receipt self-hash drift",
    )
    _require(
        row.get("receipt_sha256") == observed_self_hash,
        "dense baseline matrix receipt hash drift",
    )
    expected_scalars = {
        "schema": DENSE_RECEIPT_SCHEMA,
        "task": "offline_temporal_action_detection",
        "role": "dense_adatad_baseline",
        "trained_commit": EXPECTED_DENSE_TRAINED_COMMIT,
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
    }
    for key, expected in expected_scalars.items():
        _require(receipt.get(key) == expected, f"dense baseline receipt drift: {key}")
    for prefix in (
        "config",
        "checkpoint",
        "checkpoint_evidence",
        "training_evidence",
        "evaluation_evidence",
    ):
        _require_bound_file(
            receipt.get(f"{prefix}_path"),
            receipt.get(f"{prefix}_sha256"),
            label=f"dense receipt {prefix}",
        )
    receipt["receipt_sha256"] = observed_self_hash
    return receipt


def _validate_evaluation(
    *,
    cell: Mapping[str, Any],
    root: Path,
    expected_commit: str,
    matrix_summary_path: Path,
    matrix_summary_sha256: str,
    mechanism_gate_path: Path,
    mechanism_gate_sha256: str,
) -> dict[str, Any]:
    cell_id = str(cell["id"])
    path = root / "results" / f"{cell_id}.terminal_evaluation.json"
    payload = _load_json(path, f"{cell_id} terminal evaluation")
    unsigned = dict(payload)
    observed_self_hash = unsigned.pop("evaluation_sha256", None)
    _require(
        observed_self_hash == _canonical_sha256(unsigned),
        f"{cell_id} evaluation self-hash drift",
    )
    config = Path(str(cell["config"])).resolve()
    _require(config.is_file(), f"{cell_id} config is missing")
    _require(
        _sha256(config) == cell["config_sha256"] == payload.get("config_sha256"),
        f"{cell_id} config identity drift",
    )
    _require(
        Path(str(payload.get("config_path", ""))).resolve() == config,
        f"{cell_id} evaluation points at another config",
    )
    _require(payload.get("git_commit") == expected_commit, f"{cell_id} commit drift")
    _require(payload.get("task") == "offline_temporal_action_detection", f"{cell_id} task drift")
    _require(
        payload.get("schema_version") == "duca_r5_terminal_evaluation_v1",
        f"{cell_id} is not an R5 terminal evaluation",
    )
    identity = payload.get("training_identity")
    _require(
        isinstance(identity, Mapping)
        and identity.get("variant") == cell_id
        and identity.get("seed") == int(cell["seed"])
        and identity.get("successful_optimizer_updates") == 6000,
        f"{cell_id} lacks the exact successful-update identity",
    )
    runtime_cell = payload.get("r5_cell")
    _require(
        isinstance(runtime_cell, Mapping)
        and all(
            runtime_cell.get(key) == cell[key]
            for key in (
                "backend",
                "arm",
                "budget",
                "max_unselected_hole",
                "seed",
            )
        ),
        f"{cell_id} runtime cell identity drift",
    )
    _require(payload.get("checkpoint_epoch") == 59, f"{cell_id} is not terminal epoch 59")
    _require(payload.get("checkpoint_state_key") == "state_dict_ema", f"{cell_id} is not EMA")
    checkpoint = Path(str(payload.get("checkpoint_path", ""))).resolve()
    _require(checkpoint.is_file(), f"{cell_id} checkpoint is missing")
    _require(_sha256(checkpoint) == payload.get("checkpoint_sha256"), f"{cell_id} checkpoint drift")
    expected_prediction = Path(str(cell.get("prediction_path", ""))).resolve()
    prediction = _require_bound_file(
        payload.get("prediction_path"),
        payload.get("prediction_sha256"),
        label=f"{cell_id} raw prediction",
    )
    _require(
        prediction == expected_prediction,
        f"{cell_id} prediction path substitution",
    )
    annotation = _require_bound_file(
        payload.get("evaluation_annotation_path"),
        payload.get("evaluation_annotation_sha256"),
        label=f"{cell_id} evaluation annotation",
    )
    class_map = _require_bound_file(
        payload.get("evaluation_class_map_path"),
        payload.get("evaluation_class_map_sha256"),
        label=f"{cell_id} evaluation class map",
    )
    evaluation_config = normalize_evaluation_config(
        payload.get("evaluation_config"), expected_subset="validation"
    )
    _require(
        evaluation_config.get("ground_truth_filename") == str(annotation),
        f"{cell_id} evaluator annotation path drift",
    )
    _require(
        evaluation_config.get("blocked_videos") is None,
        f"{cell_id} is not a full validation evaluation",
    )
    _require(
        payload.get("evaluation_config_sha256")
        == canonical_sha256(evaluation_config),
        f"{cell_id} evaluation config hash drift",
    )
    evaluator = official_evaluator_identity()
    _require(payload.get("evaluator") == evaluator, f"{cell_id} evaluator identity drift")
    recomputed = recompute_official_map(
        prediction, evaluation_config, expected_subset="validation"
    )
    _require(
        recomputed.get("evaluator") == evaluator,
        f"{cell_id} recomputation did not use the official evaluator",
    )
    _require(
        recomputed.get("evaluation_config") == evaluation_config
        and recomputed.get("evaluation_config_sha256")
        == canonical_sha256(evaluation_config),
        f"{cell_id} recomputed evaluation protocol drift",
    )
    _require(
        int(payload.get("result_count", -1))
        == int(recomputed.get("result_count", -2)),
        f"{cell_id} result count drift",
    )
    _require(
        int(payload.get("video_count", -1))
        == int(recomputed.get("video_count", -2)),
        f"{cell_id} video count drift",
    )
    reported_metrics = payload.get("metrics")
    recomputed_metrics = recomputed.get("metrics")
    _require(isinstance(reported_metrics, Mapping), f"{cell_id} metrics are missing")
    _require(
        isinstance(recomputed_metrics, Mapping),
        f"{cell_id} recomputed metrics are missing",
    )
    _require_same_metrics(reported_metrics, recomputed_metrics, label=cell_id)
    training_binding = load_r5_terminal_cost_binding(
        method_name=cell_id,
        config_path=config,
        checkpoint_path=checkpoint,
        expected_commit=expected_commit,
        matrix_summary_path=matrix_summary_path,
        matrix_summary_sha256=matrix_summary_sha256,
        mechanism_gate_path=mechanism_gate_path,
        mechanism_gate_sha256=mechanism_gate_sha256,
        expected_resolved_config_sha256=str(
            payload.get("resolved_config_sha256", "")
        ),
        expected_training_identity=identity,
        expected_evaluation=payload,
    )
    _require(int(recomputed["result_count"]) > 0, f"{cell_id} has no detections")
    _require(int(recomputed["video_count"]) > 0, f"{cell_id} has no evaluated videos")
    row = {
        key: cell[key]
        for key in (
            "id",
            "backend",
            "arm",
            "budget",
            "max_unselected_hole",
            "seed",
        )
    }
    row["average_mAP"] = _finite_metric(
        recomputed_metrics, "average_mAP", cell_id
    )
    row["iou_mAP"] = {
        key: _finite_metric(recomputed_metrics, key, cell_id) for key in IOU_KEYS
    }
    row["evaluation_path"] = str(path.resolve())
    row["evaluation_sha256"] = _sha256(path)
    row["checkpoint_path"] = str(checkpoint)
    row["checkpoint_sha256"] = payload["checkpoint_sha256"]
    row["prediction_path"] = str(prediction)
    row["prediction_sha256"] = payload["prediction_sha256"]
    row["evaluation_annotation_path"] = str(annotation)
    row["evaluation_annotation_sha256"] = payload[
        "evaluation_annotation_sha256"
    ]
    row["evaluation_class_map_path"] = str(class_map)
    row["evaluation_class_map_sha256"] = payload[
        "evaluation_class_map_sha256"
    ]
    row["evaluation_config"] = evaluation_config
    row["evaluation_config_sha256"] = canonical_sha256(evaluation_config)
    row["evaluator"] = evaluator
    row["training_binding"] = training_binding
    return row


def _validate_cost(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    _require(row.get("kind") == "r5_cell", f"{row['id']} cost kind drift")
    path = Path(str(row["summary"])).resolve()
    payload = _load_json(path, str(row["id"]))
    validate_and_rebuild_profile_summary(payload)
    _require(payload.get("config_commit") == expected_commit, f"{row['id']} commit drift")
    _require(payload.get("random_init") is False, f"{row['id']} used random initialization")
    _require(payload.get("method") == row["source_cell"], f"{row['id']} method drift")
    _require(payload.get("uses_ema") is True, f"{row['id']} did not use EMA")
    binding = payload.get("r5_cost_binding")
    _require(isinstance(binding, Mapping), f"{row['id']} lacks the R5 cost binding")
    unsigned_binding = dict(binding)
    binding_self_hash = unsigned_binding.pop("binding_sha256", None)
    _require(
        binding_self_hash == _canonical_sha256(unsigned_binding),
        f"{row['id']} R5 terminal binding self-hash drift",
    )
    _require(
        payload.get("r5_cost_binding_sha256") == _canonical_sha256(binding),
        f"{row['id']} R5 cost binding self-hash drift",
    )
    _require(
        binding == source.get("training_binding"),
        f"{row['id']} is not bound to its terminal source cell",
    )
    for key, expected in (
        ("checkpoint_path", source["checkpoint_path"]),
        ("checkpoint_sha256", source["checkpoint_sha256"]),
        ("checkpoint_epoch", 59),
        ("checkpoint_state_key", "state_dict_ema"),
    ):
        observed = payload.get(key)
        if key == "checkpoint_path":
            observed = str(Path(str(observed or "")).resolve())
        _require(observed == expected, f"{row['id']} source checkpoint drift: {key}")
    latency = _finite_metric(
        payload.get("stages", {}).get("end_to_end_serial_ms", {}),
        "p50",
        str(row["id"]),
    )
    comparability = _profile_comparability_payload(
        payload, label=str(row["id"])
    )
    return {
        "id": row["id"],
        "source_cell": row["source_cell"],
        "summary_path": str(path),
        "summary_sha256": _sha256(path),
        "source_evaluation_sha256": source["evaluation_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "r5_cost_binding_sha256": payload["r5_cost_binding_sha256"],
        "comparability": comparability,
        "end_to_end_serial_ms_p50": latency,
        "selected_count_p50": _finite_metric(
            payload.get("selected_count", {}), "p50", str(row["id"])
        ),
        "peak_gpu_memory_mb_p50": _finite_metric(
            payload.get("resources", {}).get("peak_gpu_memory_mb", {}),
            "p50",
            str(row["id"]),
        ),
    }


def _validate_dense_cost(
    row: Mapping[str, Any],
    *,
    expected_evidence_commit: str,
) -> dict[str, Any]:
    _require(row.get("kind") == "dense_baseline", "dense cost kind drift")
    path = Path(str(row["summary"])).resolve()
    payload = _load_json(path, str(row["id"]))
    validate_and_rebuild_profile_summary(payload)
    receipt = _validate_dense_receipt(row)
    trained_commit = str(row.get("trained_commit", ""))
    _require(
        trained_commit == EXPECTED_DENSE_TRAINED_COMMIT,
        "dense baseline is not the frozen historical training commit",
    )
    _require(payload.get("method") == "dense-adatad", "dense baseline method drift")
    _require(payload.get("config_commit") == trained_commit, "dense config commit drift")
    _require(payload.get("trained_commit") == trained_commit, "dense trained commit drift")
    _require(
        payload.get("evidence_git_commit") == expected_evidence_commit,
        "dense evidence commit drift",
    )
    _require(payload.get("random_init") is False, "dense baseline used random initialization")
    _require(payload.get("uses_ema") is True, "dense baseline did not use EMA")

    code_binding = payload.get("inference_code_tree_binding")
    config_git_binding = payload.get("profile_config_git_binding")
    _require(
        isinstance(code_binding, Mapping)
        and code_binding.get("profile_model_loaded_from_trained_repository") is True
        and code_binding.get("profile_configs_loaded_from_trained_repository") is True,
        "dense baseline was not executed from its trained repository",
    )
    _require(
        isinstance(config_git_binding, Mapping)
        and config_git_binding.get("trained_commit") == trained_commit,
        "dense profile config lacks its trained Git binding",
    )
    trained_repository = Path(
        str(config_git_binding.get("trained_repository", ""))
    ).resolve()
    _require(
        Path(str(code_binding.get("execution_repository", ""))).resolve()
        == trained_repository
        and Path(str(code_binding.get("loaded_opentad_root", ""))).resolve()
        == trained_repository / "opentad",
        "dense model import escaped its trained repository",
    )

    expected_paths = {
        "config_path": Path(str(row["config"])).resolve(),
        "checkpoint_path": Path(str(row["checkpoint"])).resolve(),
    }
    for key, expected in expected_paths.items():
        _require(
            Path(str(payload.get(key, ""))).resolve() == expected,
            f"dense baseline {key} drift",
        )
    for key, expected in (
        ("profile_config_sha256", row["config_sha256"]),
        ("checkpoint_sha256", row["checkpoint_sha256"]),
        ("checkpoint_epoch", 59),
        ("checkpoint_state_key", "state_dict_ema"),
    ):
        _require(payload.get(key) == expected, f"dense baseline {key} drift")

    binding = payload.get("trained_checkpoint_binding")
    _require(isinstance(binding, Mapping), "dense baseline lacks checkpoint evidence")
    _require(
        payload.get("trained_checkpoint_binding_sha256")
        == _canonical_sha256(binding),
        "dense checkpoint binding canonical hash drift",
    )
    for key, expected in (
        ("role", "dense_adatad_baseline"),
        ("git_commit", trained_commit),
        ("config_path", str(expected_paths["config_path"])),
        ("config_sha256", row["config_sha256"]),
        ("checkpoint_path", str(expected_paths["checkpoint_path"])),
        ("checkpoint_sha256", row["checkpoint_sha256"]),
        ("checkpoint_epoch", 59),
        ("checkpoint_state_key", "state_dict_ema"),
        ("path", str(Path(str(row["checkpoint_evidence"])).resolve())),
        ("sha256", row["checkpoint_evidence_sha256"]),
    ):
        _require(binding.get(key) == expected, f"dense checkpoint binding drift: {key}")
    _require(
        binding.get("resolved_config_sha256")
        == payload.get("profile_resolved_config_sha256"),
        "dense resolved config binding drift",
    )
    receipt_binding_pairs = {
        "git_commit": "trained_commit",
        "config_path": "config_path",
        "config_sha256": "config_sha256",
        "resolved_config_sha256": "resolved_config_sha256",
        "checkpoint_path": "checkpoint_path",
        "checkpoint_sha256": "checkpoint_sha256",
        "checkpoint_epoch": "checkpoint_epoch",
        "checkpoint_state_key": "checkpoint_state_key",
        "path": "checkpoint_evidence_path",
        "sha256": "checkpoint_evidence_sha256",
        "training_evidence_path": "training_evidence_path",
        "training_evidence_sha256": "training_evidence_sha256",
        "evaluation_evidence_path": "evaluation_evidence_path",
        "evaluation_evidence_sha256": "evaluation_evidence_sha256",
    }
    for binding_key, receipt_key in receipt_binding_pairs.items():
        observed = binding.get(binding_key)
        expected = receipt.get(receipt_key)
        if binding_key.endswith("_path") or binding_key in {"path"}:
            observed = str(Path(str(observed or "")).resolve())
            expected = str(Path(str(expected or "")).resolve())
        _require(
            observed == expected,
            f"dense checkpoint binding differs from sealed receipt: {binding_key}",
        )

    latency = _finite_metric(
        payload.get("stages", {}).get("end_to_end_serial_ms", {}),
        "p50",
        str(row["id"]),
    )
    selected_count = _finite_metric(
        payload.get("selected_count", {}), "p50", str(row["id"])
    )
    _require(selected_count == 768.0, "dense baseline is not the full 768-point input")
    comparability = _profile_comparability_payload(
        payload, label="dense baseline"
    )
    return {
        "id": row["id"],
        "kind": "dense_baseline",
        "summary_path": str(path),
        "summary_sha256": _sha256(path),
        "trained_commit": trained_commit,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "checkpoint_evidence_sha256": row["checkpoint_evidence_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "comparability": comparability,
        "end_to_end_serial_ms_p50": latency,
        "selected_count_p50": selected_count,
        "peak_gpu_memory_mb_p50": _finite_metric(
            payload.get("resources", {}).get("peak_gpu_memory_mb", {}),
            "p50",
            str(row["id"]),
        ),
    }


def aggregate_matrix(
    *, matrix_summary: str | Path, expected_commit: str
) -> dict[str, Any]:
    summary_path = Path(matrix_summary).expanduser().resolve()
    summary = _load_json(summary_path, "R5 matrix summary")
    _require(len(expected_commit) == 40, "exact commit is required")
    _require(
        summary.get("schema") == "duca_r5_paper_matrix_v1"
        and summary.get("task") == "offline_temporal_action_detection"
        and summary.get("git_commit") == expected_commit,
        "matrix summary protocol/commit drift",
    )
    cells = summary.get("cells")
    costs = summary.get("costs")
    _require(isinstance(cells, list) and len(cells) == 24, "matrix must contain 24 cells")
    _require(
        isinstance(costs, list) and len(costs) == 8,
        "matrix must contain eight paired candidate/dense cost profiles",
    )
    r5_cost_rows = [row for row in costs if row.get("kind") == "r5_cell"]
    _require(len(r5_cost_rows) == 8, "matrix must profile all seed-3407 R5 cells")
    declared_dense = summary.get("dense_cost_baseline")
    _require(
        isinstance(declared_dense, Mapping),
        "matrix summary lacks the dense baseline receipt",
    )
    _require(
        declared_dense.get("trained_commit") == EXPECTED_DENSE_TRAINED_COMMIT,
        "dense baseline is not the frozen historical training commit",
    )
    _validate_dense_receipt(declared_dense)
    root = summary_path.parent
    matrix_digest = _load_sha256_file(
        Path(str(summary.get("matrix_summary_sha256_file", ""))).resolve(),
        target=summary_path,
        label="R5 matrix summary",
    )
    mechanism_gate_path = Path(
        str(summary.get("mechanism_gate_output", ""))
    ).resolve()
    mechanism_gate_digest = _load_sha256_file(
        Path(str(summary.get("mechanism_gate_sha256_file", ""))).resolve(),
        target=mechanism_gate_path,
        label="R5 mechanism gate",
    )
    rows = [
        _validate_evaluation(
            cell=cell,
            root=root,
            expected_commit=expected_commit,
            matrix_summary_path=summary_path,
            matrix_summary_sha256=matrix_digest,
            mechanism_gate_path=mechanism_gate_path,
            mechanism_gate_sha256=mechanism_gate_digest,
        )
        for cell in cells
    ]
    rows_by_id = {str(row["id"]): row for row in rows}
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["backend"], row["arm"], int(row["budget"]))].append(row)
        by_pair[(row["backend"], int(row["budget"]), int(row["seed"]))][
            row["arm"]
        ] = row
    aggregates = []
    for (backend, arm, budget), group in sorted(grouped.items()):
        values = [float(row["average_mAP"]) for row in group]
        _require(len(values) == 3, f"{backend}/{arm}/K{budget} lacks three seeds")
        aggregates.append(
            {
                "backend": backend,
                "arm": arm,
                "budget": budget,
                "seed_count": len(values),
                "average_mAP_mean": mean(values),
                "average_mAP_std": stdev(values),
                "average_mAP_by_seed": {
                    str(row["seed"]): row["average_mAP"] for row in group
                },
            }
        )
    paired = []
    for (backend, budget, seed), pair in sorted(by_pair.items()):
        _require(set(pair) == {"uniform", "learned"}, f"unmatched pair {backend}/K{budget}/s{seed}")
        paired.append(
            {
                "backend": backend,
                "budget": budget,
                "seed": seed,
                "learned_minus_uniform_average_mAP": (
                    pair["learned"]["average_mAP"]
                    - pair["uniform"]["average_mAP"]
                ),
            }
        )
    validated_r5_costs = [
        _validate_cost(
            row,
            source=rows_by_id[str(row["source_cell"])],
            expected_commit=expected_commit,
        )
        for row in r5_cost_rows
    ]
    paired_dense_costs = []
    for row in r5_cost_rows:
        paired_dense_costs.append(
            _validate_dense_cost(
                {
                    **dict(declared_dense),
                    "id": f"{row['id']}_paired_dense",
                    "kind": "dense_baseline",
                    "summary": row.get("paired_dense_summary"),
                },
                expected_evidence_commit=expected_commit,
            )
        )
    cost_comparisons = []
    for cost, dense_cost in zip(validated_r5_costs, paired_dense_costs):
        _require_paired_comparable(
            cost["comparability"],
            dense_cost["comparability"],
            label=str(cost["source_cell"]),
        )
        dense_latency = float(dense_cost["end_to_end_serial_ms_p50"])
        _require(dense_latency > 0.0, "paired dense latency must be positive")
        latency = float(cost["end_to_end_serial_ms_p50"])
        _require(latency > 0.0, f"{cost['source_cell']} latency must be positive")
        cost_comparisons.append(
            {
                "source_cell": cost["source_cell"],
                "profile_session_id": cost["comparability"][
                    "profile_session_id"
                ],
                "profile_pair_id": cost["comparability"]["profile_pair_id"],
                "profile_repeat_index": cost["comparability"][
                    "profile_repeat_index"
                ],
                "latency_ratio_vs_dense": latency / dense_latency,
                "latency_reduction_vs_dense": 1.0 - latency / dense_latency,
                "speedup_vs_dense": dense_latency / latency,
            }
        )

    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "matrix_summary_path": str(summary_path),
        "matrix_summary_sha256": _sha256(summary_path),
        "cell_count": len(rows),
        "cost_count": len(costs),
        "rows": rows,
        "three_seed_aggregates": aggregates,
        "paired_deltas": paired,
        "costs": validated_r5_costs,
        "dense_baseline_receipt": dict(declared_dense),
        "paired_dense_costs": paired_dense_costs,
        "cost_comparisons": cost_comparisons,
        "paper_claim_allowed": False,
        "status": "r5_raw_evidence_complete_pending_claim_adjudication",
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the complete DUCA R5 matrix")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_matrix(
        matrix_summary=args.matrix_summary,
        expected_commit=args.expected_commit,
    )
    _atomic_write(Path(args.output_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
