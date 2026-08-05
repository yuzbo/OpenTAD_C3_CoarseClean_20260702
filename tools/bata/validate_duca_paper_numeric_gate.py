from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "duca_paper_physical_exactk_numeric_gate_v1"
THRESHOLDS = {
    "fp32_fp64_slot_atol": 5.0e-5,
    "fp32_fp64_slot_rtol": 5.0e-5,
    "slot_row_mass_max_abs": 8.0e-6,
    "dual_logz_max_abs": 5.0e-5,
    "edge_flow_linf_max_abs": 2.0e-5,
    "edge_flow_per_slot_l1_max_abs": 1.0e-4,
    "fp32_fp64_gradient_atol": 2.0e-5,
    "fp32_fp64_gradient_rtol": 2.0e-3,
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_numeric_gate_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    expected_commit = str(expected_commit).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("numeric gate requires an exact commit")
    if not source.is_file():
        raise FileNotFoundError(f"numeric gate is missing: {source}")
    observed_sha = _sha256(source)
    expected_sha256 = str(expected_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("numeric gate requires an exact SHA-256")
    if observed_sha != expected_sha256:
        raise RuntimeError("numeric gate SHA-256 drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("numeric gate receipt is not a mapping")
    unsigned = dict(payload)
    content_sha = unsigned.pop("content_sha256", None)
    execution = payload.get("bounded_execution", {})
    capture = payload.get("capture", {})
    legacy = payload.get("legacy_failure_reproduction", {})
    diagnostic = payload.get("candidate_fp32_oracle_diagnostic", {})
    tensor_path = Path(str(payload.get("tensor_artifact_path", ""))).expanduser().resolve()
    short_gate = payload.get("prerequisite_real_short_window_gate", {})
    code_gate = payload.get("prerequisite_clean_linux_code_gate", {})
    config = payload.get("config", {})
    assets = payload.get("assets", {})
    rank_summary_sha = payload.get("rank_summary_sha256", {})
    numeric_fields = (
        "fp32_fp64_slot_max_abs",
        "fp32_fp64_slot_max_relative",
        "slot_row_mass_max_abs",
        "dual_logz_max_abs",
        "edge_flow_linf_max_abs",
        "edge_flow_per_slot_l1_max_abs",
        "fp32_fp64_gradient_max_abs",
        "fp32_fp64_gradient_max_relative",
        "occupancy_sum",
    )
    numeric_values_finite = all(
        math.isfinite(float(diagnostic.get(key, math.nan))) for key in numeric_fields
    )
    capture_values_finite = all(
        math.isfinite(float(capture.get(key, math.nan)))
        for key in ("score_min", "score_max", "score_mean", "score_std")
    )
    legacy_values_finite = all(
        math.isfinite(float(legacy.get(key, math.nan)))
        for key in (
            "old_raw_log_row_mass_max_abs",
            "old_fp32_normalization_envelope",
        )
    )
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "passed"
        or payload.get("fail_closed") is not True
        or payload.get("git_commit") != expected_commit
        or not str(payload.get("slurm_job_id", "")).isdigit()
        or content_sha != _canonical_sha256(unsigned)
        or code_gate.get("status") != "passed"
        or code_gate.get("git_commit") != expected_commit
        or short_gate.get("status") != "passed"
        or short_gate.get("git_commit") != expected_commit
        or int(execution.get("seed", -1)) != 5801
        or int(execution.get("world_size", -1)) != 2
        or int(execution.get("global_batch_size", -1)) != 2
        or int(execution.get("maximum_attempted_updates", -1)) != 100
        or not 1 <= int(execution.get("attempted_updates_until_trigger", -1)) <= 100
        or int(execution.get("successful_updates_until_trigger", -2))
        != int(execution.get("attempted_updates_until_trigger", -1))
        or execution.get("old_guard_triggered") is not True
        or execution.get("actual_full_model_forward_backward_optimizer_step") is not True
        or execution.get("all_training_gradients_finite") is not True
        or int(capture.get("t", -1)) != 768
        or int(capture.get("k", -1)) != 384
        or str(capture.get("score_dtype", "")) != "torch.float32"
        or capture.get("cuda_autocast_enabled") is not True
        or legacy.get("old_guard_triggered") is not True
        or float(legacy.get("old_raw_log_row_mass_max_abs", -1.0))
        <= float(legacy.get("old_fp32_normalization_envelope", math.inf))
        or diagnostic.get("passed") is not True
        or diagnostic.get("thresholds") != THRESHOLDS
        or diagnostic.get("all_gradients_finite") is not True
        or diagnostic.get("fp32_fp64_slots_allclose") is not True
        or diagnostic.get("fp32_fp64_gradients_allclose") is not True
        or diagnostic.get("hard_path_exact_identical") is not True
        or not numeric_values_finite
        or not capture_values_finite
        or not legacy_values_finite
        or int(capture.get("valid_len", -1)) != 768
        or int(diagnostic.get("effective_k", -1)) != 384
        or abs(float(diagnostic.get("occupancy_sum", math.nan)) - 384.0)
        > 2.0e-4 * 384.0
        or float(diagnostic.get("slot_row_mass_max_abs", math.inf))
        > THRESHOLDS["slot_row_mass_max_abs"]
        or float(diagnostic.get("dual_logz_max_abs", math.inf))
        > THRESHOLDS["dual_logz_max_abs"]
        or float(diagnostic.get("edge_flow_linf_max_abs", math.inf))
        > THRESHOLDS["edge_flow_linf_max_abs"]
        or float(diagnostic.get("edge_flow_per_slot_l1_max_abs", math.inf))
        > THRESHOLDS["edge_flow_per_slot_l1_max_abs"]
        or not tensor_path.is_file()
        or tensor_path.parent != source.parent
        or re.fullmatch(r"captured_solver_input\.rank[0-9]{4}\.pt", tensor_path.name)
        is None
        or _sha256(tensor_path) != payload.get("tensor_artifact_sha256")
        or set(rank_summary_sha) != {"0", "1"}
        or not re.fullmatch(r"[0-9a-f]{64}", str(config.get("runtime_resolved_sha256", "")))
        or not Path(str(config.get("path", ""))).expanduser().resolve().is_file()
        or _sha256(Path(str(config["path"])).expanduser().resolve())
        != config.get("sha256")
        or not Path(str(assets.get("train_data_path", ""))).is_dir()
        or payload.get("validation_or_test_data_used") is not False
        or payload.get("checkpoint_created") is not False
        or payload.get("prediction_generated") is not False
        or payload.get("loss_values_recorded") is not False
        or payload.get("metric_accessed") is not False
        or payload.get("paper_metric_claim_allowed") is not False
        or payload.get("paper_method_performance_evidence") is not False
        or payload.get("stage_a_release_prerequisite_satisfied") is not True
        or payload.get("stage_b_enabled") is not False
        or payload.get("official_final_consumed") is not False
    ):
        raise RuntimeError("production-like learned numeric gate contract drift")
    for index in range(2):
        rank_path = source.parent / f"rank{index:04d}.summary.json"
        if (
            not rank_path.is_file()
            or _sha256(rank_path) != rank_summary_sha[str(index)]
        ):
            raise RuntimeError("numeric gate rank-summary binding drift")
    for key in ("pretrain", "annotation", "class_map"):
        binding = assets.get(key, {})
        asset_path = Path(str(binding.get("path", ""))).expanduser().resolve()
        if (
            not asset_path.is_file()
            or _sha256(asset_path) != binding.get("sha256")
        ):
            raise RuntimeError(f"numeric gate {key} binding drift")

    import torch

    artifact = torch.load(tensor_path, map_location="cpu")
    graph = artifact.get("graph", {}) if isinstance(artifact, Mapping) else {}
    node = artifact.get("node_log_probs") if isinstance(artifact, Mapping) else None
    graph_keys = {
        "predecessor_index",
        "predecessor_valid",
        "successor_index",
        "successor_valid",
        "source_valid",
        "sink_valid",
        "max_gap_seconds",
        "edge_count",
    }
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema_version") != "duca_paper_numeric_gate_tensor_v1"
        or not torch.is_tensor(node)
        or tuple(node.shape) != (768,)
        or node.dtype != torch.float32
        or not bool(torch.isfinite(node).all().item())
        or int(artifact.get("k", -1)) != 384
        or not math.isfinite(float(artifact.get("temperature", math.nan)))
        or float(artifact.get("temperature", 0.0)) <= 0.0
        or set(graph) != graph_keys
        or not all(
            torch.is_tensor(graph[key])
            for key in graph_keys
            if key != "edge_count"
        )
        or tuple(graph["source_valid"].shape) != (768,)
        or tuple(graph["sink_valid"].shape) != (768,)
        or int(graph["predecessor_index"].shape[0]) != 768
        or int(graph["successor_index"].shape[0]) != 768
        or artifact.get("contains_gt") is not False
        or artifact.get("contains_predictions") is not False
        or artifact.get("contains_metrics") is not False
        or artifact.get("contains_loss_values") is not False
    ):
        raise RuntimeError("numeric gate tensor-artifact contract drift")
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "git_commit": expected_commit,
        "path": str(source),
        "sha256": observed_sha,
        "slurm_job_id": str(payload["slurm_job_id"]),
        "claim_scope": "engineering_learned_exactk_numeric_stability_only",
        "performance_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate_numeric_gate_artifact(
                args.receipt,
                expected_commit=args.expected_commit,
                expected_sha256=args.expected_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
