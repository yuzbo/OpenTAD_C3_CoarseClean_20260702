from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import paction_acquisition_policy as policy
from tools.bata import paction_source_samples
from tools.bata import train_paction_acquisition_policy as train_policy


SUMMARY_SCHEMA_VERSION = "c3_paction_acquisition_policy_application_v1"
READY = "C3_PACTION_POLICY_APPLICATION_READY"
BOOTSTRAP_POLICY_SOURCE = "bootstrap_paction_gap_loss_surrogate_policy"
CHECKPOINT_POLICY_SOURCE = "learned_paction_gap_loss_policy_checkpoint"
FORBIDDEN_TRUE_FLAGS = (
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
)
DEPLOY_INVISIBLE_PAYLOAD_KEYS = (
    "action_target",
    "action_labels",
    "uses_gt_for_diagnostics",
    "gt_boundaries",
    "boundaries",
    "gt_segments",
    "gt_labels",
    "ground_truth",
    "teacher",
    "teacher_logits",
    "teacher_scores",
    "teacher_predictions",
    "oracle",
    "oracle_scores",
    "oracle_selected_positions",
    "prediction_cache",
    "prediction_cache_path",
    "raw_prediction",
    "raw_predictions",
    "raw_scores",
    "raw_logits",
    "frame_signals",
    "p_action",
    "boundary_support_r1",
    "boundary_support_r2",
    "boundary_support_r4",
    "boundary_support_r8",
    "action_coverage",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: sample row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _reject_forbidden_source_flags(row: Mapping[str, Any], *, line_no: int) -> None:
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden p_action source flag {key}=true")


def _strip_deploy_invisible_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(dict(row))
    for key in DEPLOY_INVISIBLE_PAYLOAD_KEYS:
        stripped.pop(key, None)
    stripped["deploy_invisible_payload_stripped"] = True
    return stripped


def _extract_paction(row: Mapping[str, Any], *, line_no: int) -> list[float]:
    frame_signals = row.get("frame_signals")
    if isinstance(frame_signals, Mapping) and isinstance(frame_signals.get("p_action"), list):
        return [float(item) for item in frame_signals["p_action"]]
    if isinstance(row.get("p_action"), list):
        return [float(item) for item in row["p_action"]]
    raise ValueError(f"line {line_no}: p_action signal is required in frame_signals.p_action or p_action")


def _paction_positive_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    p_action_source = row.get("p_action_source") or row.get("source_p_action") or "lowres_action_probe"
    return {
        "p_action_source": str(p_action_source),
        "probe_model": row.get("probe_model"),
        "tcn_variant": row.get("tcn_variant"),
        "matrix_model_id": row.get("matrix_model_id"),
        "official_action_seg_backend": row.get("official_action_seg_backend"),
        "spatial_size": row.get("spatial_size"),
        "split": row.get("split") or row.get("subset"),
        "probe_checkpoint_sha256": row.get("probe_checkpoint_sha256"),
        "probe_manifest_sha256": row.get("probe_manifest_sha256"),
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }


def bootstrap_policy_scores(
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    dynamic_budget_buckets: Sequence[int] = policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS,
) -> tuple[list[float], list[float]]:
    features = policy.build_paction_feature_matrix(p_action, valid=valid)
    p_idx = policy.feature_index("p_action")
    abs_delta_idx = policy.feature_index("abs_delta_p_action")
    entropy_idx = policy.feature_index("entropy")
    uncertainty_idx = policy.feature_index("uncertainty")
    valid_idx = policy.feature_index("valid")
    frame_values = [
        (
            1.20 * float(row[abs_delta_idx])
            + 0.70 * float(row[entropy_idx])
            + 0.40 * float(row[uncertainty_idx])
            + 0.25 * float(row[p_idx])
        )
        * float(row[valid_idx])
        for row in features
    ]
    valid_rows = [row for row in features if float(row[valid_idx]) > 0.0]
    if not valid_rows:
        return frame_values, [0.0 for _ in dynamic_budget_buckets]
    mean_complexity = sum(
        float(row[abs_delta_idx]) + 0.50 * float(row[entropy_idx]) + 0.25 * float(row[uncertainty_idx])
        for row in valid_rows
    ) / float(len(valid_rows))
    target_idx = int(round(max(0.0, min(1.0, mean_complexity)) * float(max(0, len(dynamic_budget_buckets) - 1))))
    budget_scores = [-abs(idx - target_idx) for idx in range(len(dynamic_budget_buckets))]
    return frame_values, [float(item) for item in budget_scores]


def load_policy_checkpoint(checkpoint_path: str | Path, *, device: str = "cuda") -> tuple[policy.PActionDynamicAcquisitionPolicy, dict[str, Any]]:
    import torch

    payload = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("p_action policy checkpoint must be a mapping")
    if payload.get("schema_version") != train_policy.CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("p_action policy checkpoint schema_version mismatch")
    if payload.get("decision") != train_policy.READY:
        raise ValueError("p_action policy checkpoint decision is not ready")
    state_dict = payload.get("policy_state_dict")
    if state_dict is None:
        raise ValueError("p_action policy checkpoint is missing policy_state_dict")
    model_kwargs = dict(payload.get("model_kwargs") or {})
    if not model_kwargs:
        model_kwargs = {
            "input_dim": len(policy.PACTION_FEATURE_NAMES),
            "budget_buckets": payload.get("dynamic_budget_buckets", policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS),
        }
    model = policy.PActionDynamicAcquisitionPolicy(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dict(payload)


def checkpoint_policy_scores(
    model: policy.PActionDynamicAcquisitionPolicy,
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    device: str = "cuda",
) -> tuple[list[float], list[float]]:
    import torch

    features = policy.build_paction_feature_matrix(p_action, valid=valid)
    with torch.no_grad():
        feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
        valid_tensor = None if valid is None else torch.tensor([list(valid)], dtype=torch.bool, device=device)
        outputs = model(feature_tensor, valid_tensor)
    frame_values = [float(item) for item in outputs["frame_value"][0].detach().cpu().tolist()]
    budget_scores = [float(item) for item in outputs["budget_logits"][0].detach().cpu().tolist()]
    return frame_values, budget_scores


def run_policy_application(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    summary_json: str | Path | None = None,
    fixed_budget: int = 384,
    dynamic_budget_buckets: Sequence[int] = policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    strip_deploy_invisible_payload: bool = False,
    strict_deploy_source: bool = False,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    enriched_rows: list[dict[str, Any]] = []
    dynamic_budgets: list[int] = []
    checkpoint_payload: dict[str, Any] | None = None
    checkpoint_model: policy.PActionDynamicAcquisitionPolicy | None = None
    source = BOOTSTRAP_POLICY_SOURCE
    checkpoint_sha256: str | None = None
    if checkpoint_path is not None:
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        checkpoint_model, checkpoint_payload = load_policy_checkpoint(checkpoint_path, device=device)
        source = CHECKPOINT_POLICY_SOURCE
        dynamic_budget_buckets = checkpoint_payload.get("dynamic_budget_buckets", dynamic_budget_buckets)
    for line_no, row in enumerate(rows, start=1):
        if strict_deploy_source:
            p_action_provenance = paction_source_samples.reject_strict_deploy_source_row(
                row,
                source_name=f"{input_jsonl}:{line_no}",
                reject_payload=True,
            )
        else:
            _reject_forbidden_source_flags(row, line_no=line_no)
            p_action_provenance = _paction_positive_provenance(row)
        p_action = _extract_paction(row, line_no=line_no)
        valid_len = int(row.get("valid_len") or row.get("dense_len") or len(p_action))
        valid = [idx < valid_len for idx in range(len(p_action))]
        if checkpoint_model is None:
            frame_values, budget_scores = bootstrap_policy_scores(
                p_action,
                valid=valid,
                dynamic_budget_buckets=dynamic_budget_buckets,
            )
        else:
            frame_values, budget_scores = checkpoint_policy_scores(
                checkpoint_model,
                p_action,
                valid=valid,
                device=device,
            )
        enriched = policy.add_policy_decision_to_sample_row(
            row,
            frame_values=frame_values,
            fixed_budget=int(fixed_budget),
            dynamic_budget_scores=budget_scores,
            dynamic_budget_buckets=dynamic_budget_buckets,
        )
        enriched["paction_policy"]["source"] = source
        enriched["paction_policy"]["p_action_provenance"] = p_action_provenance
        if checkpoint_path is not None:
            enriched["paction_policy"]["checkpoint_path"] = str(checkpoint_path)
            enriched["paction_policy"]["checkpoint_sha256"] = str(checkpoint_sha256)
        enriched["paction_policy"]["frame_value_summary"] = {
            "min": min(frame_values) if frame_values else None,
            "max": max(frame_values) if frame_values else None,
            "mean": None if not frame_values else sum(frame_values) / float(len(frame_values)),
        }
        dynamic_budgets.append(int(enriched["paction_policy"]["dynamic_budget"]))
        if strip_deploy_invisible_payload:
            enriched = _strip_deploy_invisible_payload(enriched)
        enriched_rows.append(enriched)

    _write_jsonl(output_jsonl, enriched_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(enriched_rows),
        "fixed_budget": int(fixed_budget),
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "min_dynamic_budget": min(dynamic_budgets) if dynamic_budgets else None,
        "max_dynamic_budget": max(dynamic_budgets) if dynamic_budgets else None,
        "mean_dynamic_budget": None if not dynamic_budgets else sum(dynamic_budgets) / float(len(dynamic_budgets)),
        "fixed_strategy": policy.LEARNED_FIXED_STRATEGY,
        "dynamic_strategy": policy.LEARNED_DYNAMIC_STRATEGY,
        "source": source,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "strip_deploy_invisible_payload": bool(strip_deploy_invisible_payload),
        "strict_deploy_source": bool(strict_deploy_source),
        "gap_control": "learned_gap_hole_loss_no_uniform_fill",
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "loss_terms": dict(policy.DEFAULT_POLICY_LOSS_TERMS),
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a p_action acquisition policy to low-res probe sample rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--fixed-budget", type=int, default=384)
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strip-deploy-invisible-payload", action="store_true")
    parser.add_argument("--strict-deploy-source", action="store_true")
    args = parser.parse_args(argv)

    summary = run_policy_application(
        args.input_jsonl,
        args.output_jsonl,
        summary_json=args.summary_json,
        fixed_budget=int(args.fixed_budget),
        dynamic_budget_buckets=[int(item) for item in args.dynamic_budget_buckets],
        checkpoint_path=args.checkpoint_path,
        device=str(args.device),
        strip_deploy_invisible_payload=bool(args.strip_deploy_invisible_payload),
        strict_deploy_source=bool(args.strict_deploy_source),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
