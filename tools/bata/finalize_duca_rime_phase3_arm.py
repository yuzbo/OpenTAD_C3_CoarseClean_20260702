from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.duca_rime_stage_contract import (
    PHASE3_RESULT_SCHEMA,
    PHASE3_TRAIN_ARMS,
)


LEDGER_ARMS = {
    "F-bound": "fixed_bound",
    "D-shuffle": "dynamic_shuffle",
    "D-no-risk": "dynamic_no_risk",
    "AdapTok-TAD": "adaptok_tad",
    "RIME-full": "rime_full",
    "U-same-K": "uniform_same_k",
}


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {resolved}")
    return resolved, payload


def _verify_content_hash(payload: Mapping[str, Any], *, label: str) -> None:
    unsigned = dict(payload)
    embedded = unsigned.pop("content_sha256", None)
    if embedded != _canonical_sha256(unsigned):
        raise ValueError(f"{label} content hash is invalid")


def _validate_training_identity(
    terminal: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    identity = terminal.get("training_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("terminal evaluation lacks a RIME training identity")
    source_arm = str(identity.get("source_arm", ""))
    expected_source = "RIME-full" if arm == "U-same-K" else arm
    receipt_path = Path(str(identity.get("training_receipt_path", ""))).resolve()
    if (
        identity.get("evaluation_arm") != arm
        or source_arm != expected_source
        or int(identity.get("successful_detector_updates", -1)) != 6000
        or int(identity.get("research_phase", -1)) != 3
        or identity.get("official_final_subset_consumed_during_training") is not False
        or identity.get("detector_backend") != "ActionFormer"
        or identity.get("target_mean_cost") is not None
        or not receipt_path.is_file()
        or _sha256_file(receipt_path)
        != str(identity.get("training_receipt_sha256", ""))
    ):
        raise ValueError("terminal evaluation training identity is not a Phase-3 arm")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "duca_rime_phase3_training_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("arm") != expected_source
        or int(receipt.get("seed", -1)) != int(seed)
        or receipt.get("git_commit") != terminal.get("git_commit")
        or int(receipt.get("successful_detector_updates", -1)) != 6000
        or receipt.get("formal_update_audit_passed") is not True
        or receipt.get("uses_official_final") is not False
        or receipt.get("checkpoint_sha256") != terminal.get("checkpoint_sha256")
    ):
        raise ValueError("Phase-3 training receipt differs from terminal evaluation")
    audit_path = Path(str(receipt.get("training_audit_path", ""))).resolve()
    if (
        not audit_path.is_file()
        or _sha256_file(audit_path) != receipt.get("training_audit_sha256")
    ):
        raise ValueError("Phase-3 training audit artifact drifted")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    unsigned_audit = dict(audit)
    audit_sha = unsigned_audit.pop("audit_sha256", None)
    if (
        audit_sha != _canonical_sha256(unsigned_audit)
        or audit.get("status") != "complete"
        or audit.get("variant") != expected_source
        or int(audit.get("seed", -1)) != int(seed)
        or int(audit.get("research_phase", -1)) != 3
        or int(audit.get("update_audit", {}).get("successful_optimizer_updates", -1))
        != 6000
    ):
        raise ValueError("Phase-3 training audit is invalid")
    return {
        "identity": dict(identity),
        "receipt": receipt,
        "audit": audit,
    }


def finalize_arm(
    *,
    arm: str,
    seed: int,
    localization_metrics: str | Path,
    ledger_summary: str | Path | None,
    cost_evidence: str | Path | None,
    output: str | Path,
) -> dict[str, Any]:
    if arm not in set(PHASE3_TRAIN_ARMS) | {"U-same-K"}:
        raise ValueError(f"unregistered Phase-3 arm: {arm}")
    metrics_path, metrics = _load_json(localization_metrics)
    _verify_content_hash(metrics, label="localization metrics")
    terminal_path, terminal = _load_json(metrics["terminal_evaluation_path"])
    if (
        _sha256_file(terminal_path) != metrics.get("terminal_evaluation_sha256")
        or metrics.get("schema_version") != "duca_rime_localization_metrics_v1"
        or int(metrics.get("phase", -1)) != 3
        or metrics.get("variant") != arm
        or int(metrics.get("seed", -1)) != int(seed)
        or metrics.get("split_role") != "certification_development"
        or metrics.get("uses_official_final") is not False
        or metrics.get("official_final_used_for_training_or_selection") is not False
        or metrics.get("padded_to_kmax") is not False
        or metrics.get("detector_backend") != "ActionFormer"
        or float(metrics.get("target_mean_cost", math.nan)) != 384.0
        or terminal.get("variant") != arm
        or int(terminal.get("seed", -1)) != int(seed)
        or terminal.get("padded_to_kmax") is not False
    ):
        raise ValueError("localization evidence is not the requested Phase-3 arm")
    training = _validate_training_identity(terminal, arm=arm, seed=seed)
    identity = training["identity"]
    audit = training["audit"]

    k_histogram: dict[str, int] = {}
    ledger_artifact = None
    if arm in LEDGER_ARMS:
        if ledger_summary is None:
            raise ValueError(f"{arm} requires a sealed inference ledger")
        ledger_path, ledger = _load_json(ledger_summary)
        ledger_data_path = Path(str(ledger.get("path", ""))).resolve()
        if (
            ledger.get("schema_version")
            != "duca_rime_inference_ledger_summary_v1"
            or ledger.get("status") != "sealed"
            or ledger.get("arm") != LEDGER_ARMS[arm]
            or ledger.get("no_padding_ledger") is not True
            or ledger.get("official_final_labels_used_for_decision") is not False
            or not ledger_data_path.is_file()
            or _sha256_file(ledger_data_path) != ledger.get("sha256")
        ):
            raise ValueError("Phase-3 inference ledger summary is invalid")
        k_histogram = {
            str(key): int(value)
            for key, value in ledger.get("requested_k_histogram", {}).items()
        }
        if not k_histogram or any(value <= 0 for value in k_histogram.values()):
            raise ValueError("Phase-3 K histogram is empty or invalid")
        ledger_artifact = {
            "path": str(ledger_path),
            "sha256": _sha256_file(ledger_path),
            "ledger_path": str(ledger_data_path),
            "ledger_sha256": str(ledger["sha256"]),
        }
    elif ledger_summary is not None:
        raise ValueError("U-fixed must not consume a selector inference ledger")

    cost = None
    if cost_evidence is not None:
        cost_path, cost = _load_json(cost_evidence)
        _verify_content_hash(cost, label="cost evidence")
        if (
            cost.get("schema_version") != "duca_rime_paired_full_stack_cost_v1"
            or int(cost.get("research_phase", -1)) != 3
            or cost.get("arm") != arm
            or int(cost.get("seed", -1)) != int(seed)
            or cost.get("detector_backend") != "ActionFormer"
            or float(cost.get("target_mean_cost", math.nan)) != 384.0
            or cost.get("real_full_stack_measurement") is not True
            or cost.get("includes_probe_decoder_solver") is not True
        ):
            raise ValueError("paired cost evidence differs from its Phase-3 arm")
        cost = {
            **cost,
            "artifact_path": str(cost_path),
            "artifact_sha256": _sha256_file(cost_path),
        }
    if arm == "RIME-full" and cost is None:
        raise ValueError("RIME-full requires paired fixed/dense cost evidence")
    if arm != "RIME-full" and cost is not None:
        raise ValueError("only RIME-full carries the shared Phase-3 paired cost artifact")

    requested_metrics = ("avg_map", "map_0.7", "short_map", "pair_support")
    video_metrics = metrics.get("video_metrics")
    if not isinstance(video_metrics, Mapping):
        raise ValueError("Phase-3 localization video metrics are missing")
    selected_metrics = {}
    expected_videos = set(str(value) for value in metrics["evaluation_video_ids"])
    for name in requested_metrics:
        values = video_metrics.get(name)
        if not isinstance(values, Mapping) or set(map(str, values)) != expected_videos:
            raise ValueError(f"Phase-3 localization metric {name} has video drift")
        selected_metrics[name] = {
            str(key): float(value) for key, value in values.items()
        }
        if not all(math.isfinite(value) for value in selected_metrics[name].values()):
            raise ValueError(f"Phase-3 localization metric {name} is nonfinite")

    evaluation_only = arm == "U-same-K"
    payload = {
        "schema_version": PHASE3_RESULT_SCHEMA,
        "arm": arm,
        "seed": int(seed),
        "git_commit": str(metrics["git_commit"]),
        "uses_official_final": False,
        "split_assignment_sha256": str(metrics["split_assignment_sha256"]),
        "padded_to_kmax": False,
        "evaluation_video_ids": list(metrics["evaluation_video_ids"]),
        "initialization_sha256": str(identity["initialization_sha256"]),
        "training_exposure_sha256": str(identity["training_exposure_sha256"]),
        "k_histogram": k_histogram,
        "video_metrics": selected_metrics,
        "evaluation_only": evaluation_only,
        "independent_training_run": not evaluation_only,
        "successful_detector_updates": 0 if evaluation_only else 6000,
        "formal_update_audit_passed": None if evaluation_only else True,
        "training_receipt_sha256": (
            None if evaluation_only else str(identity["training_receipt_sha256"])
        ),
        "source_training_arm": "RIME-full" if evaluation_only else None,
        "source_successful_detector_updates": 6000 if evaluation_only else None,
        "source_formal_update_audit_passed": True if evaluation_only else None,
        "source_training_receipt_sha256": (
            str(identity["training_receipt_sha256"]) if evaluation_only else None
        ),
        "localization_metrics_artifact": {
            "path": str(metrics_path),
            "sha256": _sha256_file(metrics_path),
        },
        "terminal_evaluation_artifact": {
            "path": str(terminal_path),
            "sha256": _sha256_file(terminal_path),
        },
        "inference_ledger_artifact": ledger_artifact,
        "cost": cost,
        "official_final_used_for_training_or_selection": False,
        "training_audit_sha256": str(audit["audit_sha256"]),
    }
    target = Path(output).expanduser().resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to overwrite different Phase-3 arm evidence: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {"path": str(target), "sha256": _sha256_file(target), "payload": payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize one DUCA-RIME Phase-3 arm.")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--localization-metrics", required=True)
    parser.add_argument("--ledger-summary")
    parser.add_argument("--cost-evidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = finalize_arm(
        arm=args.arm,
        seed=args.seed,
        localization_metrics=args.localization_metrics,
        ledger_summary=args.ledger_summary,
        cost_evidence=args.cost_evidence,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
