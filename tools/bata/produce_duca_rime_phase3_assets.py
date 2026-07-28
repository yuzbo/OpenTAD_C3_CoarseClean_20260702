from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from tools.bata.build_duca_rime_budget_replay import (
    adaptok_test_batch_ilp,
    histogram_shuffle,
)
from tools.bata.create_duca_rime_splits import validate_rime_splits
from tools.bata.duca_rime_phase2 import _choose_budget
from tools.bata.produce_duca_rime_counterfactual_measurements import (
    FEATURE_SCHEMA,
    budget_features,
    cheap_frame_features,
)


SUMMARY_SCHEMA = "duca_rime_phase3_asset_bundle_v1"
TARGET_SCHEMA = "duca_rime_training_target_v1"
ALLOCATION_SCHEMA = "duca_rime_training_allocation_v1"
ADAPTOK_CURVE_SCHEMA = "duca_adaptok_total_loss_curve_v1"
DEFAULT_BUDGETS = (192, 256, 384, 512)


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
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _load_bound_json(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or _sha256_file(resolved) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 drift")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return resolved, payload


def _write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if count <= 0:
            raise ValueError(f"refusing to write an empty artifact: {target}")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "record_count": count,
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"path": str(target), "sha256": _sha256_file(target)}


def _predict(coefficients: Sequence[float], features: Sequence[float]) -> float:
    if len(coefficients) != len(features) + 1:
        raise ValueError("cross-fit coefficient/feature dimensions drifted")
    value = float(coefficients[0]) + sum(
        float(weight) * float(feature)
        for weight, feature in zip(coefficients[1:], features)
    )
    if not math.isfinite(value):
        raise RuntimeError("cross-fit prediction became non-finite")
    return value


def _validate_protocol(
    *,
    phase2: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    budgets: tuple[int, ...],
) -> tuple[tuple[int, ...], float]:
    costs_float = tuple(float(value) for value in protocol.get("candidate_costs", ()))
    costs = tuple(int(value) for value in costs_float)
    formal = phase2.get("formal_budget_protocols")
    bound = [
        row
        for row in formal
        if isinstance(row, Mapping)
        and float(row.get("target_mean_cost", math.nan)) == 384.0
    ] if isinstance(formal, list) else []
    if (
        phase2.get("schema_version") != "duca_rime_stage_receipt_v1"
        or phase2.get("phase") != "phase2"
        or phase2.get("gate_pass") is not True
        or phase2.get("phase3_training_authorized") is not True
        or phase2.get("official_final_subset_consumed") is not False
        or len(bound) != 1
        or Path(str(bound[0].get("path", ""))).resolve() != protocol_path
        or str(bound[0].get("sha256", "")) != protocol_sha256
        or protocol.get("schema_version") != "duca_rime_budget_protocol_v1"
        or protocol.get("gate_pass") is not True
        or protocol.get("fit_split") != "train_only"
        or protocol.get("uses_validation_or_test_labels") is not False
        or tuple(int(value) for value in protocol.get("candidate_budgets", ()))
        != budgets
        or len(costs) != len(budgets)
        or tuple(float(value) for value in costs) != costs_float
        or any(right <= left for left, right in zip(costs[:-1], costs[1:]))
        or float(protocol.get("target_mean_cost", math.nan)) != 384.0
    ):
        raise ValueError("Phase-3 assets require the hash-bound K384 Phase-2 protocol")
    return costs, float(protocol["target_mean_cost"])


def _validate_model(
    summary: Mapping[str, Any],
    *,
    split_manifest_sha256: str,
    split_assignment_sha256: str,
    budgets: tuple[int, ...],
    target_videos: set[str],
    development_videos: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models = summary.get("models")
    split = summary.get("split_manifest")
    target_model = models.get("training_targets") if isinstance(models, Mapping) else None
    o2_model = models.get("o2_decoder") if isinstance(models, Mapping) else None
    if (
        summary.get("schema_version") != "duca_rime_crossfit_record_producer_v1"
        or summary.get("status") != "produced"
        or tuple(int(value) for value in summary.get("candidate_budgets", ()))
        != budgets
        or not isinstance(split, Mapping)
        or split.get("sha256") != split_manifest_sha256
        or split.get("assignment_sha256") != split_assignment_sha256
        or not isinstance(target_model, Mapping)
        or not isinstance(o2_model, Mapping)
        or set(map(str, target_model.get("fit_video_ids", ())) ) & target_videos
        or set(map(str, o2_model.get("fit_video_ids", ()))) & development_videos
        or o2_model.get("eval_role") != "certification_development"
        or o2_model.get("runtime_decoder_api") != "decode_rime_panel"
    ):
        raise ValueError("cross-fit summary cannot produce leakage-free Phase-3 assets")
    required = (
        "utility_coefficients",
        "risk_coefficients",
        "frame_coefficients",
        "risk_absolute_residual_q95",
    )
    if any(name not in target_model or name not in o2_model for name in required):
        raise ValueError("cross-fit summary lacks one or more frozen predictors")
    return dict(target_model), dict(o2_model)


def _tensor_sample(
    item: Mapping[str, Any],
) -> tuple[Any, int, int, dict[str, Any]]:
    import torch

    inputs = item["inputs"]
    if not torch.is_tensor(inputs):
        raise TypeError("asset production requires tensor RGB inputs")
    if inputs.ndim in {2, 4, 5}:
        inputs = inputs.unsqueeze(0)
    masks = item["masks"]
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)
    temporal_len = int(masks.numel())
    valid_len = int(masks.to(dtype=torch.bool).sum().item())
    metas = item["metas"]
    if isinstance(metas, (list, tuple)):
        if len(metas) != 1:
            raise ValueError("asset production requires one sample at a time")
        metas = metas[0]
    if not isinstance(metas, Mapping):
        raise TypeError("asset production requires mapping metadata")
    return inputs, valid_len, temporal_len, dict(metas)


def _predicted_panel(
    inputs: Any,
    valid_len: int,
    *,
    budgets: tuple[int, ...],
    model: Mapping[str, Any],
) -> tuple[list[list[float]], list[float], list[float], list[float]]:
    features = cheap_frame_features(inputs, valid_len)
    panel = [
        budget_features(features, budget=budget, maximum_budget=budgets[-1])
        for budget in budgets
    ]
    utility = [
        _predict(model["utility_coefficients"], values)
        for values in panel
    ]
    risk = [
        min(
            1.0,
            max(0.0, _predict(model["risk_coefficients"], values)),
        )
        for values in panel
    ]
    hard = [
        _predict(model["frame_coefficients"], values)
        for values in features
    ]
    return features, utility, risk, hard


def produce_assets(
    *,
    config: str | Path,
    phase2_receipt: str | Path,
    phase2_receipt_sha256: str,
    budget_protocol: str | Path,
    budget_protocol_sha256: str,
    crossfit_summary: str | Path,
    crossfit_summary_sha256: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    output_root: str | Path,
    candidate_budgets: Sequence[int] = DEFAULT_BUDGETS,
    epochs: int = 60,
    seed: int = 3407,
) -> dict[str, Any]:
    from mmengine.config import Config
    from opentad.datasets import build_dataset

    budgets = tuple(int(value) for value in candidate_budgets)
    if budgets != DEFAULT_BUDGETS or int(epochs) != 60 or int(seed) != 3407:
        raise ValueError("formal Phase-3 assets are frozen to four budgets/60 epochs/seed3407")
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Phase-3 assets require a fresh root: {root}")
    root.mkdir(parents=True)

    split_path = Path(split_manifest).expanduser().resolve()
    split_validation = validate_rime_splits(
        split_path,
        expected_sha256=split_manifest_sha256,
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    target_videos = set(
        map(str, split["train_roles"]["detector_selector_train"]["videos"])
    )
    development_videos = set(
        map(str, split["train_roles"]["certification_development"]["videos"])
    )
    phase2_path, phase2 = _load_bound_json(
        phase2_receipt,
        phase2_receipt_sha256,
        label="Phase-2 receipt",
    )
    protocol_path, protocol = _load_bound_json(
        budget_protocol,
        budget_protocol_sha256,
        label="K384 budget protocol",
    )
    costs, target_mean_cost = _validate_protocol(
        phase2=phase2,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=budget_protocol_sha256,
        budgets=budgets,
    )
    summary_path, crossfit = _load_bound_json(
        crossfit_summary,
        crossfit_summary_sha256,
        label="cross-fit summary",
    )
    unsigned = dict(crossfit)
    embedded = unsigned.pop("content_sha256", None)
    if embedded != _canonical_sha256(unsigned):
        raise ValueError("cross-fit summary content hash is invalid")
    target_model, o2_model = _validate_model(
        crossfit,
        split_manifest_sha256=split_validation["manifest_sha256"],
        split_assignment_sha256=split_validation["assignment_sha256"],
        budgets=budgets,
        target_videos=target_videos,
        development_videos=development_videos,
    )

    cfg = Config.fromfile(str(Path(config).expanduser().resolve()))
    if (
        cfg.workflow.formal_protocol
        != "duca_rime_phase2_mixed_k_baseline_v1"
        or tuple(int(value) for value in cfg.duca_rime_variant.candidate_budgets)
        != budgets
    ):
        raise ValueError("asset producer requires the frozen U-mixed-K data contract")
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
    train_data_videos = {str(row[0]) for row in train_dataset.data_list}
    if train_data_videos != target_videos or len(train_dataset) != len(target_videos):
        raise RuntimeError("asset producer train dataset differs from detector_selector_train")

    model_sha = _canonical_sha256(target_model)
    targets = []
    allocations = []
    adaptok_train_curves = []
    schedule_keys = set()
    epoch_histograms: dict[str, Counter[int]] = defaultdict(Counter)
    for epoch in range(int(epochs)):
        train_dataset.set_epoch(epoch)
        for sample_index in range(len(train_dataset)):
            inputs, valid_len, temporal_len, meta = _tensor_sample(
                train_dataset[sample_index]
            )
            video = str(meta.get("video_name") or meta.get("video_id") or "")
            start = int(meta.get("window_start_frame", -1))
            observed_epoch = int(meta.get("duca_stateless_epoch", -1))
            observed_index = int(meta.get("duca_stateless_sample_index", -1))
            if (
                video not in target_videos
                or start < 0
                or observed_epoch != epoch
                or observed_index != sample_index
                or valid_len < budgets[-1]
            ):
                raise RuntimeError("stateless target schedule drifted from formal training")
            key = (video, start, epoch, sample_index)
            if key in schedule_keys:
                raise RuntimeError("duplicate stateless target schedule identity")
            schedule_keys.add(key)
            features, utility, risk, hard = _predicted_panel(
                inputs,
                valid_len,
                budgets=budgets,
                model=target_model,
            )
            if temporal_len < valid_len:
                raise RuntimeError("training mask is shorter than its valid prefix")
            hard = [*hard, *([0.0] * (temporal_len - valid_len))]
            provenance = {
                "fit_split": "train_only",
                "cross_fitted": True,
                "uses_validation_or_test": False,
                "uses_official_final": False,
                "uses_gt_for_supervision": True,
                "uses_gt_at_deployment": False,
                "uses_teacher_at_deployment": False,
                "uses_prediction_cache_at_deployment": False,
                "cheap_features_only_at_deployment": True,
                "target_role": "detector_selector_train",
                "fit_roles": list(target_model["fit_roles"]),
                "fit_video_ids": list(target_model["fit_video_ids"]),
                "eval_video_ids": [video],
                "split_manifest_sha256": split_validation["manifest_sha256"],
                "split_assignment_sha256": split_validation["assignment_sha256"],
                "crossfit_summary_sha256": crossfit_summary_sha256,
                "crossfit_model_sha256": model_sha,
                "feature_schema": FEATURE_SCHEMA,
                "complete_stateless_schedule": True,
            }
            target_row = {
                "schema_version": TARGET_SCHEMA,
                "video_id": video,
                "window_start_frame": start,
                "duca_stateless_epoch": epoch,
                "duca_stateless_sample_index": sample_index,
                "candidate_budgets": list(budgets),
                "utility_target": utility,
                "risk_target": risk,
                "target_mask": [True] * len(budgets),
                "hard_frame_utility": hard,
                "provenance": provenance,
            }
            target_row["record_sha256"] = _canonical_sha256(target_row)
            targets.append(target_row)

            risk_upper = [
                min(1.0, value + float(target_model["risk_absolute_residual_q95"]))
                for value in risk
            ]
            selected_index = _choose_budget(
                utility,
                risk_upper,
                costs,
                price=float(protocol["frozen_price"]),
                risk_weight=float(protocol["risk_weight"]),
                risk_threshold=float(protocol["risk_threshold"]),
            )
            allocation = {
                "schema_version": ALLOCATION_SCHEMA,
                "video_id": video,
                "window_start_frame": start,
                "duca_stateless_epoch": epoch,
                "duca_stateless_sample_index": sample_index,
                "requested_k": budgets[selected_index],
                "candidate_budgets": list(budgets),
                "protocol_sha256": budget_protocol_sha256,
                "target_record_sha256": target_row["record_sha256"],
                "uses_gt": False,
                "uses_teacher": False,
                "uses_prediction_cache": False,
            }
            allocation["allocation_sha256"] = _canonical_sha256(allocation)
            allocations.append(allocation)
            epoch_histograms[str(epoch)][budgets[selected_index]] += 1
            adaptok_train_curves.append(
                {
                    "schema_version": ADAPTOK_CURVE_SCHEMA,
                    "video_id": video,
                    "window_start_frame": start,
                    "duca_stateless_epoch": epoch,
                    "duca_stateless_sample_index": sample_index,
                    "candidate_budgets": list(budgets),
                    "predicted_total_loss": [-float(value) for value in utility],
                    "provenance": {
                        "uses_gt": False,
                        "uses_gt_for_supervision": True,
                        "uses_gt_at_decision": False,
                        "uses_teacher": False,
                        "uses_official_final": False,
                        "cross_fitted": True,
                        "test_batch_curve": True,
                        "partition": "detector_selector_train_epoch",
                        "crossfit_summary_sha256": crossfit_summary_sha256,
                    },
                }
            )

    if len(targets) != 6000 or len(schedule_keys) != 6000:
        raise RuntimeError("formal Phase-3 target schedule must contain exactly 6000 rows")

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=None))
    test_data_videos = {str(row[0]) for row in test_dataset.data_list}
    if test_data_videos != development_videos:
        raise RuntimeError("asset producer test dataset differs from certification_development")
    adaptok_development_curves = []
    development_keys = set()
    for sample_index in range(len(test_dataset)):
        inputs, valid_len, _temporal_len, meta = _tensor_sample(
            test_dataset[sample_index]
        )
        video = str(meta.get("video_name") or meta.get("video_id") or "")
        start = int(meta.get("window_start_frame", -1))
        key = (video, start)
        if video not in development_videos or start < 0 or key in development_keys:
            raise RuntimeError("invalid or duplicate development sliding-window identity")
        development_keys.add(key)
        _features, utility, _risk, _hard = _predicted_panel(
            inputs,
            valid_len,
            budgets=budgets,
            model=o2_model,
        )
        adaptok_development_curves.append(
            {
                "schema_version": ADAPTOK_CURVE_SCHEMA,
                "video_id": video,
                "window_start_frame": start,
                "candidate_budgets": list(budgets),
                "predicted_total_loss": [-float(value) for value in utility],
                "provenance": {
                    "uses_gt": False,
                    "uses_gt_for_supervision": True,
                    "uses_gt_at_decision": False,
                    "uses_teacher": False,
                    "uses_official_final": False,
                    "cross_fitted": True,
                    "test_batch_curve": True,
                    "partition": "certification_development_sliding_windows",
                    "crossfit_summary_sha256": crossfit_summary_sha256,
                },
            }
        )

    artifacts = {
        "training_targets": _write_jsonl(root / "training_targets.jsonl", targets),
        "rime_training_allocation": _write_jsonl(
            root / "rime_training_allocation.jsonl",
            allocations,
        ),
        "adaptok_training_curves": _write_jsonl(
            root / "adaptok_training_curves.jsonl",
            adaptok_train_curves,
        ),
        "adaptok_development_curves": _write_jsonl(
            root / "adaptok_development_curves.jsonl",
            adaptok_development_curves,
        ),
    }
    dshuffle = histogram_shuffle(
        allocations,
        seed=int(seed),
        candidate_budgets=budgets,
        source_sha256=artifacts["rime_training_allocation"]["sha256"],
    )
    artifacts["dshuffle_training_replay"] = _write_jsonl(
        root / "dshuffle_training_replay.jsonl",
        dshuffle,
    )

    adaptok_training_replay = []
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in adaptok_train_curves:
        by_epoch[int(row["duca_stateless_epoch"])].append(row)
    for epoch in range(int(epochs)):
        adaptok_training_replay.extend(
            adaptok_test_batch_ilp(
                by_epoch[epoch],
                candidate_budgets=budgets,
                candidate_costs=costs,
                target_mean_cost=target_mean_cost,
                source_sha256=artifacts["adaptok_training_curves"]["sha256"],
            )
        )
    adaptok_development_replay = adaptok_test_batch_ilp(
        adaptok_development_curves,
        candidate_budgets=budgets,
        candidate_costs=costs,
        target_mean_cost=target_mean_cost,
        source_sha256=artifacts["adaptok_development_curves"]["sha256"],
    )
    artifacts["adaptok_replay"] = _write_jsonl(
        root / "adaptok_replay.jsonl",
        [*adaptok_training_replay, *adaptok_development_replay],
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "produced",
        "claim_scope": "training_and_development_control_assets_only_no_model_result",
        "phase2_receipt": {
            "path": str(phase2_path),
            "sha256": phase2_receipt_sha256,
        },
        "budget_protocol": {
            "path": str(protocol_path),
            "sha256": budget_protocol_sha256,
            "target_mean_cost": target_mean_cost,
        },
        "crossfit_summary": {
            "path": str(summary_path),
            "sha256": crossfit_summary_sha256,
        },
        "split_manifest": {
            "path": str(split_path),
            "sha256": split_validation["manifest_sha256"],
            "assignment_sha256": split_validation["assignment_sha256"],
        },
        "candidate_budgets": list(budgets),
        "candidate_costs": list(costs),
        "stateless_seed": int(seed),
        "stateless_epoch_count": int(epochs),
        "successful_update_schedule_count": len(targets),
        "development_window_count": len(adaptok_development_curves),
        "rime_training_k_histogram": {
            str(value): sum(
                histogram.get(value, 0) for histogram in epoch_histograms.values()
            )
            for value in budgets
        },
        "rime_training_mean_cost": mean(
            int(row["requested_k"]) for row in allocations
        ),
        "artifacts": artifacts,
        "official_final_subset_consumed": False,
        "uses_gt_at_deployment": False,
        "uses_teacher_at_deployment": False,
        "uses_prediction_cache_at_deployment": False,
    }
    summary["content_sha256"] = _canonical_sha256(summary)
    summary_artifact = _write_json(root / "producer_summary.json", summary)
    summary["output_path"] = summary_artifact["path"]
    summary["output_sha256"] = summary_artifact["sha256"]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Produce the complete 6000-update DUCA-RIME target schedule and "
            "Phase-3 replay controls from frozen Phase-2 cross-fit models."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase2-receipt", required=True)
    parser.add_argument("--phase2-receipt-sha256", required=True)
    parser.add_argument("--budget-protocol", required=True)
    parser.add_argument("--budget-protocol-sha256", required=True)
    parser.add_argument("--crossfit-summary", required=True)
    parser.add_argument("--crossfit-summary-sha256", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--candidate-budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    result = produce_assets(
        config=args.config,
        phase2_receipt=args.phase2_receipt,
        phase2_receipt_sha256=args.phase2_receipt_sha256,
        budget_protocol=args.budget_protocol,
        budget_protocol_sha256=args.budget_protocol_sha256,
        crossfit_summary=args.crossfit_summary,
        crossfit_summary_sha256=args.crossfit_summary_sha256,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        output_root=args.output_root,
        candidate_budgets=args.candidate_budgets,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
