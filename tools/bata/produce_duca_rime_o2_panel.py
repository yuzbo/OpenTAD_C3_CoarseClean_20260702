from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import validate_rime_splits
from tools.bata.produce_duca_rime_counterfactual_measurements import (
    FEATURE_SCHEMA,
    MEASUREMENT_SCHEMA,
    _detector_objective,
    _load_ema,
    _sample_identity,
    _write_jsonl_atomic,
    cheap_frame_features,
)
from tools.bata.produce_duca_rime_crossfit_records import (
    DEFAULT_BUDGETS,
    SUMMARY_SCHEMA as CROSSFIT_SUMMARY_SCHEMA,
    _canonical_sha256,
    _predict,
    _read_jsonl,
    _role_maps,
    _validate_measurements,
)


METRICS_SCHEMA = "duca_rime_counterfactual_decoder_metrics_v1"
SUMMARY_SCHEMA = "duca_rime_o2_runtime_panel_producer_v1"
LEDGER_SCHEMA = "duca_rime_inference_ledger_v1"
SCORE_METRIC = "counterfactual_negative_detector_loss"
CLAIM_SCOPE = (
    "measured_detector_objective_decoder_family_regret_"
    "not_tad_map_not_localization_quality"
)
O2_ROLE = "certification_development"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"O2 artifacts never overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    text = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"path": str(target), "sha256": _sha256_file(target)}


def _load_bound_json(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or _sha256_file(resolved) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 binding drifted")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return resolved, payload


def _validate_content_sha(payload: Mapping[str, Any], *, label: str) -> None:
    claimed = str(payload.get("content_sha256", ""))
    without = dict(payload)
    without.pop("content_sha256", None)
    if len(claimed) != 64 or _canonical_sha256(without) != claimed:
        raise ValueError(f"{label} content SHA-256 is invalid")


def _observed_max_gap(
    positions: Sequence[int],
    physical_seconds: Sequence[float],
) -> float:
    selected = tuple(int(value) for value in positions)
    seconds = tuple(float(value) for value in physical_seconds)
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or selected[0] < 0
        or selected[-1] >= len(seconds)
        or not seconds
        or any(not math.isfinite(value) for value in seconds)
    ):
        raise ValueError("O2 selected positions/physical axis are invalid")
    return max(
        seconds[selected[0]] - seconds[0],
        seconds[-1] - seconds[selected[-1]],
        *(
            seconds[right] - seconds[left]
            for left, right in zip(selected[:-1], selected[1:])
        ),
    )


def _validate_training_receipt(
    receipt: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> None:
    if (
        receipt.get("schema_version")
        != "duca_rime_phase2_mixed_k_training_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("arm") != "U-mixed-K"
        or receipt.get("checkpoint_sha256") != str(checkpoint_sha256)
        or receipt.get("detector_training_exposure")
        != "mixed_k_registered_panel"
        or int(receipt.get("successful_detector_updates", -1)) != 6000
        or receipt.get("uses_official_final") is not False
    ):
        raise ValueError("O2 requires the registered 6000-update U-mixed-K receipt")


def _validate_crossfit_summary(
    summary: Mapping[str, Any],
    *,
    split_manifest_sha256: str,
    split_assignment_sha256: str,
    measurements_sha256: str,
    budgets: tuple[int, ...],
    expected_eval_videos: set[str],
) -> dict[str, Any]:
    _validate_content_sha(summary, label="cross-fit producer summary")
    model = (summary.get("models") or {}).get("o2_decoder")
    source = summary.get("source_measurements")
    split = summary.get("split_manifest")
    if (
        summary.get("schema_version") != CROSSFIT_SUMMARY_SCHEMA
        or summary.get("status") != "produced"
        or tuple(int(value) for value in summary.get("candidate_budgets", ()))
        != budgets
        or not isinstance(source, Mapping)
        or source.get("sha256") != str(measurements_sha256)
        or source.get("schema_version") != MEASUREMENT_SCHEMA
        or source.get("measurement_kind")
        != "measured_detector_counterfactual"
        or source.get("proposal_score_surrogate_utility") is not False
        or not isinstance(split, Mapping)
        or split.get("sha256") != str(split_manifest_sha256)
        or split.get("assignment_sha256") != str(split_assignment_sha256)
        or not isinstance(model, Mapping)
        or model.get("eval_role") != O2_ROLE
        or model.get("runtime_decoder_api") != "decode_rime_panel"
        or model.get("claim_scope")
        != "counterfactual_detector_objective_decoder_family_regret_not_tad_map"
    ):
        raise ValueError("cross-fit summary is not eligible for the formal O2 panel")
    fit_videos = set(map(str, model.get("fit_video_ids", ())))
    fit_roles = set(map(str, model.get("fit_roles", ())))
    coefficients = [float(value) for value in model.get("frame_coefficients", ())]
    if (
        len(fit_videos) < 3
        or bool(fit_videos & expected_eval_videos)
        or not {
            "hard_label_generation",
            "utility_risk_fit",
            "dual_risk_calibration",
        }
        <= fit_roles
        or len(coefficients) < 2
        or not all(math.isfinite(value) for value in coefficients)
    ):
        raise ValueError("O2 frame scorer is not leakage-free or finite")
    return dict(model)


def produce_o2_panel(
    *,
    config: str | Path,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    training_receipt: str | Path,
    training_receipt_sha256: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    measurements_jsonl: str | Path,
    measurements_sha256: str,
    crossfit_summary: str | Path,
    crossfit_summary_sha256: str,
    output_root: str | Path,
    candidate_budgets: Sequence[int] = DEFAULT_BUDGETS,
    weak_overlap_fraction: float = 0.50,
    seed: int = 3407,
    device: str = "cuda:0",
    num_workers: int = 2,
    backbone_pretrain: str | Path | None = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.models.duca.rime import RIME_DECODER_FAMILIES, decode_rime_panel

    budgets = tuple(int(value) for value in candidate_budgets)
    if (
        budgets != tuple(DEFAULT_BUDGETS)
        or tuple(sorted(set(budgets))) != budgets
        or any(value % 16 for value in budgets)
        or not 0.0 < float(weak_overlap_fraction) <= 1.0
    ):
        raise ValueError("formal O2 requires the registered four-budget mixed-K panel")
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"formal O2 requires a fresh output root: {output}")
    output.mkdir(parents=True)

    split_path = Path(split_manifest).expanduser().resolve()
    split_validation = validate_rime_splits(
        split_path,
        expected_sha256=split_manifest_sha256,
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    role_by_video, videos_by_role = _role_maps(split)
    expected_eval_videos = set(videos_by_role[O2_ROLE])

    measurement_path, measurement_source_rows = _read_jsonl(measurements_jsonl)
    if _sha256_file(measurement_path) != str(measurements_sha256):
        raise ValueError("O2 counterfactual measurement SHA-256 drift")
    measurement_rows = _validate_measurements(
        measurement_source_rows,
        budgets=budgets,
        assignment_sha256=split_validation["assignment_sha256"],
        role_by_video=role_by_video,
    )
    expected_rows = {
        (str(row["video_id"]), int(row["window_start_frame"])): row
        for row in measurement_rows
        if row["role"] == O2_ROLE
    }
    if (
        {video for video, _start in expected_rows} != expected_eval_videos
        or len(expected_rows) != len(expected_eval_videos)
    ):
        raise ValueError("formal O2 requires one frozen measurement per evaluation video")

    receipt_path, receipt = _load_bound_json(
        training_receipt,
        training_receipt_sha256,
        label="mixed-K training receipt",
    )
    _validate_training_receipt(receipt, checkpoint_sha256=checkpoint_sha256)
    summary_path, crossfit = _load_bound_json(
        crossfit_summary,
        crossfit_summary_sha256,
        label="cross-fit producer summary",
    )
    scorer = _validate_crossfit_summary(
        crossfit,
        split_manifest_sha256=split_validation["manifest_sha256"],
        split_assignment_sha256=split_validation["assignment_sha256"],
        measurements_sha256=measurements_sha256,
        budgets=budgets,
        expected_eval_videos=expected_eval_videos,
    )
    coefficients = [float(value) for value in scorer["frame_coefficients"]]

    config_path = Path(config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if _sha256_file(checkpoint_path) != str(checkpoint_sha256):
        raise ValueError("O2 mixed-K checkpoint SHA-256 drift")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(str(config_path))
    if (
        cfg.workflow.formal_protocol
        != "duca_rime_phase2_mixed_k_baseline_v1"
        or cfg.duca_rime_variant.arm != "U-mixed-K"
        or tuple(int(value) for value in cfg.duca_rime_variant.candidate_budgets)
        != budgets
        or cfg.model.frame_selector.rime_arm != "uniform_mixed_k"
        or cfg.model.frame_selector.actionness_source_cfg is not None
    ):
        raise ValueError("formal O2 requires the frozen U-mixed-K detector config")
    if backbone_pretrain is not None:
        pretrain = Path(backbone_pretrain).expanduser().resolve()
        cfg.model.backbone.custom.pretrain = str(pretrain)
    else:
        pretrain = Path(str(cfg.model.backbone.custom.pretrain)).expanduser().resolve()
    if not pretrain.is_file():
        raise FileNotFoundError(f"O2 VideoMAE initialization is missing: {pretrain}")

    dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
    if callable(getattr(dataset, "set_epoch", None)):
        dataset.set_epoch(0)
    dataset_videos = [str(row[0]) for row in getattr(dataset, "data_list", ())]
    if (
        len(dataset_videos) != len(set(dataset_videos))
        or set(dataset_videos) != expected_eval_videos
    ):
        raise RuntimeError(
            "O2 dataset must expose certification_development exactly once"
        )
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=int(num_workers),
    )
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal O2 detector counterfactuals require CUDA")
    torch.cuda.set_device(torch_device)
    model = build_detector(cfg.model).to(torch_device)
    checkpoint_payload, state_key = _load_ema(model, checkpoint_path)
    model.eval()
    selector = getattr(model, "frame_selector", None)
    if (
        selector is None
        or getattr(selector, "selector_variant", None) != "duca_rime_physical"
        or not callable(getattr(selector, "materialize_counterfactual_positions", None))
    ):
        raise RuntimeError("O2 model lacks the RIME materialization contract")
    frozen_normalizer = None
    if callable(getattr(model.rpn_head, "duca_set_frozen_loss_normalizer", None)):
        frozen_normalizer = model.rpn_head.loss_normalizer.detach().clone()
        model.rpn_head.duca_set_frozen_loss_normalizer(frozen_normalizer)

    scorer_identity = {
        "crossfit_summary_path": str(summary_path),
        "crossfit_summary_sha256": str(crossfit_summary_sha256),
        "frame_coefficients": coefficients,
        "fit_roles": list(scorer["fit_roles"]),
        "fit_video_ids": list(scorer["fit_video_ids"]),
        "eval_role": O2_ROLE,
        "feature_schema": FEATURE_SCHEMA,
    }
    scorer_sha256 = _canonical_sha256(scorer_identity)
    ledger_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    score_rows: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen = set()
    try:
        for data in loader:
            inputs = data["inputs"].to(torch_device, non_blocking=True)
            masks = data["masks"].to(torch_device, non_blocking=True)
            metas = [dict(value) for value in data["metas"]]
            gt_segments = [
                value.to(torch_device, non_blocking=True)
                for value in data["gt_segments"]
            ]
            gt_labels = [
                value.to(torch_device, non_blocking=True)
                for value in data["gt_labels"]
            ]
            if int(inputs.shape[0]) != 1 or len(metas) != 1:
                raise RuntimeError("formal O2 requires batch_size=1")
            video, window_start = _sample_identity(metas[0])
            identity = (video, window_start)
            expected = expected_rows.get(identity)
            if expected is None or identity in seen:
                raise RuntimeError("O2 runtime sample is absent or duplicated")
            valid = masks.to(dtype=torch.bool)
            valid_len = int(valid[0].sum().item())
            if valid_len < budgets[-1]:
                raise RuntimeError("O2 exact-K panel cannot realize Kmax without padding")
            features = cheap_frame_features(inputs, valid_len)
            if features != expected["frame_features"]:
                raise RuntimeError(
                    "O2 cheap features drifted from the frozen measurement epoch"
                )
            if len(coefficients) != len(features[0]) + 1:
                raise RuntimeError("O2 scorer/cheap-feature dimensions drifted")
            scores = [_predict(coefficients, row) for row in features]
            if not all(math.isfinite(value) for value in scores):
                raise RuntimeError("O2 frame scorer produced non-finite potentials")
            policy = torch.as_tensor(
                [scores],
                device=torch_device,
                dtype=torch.float32,
            )
            physical_seconds, _source_frames = selector._physical_axes(
                metas,
                valid,
                torch_device,
            )
            panel = decode_rime_panel(
                policy,
                physical_seconds,
                valid,
                candidate_budgets=budgets,
                weak_overlap_fraction=float(weak_overlap_fraction),
                execution_quantum=16,
            )
            seconds = [
                float(value)
                for value in physical_seconds[0, :valid_len].detach().cpu().tolist()
            ]
            for family in RIME_DECODER_FAMILIES:
                for budget in budgets:
                    decoded = panel[family][budget]
                    decoded.ledger.validate(require_no_padding=True)
                    positions = [
                        int(value)
                        for value in decoded.hard_positions[0, :budget]
                        .detach()
                        .cpu()
                        .tolist()
                    ]
                    if positions != sorted(set(positions)) or len(positions) != budget:
                        raise RuntimeError("O2 runtime decoder violated ordered exact-K")
                    materialized = selector.materialize_counterfactual_positions(
                        inputs,
                        valid,
                        metas,
                        decoded.hard_positions[:, :budget],
                        requested_k=budget,
                    )
                    detector_loss = _detector_objective(
                        model,
                        materialized,
                        gt_segments=gt_segments,
                        gt_labels=gt_labels,
                        use_amp=bool(use_amp),
                    )
                    observed_gap = _observed_max_gap(positions, seconds)
                    gap_cap = float(decoded.max_gap_seconds[0].item())
                    if observed_gap > gap_cap + 1.0e-8:
                        raise RuntimeError("O2 runtime decoder violated the physical gap")
                    score_rows[(family, budget)][video].append(-detector_loss)
                    ledger_row = {
                        "schema_version": LEDGER_SCHEMA,
                        "video_id": video,
                        "window_start_frame": window_start,
                        "dense_valid_len": valid_len,
                        "selected_dense_indices": positions,
                        "requested_k": budget,
                        "effective_k": budget,
                        "unique_k": budget,
                        "backbone_input_k": budget,
                        "padded_k": budget,
                        "observed_max_gap_seconds": observed_gap,
                        "max_gap_seconds_cap": gap_cap,
                        "max_gap_violation": False,
                        "decoder_family": family,
                        "weak_overlap_fraction": (
                            float(weak_overlap_fraction)
                            if family == "weak_overlap"
                            else None
                        ),
                        "runtime_decoder_api": "decode_rime_panel",
                        "selector_scorer_sha256": scorer_sha256,
                        "mixed_k_detector_identity_sha256": str(
                            checkpoint_sha256
                        ),
                        "split_assignment_sha256": split_validation[
                            "assignment_sha256"
                        ],
                        "split_role": O2_ROLE,
                        "uses_official_final": False,
                    }
                    ledger_row["record_sha256"] = _canonical_sha256(ledger_row)
                    ledger_rows[(family, budget)].append(ledger_row)
            seen.add(identity)
            print(
                json.dumps(
                    {
                        "o2_video": video,
                        "completed": len(seen),
                        "total": len(expected_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if frozen_normalizer is not None:
            model.rpn_head.duca_set_frozen_loss_normalizer(None)

    if seen != set(expected_rows):
        raise RuntimeError("O2 runtime panel did not cover the frozen evaluation rows")

    artifacts: dict[str, Any] = {}
    for family in RIME_DECODER_FAMILIES:
        family_root = output / family
        for budget in budgets:
            key = (family, budget)
            ledgers = sorted(
                ledger_rows[key],
                key=lambda row: (
                    str(row["video_id"]),
                    int(row["window_start_frame"]),
                ),
            )
            ledger_artifact = _write_jsonl_atomic(
                family_root / f"k{budget}.ledger.jsonl",
                ledgers,
            )
            video_scores = {
                video: mean(values)
                for video, values in sorted(score_rows[key].items())
            }
            if set(video_scores) != expected_eval_videos:
                raise RuntimeError("O2 family/budget metrics are not rectangular")
            metrics_payload = {
                "schema_version": METRICS_SCHEMA,
                "phase": 2,
                "status": "measured",
                "claim_scope": CLAIM_SCOPE,
                "score_metric": SCORE_METRIC,
                "video_metrics": {SCORE_METRIC: video_scores},
                "window_count": len(ledgers),
                "video_count": len(video_scores),
                "decoder_family": family,
                "budget": budget,
                "target_mean_cost": float(budget),
                "split_role": O2_ROLE,
                "split_assignment_sha256": split_validation["assignment_sha256"],
                "mixed_k_detector_identity_sha256": str(checkpoint_sha256),
                "selector_scorer_sha256": scorer_sha256,
                "runtime_decoder_api": "decode_rime_panel",
                "measurement_kind": "measured_detector_counterfactual",
                "detector_objective": "official_actionformer_cls_plus_reg",
                "counterfactual_score_not_tad_map": True,
                "proposal_score_surrogate_utility": False,
                "padded_to_kmax": False,
                "uses_official_final": False,
                "official_final_used_for_training_or_selection": False,
                "uses_gt_for_measurement": True,
                "uses_gt_at_deployment": False,
                "uses_teacher_at_deployment": False,
                "uses_prediction_cache_at_deployment": False,
                "ledger_sha256": ledger_artifact["sha256"],
            }
            metrics_payload["content_sha256"] = _canonical_sha256(metrics_payload)
            metrics_artifact = _atomic_json(
                family_root / f"k{budget}.metrics.json",
                metrics_payload,
            )
            artifacts[f"{family}:k{budget}"] = {
                "family": family,
                "budget": budget,
                "metrics": metrics_artifact,
                "ledger": ledger_artifact,
            }

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "produced",
        "claim_scope": CLAIM_SCOPE,
        "runtime_decoder_api": "decode_rime_panel",
        "decoder_families": list(RIME_DECODER_FAMILIES),
        "candidate_budgets": list(budgets),
        "weak_overlap_fraction": float(weak_overlap_fraction),
        "score_metric": SCORE_METRIC,
        "split_manifest": {
            "path": str(split_path),
            "sha256": split_validation["manifest_sha256"],
            "assignment_sha256": split_validation["assignment_sha256"],
            "eval_role": O2_ROLE,
        },
        "measurements": {
            "path": str(measurement_path),
            "sha256": str(measurements_sha256),
        },
        "crossfit_summary": {
            "path": str(summary_path),
            "sha256": str(crossfit_summary_sha256),
        },
        "mixed_k_training_receipt": {
            "path": str(receipt_path),
            "sha256": str(training_receipt_sha256),
        },
        "mixed_k_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": str(checkpoint_sha256),
            "epoch": int(checkpoint_payload.get("epoch", -1)),
            "state_key": state_key,
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        },
        "backbone_pretrain": {
            "path": str(pretrain),
            "sha256": _sha256_file(pretrain),
        },
        "selector_scorer": {
            **scorer_identity,
            "sha256": scorer_sha256,
        },
        "artifacts": artifacts,
        "official_final_subset_consumed": False,
        "counterfactual_score_not_tad_map": True,
        "proposal_score_surrogate_utility": False,
        "uses_gt_at_deployment": False,
        "uses_teacher_at_deployment": False,
        "uses_prediction_cache_at_deployment": False,
    }
    summary["content_sha256"] = _canonical_sha256(summary)
    artifact = _atomic_json(output / "producer_summary.json", summary)
    summary["output_path"] = artifact["path"]
    summary["output_sha256"] = artifact["sha256"]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the actual DUCA-RIME decoder families and measure their "
            "train-only ActionFormer detector-loss counterfactual panel."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--training-receipt-sha256", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--measurements-jsonl", required=True)
    parser.add_argument("--measurements-sha256", required=True)
    parser.add_argument("--crossfit-summary", required=True)
    parser.add_argument("--crossfit-summary-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--candidate-budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--weak-overlap-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-pretrain")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    result = produce_o2_panel(
        config=args.config,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        training_receipt=args.training_receipt,
        training_receipt_sha256=args.training_receipt_sha256,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        measurements_jsonl=args.measurements_jsonl,
        measurements_sha256=args.measurements_sha256,
        crossfit_summary=args.crossfit_summary,
        crossfit_summary_sha256=args.crossfit_summary_sha256,
        output_root=args.output_root,
        candidate_budgets=args.candidate_budgets,
        weak_overlap_fraction=args.weak_overlap_fraction,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        backbone_pretrain=args.backbone_pretrain,
        use_amp=not args.no_amp,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
