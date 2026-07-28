from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import validate_rime_splits
from tools.bata.duca_rime_training import validate_terminal_checkpoint_binding
from tools.bata.hrime_stage1_oracle import (
    DEFAULT_CANDIDATE_BUDGETS,
    WINDOW_OPTION_SCHEMA,
    canonical_sha256,
    canonicalize_effective_k_options,
    validate_preregistration,
)
from tools.bata.produce_duca_rime_counterfactual_measurements import (
    _detector_objective,
    _load_ema,
    _sample_identity,
    _sha256_file,
    _write_json_exclusive,
    _write_jsonl_atomic,
)


SUMMARY_SCHEMA = "hrime_stage1_window_option_producer_v1"
ORACLE_UTILITY_METRIC = "max_effective_k_detector_loss_minus_candidate_detector_loss"
ORACLE_RISK_METRIC = "positive_relative_detector_loss_degradation_from_max_effective_k"
PREDICTED_UTILITY_SOURCE = "frozen_rime_budget_controller_predicted_utility"
PREDICTED_RISK_SOURCE = "frozen_rime_budget_controller_risk_upper"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_bound_json(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    _require(
        _sha256_file(resolved) == str(expected_sha256).lower(),
        f"{label} SHA-256 drift",
    )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _git_identity(repo_root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        text=True,
    ).strip()
    _require(
        len(commit) == 40 and all(value in "0123456789abcdef" for value in commit),
        "formal Stage-1 production requires an exact Git commit",
    )
    _require(not dirty, "formal Stage-1 production requires a clean Git tree")
    return commit


def _prepare_dataset_cfg(cfg: Any, *, block_list: str) -> Any:
    dataset_cfg = copy.deepcopy(cfg.dataset.test)
    dataset_cfg.test_mode = False
    dataset_cfg.subset_name = "training"
    dataset_cfg.block_list = str(Path(block_list).expanduser().resolve())
    dataset_cfg.include_background_windows = True
    converted = 0
    collected = 0
    replay_steps = 0
    pipeline = []
    for source_step in dataset_cfg.pipeline:
        step = copy.deepcopy(source_step)
        step_type = str(step.get("type", ""))
        if step_type == "DucaRimeBudgetReplayFromJsonl":
            replay_steps += 1
            continue
        if step_type == "ConvertToTensor":
            step["keys"] = ["imgs", "gt_segments", "gt_labels"]
            converted += 1
        elif step_type == "Collect":
            step["keys"] = ["masks", "gt_segments", "gt_labels"]
            collected += 1
        pipeline.append(step)
    _require(replay_steps == 0, "window-option production forbids a budget replay")
    _require(
        converted == 1 and collected == 1,
        "window-option dataset must expose one tensor and one collection stage",
    )
    dataset_cfg.pipeline = pipeline
    return dataset_cfg


def _float_row(value: Any, *, expected: int, label: str) -> list[float]:
    values = [
        float(item)
        for item in value.detach().float().cpu().reshape(-1).tolist()
    ]
    _require(
        len(values) == int(expected) and all(math.isfinite(item) for item in values),
        f"{label} is non-finite or has the wrong width",
    )
    return values


def produce_window_options(
    *,
    config: str | Path,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    training_receipt: str | Path,
    training_receipt_sha256: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    preregistration: str | Path,
    preregistration_sha256: str,
    output_jsonl: str | Path,
    summary_json: str | Path,
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
    from opentad.models.duca.rime import decode_rime_exact_k

    _require(
        not os.environ.get("DUCA_RIME_INFERENCE_LEDGER_ROOT", "").strip(),
        "window-option production must not append an inference ledger",
    )
    repo_root = Path(__file__).resolve().parents[2]
    git_commit = _git_identity(repo_root)
    config_path = Path(config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    summary_path = Path(summary_json).expanduser().resolve()
    for target in (output_path, summary_path):
        if target.exists():
            raise FileExistsError(target)
    _require(config_path.is_file(), f"Stage-1 source config is missing: {config_path}")
    _require(checkpoint_path.is_file(), f"Stage-1 checkpoint is missing: {checkpoint_path}")
    _require(
        _sha256_file(checkpoint_path) == str(checkpoint_sha256).lower(),
        "Stage-1 checkpoint SHA-256 drift",
    )

    split_path, split = _load_bound_json(
        split_manifest,
        split_manifest_sha256,
        label="H-RIME split manifest",
    )
    split_validation = validate_rime_splits(
        split_path,
        expected_sha256=split_manifest_sha256,
    )
    prereg_path, prereg = _load_bound_json(
        preregistration,
        preregistration_sha256,
        label="H-RIME Stage-1 preregistration",
    )
    validated_prereg = validate_preregistration(
        prereg,
        expected_git_commit=git_commit,
        expected_split_manifest_sha256=split_validation["manifest_sha256"],
        expected_split_assignment_sha256=split_validation["assignment_sha256"],
    )
    budgets = tuple(int(value) for value in validated_prereg["candidate_budgets"])
    _require(
        budgets == DEFAULT_CANDIDATE_BUDGETS,
        "Stage-1 candidate budget panel drift",
    )
    development_role = validated_prereg["development_role"]
    development_binding = split["train_roles"][development_role]
    development_videos = {
        str(video) for video in development_binding["videos"]
    }
    _require(development_videos, "Stage-1 development video set is empty")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(str(config_path))
    _require(
        str(cfg.workflow.formal_protocol) == "duca_rime_physical_dynamic_k_v1"
        and str(cfg.duca_rime_variant.arm) == "RIME-full"
        and str(cfg.model.frame_selector.rime_arm) == "rime_full"
        and tuple(
            int(value) for value in cfg.model.frame_selector.candidate_budgets
        )
        == budgets
        and float(cfg.model.frame_selector.target_mean_cost) == 384.0
        and cfg.duca_rime_contract.official_final_subset_consumed is False,
        "Stage-1 option production requires the frozen Phase-3 RIME-full K384 config",
    )
    _require(
        not os.environ.get("DUCA_RIME_REPLAY_JSONL", "").strip(),
        "Stage-1 option production cannot consume a budget replay",
    )
    if backbone_pretrain is not None:
        pretrain = Path(backbone_pretrain).expanduser().resolve()
        _require(pretrain.is_file(), f"VideoMAE initialization is missing: {pretrain}")
        cfg.model.backbone.custom.pretrain = str(pretrain)
    else:
        pretrain = Path(
            str(cfg.model.backbone.custom.pretrain)
        ).expanduser().resolve()
        _require(pretrain.is_file(), f"VideoMAE initialization is missing: {pretrain}")

    dataset_cfg = _prepare_dataset_cfg(
        cfg,
        block_list=development_binding["block_list_path"],
    )
    dataset = build_dataset(dataset_cfg, default_args=dict(logger=None))
    dataset_videos = [str(row[0]) for row in getattr(dataset, "data_list", ())]
    _require(
        set(dataset_videos) == development_videos,
        "Stage-1 sliding dataset does not expose the exact development videos",
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
    _require(
        torch_device.type == "cuda" and torch.cuda.is_available(),
        "formal Stage-1 detector counterfactual production requires CUDA",
    )
    torch.cuda.set_device(torch_device)
    model = build_detector(cfg.model).to(torch_device)
    checkpoint_payload, state_key = _load_ema(model, checkpoint_path)
    training_identity = validate_terminal_checkpoint_binding(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint_payload,
        git_commit=git_commit,
        evaluation_arm="RIME-full",
        seed=int(seed),
        training_receipt_path=training_receipt,
        training_receipt_sha256=training_receipt_sha256,
    )
    _require(
        training_identity["research_phase"] == 3
        and training_identity["detector_backend"] == "ActionFormer"
        and training_identity["split_assignment_sha256"]
        == split_validation["assignment_sha256"],
        "Stage-1 requires the bound Phase-3 ActionFormer development checkpoint",
    )
    model.eval()
    selector = getattr(model, "frame_selector", None)
    _require(
        selector is not None
        and getattr(selector, "selector_variant", None) == "duca_rime_physical"
        and getattr(selector, "rime_arm", None) == "rime_full"
        and getattr(selector, "budget_controller", None) is not None
        and callable(getattr(selector, "materialize_counterfactual_positions", None)),
        "Stage-1 source checkpoint lacks the frozen RIME-full selector",
    )
    frozen_normalizer = None
    if callable(getattr(model.rpn_head, "duca_set_frozen_loss_normalizer", None)):
        frozen_normalizer = model.rpn_head.loss_normalizer.detach().clone()
        model.rpn_head.duca_set_frozen_loss_normalizer(frozen_normalizer)

    source_identity = {
        "schema_version": "hrime_stage1_window_option_source_v1",
        "git_commit": git_commit,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
        "checkpoint_state_key": state_key,
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "training_receipt_path": str(
            Path(training_receipt).expanduser().resolve()
        ),
        "training_receipt_sha256": str(training_receipt_sha256).lower(),
        "training_exposure_sha256": training_identity[
            "training_exposure_sha256"
        ],
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_validation["manifest_sha256"],
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "development_role": development_role,
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": str(preregistration_sha256).lower(),
        "backbone_pretrain_path": str(pretrain),
        "backbone_pretrain_sha256": _sha256_file(pretrain),
        "candidate_budgets": list(budgets),
        "execution_quantum": 16,
        "seed": int(seed),
        "oracle_utility_metric": ORACLE_UTILITY_METRIC,
        "oracle_risk_metric": ORACLE_RISK_METRIC,
        "predicted_utility_source": PREDICTED_UTILITY_SOURCE,
        "predicted_risk_source": PREDICTED_RISK_SOURCE,
        "dataset_config_sha256": canonical_sha256(dataset_cfg),
        "uses_official_final": False,
    }
    source_identity_sha256 = canonical_sha256(source_identity)
    rows = []
    identities = set()
    window_counts: Counter[str] = Counter()
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
            _require(
                int(inputs.shape[0]) == 1 and len(metas) == 1,
                "Stage-1 option production requires batch_size=1",
            )
            video, window_start = _sample_identity(metas[0])
            identity = (video, window_start)
            _require(
                identity not in identities and video in development_videos,
                "Stage-1 window identity is duplicate or outside development",
            )
            valid = masks.to(dtype=torch.bool)
            valid_length = int(valid[0].sum().item())
            feasible = canonicalize_effective_k_options(valid_length, budgets)

            with torch.no_grad():
                selected = selector._select(
                    inputs,
                    valid,
                    metas,
                    training=False,
                )
            state = selected["selector_outputs"]
            decision = state.get("budget_decision")
            _require(decision is not None, "RIME-full did not emit a budget decision")
            decision.validate(batch_size=1)
            predicted_utility = _float_row(
                decision.predicted_utility[0],
                expected=len(budgets),
                label="predicted utility",
            )
            predicted_risk = _float_row(
                decision.risk_upper[0],
                expected=len(budgets),
                label="predicted risk upper bound",
            )
            predicted_uncertainty = _float_row(
                decision.predicted_uncertainty[0],
                expected=len(budgets),
                label="predicted uncertainty",
            )
            policy_log_prob = state["policy_log_probabilities"].detach()
            physical_seconds = state["physical_seconds"].detach()
            option_work = []
            for choice in feasible.choices:
                nominal = int(choice.canonical_nominal_budget)
                effective = int(choice.effective_k)
                decoded = decode_rime_exact_k(
                    policy_log_prob,
                    physical_seconds,
                    valid,
                    nominal,
                    candidate_budgets=budgets,
                    decoder_family=str(selector.decoder_family),
                    weak_overlap_fraction=float(selector.weak_overlap_fraction),
                    training=False,
                    force_uniform=False,
                    require_homogeneous_execution=True,
                    execution_quantum=16,
                )
                decoded.ledger.validate(require_no_padding=True)
                _require(
                    int(decoded.effective_k[0].item()) == effective,
                    "Stage-1 decoder effective-K alias drift",
                )
                positions = decoded.hard_positions[:, :effective]
                materialized = selector.materialize_counterfactual_positions(
                    inputs,
                    valid,
                    metas,
                    positions,
                    requested_k=nominal,
                    effective_k=effective,
                    measurement_scope=(
                        "certification_development_oracle_measurement"
                    ),
                )
                loss = _detector_objective(
                    model,
                    materialized,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    use_amp=bool(use_amp),
                )
                nominal_index = budgets.index(nominal)
                option_work.append(
                    {
                        "effective_k": effective,
                        "nominal_budgets": list(choice.nominal_budgets),
                        "canonical_nominal_budget": nominal,
                        "predicted_utility": predicted_utility[nominal_index],
                        "predicted_risk": predicted_risk[nominal_index],
                        "predicted_uncertainty": predicted_uncertainty[
                            nominal_index
                        ],
                        "detector_loss": float(loss),
                        "selected_positions_sha256": canonical_sha256(
                            [
                                int(value)
                                for value in positions[0].detach().cpu().tolist()
                            ]
                        ),
                        "max_gap_seconds": float(
                            decoded.max_gap_seconds[0].detach().cpu().item()
                        ),
                    }
                )
            reference_loss = float(option_work[-1]["detector_loss"])
            denominator = max(abs(reference_loss), 1.0e-6)
            options = []
            for option in option_work:
                candidate_loss = float(option.pop("detector_loss"))
                degradation = max(0.0, candidate_loss - reference_loss)
                options.append(
                    {
                        **option,
                        "oracle_utility": float(reference_loss - candidate_loss),
                        "oracle_risk": float(degradation / denominator),
                    }
                )
            row = {
                "schema_version": WINDOW_OPTION_SCHEMA,
                "status": "measured",
                "video_id": video,
                "window_start_frame": window_start,
                "valid_length": valid_length,
                "split_role": development_role,
                "split_assignment_sha256": split_validation[
                    "assignment_sha256"
                ],
                "candidate_budgets": list(budgets),
                "options": options,
                "source_identity_sha256": source_identity_sha256,
                "provenance": {
                    "measurement_kind": (
                        "certification_development_detector_loss_oracle"
                    ),
                    "uses_official_final": False,
                    "uses_gt_for_oracle_utility": True,
                    "uses_gt_for_predicted_utility": False,
                    "uses_teacher": False,
                    "uses_prediction_cache": False,
                    "uses_raw_predictions": False,
                    "oracle_only": True,
                    "deployment_candidate": False,
                    "model_training": False,
                    "checkpoint_mutation": False,
                    "pad_to_kmax": False,
                },
            }
            row["record_sha256"] = canonical_sha256(row)
            rows.append(row)
            identities.add(identity)
            window_counts[video] += 1
            print(
                json.dumps(
                    {
                        "hrime_stage1_video": video,
                        "window_start_frame": window_start,
                        "completed_windows": len(rows),
                        "total_windows": len(dataset),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if frozen_normalizer is not None:
            model.rpn_head.duca_set_frozen_loss_normalizer(None)

    _require(
        {video for video, _start in identities} == development_videos,
        "Stage-1 options do not cover every development video",
    )
    output_artifact = _write_jsonl_atomic(output_path, rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "produced",
        "claim_scope": (
            "complete_development_window_option_measurements_"
            "not_oracle_gate_not_paper_result"
        ),
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
        "output": output_artifact,
        "video_count": len(development_videos),
        "window_count": len(rows),
        "window_count_by_video": dict(sorted(window_counts.items())),
        "candidate_budgets": list(budgets),
        "split_role": development_role,
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "uses_official_final": False,
        "oracle_only": True,
        "deployment_candidate": False,
        "authorizes_stage2_training": False,
    }
    summary["content_sha256"] = canonical_sha256(summary)
    _write_json_exclusive(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = _sha256_file(summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Produce hash-bound H-RIME Stage-1 per-window predicted and "
            "development-oracle option curves from a terminal RIME-full model."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--training-receipt-sha256", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-pretrain")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    result = produce_window_options(
        config=args.config,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        training_receipt=args.training_receipt,
        training_receipt_sha256=args.training_receipt_sha256,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        preregistration=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
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
