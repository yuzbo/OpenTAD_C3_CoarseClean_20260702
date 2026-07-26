from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics
from typing import Any

from tools.bata.export_duca_allocation_ceiling_inputs import (
    canonical_sha256,
    dataset_provenance,
    deduplicate_sliding_windows,
    git_state,
    sha256,
    write_json_exclusive,
)
from tools.bata.validate_duca_allocation_ceiling_artifact import (
    validate_artifact_receipt,
)


SCHEMA_VERSION = "duca_allocation_candidate_detector_loss_v1"
SUMMARY_SCHEMA_VERSION = "duca_allocation_candidate_detector_loss_summary_v1"


def load_ceiling_records(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            sample_id = str(row.get("sample_id"))
            if sample_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id")
            rows[sample_id] = row
    if not rows:
        raise ValueError("ceiling artifact has no records")
    return rows


def sample_id_from_meta(meta: Mapping[str, Any]) -> str:
    video_id = str(meta.get("video_name") or meta.get("video_id") or "")
    if not video_id:
        raise ValueError("dataset metadata is missing video identity")
    window_start = int(round(float(meta.get("window_start_frame", 0))))
    return f"{video_id}|{window_start}"


def candidate_dataset_indices(
    dataset: Any,
    sample_ids: Sequence[str],
) -> list[int]:
    data_list = getattr(dataset, "data_list", None)
    if not isinstance(data_list, Sequence) or isinstance(data_list, (str, bytes)):
        raise ValueError("candidate evaluator requires an auditable sliding-window data_list")
    index_by_sample: dict[str, int] = {}
    for dataset_index, item in enumerate(data_list):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) < 4
        ):
            raise ValueError("candidate evaluator requires sliding-window dataset entries")
        window = item[3]
        if (
            not isinstance(window, Sequence)
            and not hasattr(window, "__len__")
        ) or len(window) < 1:
            raise ValueError("candidate evaluator dataset window is missing")
        sample_id = f"{item[0]}|{int(round(float(window[0])))}"
        if sample_id in index_by_sample:
            raise ValueError(f"dataset emits duplicate candidate sample identity: {sample_id}")
        index_by_sample[sample_id] = dataset_index
    requested = [str(sample_id) for sample_id in sample_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("candidate ceiling sample identities must be unique")
    missing = [sample_id for sample_id in requested if sample_id not in index_by_sample]
    if missing:
        raise ValueError(
            "candidate dataset cannot resolve ceiling samples: "
            f"{missing[:5]}"
        )
    return [index_by_sample[sample_id] for sample_id in requested]


def prepare_candidate_sample(
    *,
    inputs: Any,
    masks: Any,
    meta: Mapping[str, Any],
    gt_segments: Any,
    gt_labels: Any,
    positions: Sequence[int],
    requested_budget: int,
) -> dict[str, Any]:
    import torch

    if inputs.shape[0] != 1 or masks.shape[0] != 1:
        raise ValueError("candidate preparation requires a single-sample batch")
    dense_valid_len = int(masks[0].detach().long().sum().item())
    mask_values = masks[0].detach().bool().cpu().tolist()
    if mask_values != [True] * dense_valid_len + [False] * (len(mask_values) - dense_valid_len):
        raise ValueError("candidate preparation requires one contiguous dense valid prefix")
    selected = tuple(int(value) for value in positions)
    if selected != tuple(sorted(set(selected))):
        raise ValueError("candidate positions must be unique and ordered")
    if not selected or selected[0] < 0 or selected[-1] >= dense_valid_len:
        raise ValueError("candidate positions lie outside the dense valid prefix")
    if len(selected) != min(int(requested_budget), dense_valid_len):
        raise ValueError("candidate positions violate exact effective budget")

    padded = torch.full(
        (1, int(requested_budget)),
        -1,
        device=inputs.device,
        dtype=torch.long,
    )
    padded[0, : len(selected)] = torch.as_tensor(
        selected,
        device=inputs.device,
        dtype=torch.long,
    )
    selected_inputs = _gather_raw(inputs, padded)
    selected_masks = padded >= 0
    candidate_meta = dict(meta)
    candidate_meta.update(
        {
            "irregular_selected_positions": list(selected),
            "selected_dense_indices": list(selected),
            "selected_valid_len": len(selected),
            "irregular_selected_count": len(selected),
            "irregular_selected_valid_len": len(selected),
            "irregular_dense_valid_len": dense_valid_len,
            "irregular_native_axis": True,
            "remap_gt_to_selected_axis": False,
            "gt_remapped_to_selected_axis": False,
            "pc_ot_mras_prebackbone_remap_gt_to_selected_axis": False,
            "allocation_ceiling_external_candidate": True,
        }
    )
    return {
        "inputs": selected_inputs,
        "masks": selected_masks,
        "metas": [candidate_meta],
        "gt_segments": [gt_segments],
        "gt_labels": [gt_labels],
        "dense_valid_len": dense_valid_len,
        "selected_count": len(selected),
        "selected_positions": selected,
    }


def evaluate_one_candidate(
    model: Any,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        losses = model.forward_train(
            prepared["inputs"],
            prepared["masks"],
            prepared["metas"],
            prepared["gt_segments"],
            prepared["gt_labels"],
            _duca_skip_frame_selector=True,
            _duca_counterfactual_eval=True,
        )
    if not isinstance(losses, Mapping):
        raise RuntimeError("official detector candidate pass must return a loss mapping")
    cls_terms = [
        value
        for key, value in losses.items()
        if key.endswith("loss") and "cls" in key
    ]
    reg_terms = [
        value
        for key, value in losses.items()
        if key.endswith("loss") and "reg" in key
    ]
    if not cls_terms or not reg_terms:
        raise RuntimeError("official detector candidate pass produced no cls/reg loss")
    cls_loss = sum(cls_terms)
    reg_loss = sum(reg_terms)
    detector_loss = cls_loss + reg_loss
    values = {
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "reg_loss": float(reg_loss.detach().cpu().item()),
        "detector_loss": float(detector_loss.detach().cpu().item()),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise RuntimeError("official detector candidate pass produced non-finite loss")
    debug = {}
    if hasattr(model.rpn_head, "collect_debug_state"):
        debug = dict(model.rpn_head.collect_debug_state())
    if debug.get("physical_grid_actionformer_enabled") is not True:
        raise RuntimeError("candidate detector pass did not execute physical-grid ActionFormer")
    values["physical_grid_debug"] = debug
    return values


def run_evaluation(
    *,
    config: str | Path,
    checkpoint: str | Path,
    input_jsonl: str | Path,
    ceiling_jsonl: str | Path,
    ceiling_summary_json: str | Path,
    ceiling_validation_json: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    split: str,
    family_keys: Sequence[str],
    device: str = "cuda:0",
    backbone_pretrain: str | Path | None = None,
    batch_size: int = 1,
    num_workers: int = 2,
) -> dict[str, Any]:
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models.builder import build_detector

    config_path = Path(config).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    input_path = Path(input_jsonl).resolve()
    ceiling_path = Path(ceiling_jsonl).resolve()
    ceiling_summary_path = Path(ceiling_summary_json).resolve()
    ceiling_validation_path = Path(ceiling_validation_json).resolve()
    output_path = Path(output_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("candidate detector evaluation never overwrites artifacts")
    validation = validate_artifact_receipt(
        validation_json=ceiling_validation_path,
        input_jsonl=input_path,
        output_jsonl=ceiling_path,
        summary_json=ceiling_summary_path,
        require_gt_solver_replay=True,
    )
    ceiling = load_ceiling_records(ceiling_path)
    cfg = Config.fromfile(str(config_path))
    configured_pretrain = (
        backbone_pretrain
        if backbone_pretrain is not None
        else cfg.model.backbone.custom.get("pretrain")
    )
    pretrain_path = Path(str(configured_pretrain or "")).expanduser().resolve()
    if not pretrain_path.is_file():
        raise FileNotFoundError(f"backbone pretrain is missing: {pretrain_path}")
    cfg.model.backbone.custom.pretrain = str(pretrain_path)
    contract = cfg.get("allocation_ceiling_evaluator_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("evaluator config must declare allocation_ceiling_evaluator_contract")
    if contract.get("physical_grid_actionformer") is not True:
        raise ValueError("evaluator config must require physical-grid ActionFormer")
    if contract.get("dense_axis_gt") is not True or contract.get("selected_axis_gt_remap") is not False:
        raise ValueError("evaluator config must preserve dense-axis GT")
    if split not in cfg.dataset or split not in cfg.solver:
        raise ValueError(f"evaluator config must define dataset.{split} and solver.{split}")

    dataset = build_dataset(cfg.dataset[split], default_args=dict(logger=None))
    deduplicate_sliding_windows(dataset)
    dataset_source = dataset_provenance(cfg.dataset[split], dataset)
    candidate_indices = candidate_dataset_indices(dataset, tuple(ceiling))
    dataset = torch.utils.data.Subset(dataset, candidate_indices)
    loader_cfg = dict(cfg.solver[split])
    loader_cfg["batch_size"] = int(batch_size)
    loader_cfg["num_workers"] = int(num_workers)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    model = build_detector(cfg.model).to(torch_device)
    checkpoint_payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("checkpoint must be a mapping")
    state_key = "state_dict_ema"
    state = checkpoint_payload.get(state_key)
    if not isinstance(state, Mapping):
        raise ValueError("candidate evaluator requires terminal state_dict_ema")
    normalized = {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(normalized, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    if not getattr(model.rpn_head, "physical_grid_enabled", False):
        raise RuntimeError("built model did not enable physical-grid ActionFormer")
    frozen_normalizer = model.rpn_head.loss_normalizer.detach().clone()
    model.rpn_head.duca_set_frozen_loss_normalizer(frozen_normalizer)

    source = {
        **git_state(config_path.parents[3]),
        **dataset_source,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_state_key": state_key,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "backbone_pretrain": str(pretrain_path),
        "backbone_pretrain_sha256": sha256(pretrain_path),
        "ceiling_jsonl": str(ceiling_path),
        "ceiling_jsonl_sha256": sha256(ceiling_path),
        "ceiling_summary_json": str(ceiling_summary_path),
        "ceiling_summary_json_sha256": sha256(ceiling_summary_path),
        "ceiling_validation_json": str(ceiling_validation_path),
        "ceiling_validation_json_sha256": sha256(ceiling_validation_path),
        "ceiling_validation": validation,
        "split": split,
        "frozen_loss_normalizer": float(frozen_normalizer.cpu().item()),
    }
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for data in loader:
                inputs = data["inputs"].to(torch_device, non_blocking=True)
                masks = data["masks"].to(torch_device, non_blocking=True)
                metas = [dict(item) for item in data["metas"]]
                gt_segments = data.get("gt_segments")
                gt_labels = data.get("gt_labels")
                if gt_segments is None or gt_labels is None:
                    raise ValueError("candidate loss evaluation requires GT for evaluation only")
                for batch_index, meta in enumerate(metas):
                    sample_id = sample_id_from_meta(meta)
                    if sample_id not in ceiling:
                        continue
                    if sample_id in seen_samples:
                        raise ValueError(f"dataset emitted duplicate candidate sample {sample_id}")
                    candidate_map = {
                        str(family["family_key"]): family
                        for family in ceiling[sample_id]["families"]
                    }
                    for family_key in family_keys:
                        if family_key not in candidate_map:
                            raise ValueError(f"{sample_id}: missing requested family {family_key}")
                        family = candidate_map[family_key]
                        prepared = prepare_candidate_sample(
                            inputs=inputs[batch_index : batch_index + 1],
                            masks=masks[batch_index : batch_index + 1],
                            meta=meta,
                            gt_segments=gt_segments[batch_index].to(torch_device),
                            gt_labels=gt_labels[batch_index].to(torch_device),
                            positions=family["positions"],
                            requested_budget=int(ceiling[sample_id]["requested_budget"]),
                        )
                        losses = evaluate_one_candidate(model, prepared)
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "sample_id": sample_id,
                            "video_id": ceiling[sample_id]["video_id"],
                            "family_key": family_key,
                            "selected_positions": list(prepared["selected_positions"]),
                            "selected_count": prepared["selected_count"],
                            "dense_valid_len": prepared["dense_valid_len"],
                            "privileged": bool(family["privileged"]),
                            "deployable": bool(family["deployable"]),
                            "losses": losses,
                            "source": source,
                            "contract": {
                                "model_training": False,
                                "checkpoint_mutation": False,
                                "dense_axis_gt": True,
                                "selected_axis_gt_remap": False,
                                "physical_grid_actionformer": True,
                                "gt_used_for_selection": bool(family["privileged"]),
                                "detector_loss_is_empirical_not_combinatorial_oracle": True,
                            },
                        }
                        row["record_sha256"] = canonical_sha256(row)
                        rows.append(row)
                        handle.write(
                            json.dumps(
                                row,
                                sort_keys=True,
                                ensure_ascii=True,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    seen_samples.add(sample_id)
        expected_samples = set(ceiling)
        if seen_samples != expected_samples:
            missing = sorted(expected_samples - seen_samples)
            raise ValueError(
                "candidate detector evaluation did not visit every artifact sample: "
                f"missing={missing[:5]}"
            )
        temporary.replace(output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        model.rpn_head.duca_set_frozen_loss_normalizer(None)

    summary = _summarize(
        rows,
        output_path=output_path,
        source=source,
        requested_family_keys=tuple(family_keys),
    )
    write_json_exclusive(summary_path, summary)
    return summary


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    source: Mapping[str, Any],
    requested_family_keys: Sequence[str],
) -> dict[str, Any]:
    family_rows: dict[str, list[Mapping[str, Any]]] = {
        key: [] for key in requested_family_keys
    }
    for row in rows:
        family_rows[str(row["family_key"])].append(row)
    means = {}
    for key, values in family_rows.items():
        if not values:
            raise ValueError(f"candidate detector evaluation has no rows for {key}")
        means[key] = {
            "sample_count": len(values),
            "mean_cls_loss": _mean(row["losses"]["cls_loss"] for row in values),
            "mean_reg_loss": _mean(row["losses"]["reg_loss"] for row in values),
            "mean_detector_loss": _mean(row["losses"]["detector_loss"] for row in values),
        }
    uniform = means.get("A_exact_uniform")
    if uniform is not None:
        uniform_by_sample = {
            str(row["sample_id"]): float(row["losses"]["detector_loss"])
            for row in family_rows["A_exact_uniform"]
        }
        for key, value in means.items():
            candidate_by_sample = {
                str(row["sample_id"]): float(row["losses"]["detector_loss"])
                for row in family_rows[key]
            }
            if set(candidate_by_sample) != set(uniform_by_sample):
                raise ValueError(f"candidate detector sample set differs from uniform for {key}")
            paired = [
                uniform_by_sample[sample_id] - candidate_by_sample[sample_id]
                for sample_id in sorted(uniform_by_sample)
            ]
            value["gain_vs_uniform"] = sum(paired) / len(paired)
            value["paired_vs_uniform"] = {
                "sample_count": len(paired),
                "mean_detector_loss_gain": sum(paired) / len(paired),
                "median_detector_loss_gain": statistics.median(paired),
                "win_fraction": sum(delta > 0.0 for delta in paired) / len(paired),
                "tie_fraction": sum(delta == 0.0 for delta in paired) / len(paired),
                "loss_fraction": sum(delta < 0.0 for delta in paired) / len(paired),
                "min_gain": min(paired),
                "max_gain": max(paired),
            }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "diagnostic_role": "frozen_physical_grid_detector_loss_not_mAP_oracle",
        "sample_count": len({str(row["sample_id"]) for row in rows}),
        "row_count": len(rows),
        "requested_family_keys": list(requested_family_keys),
        "families": means,
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": sha256(output_path),
        "source": dict(source),
        "contract": {
            "model_training": False,
            "dense_axis_gt": True,
            "selected_axis_gt_remap": False,
            "physical_grid_actionformer": True,
            "mAP_evaluated": False,
            "paper_claim_allowed": False,
        },
    }


def _gather_raw(inputs: Any, positions: Any) -> Any:
    import torch

    if inputs.ndim not in (3, 5, 6):
        raise ValueError("candidate gather requires [B,C,T], [B,C,T,H,W], or [B,N,C,T,H,W]")
    time_dim = 2 if inputs.ndim in (3, 5) else 3
    view = [positions.shape[0]] + [1] * (inputs.ndim - 1)
    view[time_dim] = positions.shape[1]
    expand = list(inputs.shape)
    expand[time_dim] = positions.shape[1]
    valid = positions >= 0
    index = positions.clamp_min(0).view(view).expand(expand)
    gathered = torch.gather(inputs, time_dim, index)
    mask_view = [positions.shape[0]] + [1] * (inputs.ndim - 1)
    mask_view[time_dim] = positions.shape[1]
    return gathered * valid.view(mask_view).to(dtype=gathered.dtype)


def _mean(values) -> float:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("candidate loss summary requires finite non-empty values")
    return sum(numbers) / len(numbers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DUCA allocation candidates with a frozen physical-grid AdaTAD detector."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--ceiling-jsonl", required=True)
    parser.add_argument("--ceiling-summary-json", required=True)
    parser.add_argument("--ceiling-validation-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument(
        "--family-keys",
        nargs="+",
        default=[
            "A_exact_uniform",
            "D_deploy_score",
            "D_privileged_gt_ceiling",
            "E_privileged_unrestricted_gt",
        ],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-pretrain")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    summary = run_evaluation(
        config=args.config,
        checkpoint=args.checkpoint,
        input_jsonl=args.input_jsonl,
        ceiling_jsonl=args.ceiling_jsonl,
        ceiling_summary_json=args.ceiling_summary_json,
        ceiling_validation_json=args.ceiling_validation_json,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        split=args.split,
        family_keys=args.family_keys,
        device=args.device,
        backbone_pretrain=args.backbone_pretrain,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
