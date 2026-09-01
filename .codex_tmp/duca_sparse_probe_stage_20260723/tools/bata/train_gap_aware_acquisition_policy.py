from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import gas_vt_paction_policy as gas_vt
from tools.bata import train_paction_acquisition_policy as base_train


SUMMARY_SCHEMA_VERSION = "c3_gas_vt_policy_train_v1"
CHECKPOINT_SCHEMA_VERSION = "c3_gas_vt_policy_checkpoint_v1"
READY = "C3_GAS_VT_POLICY_TRAIN_READY"


_read_jsonl = base_train._read_jsonl
_write_json = base_train._write_json
_sha256_file = base_train._sha256_file
_git_sha = base_train._git_sha
_validate_source_row = base_train._validate_source_row
_extract_paction = base_train._extract_paction
_extract_action_target = base_train._extract_action_target
_extract_boundaries = base_train._extract_boundaries
_valid_len = base_train._valid_len


def _action_interior_bins_from_target(
    action_target: Sequence[Any],
    *,
    valid_len: int,
    bins_per_segment: int = 4,
) -> list[list[float]]:
    length = len(action_target)
    bins: list[list[float]] = []
    idx = 0
    while idx < int(valid_len):
        if float(action_target[idx]) < 0.5:
            idx += 1
            continue
        start = idx
        while idx < int(valid_len) and float(action_target[idx]) >= 0.5:
            idx += 1
        end = idx
        segment_len = max(1, end - start)
        bin_count = max(1, min(int(bins_per_segment), segment_len))
        for bin_idx in range(bin_count):
            bin_start = start + int(round(bin_idx * segment_len / float(bin_count)))
            bin_end = start + int(round((bin_idx + 1) * segment_len / float(bin_count)))
            bin_end = max(bin_start + 1, min(end, bin_end))
            mask = [0.0 for _ in range(length)]
            for pos in range(bin_start, bin_end):
                mask[pos] = 1.0
            bins.append(mask)
    return bins


def _row_with_inferred_split(
    row: Mapping[str, Any],
    *,
    expected_split: str | None,
    allow_missing_split_from_source_path: bool,
) -> tuple[Mapping[str, Any], bool, bool]:
    patched: dict[str, Any] | None = None
    inferred_split = False
    if not expected_split or base_train._row_split(row) is not None:
        return row, inferred_split, False
    if allow_missing_split_from_source_path:
        patched = dict(row)
        patched["split"] = str(expected_split)
        patched["inferred_split_from_source_path"] = True
        inferred_split = True
    return patched or row, inferred_split, False


def _row_for_source_validation(
    row: Mapping[str, Any],
    *,
    expected_split: str | None,
    allow_missing_split_from_source_path: bool,
    allow_gt_diagnostics_in_training_source: bool,
) -> tuple[Mapping[str, Any], bool, bool]:
    row_for_training, inferred_split, _ = _row_with_inferred_split(
        row,
        expected_split=expected_split,
        allow_missing_split_from_source_path=allow_missing_split_from_source_path,
    )
    expected_is_training = bool(expected_split and base_train._split_matches("training", str(expected_split)))
    gt_diagnostics_allowed = (
        bool(allow_gt_diagnostics_in_training_source)
        and expected_is_training
        and base_train._is_true(row_for_training.get("uses_gt_for_diagnostics", False))
    )
    if gt_diagnostics_allowed:
        patched = dict(row_for_training)
        patched["uses_gt_for_diagnostics"] = False
        patched["allowed_gt_diagnostics_in_training_source"] = True
        row_for_training = patched
    return row_for_training, inferred_split, gt_diagnostics_allowed


def _prepared_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dynamic_budget_buckets: Sequence[int],
    expected_split: str | None,
    allow_missing_split_from_source_path: bool = False,
    allow_gt_diagnostics_in_training_source: bool = False,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for line_no, row in enumerate(rows, start=1):
        row_for_training, inferred_split, allowed_gt_diagnostics = _row_for_source_validation(
            row,
            expected_split=expected_split,
            allow_missing_split_from_source_path=allow_missing_split_from_source_path,
            allow_gt_diagnostics_in_training_source=allow_gt_diagnostics_in_training_source,
        )
        _validate_source_row(row_for_training, line_no=line_no, expected_split=expected_split)
        p_action = _extract_paction(row_for_training, line_no=line_no)
        valid_len = _valid_len(row_for_training, paction_len=len(p_action))
        action_target = _extract_action_target(row_for_training, line_no=line_no, length=len(p_action))
        boundaries = _extract_boundaries(row_for_training, length=len(p_action))
        boundary_target = [0.0 for _ in p_action]
        for idx in boundaries:
            boundary_target[idx] = 1.0
        dynamic_budget_target = base_train._budget_target_from_row(
            row_for_training,
            action_target=action_target,
            boundaries=boundaries,
            valid_len=valid_len,
            dynamic_budget_buckets=dynamic_budget_buckets,
        )
        prepared.append(
            {
                "features": gas_vt.build_gap_aware_feature_matrix(
                    p_action,
                    valid=[idx < valid_len for idx in range(len(p_action))],
                    target_budget=dynamic_budget_target,
                ),
                "action_target": action_target,
                "boundary_target": boundary_target,
                "action_interior_bins": _action_interior_bins_from_target(action_target, valid_len=valid_len),
                "valid_len": valid_len,
                "dynamic_budget_target": dynamic_budget_target,
                "inferred_split_from_source_path": bool(inferred_split),
                "allowed_gt_diagnostics_in_training_source": bool(allowed_gt_diagnostics),
            }
        )
    return prepared


def _batch_to_tensors(batch: Sequence[Mapping[str, Any]], *, device: str, dynamic_budget_buckets: Sequence[int]):
    import torch

    max_len = max(len(row["features"]) for row in batch)
    feature_dim = len(gas_vt.GAS_VT_FEATURE_NAMES)
    features = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32, device=device)
    action = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
    boundary = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
    valid = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
    max_bins = max((len(row.get("action_interior_bins") or []) for row in batch), default=0)
    action_interior_bins = torch.zeros((len(batch), max_bins, max_len), dtype=torch.float32, device=device)
    buckets = [int(item) for item in dynamic_budget_buckets]
    budget_indices: list[int] = []
    budget_targets: list[int] = []
    for row_idx, row in enumerate(batch):
        length = len(row["features"])
        features[row_idx, :length] = torch.tensor(row["features"], dtype=torch.float32, device=device)
        action[row_idx, :length] = torch.tensor(row["action_target"], dtype=torch.float32, device=device)
        boundary[row_idx, :length] = torch.tensor(row["boundary_target"], dtype=torch.float32, device=device)
        for bin_idx, bin_mask in enumerate(row.get("action_interior_bins") or []):
            action_interior_bins[row_idx, bin_idx, :length] = torch.tensor(bin_mask, dtype=torch.float32, device=device)
        valid[row_idx, : int(row["valid_len"])] = True
        target_budget = int(row["dynamic_budget_target"])
        budget_targets.append(target_budget)
        budget_indices.append(min(range(len(buckets)), key=lambda idx: (abs(buckets[idx] - target_budget), idx)))
    return {
        "features": features,
        "action": action,
        "boundary": boundary,
        "action_interior_bins": action_interior_bins,
        "valid": valid,
        "budget_indices": torch.tensor(budget_indices, dtype=torch.long, device=device),
        "budget_targets": torch.tensor(budget_targets, dtype=torch.float32, device=device),
    }


def _run_epoch(
    *,
    model: gas_vt.GapAwareSequentialAcquisitionPolicy,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    device: str,
    dynamic_budget_buckets: Sequence[int],
    optimizer: Any | None,
    budget_ce_loss_weight: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    batches = 0
    for start in range(0, len(rows), max(1, int(batch_size))):
        batch = rows[start : start + max(1, int(batch_size))]
        tensors = _batch_to_tensors(batch, device=device, dynamic_budget_buckets=dynamic_budget_buckets)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(tensors["features"], tensors["valid"], target_budget=tensors["budget_targets"])
            losses = gas_vt.gas_vt_training_objective(
                outputs,
                action_target=tensors["action"],
                boundary_target=tensors["boundary"],
                valid=tensors["valid"],
                target_budget=tensors["budget_targets"],
                action_interior_bins=tensors["action_interior_bins"],
            )
            budget_ce = F.cross_entropy(outputs["budget_logits"], tensors["budget_indices"])
            total_loss = losses["total_loss"] + budget_ce * float(budget_ce_loss_weight)
            if optimizer is not None:
                total_loss.backward()
                optimizer.step()
        scalars = {key: float(value.detach().cpu().item()) for key, value in losses.items() if key.endswith("_loss") and hasattr(value, "detach")}
        scalars["budget_ce_loss"] = float(budget_ce.detach().cpu().item())
        scalars["optimized_loss"] = float(total_loss.detach().cpu().item())
        for key, value in scalars.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / float(max(1, batches)) for key, value in totals.items()}


def run_training(
    train_jsonl: str | Path,
    *,
    out_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    summary_json: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dynamic_budget_buckets: Sequence[int] = gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.10,
    budget_ce_loss_weight: float = 0.25,
    device: str = "cuda",
    seed: int = 0,
    expected_split: str | None = "training",
    allow_missing_split_from_source_path: bool = False,
    allow_gt_diagnostics_in_training_source: bool = False,
) -> dict[str, Any]:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path is not None else out_path / "gas_vt_policy.pth"
    buckets = [int(item) for item in dynamic_budget_buckets]
    train_rows = _prepared_rows(
        _read_jsonl(train_jsonl),
        dynamic_budget_buckets=buckets,
        expected_split=expected_split,
        allow_missing_split_from_source_path=allow_missing_split_from_source_path,
        allow_gt_diagnostics_in_training_source=allow_gt_diagnostics_in_training_source,
    )
    model_kwargs = {
        "input_dim": len(gas_vt.GAS_VT_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "budget_buckets": buckets,
        "dropout": float(dropout),
    }
    model = gas_vt.GapAwareSequentialAcquisitionPolicy(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    history: list[dict[str, Any]] = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        random.shuffle(train_rows)
        history.append(
            {
                "epoch": int(epoch),
                "train": _run_epoch(
                    model=model,
                    rows=train_rows,
                    batch_size=int(batch_size),
                    device=device,
                    dynamic_budget_buckets=buckets,
                    optimizer=optimizer,
                    budget_ce_loss_weight=float(budget_ce_loss_weight),
                ),
            }
        )
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "decision": READY,
        "policy_family": "GAS-VT",
        "policy_source": gas_vt.GAS_VT_CHECKPOINT_POLICY_SOURCE,
        "policy_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
        "dynamic_budget_buckets": buckets,
        "loss_terms": dict(gas_vt.DEFAULT_GAS_VT_LOSS_TERMS),
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "git_sha": _git_sha(),
        "history": history,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "policy_family": "GAS-VT",
        "train_jsonl": str(train_jsonl),
        "out_dir": str(out_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "train_jsonl_sha256": _sha256_file(train_jsonl),
        "dynamic_budget_buckets": buckets,
        "expected_split": expected_split,
        "allow_missing_split_from_source_path": bool(allow_missing_split_from_source_path),
        "allow_gt_diagnostics_in_training_source": bool(allow_gt_diagnostics_in_training_source),
        "inferred_split_rows": int(sum(1 for row in train_rows if row.get("inferred_split_from_source_path"))),
        "allowed_gt_diagnostics_rows": int(sum(1 for row in train_rows if row.get("allowed_gt_diagnostics_in_training_source"))),
        "train_row_count": int(len(train_rows)),
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "final_train": history[-1]["train"] if history else {},
        "history": history,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a GAS-VT gap-aware sequential acquisition policy.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--summary-json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--budget-ce-loss-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-split", default="training")
    parser.add_argument(
        "--allow-missing-split-from-source-path",
        action="store_true",
        help=(
            "Allow source JSONL rows without an explicit split only when the caller "
            "is already passing the split-specific train/val/test path. Explicit "
            "wrong splits still fail closed."
        ),
    )
    parser.add_argument(
        "--allow-gt-diagnostics-in-training-source",
        action="store_true",
        help=(
            "Allow a training split source JSONL to carry uses_gt_for_diagnostics=true "
            "for exported diagnostic metrics. This does not allow uses_gt/teacher/"
            "oracle/cache/raw-prediction flags and must not be used for val/test "
            "policy inputs."
        ),
    )
    args = parser.parse_args(argv)
    summary = run_training(
        args.train_jsonl,
        out_dir=args.out_dir,
        checkpoint_path=args.checkpoint_path,
        summary_json=args.summary_json,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dynamic_budget_buckets=args.dynamic_budget_buckets,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        budget_ce_loss_weight=args.budget_ce_loss_weight,
        device=args.device,
        seed=args.seed,
        expected_split=args.expected_split,
        allow_missing_split_from_source_path=args.allow_missing_split_from_source_path,
        allow_gt_diagnostics_in_training_source=args.allow_gt_diagnostics_in_training_source,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
