from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import train_paction_acquisition_policy as base_train


SUMMARY_SCHEMA_VERSION = "c3_detector_aware_policy_train_v1"
CHECKPOINT_SCHEMA_VERSION = "c3_detector_aware_policy_checkpoint_v1"
READY = "C3_DETECTOR_AWARE_POLICY_TRAIN_READY"


_read_jsonl = base_train._read_jsonl
_write_json = base_train._write_json
_sha256_file = base_train._sha256_file
_git_sha = base_train._git_sha
_validate_source_row = base_train._validate_source_row
_extract_paction = base_train._extract_paction
_valid_len = base_train._valid_len


def _extract_teacher_utility(row: Mapping[str, Any], *, line_no: int, length: int) -> list[float]:
    provenance = row.get("teacher_utility_provenance")
    if not isinstance(provenance, Mapping):
        provenance = (row.get("teacher_utility") or {}).get("provenance") if isinstance(row.get("teacher_utility"), Mapping) else None
    if not isinstance(provenance, Mapping) or provenance.get("split_scope") != "train_only":
        raise ValueError(f"line {line_no}: teacher utility must be train_only")
    raw = None
    teacher_utility = row.get("teacher_utility")
    if isinstance(teacher_utility, Mapping) and isinstance(teacher_utility.get("signed_frame_utility"), list):
        raw = teacher_utility.get("signed_frame_utility")
    if raw is None and isinstance(row.get("signed_frame_utility"), list):
        raw = row.get("signed_frame_utility")
    if raw is None and isinstance(teacher_utility, Mapping) and isinstance(teacher_utility.get("frame_utility"), list):
        raw = teacher_utility.get("frame_utility")
    if raw is None and isinstance(row.get("frame_utility"), list):
        raw = row.get("frame_utility")
    if not isinstance(raw, list):
        raise ValueError(f"line {line_no}: train-only teacher frame_utility is required")
    if len(raw) != int(length):
        raise ValueError(f"line {line_no}: teacher frame_utility length must match p_action length")
    return [max(-1.0, min(1.0, float(item))) for item in raw]


def _marginal_gain_target(utility: Sequence[float]) -> list[float]:
    return [abs(max(-1.0, min(1.0, float(item)))) for item in utility]


def _utility_budget_target(
    utility: Sequence[float],
    *,
    valid_len: int,
    dynamic_budget_buckets: Sequence[int],
    coverage_target: float = 0.80,
) -> int:
    valid_values = [abs(max(-1.0, min(1.0, float(item)))) for item in utility[: int(valid_len)]]
    total = sum(valid_values)
    if total <= 0.0:
        return min(max(1, min(int(item) for item in dynamic_budget_buckets)), int(valid_len))
    ranked = sorted(valid_values, reverse=True)
    buckets = sorted(min(max(1, int(item)), int(valid_len)) for item in dynamic_budget_buckets)
    for bucket in buckets:
        if sum(ranked[:bucket]) / total >= float(coverage_target):
            return int(bucket)
    return int(buckets[-1])


def _prepared_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dynamic_budget_buckets: Sequence[int],
    expected_split: str | None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for line_no, row in enumerate(rows, start=1):
        _validate_source_row(row, line_no=line_no, expected_split=expected_split)
        p_action = _extract_paction(row, line_no=line_no)
        valid_len = _valid_len(row, paction_len=len(p_action))
        utility = _extract_teacher_utility(row, line_no=line_no, length=len(p_action))
        gain = _marginal_gain_target(utility)
        dynamic_budget_target = _utility_budget_target(
            gain,
            valid_len=valid_len,
            dynamic_budget_buckets=dynamic_budget_buckets,
        )
        prepared.append(
            {
                "features": detector_policy.build_detector_aware_feature_matrix(
                    p_action,
                    valid=[idx < valid_len for idx in range(len(p_action))],
                    target_budget=dynamic_budget_target,
                ),
                "detector_utility_target": utility,
                "detector_marginal_gain_target": gain,
                "valid_len": valid_len,
                "dynamic_budget_target": dynamic_budget_target,
                "dynamic_gain_calibration": dict(detector_policy.DEFAULT_DYNAMIC_GAIN_CALIBRATION),
            }
        )
    return prepared


def _batch_to_tensors(batch: Sequence[Mapping[str, Any]], *, device: str, dynamic_budget_buckets: Sequence[int]):
    import torch

    max_len = max(len(row["features"]) for row in batch)
    feature_dim = len(detector_policy.DETECTOR_AWARE_FEATURE_NAMES)
    features = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32, device=device)
    utility = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
    gain = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
    valid = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
    buckets = [int(item) for item in dynamic_budget_buckets]
    budget_indices: list[int] = []
    budget_targets: list[int] = []
    for row_idx, row in enumerate(batch):
        length = len(row["features"])
        features[row_idx, :length] = torch.tensor(row["features"], dtype=torch.float32, device=device)
        utility[row_idx, :length] = torch.tensor(row["detector_utility_target"], dtype=torch.float32, device=device)
        gain[row_idx, :length] = torch.tensor(row["detector_marginal_gain_target"], dtype=torch.float32, device=device)
        valid[row_idx, : int(row["valid_len"])] = True
        target_budget = int(row["dynamic_budget_target"])
        budget_targets.append(target_budget)
        budget_indices.append(min(range(len(buckets)), key=lambda idx: (abs(buckets[idx] - target_budget), idx)))
    return {
        "features": features,
        "utility": utility,
        "gain": gain,
        "valid": valid,
        "budget_indices": torch.tensor(budget_indices, dtype=torch.long, device=device),
        "budget_targets": torch.tensor(budget_targets, dtype=torch.float32, device=device),
    }


def _run_epoch(
    *,
    model: detector_policy.DetectorAwareSequentialAcquisitionPolicy,
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
            losses = detector_policy.detector_aware_training_objective(
                outputs,
                detector_utility_target=tensors["utility"],
                detector_gain_target=tensors["gain"],
                valid=tensors["valid"],
                target_budget=tensors["budget_targets"],
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
    dynamic_budget_buckets: Sequence[int] = detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.10,
    budget_ce_loss_weight: float = 0.25,
    device: str = "cuda",
    seed: int = 0,
    expected_split: str | None = "training",
) -> dict[str, Any]:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path is not None else out_path / "detector_aware_policy.pth"
    buckets = [int(item) for item in dynamic_budget_buckets]
    train_rows = _prepared_rows(_read_jsonl(train_jsonl), dynamic_budget_buckets=buckets, expected_split=expected_split)
    model_kwargs = {
        "input_dim": len(detector_policy.DETECTOR_AWARE_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "budget_buckets": buckets,
        "dropout": float(dropout),
    }
    model = detector_policy.DetectorAwareSequentialAcquisitionPolicy(**model_kwargs).to(device)
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
        "policy_family": "detector_aware_offline_selector",
        "stage_label": detector_policy.STAGE_LABEL,
        "policy_source": detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
        "train_jsonl": str(train_jsonl),
        "train_jsonl_sha256": _sha256_file(train_jsonl),
        "policy_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
        "dynamic_budget_buckets": buckets,
        "utility_semantics": "signed_detector_utility_v1",
        "signed_utility_supported": True,
        "dynamic_gain_calibration": dict(detector_policy.DEFAULT_DYNAMIC_GAIN_CALIBRATION),
        "loss_terms": dict(detector_policy.DEFAULT_DETECTOR_AWARE_LOSS_TERMS),
        "teacher_target_scope": "train_only",
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "end_to_end": False,
        "git_sha": _git_sha(),
        "history": history,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "policy_family": "detector_aware_offline_selector",
        "stage_label": detector_policy.STAGE_LABEL,
        "train_jsonl": str(train_jsonl),
        "out_dir": str(out_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "train_jsonl_sha256": _sha256_file(train_jsonl),
        "dynamic_budget_buckets": buckets,
        "utility_semantics": "signed_detector_utility_v1",
        "signed_utility_supported": True,
        "dynamic_gain_calibration": dict(detector_policy.DEFAULT_DYNAMIC_GAIN_CALIBRATION),
        "teacher_target_scope": "train_only",
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "end_to_end": False,
        "final_train": history[-1]["train"] if history else {},
        "history": history,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Stage-2 detector-aware offline selector from train-only teacher utility.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--summary-json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--budget-ce-loss-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-split", default="training")
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
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
