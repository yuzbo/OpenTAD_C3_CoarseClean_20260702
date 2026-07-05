from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import paction_acquisition_policy as policy


SUMMARY_SCHEMA_VERSION = "c3_paction_acquisition_policy_train_v1"
CHECKPOINT_SCHEMA_VERSION = "c3_paction_acquisition_policy_checkpoint_v1"
READY = "C3_PACTION_POLICY_TRAIN_READY"


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
        raise ValueError(f"sample JSONL has no rows: {path}")
    return rows


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


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _extract_paction(row: Mapping[str, Any], *, line_no: int) -> list[float]:
    frame_signals = row.get("frame_signals")
    if isinstance(frame_signals, Mapping) and isinstance(frame_signals.get("p_action"), list):
        return [float(item) for item in frame_signals["p_action"]]
    if isinstance(row.get("p_action"), list):
        return [float(item) for item in row["p_action"]]
    raise ValueError(f"line {line_no}: p_action signal is required")


def _extract_action_target(row: Mapping[str, Any], *, line_no: int, length: int) -> list[float]:
    raw = row.get("action_target")
    if raw is None:
        raw = row.get("action_labels")
    if not isinstance(raw, list):
        raise ValueError(f"line {line_no}: action_target is required for policy training")
    if len(raw) != int(length):
        raise ValueError(f"line {line_no}: action_target length must match p_action length")
    return [1.0 if float(item) >= 0.5 else 0.0 for item in raw]


def _extract_boundaries(row: Mapping[str, Any], *, length: int) -> list[int]:
    raw = row.get("gt_boundaries")
    if raw is None:
        raw = row.get("boundaries")
    boundaries: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            idx = int(round(float(item)))
            if 0 <= idx < int(length):
                boundaries.append(idx)
    if not boundaries and isinstance(row.get("gt_segments"), list):
        for segment in row["gt_segments"]:
            if isinstance(segment, list) and len(segment) >= 2:
                for item in (segment[0], segment[1]):
                    idx = int(round(float(item)))
                    if 0 <= idx < int(length):
                        boundaries.append(idx)
    return sorted(set(boundaries))


def _valid_len(row: Mapping[str, Any], *, paction_len: int) -> int:
    valid_len = int(row.get("valid_len") or row.get("dense_len") or paction_len)
    if valid_len <= 0:
        raise ValueError("valid_len must be positive")
    return min(valid_len, int(paction_len))


def _budget_target_from_row(
    row: Mapping[str, Any],
    *,
    action_target: Sequence[float],
    boundaries: Sequence[int],
    valid_len: int,
    dynamic_budget_buckets: Sequence[int],
) -> int:
    for key in ("dynamic_budget_target", "paction_budget_target", "budget_target"):
        if key in row and row[key] is not None:
            target = int(row[key])
            return min(max(1, target), int(valid_len))
    positive_count = sum(1 for item in action_target[: int(valid_len)] if float(item) >= 0.5)
    boundary_cover_need = len(set(int(item) for item in boundaries)) * (2 * int(policy.DEFAULT_BOUNDARY_LOSS_RADIUS) + 1)
    action_cover_need = int(math.ceil(0.50 * float(positive_count)))
    gap_cover_need = int(math.ceil(float(valid_len) / float(max(1, policy.DEFAULT_GAP_LOSS_MAX_GAP + 1))))
    required = max(1, boundary_cover_need, action_cover_need, gap_cover_need)
    for bucket in sorted(int(item) for item in dynamic_budget_buckets):
        if bucket >= required:
            return min(bucket, int(valid_len))
    return min(max(int(item) for item in dynamic_budget_buckets), int(valid_len))


def _prepared_rows(rows: Sequence[Mapping[str, Any]], *, dynamic_budget_buckets: Sequence[int]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for line_no, row in enumerate(rows, start=1):
        p_action = _extract_paction(row, line_no=line_no)
        valid_len = _valid_len(row, paction_len=len(p_action))
        action_target = _extract_action_target(row, line_no=line_no, length=len(p_action))
        boundaries = _extract_boundaries(row, length=len(p_action))
        boundary_target = [0.0 for _ in p_action]
        for idx in boundaries:
            boundary_target[idx] = 1.0
        dynamic_budget_target = _budget_target_from_row(
            row,
            action_target=action_target,
            boundaries=boundaries,
            valid_len=valid_len,
            dynamic_budget_buckets=dynamic_budget_buckets,
        )
        prepared.append(
            {
                "features": policy.build_paction_feature_matrix(
                    p_action,
                    valid=[idx < valid_len for idx in range(len(p_action))],
                ),
                "action_target": action_target,
                "boundary_target": boundary_target,
                "valid_len": valid_len,
                "dynamic_budget_target": dynamic_budget_target,
            }
        )
    return prepared


def _batch_to_tensors(batch: Sequence[Mapping[str, Any]], *, device: str, dynamic_budget_buckets: Sequence[int]):
    import torch

    max_len = max(len(row["features"]) for row in batch)
    feature_dim = len(policy.PACTION_FEATURE_NAMES)
    features = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32, device=device)
    action = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
    boundary = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
    valid = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
    budget_indices: list[int] = []
    buckets = [int(item) for item in dynamic_budget_buckets]
    for row_idx, row in enumerate(batch):
        row_features = row["features"]
        length = len(row_features)
        features[row_idx, :length] = torch.tensor(row_features, dtype=torch.float32, device=device)
        action[row_idx, :length] = torch.tensor(row["action_target"], dtype=torch.float32, device=device)
        boundary[row_idx, :length] = torch.tensor(row["boundary_target"], dtype=torch.float32, device=device)
        valid[row_idx, : int(row["valid_len"])] = True
        target_budget = int(row["dynamic_budget_target"])
        closest_idx = min(range(len(buckets)), key=lambda idx: (abs(buckets[idx] - target_budget), idx))
        budget_indices.append(int(closest_idx))
    return {
        "features": features,
        "action": action,
        "boundary": boundary,
        "valid": valid,
        "budget_indices": torch.tensor(budget_indices, dtype=torch.long, device=device),
        "budget_targets": torch.tensor([int(row["dynamic_budget_target"]) for row in batch], dtype=torch.float32, device=device),
    }


def _run_epoch(
    *,
    model: policy.PActionDynamicAcquisitionPolicy,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    device: str,
    dynamic_budget_buckets: Sequence[int],
    optimizer: Any | None,
    gap_loss_max_gap: int,
    budget_ce_loss_weight: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    if optimizer is None:
        model.eval()
    else:
        model.train()
    totals: dict[str, float] = {}
    batch_count = 0
    for start in range(0, len(rows), max(1, int(batch_size))):
        batch = rows[start : start + max(1, int(batch_size))]
        tensors = _batch_to_tensors(batch, device=device, dynamic_budget_buckets=dynamic_budget_buckets)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(tensors["features"], tensors["valid"])
            losses = policy.paction_gap_loss_training_objective(
                outputs,
                action_target=tensors["action"],
                boundary_target=tensors["boundary"],
                valid=tensors["valid"],
                target_budget=tensors["budget_targets"],
                gap_loss_max_gap=int(gap_loss_max_gap),
            )
            budget_ce = F.cross_entropy(outputs["budget_logits"], tensors["budget_indices"])
            total_loss = losses["total_loss"] + budget_ce * float(budget_ce_loss_weight)
            if optimizer is not None:
                total_loss.backward()
                optimizer.step()
        scalar_losses = {
            key: float(value.detach().cpu().item())
            for key, value in losses.items()
            if key.endswith("_loss") and hasattr(value, "detach")
        }
        scalar_losses["budget_ce_loss"] = float(budget_ce.detach().cpu().item())
        scalar_losses["optimized_loss"] = float(total_loss.detach().cpu().item())
        for key, value in scalar_losses.items():
            totals[key] = totals.get(key, 0.0) + value
        batch_count += 1
    return {key: value / float(max(1, batch_count)) for key, value in totals.items()}


def run_training(
    train_jsonl: str | Path,
    *,
    out_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    summary_json: str | Path | None = None,
    val_jsonl: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    fixed_budget: int = 384,
    dynamic_budget_buckets: Sequence[int] = policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.10,
    gap_loss_max_gap: int = policy.DEFAULT_GAP_LOSS_MAX_GAP,
    budget_ce_loss_weight: float = 0.25,
    device: str = "cuda",
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path is not None else out_path / "paction_policy.pth"
    buckets = [int(item) for item in dynamic_budget_buckets]
    train_rows = _prepared_rows(_read_jsonl(train_jsonl), dynamic_budget_buckets=buckets)
    val_rows = _prepared_rows(_read_jsonl(val_jsonl), dynamic_budget_buckets=buckets) if val_jsonl is not None else []
    model_kwargs = {
        "input_dim": len(policy.PACTION_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "budget_buckets": buckets,
        "dropout": float(dropout),
    }
    model = policy.PActionDynamicAcquisitionPolicy(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    history: list[dict[str, Any]] = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        random.shuffle(train_rows)
        train_metrics = _run_epoch(
            model=model,
            rows=train_rows,
            batch_size=int(batch_size),
            device=device,
            dynamic_budget_buckets=buckets,
            optimizer=optimizer,
            gap_loss_max_gap=int(gap_loss_max_gap),
            budget_ce_loss_weight=float(budget_ce_loss_weight),
        )
        epoch_row: dict[str, Any] = {"epoch": int(epoch), "train": train_metrics}
        if val_rows:
            with torch.no_grad():
                epoch_row["val"] = _run_epoch(
                    model=model,
                    rows=val_rows,
                    batch_size=int(batch_size),
                    device=device,
                    dynamic_budget_buckets=buckets,
                    optimizer=None,
                    gap_loss_max_gap=int(gap_loss_max_gap),
                    budget_ce_loss_weight=float(budget_ce_loss_weight),
                )
        history.append(epoch_row)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "decision": READY,
        "policy_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
        "hparams": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "fixed_budget": int(fixed_budget),
            "dynamic_budget_buckets": buckets,
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "gap_loss_max_gap": int(gap_loss_max_gap),
            "budget_ce_loss_weight": float(budget_ce_loss_weight),
            "device": str(device),
            "seed": int(seed),
        },
        "data": {
            "train_jsonl": str(train_jsonl),
            "train_jsonl_sha256": _sha256_file(train_jsonl),
            "val_jsonl": None if val_jsonl is None else str(val_jsonl),
            "val_jsonl_sha256": None if val_jsonl is None else _sha256_file(val_jsonl),
            "train_row_count": len(train_rows),
            "val_row_count": len(val_rows),
        },
        "git_sha": _git_sha(),
        "torch_initial_seed": int(torch.initial_seed()),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "fixed_budget": int(fixed_budget),
        "dynamic_budget_buckets": buckets,
        "gap_loss_max_gap": int(gap_loss_max_gap),
        "loss_terms": dict(policy.DEFAULT_POLICY_LOSS_TERMS),
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "gap_control": "learned_gap_hole_loss_no_uniform_fill",
        "history": history,
    }
    torch.save(checkpoint_payload, checkpoint)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": None if val_jsonl is None else str(val_jsonl),
        "out_dir": str(out_path),
        "checkpoint_path": str(checkpoint),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "fixed_budget": int(fixed_budget),
        "dynamic_budget_buckets": buckets,
        "gap_control": "learned_gap_hole_loss_no_uniform_fill",
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "loss_terms": dict(policy.DEFAULT_POLICY_LOSS_TERMS),
        "final_train": history[-1]["train"] if history else {},
        "final_val": history[-1].get("val") if history and "val" in history[-1] else None,
        "history": history,
        "train_jsonl_sha256": _sha256_file(train_jsonl),
        "val_jsonl_sha256": None if val_jsonl is None else _sha256_file(val_jsonl),
        "git_sha": checkpoint_payload["git_sha"],
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the learned p_action gap/hole acquisition policy.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--summary-json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fixed-budget", type=int, default=384)
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--budget-buckets", type=int, nargs="+", dest="dynamic_budget_buckets", default=argparse.SUPPRESS)
    parser.add_argument("--policy-mode", default="gap_loss", choices=["gap_loss", "learned_gap_loss", "dynamic_budget"])
    parser.add_argument("--selection-strategy", default="learned_paction_gap_loss_value")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--gap-loss-max-gap", type=int, default=policy.DEFAULT_GAP_LOSS_MAX_GAP)
    parser.add_argument("--budget-ce-loss-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_training(
        args.train_jsonl,
        out_dir=args.out_dir,
        checkpoint_path=args.checkpoint_path,
        summary_json=args.summary_json,
        val_jsonl=args.val_jsonl,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        fixed_budget=int(args.fixed_budget),
        dynamic_budget_buckets=[int(item) for item in args.dynamic_budget_buckets],
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        gap_loss_max_gap=int(args.gap_loss_max_gap),
        budget_ce_loss_weight=float(args.budget_ce_loss_weight),
        device=str(args.device),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
