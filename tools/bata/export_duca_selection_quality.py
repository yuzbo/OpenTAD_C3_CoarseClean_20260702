from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.analyze_duca_selection_quality import RECORD_SCHEMA_VERSION


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selector_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }
    selected = {
        key[len("frame_selector.") :]: value
        for key, value in normalized.items()
        if key.startswith("frame_selector.")
    }
    if not selected:
        raise ValueError("checkpoint has no frame_selector.* parameters")
    return selected


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _find_git_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _checkpoint_state(checkpoint: Mapping[str, Any], use_ema: str) -> tuple[str, Mapping[str, Any]]:
    if use_ema == "true":
        key = "state_dict_ema"
    elif use_ema == "false":
        key = "state_dict"
    else:
        key = "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict"
    state = checkpoint.get(key)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint is missing mapping {key}")
    return key, state


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _strict_float_row(value: Any, index: int, valid_len: int) -> list[float]:
    if value is None:
        raise ValueError("selector output is missing a required score tensor")
    row = value[index].detach().float().cpu().tolist()
    if len(row) < valid_len:
        raise ValueError("selector score tensor is shorter than valid_len")
    return [round(float(item), 7) for item in row[:valid_len]]


def _sample_id(meta: Mapping[str, Any], fallback: int) -> str:
    video_id = str(meta.get("video_name") or meta.get("video_id") or f"sample_{fallback:06d}")
    window = int(meta.get("window_start_frame", 0))
    return f"{video_id}|{window}"


def _records_from_batch(
    *,
    selector_output: Mapping[str, Any],
    masks: Any,
    gt_segments: Sequence[Any],
    metas: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    seen_count: int,
) -> list[dict[str, Any]]:
    scores = selector_output.get("selector_outputs")
    if not isinstance(scores, Mapping):
        raise ValueError("frame selector output is missing selector_outputs")
    grid = scores.get("grid")
    if grid is None or not hasattr(grid, "selected_positions"):
        raise ValueError("selector_outputs is missing SparseTemporalGrid")
    selected_rows = grid.selected_positions.detach().cpu().long()
    repair_rows = grid.metadata.get("max_gap_repair", []) if isinstance(grid.metadata, Mapping) else []
    records: list[dict[str, Any]] = []
    for idx, meta in enumerate(metas):
        valid_len = int(masks[idx].detach().long().sum().cpu().item())
        selected = [int(item) for item in selected_rows[idx].tolist() if 0 <= int(item) < valid_len]
        gt = _to_list(gt_segments[idx]) if idx < len(gt_segments) else []
        sample_id = _sample_id(meta, seen_count + idx)
        repair = repair_rows[idx] if idx < len(repair_rows) and isinstance(repair_rows[idx], Mapping) else {}
        records.append(
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "sample_id": sample_id,
                "video_id": str(meta.get("video_name") or meta.get("video_id") or sample_id.split("|")[0]),
                "window_start_frame": int(meta.get("window_start_frame", 0)),
                "snippet_stride": int(meta.get("snippet_stride", 1)),
                "valid_len": valid_len,
                "budget": len(selected),
                "gt_segments": [[round(float(pair[0]), 6), round(float(pair[1]), 6)] for pair in gt],
                "p_action": _strict_float_row(scores.get("p_action"), idx, valid_len),
                "actionness_logits": _strict_float_row(scores.get("actionness_logits"), idx, valid_len),
                "transition_policy_scores": _strict_float_row(
                    scores.get("transition_policy_scores", scores.get("center_scores")), idx, valid_len
                ),
                "raw_transition_scores": _strict_float_row(scores.get("transition_score"), idx, valid_len),
                "abs_delta_p_action": _strict_float_row(scores.get("abs_delta_p_action"), idx, valid_len),
                "uncertainty": _strict_float_row(scores.get("uncertainty"), idx, valid_len),
                "selected_positions": selected,
                "decode_diagnostics": {
                    "repair_count": int(repair.get("repair_count", 0)),
                    "repair_enabled": bool(repair.get("enabled", False)),
                    "repair_feasible": bool(repair.get("feasible", True)),
                    "repair_satisfied": bool(repair.get("satisfied", True)),
                    "max_hole_before": repair.get("max_hole_before"),
                    "max_hole_after": repair.get("max_hole_after"),
                },
                "selection_path": str(scores.get("selection_path", "unknown")),
                "policy_mix_alpha": float(scores.get("policy_mix_alpha", 1.0)),
                "source": dict(source),
                "gt_role": "evaluation_only_not_selector_input",
            }
        )
    return records


def export_records(
    *,
    config: str | Path,
    checkpoint: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    split: str = "val",
    device: str = "cuda:0",
    use_ema: str = "auto",
    use_amp: bool = True,
    batch_size: int | None = None,
    num_workers: int | None = None,
    limit_batches: int = 0,
    seed: int = 3407,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models.builder import build_selector

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_path = Path(config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    summary_path = Path(summary_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = _find_git_root(config_path.parent) or Path.cwd().resolve()
    cfg = Config.fromfile(str(config_path))
    if split not in cfg.dataset or split not in cfg.solver:
        raise ValueError(f"config must define dataset.{split} and solver.{split}")
    dataset = build_dataset(cfg.dataset[split], default_args=dict(logger=None))
    loader_cfg = dict(cfg.solver[split])
    if batch_size is not None:
        loader_cfg["batch_size"] = int(batch_size)
    if num_workers is not None:
        loader_cfg["num_workers"] = int(num_workers)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    selector = build_selector(cfg.model.frame_selector)
    checkpoint_payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("checkpoint must be a mapping")
    state_key, full_state = _checkpoint_state(checkpoint_payload, use_ema=use_ema)
    incompatible = selector.load_state_dict(selector_state_dict(full_state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"selector checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    selector = selector.to(torch_device).eval()
    source = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_state_key": state_key,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "git_commit": _git_commit(repo_root),
        "split": split,
        "selector_only_inference": True,
        "detector_backbone_executed": False,
        "uses_gt_for_selection": False,
        "seed": int(seed),
    }
    row_count = 0
    sample_count = 0
    with output_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if int(limit_batches) > 0 and batch_idx >= int(limit_batches):
                break
            inputs = data["inputs"].to(torch_device, non_blocking=True)
            masks = data["masks"].to(torch_device, non_blocking=True)
            metas = [dict(item) for item in data.get("metas", [{} for _ in range(inputs.shape[0])])]
            gt_segments = data.get("gt_segments", [[] for _ in range(inputs.shape[0])])
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=bool(use_amp and torch_device.type == "cuda")):
                # GT is intentionally absent from this call. It is consumed only below for evaluation records.
                output = selector.forward_test(inputs=inputs, masks=masks, metas=metas)
            records = _records_from_batch(
                selector_output=output,
                masks=masks,
                gt_segments=gt_segments,
                metas=metas,
                source=source,
                seen_count=sample_count,
            )
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            row_count += 1
            sample_count += len(records)
            if batch_idx % 20 == 0:
                print(json.dumps({"batch": batch_idx, "samples": sample_count}, sort_keys=True), flush=True)
    summary = {
        "schema_version": "duca_selection_quality_export_summary_v1",
        "output_jsonl": str(output_path),
        "row_count": row_count,
        "sample_count": sample_count,
        "limit_batches": int(limit_batches),
        "seed": int(seed),
        "source": source,
        "decision_contract": {
            "gt_passed_to_selector": False,
            "teacher_passed_to_selector": False,
            "raw_prediction_passed_to_selector": False,
            "detector_backbone_executed": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export DUCA selector quality records from a real checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-ema", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    summary = export_records(
        config=args.config,
        checkpoint=args.checkpoint,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        split=args.split,
        device=args.device,
        use_ema=args.use_ema,
        use_amp=not args.no_amp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_batches=args.limit_batches,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
