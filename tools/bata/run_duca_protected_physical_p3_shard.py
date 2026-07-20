from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config
from torch.utils.data import Subset

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from tools.bata.duca_protected_physical_p3 import (
    DURATION_STRATA,
    boundary_distance_stratum,
    deterministic_quartile_swaps,
    legal_single_swaps,
    stratified_window_manifest,
)
from tools.bata.duca_protected_physical_training import (
    canonical_sha256,
    sha256_file,
)
from tools.bata.run_duca_protected_physical_full_model_gate import (
    _bind_runtime,
    _capture_mutable_state,
    _cuda_batch,
    _hard_gather,
    _require,
    _restore_mutable_state,
)


SCHEMA = "duca_protected_physical_p3_shard_v1"


def _load_protocol(path: str, expected_sha256: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), "P0 protocol manifest is missing")
    _require(sha256_file(resolved) == expected_sha256, "P0 manifest hash drift")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(
        payload.get("schema") == "duca_protected_physical_protocol_manifest_v1"
        and payload.get("ok") is True,
        "P0 manifest did not pass",
    )
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _parameter_versions(model) -> dict[str, int]:
    return {name: int(parameter._version) for name, parameter in model.named_parameters()}


def _buffers_equal(model, state: Mapping[str, Any]) -> bool:
    current = dict(model.named_buffers())
    return all(
        name in current and torch.equal(current[name], expected)
        for name, expected in state["buffers"].items()
    )


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _mutable_state_equal(model, state: Mapping[str, Any]) -> bool:
    modules = dict(model.named_modules())
    if set(state["module_training"]) != set(modules):
        return False
    if any(
        bool(modules[name].training) != bool(expected)
        for name, expected in state["module_training"].items()
    ):
        return False
    for name, expected in state["custom_replay"].items():
        module = modules.get(name)
        if module is None:
            return False
        observed = module.capture_amp_replay_state()
        if not _nested_equal(observed, expected):
            return False
    return _buffers_equal(model, state) and _rng_equal(state)


def _rng_equal(state: Mapping[str, Any]) -> bool:
    numpy_now = np.random.get_state()
    numpy_expected = state["numpy_rng"]
    return (
        torch.equal(torch.random.get_rng_state(), state["cpu_rng"])
        and all(
            torch.equal(left, right)
            for left, right in zip(
                torch.cuda.get_rng_state_all(),
                state["cuda_rng"],
            )
        )
        and random.getstate() == state["python_rng"]
        and numpy_now[0] == numpy_expected[0]
        and np.array_equal(numpy_now[1], numpy_expected[1])
        and numpy_now[2:] == numpy_expected[2:]
    )


def _evaluate_hard(
    model,
    *,
    materialized: Mapping[str, Any],
    batch: Mapping[str, Any],
    mutable_state: Mapping[str, Any],
    captured_backbone_inputs: list[torch.Tensor],
) -> tuple[float, bool, bool]:
    _restore_mutable_state(model, mutable_state)
    captured_backbone_inputs.clear()
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    ):
        losses = model.forward_train(
            materialized["inputs"],
            materialized["masks"],
            copy.deepcopy(materialized["metas"]),
            batch["gt_segments"],
            batch["gt_labels"],
            _duca_skip_frame_selector=True,
        )
        objective = model._duca_detector_objective(losses)
    _require(bool(torch.isfinite(objective).item()), "P3 hard loss is non-finite")
    _require(
        len(captured_backbone_inputs) == 1,
        "P3 hard replay did not execute the backbone exactly once",
    )
    hard_equal = torch.equal(
        captured_backbone_inputs[0],
        materialized["inputs"],
    )
    _restore_mutable_state(model, mutable_state)
    restoration_ok = _mutable_state_equal(model, mutable_state)
    return float(objective.detach().float().item()), hard_equal, restoration_ok


def _gap_metrics(
    positions: torch.Tensor,
    *,
    source_frames: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_len: int,
) -> dict[str, float]:
    active = positions.detach().cpu().numpy().astype(np.int64)
    index_axis = np.arange(valid_len, dtype=np.float64)
    source = source_frames[:valid_len].detach().cpu().numpy()
    seconds = physical_seconds[:valid_len].detach().cpu().numpy()

    def intervals(axis):
        values = axis[active]
        return np.concatenate(
            (
                np.asarray([values[0] - axis[0]]),
                np.diff(values),
                np.asarray([axis[-1] - values[-1]]),
            )
        )

    return {
        "max_candidate_index_interval": float(np.max(intervals(index_axis))),
        "max_source_frame_interval": float(np.max(intervals(source))),
        "max_seconds_interval": float(np.max(intervals(seconds))),
        "left_edge_seconds_interval": float(seconds[active[0]] - seconds[0]),
        "right_edge_seconds_interval": float(seconds[-1] - seconds[active[-1]]),
    }


def _true_boundary_validity(
    gt_segments,
    true_boundary_candidate_positions,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    true_boundaries = torch.as_tensor(
        true_boundary_candidate_positions,
        device=device,
        dtype=torch.float32,
    ).reshape(-1)
    _require(
        true_boundaries.numel() > 0,
        "P3 manifest has no original-annotation boundaries",
    )
    output = []
    for segments in gt_segments:
        endpoints = torch.as_tensor(
            segments,
            device=device,
            dtype=torch.float32,
        ).reshape(-1, 2)
        if endpoints.numel() == 0:
            output.append(torch.zeros_like(endpoints, dtype=torch.bool))
            continue
        distances = (
            endpoints[..., None] - true_boundaries.reshape(1, 1, -1)
        ).abs()
        output.append(distances.amin(dim=-1) <= 1.0e-4)
    return output


def run_shard(
    *,
    stratum: str,
    config_path: str,
    expected_commit: str,
    protocol_manifest: str,
    protocol_manifest_sha256: str,
    adatad_pretrain: str,
    adatad_pretrain_sha256: str,
    output_json: str,
) -> dict[str, Any]:
    _require(stratum in DURATION_STRATA, f"invalid P3 stratum {stratum!r}")
    output = Path(output_json).expanduser().resolve()
    _require(not output.exists(), "refusing to overwrite P3 shard evidence")
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("P3 evidence must be outside the Git worktree")
    runtime = _bind_runtime(expected_commit)
    protocol = _load_protocol(protocol_manifest, protocol_manifest_sha256)
    _require(protocol["git_commit"] == expected_commit, "P0 commit drift")
    _require(
        protocol.get("git_tree") == runtime["git_tree"],
        "P0 Git tree differs from the P3 tree",
    )

    cfg_path = (ROOT / config_path).resolve()
    cfg = Config.fromfile(str(cfg_path))
    _require(cfg.model.frame_selector.arm == "protected_e2e", "P3 must use main arm")
    _require(cfg.dataset.val is None, "P3 config exposes validation")
    expected_p3 = protocol["p3_population"]
    _require(
        sha256_file(cfg_path) == expected_p3["config_sha256"],
        "P3 config differs from P0",
    )
    pretrain = Path(adatad_pretrain).expanduser().resolve()
    _require(pretrain.is_file(), "P3 VideoMAE-S pretrain is missing")
    _require(
        sha256_file(pretrain) == adatad_pretrain_sha256,
        "P3 VideoMAE-S pretrain hash drift",
    )
    _require(
        protocol.get("videomae_pretrain", {}).get("sha256")
        == adatad_pretrain_sha256,
        "P3 VideoMAE-S pretrain differs from P0",
    )

    dataset = build_dataset(
        copy.deepcopy(cfg.dataset.train),
        default_args={"logger": None},
    )
    window_manifest = stratified_window_manifest(dataset)
    _require(
        canonical_sha256(window_manifest) == expected_p3["windows_sha256"],
        "P3 window population differs from P0",
    )
    shard_windows = [
        row for row in window_manifest if row["duration_stratum"] == stratum
    ]
    _require(len(shard_windows) == 16, "P3 shard must contain 16 windows")
    loader = build_dataloader(
        Subset(
            dataset,
            [int(row["dataset_index"]) for row in shard_windows],
        ),
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **copy.deepcopy(cfg.solver.train),
    )

    _seed_everything(3407)
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = str(pretrain)
    model = build_detector(model_cfg).to("cuda:0")
    model.train()
    selector = model.frame_selector
    selector.capture_policy_score_gradients = True
    model.rpn_head.duca_set_frozen_loss_normalizer(
        model.rpn_head.loss_normalizer.detach().clone()
    )
    parameter_versions = _parameter_versions(model)
    initial_state = _capture_mutable_state(model)
    rows = []
    windows = []
    captured_backbone_inputs: list[torch.Tensor] = []

    def capture_backbone_input(_module, args):
        captured_backbone_inputs.append(args[0].detach().clone())

    hook = model.backbone.register_forward_pre_hook(capture_backbone_input)
    try:
        for manifest_row, raw_batch in zip(shard_windows, loader):
            batch = _cuda_batch(raw_batch)
            meta = batch["metas"][0]
            _require(
                str(meta["video_name"]) == manifest_row["video_id"],
                "P3 loader video order drift",
            )
            _require(
                int(meta["window_start_frame"]) == manifest_row["window_start"],
                "P3 loader window order drift",
            )
            valid_len = int(batch["masks"][0].sum().item())
            _require(
                valid_len == int(manifest_row["valid_len"]) and valid_len > 384,
                "P3 valid length differs from the frozen window population",
            )
            _require(
                manifest_row.get("boundary_source")
                == "original_uncropped_annotation",
                "P3 boundary source is not the original annotation",
            )
            batch["gt_boundary_validity"] = _true_boundary_validity(
                batch["gt_segments"],
                manifest_row["true_boundary_candidate_positions"],
                device=batch["inputs"].device,
            )
            _restore_mutable_state(model, initial_state)
            model.zero_grad(set_to_none=True)
            captured_backbone_inputs.clear()
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
                cache_enabled=False,
            ):
                losses = model.forward_train(
                    batch["inputs"],
                    batch["masks"],
                    batch["metas"],
                    batch["gt_segments"],
                    batch["gt_labels"],
                    gt_boundary_validity=batch["gt_boundary_validity"],
                )
                detector_objective = model._duca_detector_objective(losses)
            scaler.scale(detector_objective).backward()
            policy_scores = selector._last_policy_scores
            _require(
                policy_scores is not None and policy_scores.grad is not None,
                "P3 detector loss did not reach policy scores",
            )
            score_gradient = (
                policy_scores.grad.detach().float() / float(scaler.get_scale())
            )
            _require(
                bool(torch.isfinite(score_gradient).all().item()),
                "P3 score gradient is non-finite",
            )
            positions = selector._last_selected_positions
            _require(positions is not None, "P3 hard positions are missing")
            _require(
                len(captured_backbone_inputs) == 1,
                "P3 ST forward did not execute backbone exactly once",
            )
            expected_hard = _hard_gather(batch["inputs"], positions)
            _require(
                torch.equal(captured_backbone_inputs[0], expected_hard),
                "P3 ST detector input is not exact hard gather",
            )
            fixed = selector.materialize_hard_positions(
                batch["inputs"],
                batch["masks"],
                batch["metas"],
                positions,
            )
            base_loss, base_hard_equal, base_restore = _evaluate_hard(
                model,
                materialized=fixed,
                batch=batch,
                mutable_state=initial_state,
                captured_backbone_inputs=captured_backbone_inputs,
            )
            _require(base_hard_equal, "P3 base hard-forward equality failed")
            _require(base_restore, "P3 base restoration failed")
            _require(
                abs(base_loss - float(detector_objective.detach().float().item()))
                <= 1.0e-6,
                "P3 ST and hard-only base losses disagree",
            )

            effective_k = min(384, valid_len)
            active_positions = (
                positions[0, :effective_k].detach().cpu().tolist()
            )
            physical_seconds = fixed["physical_seconds"][0]
            source_frames = fixed["decoded_source_frames"][0]
            cap = float(fixed["max_gap_seconds"][0].item())
            legal = legal_single_swaps(
                active_positions,
                physical_seconds.detach().cpu().tolist(),
                valid_len,
                cap,
            )
            sampled = deterministic_quartile_swaps(
                legal,
                score_gradient[0, :valid_len].detach().cpu().tolist(),
                video_id=manifest_row["video_id"],
                window_start=manifest_row["window_start"],
            )
            true_candidate_endpoints = np.asarray(
                manifest_row["true_boundary_candidate_positions"],
                dtype=np.float64,
            )
            true_source_endpoints = np.asarray(
                manifest_row["true_boundary_source_frames"],
                dtype=np.float64,
            )
            true_seconds_endpoints = np.asarray(
                manifest_row["true_boundary_seconds"],
                dtype=np.float64,
            )
            _require(
                true_candidate_endpoints.size > 0
                and true_candidate_endpoints.size
                == true_source_endpoints.size
                == true_seconds_endpoints.size,
                "P3 true-boundary arrays are empty or misaligned",
            )
            source_axis = (
                source_frames[:valid_len].detach().float().cpu().numpy()
            )
            seconds_axis = (
                physical_seconds[:valid_len].detach().float().cpu().numpy()
            )
            window_rows = []
            for sampled_row in sampled:
                candidate = sorted(
                    (set(active_positions) - {sampled_row["removed"]})
                    | {sampled_row["incoming"]}
                )
                candidate_active = torch.tensor(
                    candidate,
                    device=batch["inputs"].device,
                    dtype=torch.long,
                )
                candidate_tensor = torch.full(
                    (1, 384),
                    -1,
                    device=batch["inputs"].device,
                    dtype=torch.long,
                )
                candidate_tensor[0, :effective_k] = candidate_active
                excluded_reason = None
                violation_count = 0
                restoration_ok = False
                hard_equal = False
                actual_delta = float("nan")
                try:
                    materialized = selector.materialize_hard_positions(
                        batch["inputs"],
                        batch["masks"],
                        batch["metas"],
                        candidate_tensor,
                    )
                    candidate_loss, hard_equal, restoration_ok = _evaluate_hard(
                        model,
                        materialized=materialized,
                        batch=batch,
                        mutable_state=initial_state,
                        captured_backbone_inputs=captured_backbone_inputs,
                    )
                    actual_delta = candidate_loss - base_loss
                    gap_metrics = _gap_metrics(
                        candidate_tensor[0],
                        source_frames=source_frames,
                        physical_seconds=physical_seconds,
                        valid_len=valid_len,
                    )
                    if gap_metrics["max_seconds_interval"] > cap + 1.0e-9:
                        violation_count = 1
                        excluded_reason = "physical_cap_violation"
                except Exception as exc:
                    gap_metrics = {}
                    violation_count = 1
                    excluded_reason = (
                        f"{exc.__class__.__name__}:{str(exc)}"
                    )
                add_boundary_distance = float(
                    np.min(
                        np.abs(
                            true_candidate_endpoints
                            - sampled_row["incoming"]
                        )
                    )
                )
                remove_boundary_distance = float(
                    np.min(
                        np.abs(
                            true_candidate_endpoints
                            - sampled_row["removed"]
                        )
                    )
                )
                add_source_distance = float(
                    np.min(
                        np.abs(
                            true_source_endpoints
                            - source_axis[sampled_row["incoming"]]
                        )
                    )
                )
                remove_source_distance = float(
                    np.min(
                        np.abs(
                            true_source_endpoints
                            - source_axis[sampled_row["removed"]]
                        )
                    )
                )
                add_seconds_distance = float(
                    np.min(
                        np.abs(
                            true_seconds_endpoints
                            - seconds_axis[sampled_row["incoming"]]
                        )
                    )
                )
                remove_seconds_distance = float(
                    np.min(
                        np.abs(
                            true_seconds_endpoints
                            - seconds_axis[sampled_row["removed"]]
                        )
                    )
                )
                row = {
                    **sampled_row,
                    "video_id": manifest_row["video_id"],
                    "window_start": int(manifest_row["window_start"]),
                    "dataset_index": int(manifest_row["dataset_index"]),
                    "duration_stratum": stratum,
                    "shortest_action_duration_seconds": float(
                        manifest_row["shortest_action_duration_seconds"]
                    ),
                    "valid_len": valid_len,
                    "window_kind": manifest_row["window_kind"],
                    "boundary_source": manifest_row["boundary_source"],
                    "incoming_boundary_distance_candidate_indices": (
                        add_boundary_distance
                    ),
                    "removed_boundary_distance_candidate_indices": (
                        remove_boundary_distance
                    ),
                    "boundary_distance_gain_candidate_indices": (
                        remove_boundary_distance - add_boundary_distance
                    ),
                    "incoming_boundary_distance_source_frames": (
                        add_source_distance
                    ),
                    "removed_boundary_distance_source_frames": (
                        remove_source_distance
                    ),
                    "incoming_boundary_distance_seconds": add_seconds_distance,
                    "removed_boundary_distance_seconds": (
                        remove_seconds_distance
                    ),
                    "boundary_distance_gain_seconds": (
                        remove_seconds_distance - add_seconds_distance
                    ),
                    "boundary_distance_stratum": boundary_distance_stratum(
                        add_seconds_distance
                    ),
                    "base_loss": base_loss,
                    "actual_delta": actual_delta,
                    "hard_forward_equal": bool(hard_equal),
                    "physical_violation_count": int(violation_count),
                    "restoration_mismatch": not bool(restoration_ok),
                    "excluded_reason": excluded_reason,
                    "repeated_base_loss_abs_error": 0.0,
                    "max_gap_seconds_cap": cap,
                    **gap_metrics,
                }
                rows.append(row)
                window_rows.append(row)

            repeated_base_loss, repeated_hard_equal, repeated_restore = (
                _evaluate_hard(
                    model,
                    materialized=fixed,
                    batch=batch,
                    mutable_state=initial_state,
                    captured_backbone_inputs=captured_backbone_inputs,
                )
            )
            repeated_error = abs(repeated_base_loss - base_loss)
            for row in window_rows:
                row["repeated_base_loss_abs_error"] = repeated_error
                row["hard_forward_equal"] = bool(
                    row["hard_forward_equal"] and repeated_hard_equal
                )
                row["restoration_mismatch"] = bool(
                    row["restoration_mismatch"] or not repeated_restore
                )
            windows.append(
                {
                    **manifest_row,
                    "base_loss": base_loss,
                    "repeated_base_loss": repeated_base_loss,
                    "repeated_base_loss_abs_error": repeated_error,
                    "legal_swap_count": len(legal),
                    "sampled_swap_count": len(window_rows),
                    "score_gradient_l1": float(
                        score_gradient[0, :valid_len].abs().sum().item()
                    ),
                    "hard_forward_equal": bool(
                        base_hard_equal and repeated_hard_equal
                    ),
                }
            )
            model.zero_grad(set_to_none=True)
    finally:
        hook.remove()
        model.rpn_head.duca_set_frozen_loss_normalizer(None)

    _require(len(windows) == 16, "P3 shard window count drift")
    _require(len(rows) == 192, "P3 shard swap count drift")
    _require(
        _parameter_versions(model) == parameter_versions,
        "P3 modified model parameters without an optimizer step",
    )
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "runtime": runtime,
        "stratum": stratum,
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "adatad_pretrain": {
            "path": str(pretrain),
            "sha256": adatad_pretrain_sha256,
        },
        "protocol_manifest_path": str(
            Path(protocol_manifest).expanduser().resolve()
        ),
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "optimizer_step": 0,
        "seed": 3407,
        "loss_normalizer_frozen": True,
        "train_split_only": True,
        "test_loader_built": False,
        "checkpoint_written": False,
        "windows": windows,
        "rows": rows,
        "row_sha256": canonical_sha256(rows),
        "paper_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", required=True, choices=DURATION_STRATA)
    parser.add_argument(
        "--config",
        default="configs/adatad/thumos/duca_protected_physical_p3_train_windows.py",
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--protocol-manifest-sha256", required=True)
    parser.add_argument("--adatad-pretrain", required=True)
    parser.add_argument("--adatad-pretrain-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        run_shard(
            stratum=args.stratum,
            config_path=args.config,
            expected_commit=args.expected_commit,
            protocol_manifest=args.protocol_manifest,
            protocol_manifest_sha256=args.protocol_manifest_sha256,
            adatad_pretrain=args.adatad_pretrain,
            adatad_pretrain_sha256=args.adatad_pretrain_sha256,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
