from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from ..builder import SELECTORS
from .duca_online_frame_selector import DucaOnlineFrameSelector
from ..utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS


@SELECTORS.register_module()
class DucaAllocationArtifactReplaySelector(DucaOnlineFrameSelector):
    """Evaluation-only replay of a hash-bound allocation-family artifact."""

    def __init__(
        self,
        *args: Any,
        artifact_path: str,
        artifact_sha256: str,
        family_key: str,
        allow_privileged_family: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.artifact_path = str(Path(artifact_path).expanduser().resolve())
        self.artifact_sha256 = str(artifact_sha256).lower()
        self.family_key = str(family_key)
        self.allow_privileged_family = bool(allow_privileged_family)
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise ValueError("allocation replay requires an exact artifact SHA-256")
        path = Path(self.artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"allocation replay artifact is missing: {path}")
        if _sha256(path) != self.artifact_sha256:
            raise ValueError("allocation replay artifact SHA-256 mismatch")
        self._allocation_rows = _load_family_rows(
            path,
            family_key=self.family_key,
            allow_privileged=self.allow_privileged_family,
        )

    def forward_train(self, *args: Any, **kwargs: Any):
        raise RuntimeError("allocation artifact replay is evaluation-only and cannot train")

    def forward_test(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas=None,
        budget=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del budget, kwargs
        if metas is None or not isinstance(metas, (list, tuple)):
            raise ValueError("allocation replay requires per-sample metadata")
        if len(metas) != int(inputs.shape[0]):
            raise ValueError("allocation replay metadata batch mismatch")
        dense_masks = masks.to(device=inputs.device, dtype=torch.bool)
        selected_rows: list[tuple[int, ...]] = []
        updated_metas: list[dict[str, Any]] = []
        effective_budgets: list[int] = []
        for batch_index, source_meta in enumerate(metas):
            if not isinstance(source_meta, Mapping):
                raise ValueError("allocation replay metadata entries must be mappings")
            valid_len = int(dense_masks[batch_index].long().sum().item())
            dense_values = dense_masks[batch_index].detach().cpu().tolist()
            if dense_values != [True] * valid_len + [False] * (len(dense_values) - valid_len):
                raise ValueError("allocation replay requires one contiguous dense valid prefix")
            sample_id = _sample_id(source_meta)
            row = self._allocation_rows.get(sample_id)
            if row is None:
                raise KeyError(f"allocation replay has no row for {sample_id}")
            if int(row["valid_len"]) != valid_len:
                raise ValueError(f"allocation replay valid_len mismatch for {sample_id}")
            positions = tuple(int(value) for value in row["positions"])
            expected_budget = min(self.budget, valid_len)
            if len(positions) != expected_budget:
                raise ValueError(f"allocation replay exact-K mismatch for {sample_id}")
            if positions != tuple(sorted(set(positions))):
                raise ValueError(f"allocation replay positions are not ordered and unique for {sample_id}")
            if positions[0] < 0 or positions[-1] >= valid_len:
                raise ValueError(f"allocation replay positions leave the valid prefix for {sample_id}")
            selected_rows.append(positions)
            effective_budgets.append(len(positions))
            meta = dict(source_meta)
            selected_axis = self.detector_output_coordinate_space == SELECTED_AXIS
            remap = {
                "source": SELECTED_AXIS,
                "target": TRUE_TIME_AXIS,
                "selected_to_original": {
                    int(axis): int(position)
                    for axis, position in enumerate(positions)
                },
                "original_to_selected": {
                    int(position): int(axis)
                    for axis, position in enumerate(positions)
                },
                "selected_axis_to_true_time_dense_index": list(positions),
                "acquisition_positions": list(positions),
            }
            meta.update(
                {
                    "irregular_selected_positions": list(positions),
                    "selected_dense_indices": list(positions),
                    "selected_valid_len": len(positions),
                    "irregular_selected_count": len(positions),
                    "irregular_selected_valid_len": len(positions),
                    "irregular_dense_valid_len": valid_len,
                    "irregular_native_axis": True,
                    "remap_gt_to_selected_axis": bool(selected_axis),
                    "gt_remapped_to_selected_axis": False,
                    "pc_ot_mras_prebackbone_remap_gt_to_selected_axis": bool(
                        selected_axis
                    ),
                    "detector_output_coordinate_space": self.detector_output_coordinate_space,
                    "detector_prediction_inverse_map_required": bool(selected_axis),
                    "selected_axis_to_true_time_dense_index": list(positions),
                    "truetime_selected_positions": list(positions),
                    "truetime_dense_len": int(
                        inputs.shape[2 if inputs.ndim in (3, 5) else 3]
                    ),
                    "truetime_dense_valid_len": valid_len,
                    "duca_online_selected_positions": list(positions),
                    "duca_online_selected_axis_remap": remap,
                    "allocation_replay_family_key": self.family_key,
                    "allocation_replay_artifact_sha256": self.artifact_sha256,
                    "allocation_replay_privileged": bool(row["privileged"]),
                }
            )
            updated_metas.append(meta)

        padded = torch.full(
            (int(inputs.shape[0]), self.budget),
            -1,
            device=inputs.device,
            dtype=torch.long,
        )
        for batch_index, positions in enumerate(selected_rows):
            padded[batch_index, : len(positions)] = torch.as_tensor(
                positions,
                device=inputs.device,
                dtype=torch.long,
            )
        selected_inputs = _gather_raw(inputs, padded)
        selected_masks = padded >= 0
        self.last_forward_summary = {
            "selection_scope": "offline_full_window_artifact_replay_diagnostic",
            "family_key": self.family_key,
            "artifact_sha256": self.artifact_sha256,
            "effective_budget": effective_budgets,
            "uses_gt_at_runtime": False,
            "detector_output_coordinate_space": self.detector_output_coordinate_space,
            "paper_deployable": False,
        }
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": updated_metas,
            "selector_outputs": {
                "selection_path": "hash_bound_allocation_artifact_replay",
                "selected_positions": padded,
                "effective_budget": effective_budgets,
            },
        }


def _load_family_rows(
    path: Path,
    *,
    family_key: str,
    allow_privileged: bool,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "duca_allocation_family_ceiling_record_v1":
                raise ValueError(f"{path}:{line_number}: unsupported allocation schema")
            recorded_hash = record.get("record_sha256")
            unhashed = dict(record)
            unhashed.pop("record_sha256", None)
            if not isinstance(recorded_hash, str) or _canonical_sha256(unhashed) != recorded_hash:
                raise ValueError(f"{path}:{line_number}: allocation record SHA-256 mismatch")
            candidates = [
                family
                for family in record.get("families", [])
                if family.get("family_key") == family_key
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"{path}:{line_number}: family {family_key} is missing or duplicated"
                )
            family = candidates[0]
            if family.get("exact") is not True or family.get("solver_status") != "OPTIMAL":
                raise ValueError(f"{path}:{line_number}: replay family is not exact OPTIMAL")
            privileged = bool(family.get("privileged"))
            deployable = bool(family.get("deployable"))
            if privileged and not allow_privileged:
                raise ValueError(
                    f"{path}:{line_number}: privileged family requires explicit diagnostic opt-in"
                )
            if not privileged and not deployable:
                raise ValueError(f"{path}:{line_number}: nonprivileged replay family is not deployable")
            sample_id = str(record.get("sample_id"))
            if not sample_id or sample_id in rows:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate sample_id")
            positions = [int(value) for value in family.get("positions", [])]
            if any(not math.isfinite(float(value)) for value in positions):
                raise ValueError(f"{path}:{line_number}: non-finite replay position")
            rows[sample_id] = {
                "valid_len": int(record["valid_len"]),
                "positions": positions,
                "privileged": privileged,
                "deployable": deployable,
            }
    if not rows:
        raise ValueError("allocation replay artifact has no records")
    return rows


def _sample_id(meta: Mapping[str, Any]) -> str:
    video_id = str(meta.get("video_name") or meta.get("video_id") or "")
    if not video_id:
        raise ValueError("allocation replay metadata lacks video identity")
    window_start = int(round(float(meta.get("window_start_frame", 0))))
    return f"{video_id}|{window_start}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gather_raw(inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if inputs.ndim not in (3, 5, 6):
        raise ValueError("allocation replay supports [B,C,T], [B,C,T,H,W], or [B,N,C,T,H,W]")
    time_dim = 2 if inputs.ndim in (3, 5) else 3
    view = [positions.shape[0]] + [1] * (inputs.ndim - 1)
    view[time_dim] = positions.shape[1]
    expand = list(inputs.shape)
    expand[time_dim] = positions.shape[1]
    active = positions >= 0
    index = positions.clamp_min(0).view(view).expand(expand)
    selected = torch.gather(inputs, time_dim, index)
    mask_view = [positions.shape[0]] + [1] * (inputs.ndim - 1)
    mask_view[time_dim] = positions.shape[1]
    return selected * active.view(mask_view).to(dtype=selected.dtype)
