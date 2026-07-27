import copy
import hashlib
import json
import math
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F
from .boundary_acquisition import load_value_transport_selection_ledger
from tools.bata import paction_budget_contract


def _stable_string_seed(value):
    if value is None:
        value = "unknown"
    if not isinstance(value, str):
        value = str(value)
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


@PIPELINES.register_module()
class PrepareVideoInfo:
    def __init__(self, format="mp4", modality="RGB", prefix=""):
        self.format = format
        self.modality = modality
        self.prefix = prefix

    def __call__(self, results):
        results["modality"] = self.modality
        results["filename"] = os.path.join(
            results["data_path"],
            self.prefix + results["video_name"] + "." + self.format,
        )
        return results


@PIPELINES.register_module()
class DucaExternalActionnessFromJsonl:
    """Attach deployable external actionness to metas for DUCA online selection."""

    def __init__(
        self,
        actionness_jsonl,
        p_action_key="duca_external_p_action",
        logits_key="duca_external_actionness_logits",
        valid_key="duca_external_actionness_valid",
        provenance_key="duca_external_actionness_provenance",
        source_key="duca_external_actionness_source",
        observation_times_key="duca_external_actionness_observation_times",
        allow_missing=False,
    ):
        self.actionness_jsonl = os.path.expandvars(os.path.expanduser(str(actionness_jsonl)))
        self.p_action_key = str(p_action_key)
        self.logits_key = str(logits_key)
        self.valid_key = str(valid_key)
        self.provenance_key = str(provenance_key)
        self.source_key = str(source_key)
        self.observation_times_key = str(observation_times_key)
        self.allow_missing = bool(allow_missing)
        self._index = self._load_actionness_jsonl(self.actionness_jsonl)

    @staticmethod
    def _logit(prob):
        clipped = min(1.0 - 1e-6, max(1e-6, float(prob)))
        return math.log(clipped / (1.0 - clipped))

    @staticmethod
    def _forbidden_true(value):
        if value is True:
            return True
        return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}

    @classmethod
    def _validate_provenance(cls, provenance, *, source_name):
        if not isinstance(provenance, dict):
            raise ValueError(f"{source_name}: source_provenance must be an object")
        for key in ("thumos_trained", "uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
            if provenance.get(key) is not False:
                raise ValueError(f"{source_name}: source_provenance.{key} must be false")
        if provenance.get("calibration_split") not in (None, "", "none", "train_only"):
            raise ValueError(f"{source_name}: source_provenance.calibration_split must be none/train_only")
        for key in ("uses_raw_prediction", "uses_oracle", "uses_cache", "training_only"):
            if cls._forbidden_true(provenance.get(key, False)):
                raise ValueError(f"{source_name}: forbidden source_provenance flag {key}=true")

    @classmethod
    def _load_actionness_jsonl(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"external actionness JSONL missing: {path}")
        grouped = {}
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: row must be an object")
                video_id = str(row.get("video_id") or row.get("video_name") or "")
                if not video_id:
                    raise ValueError(f"{path}:{line_no}: missing video_id")
                provenance = row.get("source_provenance")
                cls._validate_provenance(provenance, source_name=f"{path}:{line_no}")
                if row.get("valid") is not True:
                    continue
                grouped.setdefault(video_id, []).append(
                    {
                        "time": float(row["original_time"]),
                        "p_action": float(row["p_action"]),
                        "logit": float(row.get("logit", cls._logit(float(row["p_action"])))),
                        "source_name": str(row.get("source_name") or provenance.get("source_name") or "external_actionness"),
                        "provenance": dict(provenance),
                    }
                )
        if not grouped:
            raise ValueError(f"external actionness JSONL has no valid rows: {path}")
        out = {}
        for video_id, rows in grouped.items():
            rows = sorted(rows, key=lambda item: item["time"])
            times = np.asarray([item["time"] for item in rows], dtype=np.float32)
            if np.unique(times).size != times.size:
                raise ValueError(f"{path}: duplicate original_time values for video_id={video_id}")
            provenance = rows[0]["provenance"]
            source_name = rows[0]["source_name"]
            for item in rows[1:]:
                if item["source_name"] != source_name:
                    raise ValueError(f"{path}: mixed source_name for video_id={video_id}")
                for key in ("thumos_trained", "uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
                    if item["provenance"].get(key) != provenance.get(key):
                        raise ValueError(f"{path}: mixed provenance field {key} for video_id={video_id}")
            out[video_id] = {
                "times": times,
                "p_action": np.asarray([item["p_action"] for item in rows], dtype=np.float32),
                "logits": np.asarray([item["logit"] for item in rows], dtype=np.float32),
                "source_name": source_name,
                "provenance": provenance,
            }
        return out

    @staticmethod
    def _observation_times(results):
        if "frame_inds" not in results:
            raise ValueError("external actionness alignment requires frame_inds")
        fps = float(results.get("avg_fps", results.get("fps", 0.0)))
        if fps <= 0.0:
            raise ValueError("external actionness alignment requires positive avg_fps/fps")
        frame_inds = results["frame_inds"]
        if torch.is_tensor(frame_inds):
            frames = frame_inds.detach().cpu().numpy()
        else:
            frames = np.asarray(frame_inds)
        frames = frames.reshape(-1).astype(np.float32)
        if frames.size <= 0:
            raise ValueError("external actionness alignment received empty frame_inds")
        masks = results.get("masks")
        if masks is None:
            obs_len = int(frames.size)
        elif torch.is_tensor(masks):
            obs_len = int(masks.numel())
        else:
            obs_len = int(np.asarray(masks).size)
        if obs_len <= 0:
            raise ValueError("external actionness alignment received empty masks")
        if frames.size == obs_len:
            centers = frames
        elif frames.size % obs_len == 0:
            centers = frames.reshape(obs_len, frames.size // obs_len).mean(axis=1)
        else:
            raise ValueError(
                f"frame_inds length {frames.size} must equal or be divisible by observation length {obs_len}"
            )
        return centers / fps

    @staticmethod
    def _mask_valid(results, obs_len):
        masks = results.get("masks")
        if masks is None:
            return [True] * int(obs_len)
        if torch.is_tensor(masks):
            values = masks.detach().cpu().bool().reshape(-1).tolist()
        else:
            values = np.asarray(masks).astype(bool).reshape(-1).tolist()
        if len(values) != int(obs_len):
            raise ValueError("external actionness mask length must match observation length")
        return [bool(item) for item in values]

    def __call__(self, results):
        video_id = str(results.get("video_name") or results.get("video_id") or "")
        if not video_id:
            raise ValueError("external actionness requires video_name/video_id")
        entry = self._index.get(video_id)
        if entry is None:
            if self.allow_missing:
                return results
            raise ValueError(f"missing external actionness for video_id={video_id}")
        observation_times = self._observation_times(results)
        p_action = np.interp(observation_times, entry["times"], entry["p_action"]).astype(np.float32)
        logits = np.interp(observation_times, entry["times"], entry["logits"]).astype(np.float32)
        valid = self._mask_valid(results, len(observation_times))
        results[self.p_action_key] = [float(item) for item in p_action.tolist()]
        results[self.logits_key] = [float(item) for item in logits.tolist()]
        results[self.valid_key] = valid
        results[self.provenance_key] = dict(entry["provenance"])
        results[self.source_key] = str(entry["source_name"])
        results[self.observation_times_key] = [float(item) for item in observation_times.tolist()]
        results["duca_external_actionness_jsonl"] = self.actionness_jsonl
        return results


@PIPELINES.register_module()
class DucaRimeTargetsFromJsonl:
    """Attach cross-fitted train-only RIME targets to one deterministic window."""

    def __init__(
        self,
        targets_jsonl,
        targets_sha256,
        candidate_budgets=(192, 256, 384, 512),
    ):
        self.targets_jsonl = os.path.abspath(
            os.path.expandvars(os.path.expanduser(str(targets_jsonl)))
        )
        self.targets_sha256 = str(targets_sha256).lower()
        self.candidate_budgets = tuple(int(value) for value in candidate_budgets)
        if not os.path.isfile(self.targets_jsonl):
            raise FileNotFoundError(f"RIME target JSONL missing: {self.targets_jsonl}")
        digest = hashlib.sha256()
        with open(self.targets_jsonl, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if not self.targets_sha256 or digest.hexdigest() != self.targets_sha256:
            raise ValueError("RIME target JSONL SHA-256 is required and must match")
        self._index = self._load(self.targets_jsonl)

    def _load(self, path):
        index = {}
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                prefix = f"{path}:{line_number}"
                if row.get("schema_version") != "duca_rime_training_target_v1":
                    raise ValueError(f"{prefix}: unsupported RIME target schema")
                provenance = row.get("provenance")
                if not isinstance(provenance, dict):
                    raise ValueError(f"{prefix}: provenance must be an object")
                if provenance.get("fit_split") not in {"train", "training", "train_only"}:
                    raise ValueError(f"{prefix}: RIME targets must be fit train-only")
                if provenance.get("cross_fitted") is not True:
                    raise ValueError(f"{prefix}: RIME targets must be cross-fitted")
                if provenance.get("uses_validation_or_test") is not False:
                    raise ValueError(f"{prefix}: validation/test leakage is forbidden")
                budgets = tuple(int(value) for value in row.get("candidate_budgets", ()))
                if budgets != self.candidate_budgets:
                    raise ValueError(f"{prefix}: candidate budgets disagree")
                video_id = str(row.get("video_id") or row.get("video_name") or "")
                if not video_id:
                    raise ValueError(f"{prefix}: video_id is required")
                window_start = int(row.get("window_start_frame", -1))
                if window_start < 0:
                    raise ValueError(f"{prefix}: window_start_frame must be non-negative")
                utility = np.asarray(row.get("utility_target"), dtype=np.float32)
                risk = np.asarray(row.get("risk_target"), dtype=np.float32)
                mask = np.asarray(
                    row.get("target_mask", [True] * len(budgets)),
                    dtype=bool,
                )
                expected_shape = (len(budgets),)
                if (
                    utility.shape != expected_shape
                    or risk.shape != expected_shape
                    or mask.shape != expected_shape
                ):
                    raise ValueError(f"{prefix}: per-K targets must align with budgets")
                if not np.isfinite(utility[mask]).all() or not np.isfinite(risk[mask]).all():
                    raise ValueError(f"{prefix}: active targets must be finite")
                if np.any((risk[mask] < 0.0) | (risk[mask] > 1.0)):
                    raise ValueError(f"{prefix}: active risk targets must lie in [0,1]")
                hard_utility = row.get("hard_frame_utility")
                if hard_utility is not None:
                    hard_utility = np.asarray(hard_utility, dtype=np.float32)
                    if hard_utility.ndim != 1 or not np.isfinite(hard_utility).all():
                        raise ValueError(f"{prefix}: hard frame utility must be finite [T]")
                key = (video_id, window_start)
                if key in index:
                    raise ValueError(f"{prefix}: duplicate RIME target window {key}")
                index[key] = {
                    "rime_utility_target": utility,
                    "rime_risk_target": risk,
                    "rime_target_mask": mask,
                    "rime_hard_frame_utility": hard_utility,
                    "rime_target_provenance": provenance,
                }
        if not index:
            raise ValueError(f"RIME target JSONL contains no records: {path}")
        return index

    def __call__(self, results):
        video_id = str(results.get("video_name") or results.get("video_id") or "")
        window_start = int(results.get("window_start_frame", 0))
        key = (video_id, window_start)
        if key not in self._index:
            raise ValueError(f"missing cross-fitted RIME targets for window {key}")
        entry = self._index[key]
        hard_utility = entry["rime_hard_frame_utility"]
        if hard_utility is not None:
            masks = results.get("masks")
            temporal_len = int(np.asarray(masks).size) if masks is not None else int(
                np.asarray(results["frame_inds"]).shape[0]
            )
            if hard_utility.shape != (temporal_len,):
                raise ValueError(
                    "RIME hard-frame utility length must match the dense candidate axis"
                )
        for key_name, value in entry.items():
            if value is not None:
                results[key_name] = value.copy() if hasattr(value, "copy") else dict(value)
        results["rime_targets_jsonl"] = self.targets_jsonl
        results["rime_targets_sha256"] = self.targets_sha256
        return results


@PIPELINES.register_module()
class DucaRimeBudgetReplayFromJsonl:
    """Attach an immutable evaluation/control K assignment without predictions."""

    def __init__(self, replay_jsonl, replay_sha256, candidate_budgets=(192, 256, 384, 512)):
        self.replay_jsonl = os.path.abspath(
            os.path.expandvars(os.path.expanduser(str(replay_jsonl)))
        )
        self.replay_sha256 = str(replay_sha256).lower()
        self.candidate_budgets = tuple(int(value) for value in candidate_budgets)
        if not os.path.isfile(self.replay_jsonl):
            raise FileNotFoundError(f"RIME budget replay JSONL missing: {self.replay_jsonl}")
        digest = hashlib.sha256()
        with open(self.replay_jsonl, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if not self.replay_sha256 or digest.hexdigest() != self.replay_sha256:
            raise ValueError("RIME budget replay SHA-256 is required and must match")
        self._index = {}
        with open(self.replay_jsonl, "r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                prefix = f"{self.replay_jsonl}:{line_number}"
                if row.get("schema_version") != "duca_rime_budget_replay_v1":
                    raise ValueError(f"{prefix}: unsupported replay schema")
                video = str(row.get("video_id") or row.get("video_name") or "")
                window_start = int(row.get("window_start_frame", -1))
                requested_k = int(row.get("requested_k", -1))
                provenance = row.get("provenance")
                if (
                    not video
                    or window_start < 0
                    or requested_k not in self.candidate_budgets
                    or not isinstance(provenance, dict)
                ):
                    raise ValueError(f"{prefix}: invalid replay record")
                if any(
                    bool(provenance.get(key, False))
                    for key in ("uses_gt", "uses_teacher", "uses_prediction_cache")
                ):
                    raise ValueError(f"{prefix}: contaminated replay provenance")
                key = (video, window_start)
                if key in self._index:
                    raise ValueError(f"{prefix}: duplicate replay window {key}")
                self._index[key] = (requested_k, dict(provenance))
        if not self._index:
            raise ValueError("RIME budget replay JSONL contains no records")

    def __call__(self, results):
        key = (
            str(results.get("video_name") or results.get("video_id") or ""),
            int(results.get("window_start_frame", 0)),
        )
        if key not in self._index:
            raise ValueError(f"missing RIME budget replay for window {key}")
        requested_k, provenance = self._index[key]
        results["rime_requested_k_replay"] = int(requested_k)
        results["rime_requested_k_replay_provenance"] = dict(provenance)
        results["rime_budget_replay_jsonl"] = self.replay_jsonl
        results["rime_budget_replay_sha256"] = self.replay_sha256
        return results


@PIPELINES.register_module()
class LoadSnippetFrames:
    """Load the snippet frame, the output should follows the format:
    snippet_num x channel x clip_len x height x width
    """

    def __init__(
        self,
        clip_len,
        frame_interval=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.method = method  # resize or padding or sliding window
        # todo: support to  change FPS
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            snippet_num = results["resize_length"]
            snippet_stride = total_frames / snippet_num
            snippet_center = np.arange(
                snippet_stride / 2 - 0.5,
                total_frames + snippet_stride / 2 - 0.5,
                snippet_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            snippet_num = self.trunc_len
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            # trunc the snippet_center
            snippet_center, gt_segments, gt_labels = self.random_trunc(
                snippet_center,
                trunc_len=snippet_num,
                gt_segments=results["gt_segments"],
                gt_labels=results["gt_labels"],
            )

            # update the gt_segments
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

            # pad the snippet_center
            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        elif self.method == "sliding_window":
            snippet_num = results["window_size"]
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            start_idx = min(results["feature_start_idx"], len(snippet_center))
            end_idx = min((results["feature_end_idx"] + 1), len(snippet_center))

            snippet_center = snippet_center[start_idx:end_idx]

            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()
        elif self.method == "padding":
            raise NotImplementedError

        # extend snippet center to a clip
        clip_idxs = np.arange(-(self.clip_len // 2), self.clip_len // 2)
        frame_idxs = snippet_center[:, None] + self.frame_interval * clip_idxs[None, :]  # [snippet_num, clip_len]

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == snippet_num, "snippet center number should be equal to snippet number"
        assert frame_idxs.shape[1] == self.clip_len, "snippet length should be equal to clip length"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = snippet_num
        results["clip_len"] = self.clip_len
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class LoadFrames:
    def __init__(
        self,
        num_clips=1,
        scale_factor=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
        keep_ratio=0.5,
        method_base=None,
        target_len=None,
        source_len=None,
        remap_gt_to_selected_axis=True,
        bata_value_transport_ledger_path=None,
        bata_value_transport_allow_missing_fallback=False,
        bata_value_transport_require_deployable=True,
        bata_value_transport_require_selected_count=None,
        bata_value_transport_allow_short_valid_ratio_count=False,
        bata_value_transport_source="pc_ot_mras_frontend_hard_positions",
        bata_value_transport_config_hash="",
        bata_value_transport_use_expanded_positions=False,
        emit_boundary_validity=False,
    ):
        self.num_clips = num_clips
        self.scale_factor = scale_factor  # multiply by the frame number, if backbone has downsampling
        self.method = method  # resize or padding or random_trunc or sliding_window or value-transport ledger
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.keep_ratio = keep_ratio
        self.method_base = method_base
        self.target_len = target_len
        self.source_len = source_len
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.bata_value_transport_ledger_path = bata_value_transport_ledger_path
        self.bata_value_transport_allow_missing_fallback = bool(bata_value_transport_allow_missing_fallback)
        self.bata_value_transport_require_deployable = bool(bata_value_transport_require_deployable)
        self.bata_value_transport_require_selected_count = bata_value_transport_require_selected_count
        self.bata_value_transport_allow_short_valid_ratio_count = bool(
            bata_value_transport_allow_short_valid_ratio_count
        )
        self.bata_value_transport_source = bata_value_transport_source
        self.bata_value_transport_config_hash = bata_value_transport_config_hash
        self.bata_value_transport_use_expanded_positions = bool(bata_value_transport_use_expanded_positions)
        self.emit_boundary_validity = bool(emit_boundary_validity)
        self._bata_value_transport_ledger = None

    def random_trunc(
        self,
        feats,
        trunc_len,
        gt_segments,
        gt_labels,
        offset=0,
        max_num_trials=200,
        return_boundary_validity=False,
    ):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        def pack(out_feats, out_segments, out_labels, validity):
            if return_boundary_validity:
                return out_feats, out_segments, out_labels, validity
            return out_feats, out_segments, out_labels

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                validity = np.ones((num_segs, 2), dtype=np.bool_)
                return pack(feats, gt_segments, gt_labels, validity)
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    validity = np.ones((num_segs, 2), dtype=np.bool_)
                    return pack(feats, gt_segments, gt_labels, validity)

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        original_segments = gt_segments[seg_idx]
        boundary_validity = np.stack(
            (
                np.isclose(
                    left[seg_idx],
                    original_segments[:, 0],
                    rtol=0.0,
                    atol=1.0e-6,
                ),
                np.isclose(
                    right[seg_idx],
                    original_segments[:, 1],
                    rtol=0.0,
                    atol=1.0e-6,
                ),
            ),
            axis=1,
        ).astype(np.bool_)
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return pack(feats, gt_segments, gt_labels, boundary_validity)

    def _map_coord_to_selected_axis(self, coord, kept_positions, valid_len):
        if kept_positions.size == 0:
            return 0.0
        xp = np.concatenate([kept_positions.astype(np.float32), np.array([float(valid_len)], dtype=np.float32)])
        fp = np.concatenate(
            [np.arange(kept_positions.size, dtype=np.float32), np.array([float(kept_positions.size)], dtype=np.float32)]
        )
        coord = float(np.clip(coord, 0.0, float(valid_len)))
        return float(np.interp(coord, xp, fp))

    def _remap_gt_to_selected_axis(self, gt_segments, gt_labels, kept_positions, valid_len):
        if gt_segments is None or gt_labels is None or len(gt_segments) == 0 or kept_positions.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        remapped_segments = []
        remapped_labels = []
        max_coord = float(kept_positions.size)
        for idx, seg in enumerate(gt_segments):
            start = self._map_coord_to_selected_axis(seg[0], kept_positions, valid_len)
            end = self._map_coord_to_selected_axis(seg[1], kept_positions, valid_len)
            start = float(np.clip(start, 0.0, max_coord))
            end = float(np.clip(end, 0.0, max_coord))
            if end <= start:
                end = min(max_coord, start + 1e-3)
            if end > start:
                remapped_segments.append([start, end])
                remapped_labels.append(int(gt_labels[idx]))

        if len(remapped_segments) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.asarray(remapped_segments, dtype=np.float32), np.asarray(remapped_labels, dtype=np.int32)

    def _set_irregular_axis_meta(self, results, kept_positions, valid_len):
        scale = float(max(self.scale_factor, 1))
        selected_positions = np.asarray(kept_positions, dtype=np.float32) / scale
        results["irregular_selected_positions"] = selected_positions
        results["selected_dense_indices"] = selected_positions
        results["selected_valid_len"] = int(len(kept_positions))
        results["irregular_selected_valid_len"] = float(valid_len) / scale
        results["irregular_dense_valid_len"] = float(valid_len) / scale
        results["remap_gt_to_selected_axis"] = bool(self.remap_gt_to_selected_axis)
        results["gt_remapped_to_selected_axis"] = bool(self.remap_gt_to_selected_axis)
        results["irregular_native_axis"] = bool(not self.remap_gt_to_selected_axis)

    def _select_random_fixed_positions(self, valid_len, target_frame_num, sample_key):
        valid_len = int(valid_len)
        target_frame_num = int(target_frame_num)
        if valid_len <= 0 or target_frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)
        if target_frame_num >= valid_len:
            return np.arange(valid_len, dtype=np.int64)
        rng = np.random.RandomState(_stable_string_seed(sample_key))
        return np.sort(rng.choice(valid_len, size=target_frame_num, replace=False)).astype(np.int64)

    def _exact_uniform_dense_positions(self, valid_len, dense_frame_num, frame_num):
        valid_len = int(valid_len)
        dense_frame_num = int(dense_frame_num)
        frame_num = int(frame_num)
        if valid_len <= 0 or dense_frame_num <= 0 or frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)
        selected_valid_len = int(np.ceil(float(valid_len) * float(frame_num) / float(dense_frame_num)))
        selected_valid_len = max(1, min(selected_valid_len, frame_num))
        step = float(dense_frame_num) / float(frame_num)
        positions = [int(round(float(idx) * step)) for idx in range(selected_valid_len)]
        positions = [int(np.clip(pos, 0, valid_len - 1)) for pos in positions]
        return np.asarray(sorted(dict.fromkeys(positions)), dtype=np.int64)

    def _canonical_exact_uniform_positions(self, valid_len, target_frame_num):
        valid_len = int(valid_len)
        target_frame_num = min(int(target_frame_num), valid_len)
        if valid_len <= 0 or target_frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)
        if target_frame_num == 1:
            return np.zeros((1,), dtype=np.int64)
        denominator = target_frame_num - 1
        positions = []
        for index in range(target_frame_num):
            numerator = index * (valid_len - 1)
            quotient, remainder = divmod(numerator, denominator)
            if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1):
                quotient += 1
            positions.append(quotient)
        result = np.asarray(positions, dtype=np.int64)
        if np.unique(result).size != target_frame_num:
            raise RuntimeError("canonical exact-uniform positions must be unique")
        return result

    def _value_transport_ledger(self):
        if self._bata_value_transport_ledger is None:
            self._bata_value_transport_ledger = load_value_transport_selection_ledger(
                self.bata_value_transport_ledger_path,
                require_deployable=self.bata_value_transport_require_deployable,
            )
        return self._bata_value_transport_ledger

    def _value_transport_sample_id(self, results):
        if "window_start_frame" not in results:
            raise ValueError("value_transport_ledger_subsample requires window_start_frame in results")
        return f"{results.get('video_name', 'unknown')}|{int(results['window_start_frame'])}"

    def _value_transport_metadata(self, row, key):
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        return row.get(key, diagnostics.get(key))

    def _assert_value_transport_ledger_contract(self, row, sample_id, valid_len, dense_frame_num, target_len):
        if bool(row.get("fallback_missing_ledger", False)):
            return
        if self.bata_value_transport_require_deployable:
            if row.get("deploy_selection_ledger") is not True:
                raise ValueError(f"value-transport ledger sample_id={sample_id} must set deploy_selection_ledger=true")
            if row.get("diagnostic_only") is True:
                raise ValueError(f"value-transport ledger sample_id={sample_id} is diagnostic_only")
        for key in (
            "uses_gt",
            "uses_gt_for_diagnostics",
            "uses_teacher",
            "uses_oracle",
            "uses_cache",
            "uses_prediction_cache",
            "uses_raw_prediction",
            "uses_checkpoint",
            "prediction_uses_gt",
            "training_only",
        ):
            value = row.get(key, False)
            if value is True or (isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}):
                raise ValueError(f"value-transport ledger sample_id={sample_id} forbidden flag {key}=true")
        row_target_len = row.get("target_len")
        if row_target_len is not None and int(row_target_len) != int(target_len):
            raise ValueError(
                f"value-transport ledger sample_id={sample_id} target_len={row_target_len} "
                f"does not match configured target_len={target_len}"
            )
        row_dense_len = row.get("dense_len")
        if row_dense_len is not None and int(row_dense_len) != int(dense_frame_num):
            raise ValueError(
                f"value-transport ledger sample_id={sample_id} dense_len={row_dense_len} "
                f"does not match dense_frame_num={dense_frame_num}"
            )
        row_valid_len = row.get("valid_len")
        if row_valid_len is not None and int(row_valid_len) != int(valid_len):
            raise ValueError(
                f"value-transport ledger sample_id={sample_id} valid_len={row_valid_len} "
                f"does not match runtime valid_len={valid_len}"
            )
        expected_source = str(self.bata_value_transport_source or "")
        row_policy_source = self._value_transport_metadata(row, "policy_source")
        if expected_source:
            if row_policy_source != expected_source:
                raise ValueError(
                    f"value-transport ledger sample_id={sample_id} policy_source={row_policy_source} "
                    f"does not match configured source={expected_source}"
                )
        expected_hash = str(self.bata_value_transport_config_hash or "")
        if expected_hash:
            row_checkpoint_sha = self._value_transport_metadata(row, "policy_checkpoint_sha256")
            if row_checkpoint_sha != expected_hash:
                raise ValueError(
                    f"value-transport ledger sample_id={sample_id} policy_checkpoint_sha256 mismatch"
                )

    def _lookup_value_transport_positions(self, results, valid_len, dense_frame_num, frame_num, target_len):
        sample_id = self._value_transport_sample_id(results)
        row = self._value_transport_ledger().get(sample_id)
        if row is None:
            if not self.bata_value_transport_allow_missing_fallback:
                raise KeyError(f"value-transport ledger missing sample_id={sample_id}")
            positions = self._exact_uniform_dense_positions(valid_len, dense_frame_num, frame_num)
            return positions, dict(sample_id=sample_id, fallback_missing_ledger=True)
        self._assert_value_transport_ledger_contract(
            row,
            sample_id,
            valid_len=valid_len,
            dense_frame_num=dense_frame_num,
            target_len=target_len,
        )
        positions = row["selected_positions"].astype(np.int64, copy=False).reshape(-1)
        if positions.size == 0:
            raise ValueError(f"value-transport ledger sample_id={sample_id} selected no positions")
        if positions[0] < 0 or positions[-1] >= int(valid_len):
            raise ValueError(f"value-transport ledger sample_id={sample_id} positions exceed valid dense window")
        return positions, row

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            frame_num = results["resize_length"] * self.scale_factor
            frame_stride = total_frames / frame_num
            frame_idxs = np.arange(
                frame_stride / 2 - 0.5,
                total_frames + frame_stride / 2 - 0.5,
                frame_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()  # should not multiply by scale_factor

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            frame_num = self.trunc_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            # trunc the frame_idxs
            truncation = self.random_trunc(
                frame_idxs,
                trunc_len=frame_num,
                gt_segments=results["gt_segments"] * self.scale_factor,  # gt segment should be mapped to frame level
                gt_labels=results["gt_labels"],
                return_boundary_validity=self.emit_boundary_validity,
            )
            if self.emit_boundary_validity:
                (
                    frame_idxs,
                    gt_segments,
                    gt_labels,
                    gt_boundary_validity,
                ) = truncation
                results["gt_boundary_validity"] = gt_boundary_validity
            else:
                frame_idxs, gt_segments, gt_labels = truncation
            results["gt_segments"] = gt_segments / self.scale_factor  # convert back to original scale
            results["gt_labels"] = gt_labels

            # pad the frame_idxs
            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(self.trunc_len - valid_len)]).bool()
            else:
                masks = torch.ones(self.trunc_len).bool()

        elif self.method == "sliding_window":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            window_size = results["window_size"]
            frame_num = window_size * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            start_idx = min(results["feature_start_idx"] * self.scale_factor, len(frame_idxs))
            end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(frame_idxs))

            frame_idxs = frame_idxs[start_idx:end_idx]

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)]).bool()
            else:
                masks = torch.ones(window_size).bool()

        elif self.method in {"random_fixed_subsample", "exact_uniform_fixed_subsample"}:
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            keep_ratio = float(self.keep_ratio)
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            gt_segments = results["gt_segments"] * self.scale_factor if "gt_segments" in results else None
            gt_labels = results["gt_labels"] if "gt_labels" in results else None

            if self.method_base == "random_trunc":
                if gt_segments is None or gt_labels is None:
                    raise ValueError("random_fixed_subsample with random_trunc requires gt_segments and gt_labels")
                if self.trunc_len is None and self.target_len is None:
                    raise ValueError(
                        "random_fixed_subsample requires trunc_len or target_len when method_base='random_trunc'"
                    )
                target_len = int(self.target_len) if self.target_len is not None else int(self.trunc_len)
                source_len = int(self.source_len) if self.source_len is not None else int(round(target_len / max(keep_ratio, 1e-6)))
                frame_num = target_len * self.scale_factor
                dense_frame_num = source_len * self.scale_factor
                dense_window, gt_segments, gt_labels = self.random_trunc(
                    dense_frame_idxs,
                    trunc_len=dense_frame_num,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                )
            elif self.method_base == "sliding_window":
                if "window_size" not in results:
                    raise ValueError("random_fixed_subsample with sliding_window requires window_size in results")
                dense_window_len = int(results["window_size"])
                target_len = int(self.target_len) if self.target_len is not None else int(round(dense_window_len * keep_ratio))
                frame_num = target_len * self.scale_factor
                dense_frame_num = dense_window_len * self.scale_factor
                start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
                end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
                dense_window = dense_frame_idxs[start_idx:end_idx]
            else:
                raise ValueError("random_fixed_subsample requires method_base='random_trunc' or 'sliding_window'")

            valid_len = int(len(dense_window))
            if valid_len <= 0:
                raise RuntimeError("random_fixed_subsample received an empty dense window")

            if self.method == "exact_uniform_fixed_subsample":
                keep_positions = self._canonical_exact_uniform_positions(valid_len, frame_num)
                results["fixed_subsample_policy"] = "canonical_round_endpoint_exact_uniform"
            else:
                sample_key = (
                    f"{results.get('video_name', 'unknown')}|random_fixed|"
                    f"{int(dense_window[0])}|{int(dense_window[-1])}|{valid_len}|{frame_num}"
                )
                keep_positions = self._select_random_fixed_positions(valid_len, frame_num, sample_key)
                results["fixed_subsample_policy"] = "deterministic_sample_key_random"
            if keep_positions.size == 0:
                keep_positions = np.array([0], dtype=np.int64)

            frame_idxs = dense_window[keep_positions]
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

            if gt_segments is not None and gt_labels is not None:
                if self.remap_gt_to_selected_axis:
                    gt_segments, gt_labels = self._remap_gt_to_selected_axis(
                        gt_segments=gt_segments,
                        gt_labels=gt_labels,
                        kept_positions=keep_positions,
                        valid_len=valid_len,
                    )
                results["gt_segments"] = gt_segments / self.scale_factor
                results["gt_labels"] = gt_labels

            if len(frame_idxs) < frame_num:
                valid_mask_len = min(
                    int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
                    int(np.ceil(frame_num / max(self.scale_factor, 1))),
                )
                target_mask_len = int(np.ceil(frame_num / self.scale_factor))
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
            else:
                masks = torch.ones(int(np.ceil(frame_num / self.scale_factor))).bool()

        elif self.method == "bata_value_transport_ledger_subsample":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"
            if self.method_base != "sliding_window":
                raise ValueError("bata_value_transport_ledger_subsample currently supports method_base='sliding_window'")
            if "window_size" not in results:
                raise ValueError("bata_value_transport_ledger_subsample requires window_size in results")

            dense_window_len = int(results["window_size"])
            target_len = int(self.target_len) if self.target_len is not None else int(round(dense_window_len * float(self.keep_ratio)))
            frame_num = target_len * self.scale_factor
            dense_frame_num = dense_window_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
            end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
            dense_window = dense_frame_idxs[start_idx:end_idx]
            valid_len = int(len(dense_window))
            if valid_len <= 0:
                raise RuntimeError("bata_value_transport_ledger_subsample received an empty dense window")

            keep_positions, ledger_row = self._lookup_value_transport_positions(
                results,
                valid_len=valid_len,
                dense_frame_num=dense_frame_num,
                frame_num=frame_num,
                target_len=target_len,
            )
            if self.bata_value_transport_use_expanded_positions and "expanded_selected_positions" in ledger_row:
                keep_positions = np.asarray(
                    [int(item) for item in ledger_row["expanded_selected_positions"]],
                    dtype=np.int64,
                )
                if keep_positions.size == 0:
                    raise ValueError("expanded_selected_positions must not be empty")
                if keep_positions.tolist() != sorted(set(int(item) for item in keep_positions.tolist())):
                    raise ValueError("expanded_selected_positions must be sorted unique")
                if keep_positions[0] < 0 or keep_positions[-1] >= valid_len:
                    raise ValueError("expanded_selected_positions exceed valid_len")
            if keep_positions.size > int(frame_num):
                sample_id = ledger_row.get("sample_id", self._value_transport_sample_id(results))
                raise ValueError(
                    f"value-transport ledger sample_id={sample_id} selects {keep_positions.size} positions "
                    f"but target_len={target_len} allows only {frame_num} frame indices"
                )
            required_count = self.bata_value_transport_require_selected_count
            if required_count is True:
                required_count = int(frame_num)
            elif required_count in (False, None):
                required_count = None
            else:
                required_count = int(required_count)
            if required_count is not None and keep_positions.size != required_count:
                expected_required_count = paction_budget_contract.expected_selected_count(
                    required_count,
                    valid_len=int(valid_len),
                    dense_len=int(dense_frame_num),
                    allow_short_valid_ratio_count=bool(self.bata_value_transport_allow_short_valid_ratio_count),
                )
                if keep_positions.size == expected_required_count:
                    required_count = expected_required_count
                else:
                    sample_id = ledger_row.get("sample_id", self._value_transport_sample_id(results))
                    raise ValueError(
                        f"value-transport ledger sample_id={sample_id} selects {keep_positions.size} positions "
                        f"but bata_value_transport_require_selected_count={required_count} "
                        f"(expected={expected_required_count}, valid_len={valid_len}, dense_frame_num={dense_frame_num})"
                    )
            if required_count is not None and keep_positions.size != required_count:
                sample_id = ledger_row.get("sample_id", self._value_transport_sample_id(results))
                raise ValueError(
                    f"value-transport ledger sample_id={sample_id} selects {keep_positions.size} positions "
                    f"but bata_value_transport_require_selected_count={required_count}"
                )

            frame_idxs = dense_window[keep_positions]
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

            gt_segments = results["gt_segments"] * self.scale_factor if "gt_segments" in results else None
            gt_labels = results["gt_labels"] if "gt_labels" in results else None
            if gt_segments is not None and gt_labels is not None:
                if self.remap_gt_to_selected_axis:
                    gt_segments, gt_labels = self._remap_gt_to_selected_axis(
                        gt_segments=gt_segments,
                        gt_labels=gt_labels,
                        kept_positions=keep_positions,
                        valid_len=valid_len,
                    )
                results["gt_segments"] = gt_segments / self.scale_factor
                results["gt_labels"] = gt_labels

            results["bata_selected_dense_indices"] = keep_positions.astype(np.int64)
            results["bata_value_transport_selection_row"] = {
                key: value
                for key, value in dict(ledger_row).items()
                if key
                in {
                    "sample_id",
                    "schema_version",
                    "route",
                    "route_variant",
                    "policy",
                    "valid_len",
                    "dense_len",
                    "target_len",
                    "selected_count",
                    "selected_positions_unit",
                    "selected_positions_are_centers",
                    "context_radius_unit",
                    "context_radius_range",
                    "context_radius_by_position",
                    "context_radius_float_by_position",
                    "selected_observations",
                    "expanded_selected_positions",
                    "expanded_selected_count",
                    "diagnostics",
                    "diagnostic_only",
                    "training_only",
                    "diagnostic_uses_train_utility_for_audit",
                    "deploy_selection_ledger",
                    "prediction_uses_gt",
                    "uses_gt",
                    "uses_teacher",
                    "uses_oracle",
                    "uses_cache",
                    "uses_prediction_cache",
                    "uses_raw_prediction",
                    "uses_checkpoint",
                }
            }
            results["bata_score_source"] = self.bata_value_transport_source
            results["bata_value_transport_config_hash"] = self.bata_value_transport_config_hash
            results["bata_diagnostic_only"] = bool(ledger_row.get("diagnostic_only", False))

            if len(frame_idxs) < frame_num:
                valid_mask_len = min(
                    int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
                    int(np.ceil(frame_num / max(self.scale_factor, 1))),
                )
                target_mask_len = int(np.ceil(frame_num / self.scale_factor))
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
            else:
                masks = torch.ones(int(np.ceil(frame_num / self.scale_factor))).bool()

        elif self.method == "padding":
            raise NotImplementedError

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == frame_num, "snippet center number should be equal to snippet number"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = self.num_clips
        results["clip_len"] = frame_num // self.num_clips
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class Interpolate:
    def __init__(self, keys, size=128, mode="linear"):
        self.keys = keys
        self.size = size
        self.mode = mode

    def __call__(self, results):
        for key in self.keys:
            if results[key].shape[2:] != self.size:
                results[key] = F.interpolate(
                    results[key],
                    size=self.size,
                    mode=self.mode,
                    align_corners=False,
                )
        return results
