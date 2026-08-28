import copy
import json
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F


@PIPELINES.register_module()
class LoadFeats:
    def __init__(self, feat_format, prefix="", suffix=""):
        self.feat_format = feat_format
        self.prefix = prefix
        self.suffix = suffix
        # check feat format
        if isinstance(self.feat_format, str):
            self.check_feat_format(self.feat_format)
        elif isinstance(self.feat_format, list):
            for feat_format in self.feat_format:
                self.check_feat_format(feat_format)

    def check_feat_format(self, feat_format):
        assert feat_format in ["npy", "npz", "pt", "csv", "pkl"], print(f"not support {feat_format}")

    def read_from_tensor(self, file_path):
        feats = torch.load(file_path).float()
        return feats

    def read_from_npy(self, file_path):
        feats = np.load(file_path).astype(np.float32)
        return feats

    def read_from_npz(self, file_path):
        feats = np.load(file_path)["feats"].astype(np.float32)
        return feats

    def read_from_csv(self, file_path):
        feats = pd.read_csv(file_path, dtype="float32").to_numpy()
        feats = feats.astype(np.float32)
        return feats

    def read_from_pkl(self, file_path):
        feats = pickle.load(open(file_path, "rb"))
        feats = feats.astype(np.float32)
        return feats

    def load_single_feat(self, file_path, feat_format):
        try:
            if feat_format == "npy":
                feats = self.read_from_npy(file_path)
            elif feat_format == "npz":
                feats = self.read_from_npz(file_path)
            elif feat_format == "pt":
                feats = self.read_from_tensor(file_path)
            elif feat_format == "csv":
                feats = self.read_from_csv(file_path)
            elif feat_format == "pkl":
                feats = self.read_from_pkl(file_path)
        except:
            print("Missing data:", file_path)
            exit()
        return feats

    def __call__(self, results):
        video_name = results["video_name"]

        if isinstance(results["data_path"], str):
            file_path = os.path.join(results["data_path"], f"{self.prefix}{video_name}{self.suffix}.{self.feat_format}")
            feats = self.load_single_feat(file_path, self.feat_format)
        elif isinstance(results["data_path"], list):
            feats = []

            # check if the feat_format is a list
            if isinstance(self.feat_format, str):
                self.feat_format = [self.feat_format] * len(results["data_path"])

            for data_path, feat_format in zip(results["data_path"], self.feat_format):
                file_path = os.path.join(data_path, f"{self.prefix}{video_name}{self.suffix}.{feat_format}")
                feats.append(self.load_single_feat(file_path, feat_format))

            max_len = max([feat.shape[0] for feat in feats])
            for i in range(len(feats)):
                if feats[i].shape[0] != max_len:
                    # assume the first dimension is T
                    tmp_feat = F.interpolate(
                        torch.Tensor(feats[i]).permute(1, 0).unsqueeze(0),
                        size=max_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)
                    feats[i] = tmp_feat.permute(1, 0).numpy()
            feats = np.concatenate(feats, axis=1)

        # sample the feature
        sample_stride = results.get("sample_stride", 1)
        if sample_stride > 1:
            feats = feats[::sample_stride]

        results["feats"] = feats
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"feat_format={self.feat_format}"
        return repr_str


@PIPELINES.register_module()
class SlidingWindowTrunc:
    """This is used for sliding window dataset, which will give a window start and window end in the result dict,
    and we will extract the window features, also pad to fixed length"""

    def __init__(self, with_mask=True):
        self.with_mask = with_mask

    def __call__(self, results):
        assert "window_size" in results.keys(), "should have window_size as a key"
        assert isinstance(results["feats"], torch.Tensor)
        window_size = results["window_size"]

        feats_length = results["feats"].shape[0]
        start_idx = min(results["feature_start_idx"], feats_length)
        end_idx = min(results["feature_end_idx"] + 1, feats_length)

        window_feats = results["feats"][start_idx:end_idx]
        valid_len = window_feats.shape[0]

        # if the valid window is smaller than window size, pad with -1
        if valid_len < window_size:
            pad_data = torch.zeros(window_size - valid_len, window_feats.shape[1])
            window_feats = torch.cat((window_feats, pad_data), dim=0)

        # if we need padding mask (valid is 1, pad is 0)
        if self.with_mask:
            if valid_len < window_size:
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)])
            else:
                masks = torch.ones(window_size)
            results["masks"] = masks.bool()

        results["feats"] = window_feats.float()
        return results


@PIPELINES.register_module()
class LoadDucaWindowBudgetFrames:
    """Load the table-frozen exact-K RGB observations for one THUMOS window."""

    _ARMS = {"fixed384", "semantic", "permuted_control"}

    def __init__(self, table_path, arm, detector_length=384, clip_len=16, scale_factor=1):
        self.table_path = os.path.abspath(os.path.expanduser(str(table_path)))
        self.arm = str(arm)
        self.detector_length = int(detector_length)
        self.clip_len = int(clip_len)
        self.scale_factor = int(scale_factor)
        if self.arm not in self._ARMS:
            raise ValueError(f"unsupported DUCA budget arm: {self.arm}")
        if self.detector_length <= 1 or self.clip_len <= 0 or self.scale_factor != 1:
            raise ValueError("DUCA budget frames require detector_length>1, clip_len>0 and scale_factor=1")
        if not os.path.isfile(self.table_path):
            raise FileNotFoundError(self.table_path)
        self.rows = self._read_table(self.table_path)

    @staticmethod
    def _read_table(path):
        rows = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (
                    str(row["split"]),
                    str(row["video_name"]),
                    int(row["window_index"]),
                )
                if key in rows:
                    raise ValueError(f"duplicate DUCA window-table key {key} at line {line_number}")
                rows[key] = row
        if not rows:
            raise ValueError("DUCA window table is empty")
        return rows

    @staticmethod
    def _selected_to_detector_grid(selected_positions, detector_length):
        selected = np.asarray(selected_positions, dtype=np.float64)
        if selected.ndim != 1 or selected.size < 2 or selected.size % 2 != 0:
            raise ValueError("selected positions must be a non-empty even 1-D sequence")
        if np.any(np.diff(selected) <= 0):
            raise ValueError("selected positions must be strictly increasing")
        tubelet_centers = selected.reshape(-1, 2).mean(axis=1)
        source_axis = np.arange(tubelet_centers.size, dtype=np.float64)
        target_axis = np.linspace(0.0, float(tubelet_centers.size - 1), detector_length)
        detector_grid = np.interp(target_axis, source_axis, tubelet_centers)
        if np.any(np.diff(detector_grid) <= 0):
            raise ValueError("detector physical-time grid must be strictly increasing")
        return detector_grid.astype(np.float32)

    @staticmethod
    def _true_to_detector(values, detector_grid, valid_len):
        true_knots = np.asarray(detector_grid, dtype=np.float64)
        detector_knots = np.arange(true_knots.size, dtype=np.float64)
        if true_knots[0] > 0.0:
            true_knots = np.concatenate(([0.0], true_knots))
            detector_knots = np.concatenate(([-1.0], detector_knots))
        if true_knots[-1] < float(valid_len):
            true_knots = np.concatenate((true_knots, [float(valid_len)]))
            detector_knots = np.concatenate((detector_knots, [float(detector_grid.size)]))
        return np.interp(values, true_knots, detector_knots).astype(np.float32)

    def __call__(self, results):
        split = str(results["duca_split"])
        video_name = str(results["video_name"])
        window_index = int(results["duca_window_index"])
        key = (split, video_name, window_index)
        if key not in self.rows:
            raise KeyError(f"DUCA window table has no row for {key}")
        row = self.rows[key]
        if int(row["window_count"]) != int(results["duca_window_count"]):
            raise ValueError(f"DUCA window count mismatch for {key}")
        if int(row["window_start_frame"]) != int(results["window_start_frame"]):
            raise ValueError(f"DUCA window start mismatch for {key}")
        if int(row["window_end_frame"]) != int(results["window_end_frame"]):
            raise ValueError(f"DUCA window end mismatch for {key}")

        requested_k = 384 if self.arm == "fixed384" else int(row[f"{self.arm}_budget"])
        if requested_k not in {256, 384, 512} or requested_k % self.clip_len != 0:
            raise ValueError(f"invalid DUCA exact-K budget {requested_k} for {key}")
        selected_positions = np.asarray(
            row["positions_by_budget"][str(requested_k)], dtype=np.int64
        )
        if selected_positions.shape != (requested_k,):
            raise ValueError(f"DUCA row {key} does not contain exact K={requested_k} positions")
        if selected_positions.tolist() != sorted(set(selected_positions.tolist())):
            raise ValueError(f"DUCA selected positions must be sorted unique for {key}")

        total_frames = int(results["total_frames"])
        frame_stride = int(results["snippet_stride"])
        dense_frame_indices = np.arange(0, total_frames, frame_stride)
        start = min(int(results["feature_start_idx"]), len(dense_frame_indices))
        end = min(int(results["feature_end_idx"]) + 1, len(dense_frame_indices))
        dense_window = dense_frame_indices[start:end]
        valid_len = int(dense_window.size)
        if valid_len < requested_k:
            raise RuntimeError(
                f"DUCA exact-K={requested_k} exceeds valid window length {valid_len} for {key}"
            )
        if selected_positions[0] < 0 or selected_positions[-1] >= valid_len:
            raise ValueError(f"DUCA selected positions exceed valid window for {key}")

        detector_grid = self._selected_to_detector_grid(selected_positions, self.detector_length)
        if "gt_segments" in results:
            results["gt_segments"] = self._true_to_detector(
                np.asarray(results["gt_segments"], dtype=np.float32),
                detector_grid,
                valid_len,
            )
            results["gt_remapped_to_selected_axis"] = True
            results["gt_coordinate_space"] = "selected_axis_index"

        results["frame_inds"] = dense_window[selected_positions].astype(np.int64)
        results["num_clips"] = requested_k // self.clip_len
        results["clip_len"] = self.clip_len
        results["masks"] = torch.ones(self.detector_length, dtype=torch.bool)
        results["duca_budget_arm"] = self.arm
        results["duca_requested_k"] = requested_k
        results["duca_effective_k"] = requested_k
        results["duca_unique_k"] = int(selected_positions.size)
        results["duca_actual_backbone_input_k"] = requested_k
        results["duca_actual_backbone_chunks"] = requested_k // self.clip_len
        results["duca_detector_length"] = self.detector_length
        # This becomes true only after the backbone has actually consumed the
        # requested K without global-max padding.
        results["duca_dynamic_compute_realized"] = False
        results["duca_acquisition_positions"] = selected_positions.tolist()
        results["selected_axis_to_true_time_dense_index"] = detector_grid.tolist()
        results["irregular_selected_positions"] = detector_grid.tolist()
        results["irregular_selected_count"] = self.detector_length
        results["irregular_selected_valid_len"] = self.detector_length
        results["irregular_dense_valid_len"] = valid_len
        results["truetime_dense_len"] = int(results["window_size"])
        results["truetime_dense_valid_len"] = valid_len
        results["detector_prediction_inverse_map_required"] = True
        results["detector_output_coordinate_space"] = "selected_axis_index"
        results["irregular_native_axis"] = False
        results["duca_window_table_path"] = self.table_path
        return results


@PIPELINES.register_module()
class RandomTrunc:
    """Crops features within a window such that they have a large overlap with ground truth segments.
    Withing the cropping ratio, the length is sampled."""

    def __init__(
        self,
        trunc_len,
        trunc_thresh,
        crop_ratio=None,
        max_num_trials=200,
        has_action=True,
        no_trunc=False,
        pad_value=0,
        channel_first=False,
    ):
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.max_num_trials = max_num_trials
        self.has_action = has_action
        self.no_trunc = no_trunc
        self.pad_value = pad_value
        self.channel_first = channel_first

    def trunc_features(self, feats, gt_segments, gt_labels, offset):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = self.trunc_len
        if feat_len <= self.trunc_len:
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
        for _ in range(self.max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = torch.as_tensor([st, ed], dtype=torch.float32)

            # compute the intersection between the sampled window and all segments
            window = window[None].repeat(num_segs, 1)
            left = torch.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = torch.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = (right - left).clamp(min=0)
            area_segs = torch.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            if self.no_trunc:
                # with at least one action and not truncating any actions
                seg_trunc_idx = (inter_ratio > 0.0) & (inter_ratio < 1.0)
                if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                    break
            elif self.has_action:
                # with at least one action
                if seg_idx.sum().item() > 0:
                    break
            else:
                # without any constraints
                break

        feats = feats[st:ed, :]  # [T,C]
        gt_segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def pad_features(self, feats):
        feat_len = feats.shape[0]
        if feat_len < self.trunc_len:
            feats_pad = torch.ones((self.trunc_len - feat_len,) + feats.shape[1:]) * self.pad_value
            feats = torch.cat([feats, feats_pad], dim=0)
            masks = torch.cat([torch.ones(feat_len), torch.zeros(self.trunc_len - feat_len)])
            return feats, masks
        else:
            return feats, torch.ones(feat_len)

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        offset = 0

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [C,T] -> [T,C]

        # truncate the features
        feats, gt_segments, gt_labels = self.trunc_features(
            results["feats"],
            results["gt_segments"],
            results["gt_labels"],
            offset,
        )

        # pad the features to the fixed length
        feats, masks = self.pad_features(feats)

        results["feats"] = feats.float()
        results["masks"] = masks.bool()
        results["gt_segments"] = gt_segments
        results["gt_labels"] = gt_labels

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [T,C] -> [C,T]
        return results
