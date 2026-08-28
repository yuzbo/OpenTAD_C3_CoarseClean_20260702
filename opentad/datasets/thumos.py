import hashlib
import random

import numpy as np
import torch
from copy import deepcopy
from .base import (
    PaddingDataset,
    SlidingWindowDataset,
    compute_gt_completeness,
    filter_same_annotation,
)
from .builder import DATASETS


@DATASETS.register_module()
class ThumosSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                # trunc window setting
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                # sliding post process setting
                fps=video_info["frame"] / video_info["duration"],
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # training setting
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class DucaVideoGroupedThumosSlidingDataset(ThumosSlidingDataset):
    """Canonical THUMOS sliding windows grouped by source video for training.

    Every dataset item contains all windows from one video.  This makes a
    two-item dataloader batch exactly a two-video logical optimization unit,
    while the collate function can still flatten the windows for the detector.
    """

    def __init__(self, *args, stateless_seed=3407, group_by_video=True, **kwargs):
        self.stateless_seed = int(stateless_seed)
        self.group_by_video = bool(group_by_video)
        self._duca_epoch = 0
        super().__init__(*args, **kwargs)
        self._window_data_list = list(self.data_list)
        if self.group_by_video:
            grouped = []
            current_name = None
            current = []
            for window_index, record in enumerate(self._window_data_list):
                video_name = str(record[0])
                if current_name is not None and video_name != current_name:
                    grouped.append(current)
                    current = []
                current_name = video_name
                current.append(window_index)
            if current:
                grouped.append(current)
            self.data_list = grouped

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("DUCA dataset epoch must be non-negative")
        self._duca_epoch = epoch

    def split_video_to_windows(self, video_name, video_info, video_anno):
        num_frames = (
            int(video_info["duration"] * self.fps)
            if self.fps > 0
            else int(video_info["frame"])
        )
        centers = np.arange(0, num_frames, self.snippet_stride)
        snippet_count = int(len(centers))
        if snippet_count <= 0:
            return []
        if snippet_count <= self.window_size:
            starts = [0]
        else:
            final_start = snippet_count - self.window_size
            starts = list(range(0, final_start + 1, self.window_stride))
            if starts[-1] != final_start:
                starts.append(final_start)

        records = []
        window_count = len(starts)
        for window_index, start in enumerate(starts):
            end = min(start + self.window_size, snippet_count)
            window_centers = centers[start:end]
            if video_anno == {}:
                window_anno = {}
            else:
                gt_segments = video_anno["gt_segments"]
                gt_labels = video_anno["gt_labels"]
                anchor = np.asarray([window_centers[0], window_centers[-1]])
                completeness, clipped = compute_gt_completeness(gt_segments, anchor)
                valid = completeness > self.ioa_thresh
                endpoint_validity = np.stack(
                    (
                        np.isclose(clipped[:, 0], gt_segments[:, 0], rtol=0.0, atol=1.0e-6),
                        np.isclose(clipped[:, 1], gt_segments[:, 1], rtol=0.0, atol=1.0e-6),
                    ),
                    axis=1,
                ).astype(np.bool_)
                window_anno = dict(
                    gt_segments=clipped[valid].astype(np.float32, copy=False),
                    gt_labels=gt_labels[valid],
                    gt_boundary_validity=endpoint_validity[valid],
                )
            records.append(
                [
                    video_name,
                    video_info,
                    window_anno,
                    window_centers,
                    window_index,
                    window_count,
                ]
            )
        return records

    def _window_seed(self, record):
        payload = (
            f"{self.stateless_seed}|{self._duca_epoch}|{record[0]}|"
            f"{int(record[3][0])}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)

    def _get_window(self, window_index):
        record = self._window_data_list[int(window_index)]
        seed = self._window_seed(record)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            video_name, video_info, video_anno, centers, index, count = record
            if video_anno != {}:
                video_anno = deepcopy(video_anno)
                video_anno["gt_segments"] = (
                    video_anno["gt_segments"] - centers[0] - self.offset_frames
                ) / self.snippet_stride
            results = self.pipeline(
                dict(
                    video_name=video_name,
                    data_path=self.data_path,
                    window_size=self.window_size,
                    feature_start_idx=int(centers[0] / self.snippet_stride),
                    feature_end_idx=int(centers[-1] / self.snippet_stride),
                    sample_stride=self.sample_stride,
                    fps=video_info["frame"] / video_info["duration"],
                    snippet_stride=self.snippet_stride,
                    window_start_frame=centers[0],
                    window_end_frame=centers[-1],
                    duration=video_info["duration"],
                    offset_frames=self.offset_frames,
                    duca_split=str(self.subset_name),
                    duca_window_index=int(index),
                    duca_window_count=int(count),
                    duca_stateless_seed=int(seed),
                    duca_stateless_epoch=int(self._duca_epoch),
                    **video_anno,
                )
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        return results

    def __getitem__(self, index):
        if self.group_by_video:
            return [self._get_window(window_index) for window_index in self.data_list[index]]
        return self._get_window(index)


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=video_info["frame"] / video_info["duration"],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results
