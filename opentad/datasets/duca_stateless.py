from __future__ import annotations

import hashlib
import random
from copy import deepcopy

import numpy as np
import torch

from .builder import DATASETS
from .thumos import ThumosPaddingDataset


@DATASETS.register_module()
class DucaStatelessThumosPaddingDataset(ThumosPaddingDataset):
    """THUMOS training videos with per-(seed, epoch, sample) augmentation RNG."""

    def __init__(self, *args, stateless_seed: int = 3407, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateless_seed = int(stateless_seed)
        self._duca_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("DUCA stateless dataset epoch must be non-negative")
        self._duca_epoch = epoch

    def sample_seed(self, index: int) -> int:
        index = int(index)
        if index < 0 or index >= len(self.data_list):
            raise IndexError(index)
        video_name = str(self.data_list[index][0])
        payload = (
            f"{self.stateless_seed}|{self._duca_epoch}|{video_name}|{index}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
            2**32
        )

    def __getitem__(self, index):
        seed = self.sample_seed(index)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            video_name, video_info, video_anno = self.data_list[index]
            if video_anno != {}:
                video_anno = deepcopy(video_anno)
                video_anno["gt_segments"] = (
                    video_anno["gt_segments"] - self.offset_frames
                )
                video_anno["gt_segments"] = (
                    video_anno["gt_segments"] / self.snippet_stride
                )
            item = self.pipeline(
                dict(
                    video_name=video_name,
                    data_path=self.data_path,
                    sample_stride=self.sample_stride,
                    snippet_stride=self.snippet_stride,
                    fps=video_info["frame"] / video_info["duration"],
                    duration=video_info["duration"],
                    offset_frames=self.offset_frames,
                    duca_stateless_seed=int(seed),
                    duca_stateless_epoch=int(self._duca_epoch),
                    duca_stateless_sample_index=int(index),
                    **video_anno,
                )
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        metas = item.get("metas")
        if isinstance(metas, dict):
            metas["duca_stateless_seed"] = int(seed)
            metas["duca_stateless_epoch"] = int(self._duca_epoch)
            metas["duca_stateless_sample_index"] = int(index)
        return item


__all__ = ["DucaStatelessThumosPaddingDataset"]
