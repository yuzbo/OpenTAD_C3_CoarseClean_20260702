import numpy as np
import torch

from geosparse_ext.data import CaptureGeoSparseSupport
from geosparse_ext.matrix import base
from geosparse_ext.protocol import resolve_opentad_config


def test_official_recipe_keeps_full_grid_augmentation_and_scheduler():
    cfg = resolve_opentad_config(dict(dataset="thumos14", model=base("A")),
        dict(checkpoints=dict(videomae_b="recognition.pth"), class_map="classes.txt",
             video_roots=dict(training="all train", validation="all test")), "original_annotations.json")
    assert cfg.scheduler.max_epoch == 100 and cfg.scheduler.warmup_epoch == 5
    assert cfg.workflow.end_epoch == 60 and cfg.model.projection.max_seq_len == 768
    assert cfg.solver.train.batch_size == 2
    assert any(p.type == "GeoSparseImgAug" and p.transforms == "default" for p in cfg.dataset.train.pipeline)
    assert any(p.type == "mmaction.Resize" and p.scale == (160, 160) for p in cfg.dataset.train.pipeline)
    assert cfg.dataset.train.ann_file == cfg.dataset.test.ann_file == "original_annotations.json"
    assert cfg.dataset.test.subset_name == "validation" and cfg.dataset.test.window_overlap_ratio == .5


def test_pts_capture_never_replaces_official_sampled_indices():
    class Reader:
        def get_frame_timestamp(self, ids):
            # Deliberately irregular PTS, which previously triggered re-sampling.
            return np.stack((np.asarray(ids) * .07, np.asarray(ids) * .07 + .03), -1)
    ids = np.array([0, 4, 8, 12])
    masks = torch.tensor([True, True, True, False])
    sample = dict(frame_inds=ids.copy(), video_reader=Reader(), fps=25., snippet_stride=4,
                  masks=masks.clone(), filename="real input.mp4")
    output = CaptureGeoSparseSupport()(sample)
    assert np.array_equal(output["frame_inds"], ids)
    assert torch.equal(output["masks"], masks)
    assert output["geosparse_support"]["remapped_source_frames"] == 0
