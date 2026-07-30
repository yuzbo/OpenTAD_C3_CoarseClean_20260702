import os
import json
import hashlib
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("thumos")
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        video_id_manifest=None,
        video_id_manifest_sha256=None,
        video_id_manifest_subset=None,
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.video_id_filter = None
        self.video_id_manifest_info = None
        manifest_fields = (
            video_id_manifest,
            video_id_manifest_sha256,
            video_id_manifest_subset,
        )
        if any(value is not None for value in manifest_fields):
            if not all(value is not None for value in manifest_fields):
                raise ValueError(
                    "THUMOS internal holdout requires manifest path, SHA-256, "
                    "and subset together"
                )
            manifest_path = os.path.expandvars(video_id_manifest)
            expected_sha256 = os.path.expandvars(
                video_id_manifest_sha256
            )
            manifest_subset = os.path.expandvars(
                video_id_manifest_subset
            )
            if manifest_subset not in ("train", "holdout"):
                raise ValueError(
                    "THUMOS internal holdout subset must be train or holdout"
                )
            with open(manifest_path, "rb") as fid:
                manifest_bytes = fid.read()
            actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "THUMOS internal holdout manifest SHA-256 mismatch"
                )
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            if (
                manifest.get("schema_version")
                != "actionformer_dcsr_internal_holdout_v1"
            ):
                raise ValueError(
                    "unsupported THUMOS internal holdout manifest schema"
                )
            with open(self.json_file, "rb") as fid:
                annotation_sha256 = hashlib.sha256(fid.read()).hexdigest()
            if (
                manifest.get("source_annotation_sha256")
                != annotation_sha256
            ):
                raise ValueError(
                    "THUMOS internal holdout annotation identity mismatch"
                )
            if set(self.split) != {"validation"}:
                raise ValueError(
                    "THUMOS internal holdout may only filter validation"
                )
            manifest_key = manifest_subset + "_video_ids"
            video_ids = manifest.get(manifest_key)
            if not isinstance(video_ids, list) or not video_ids:
                raise ValueError(
                    "THUMOS internal holdout manifest has no requested IDs"
                )
            if len(video_ids) != len(set(video_ids)):
                raise ValueError(
                    "THUMOS internal holdout manifest contains duplicate IDs"
                )
            self.video_id_filter = frozenset(video_ids)
            self.video_id_manifest_info = {
                "path": manifest_path,
                "sha256": actual_sha256,
                "subset": manifest_subset,
            }

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict
        if self.video_id_filter is not None:
            loaded_ids = {item["id"] for item in self.data_list}
            missing_ids = self.video_id_filter - loaded_ids
            if missing_ids:
                raise ValueError(
                    "THUMOS internal holdout IDs missing features: {:s}".format(
                        ", ".join(sorted(missing_ids))
                    )
                )
            self.evaluation_video_ids = tuple(sorted(loaded_ids))
        else:
            self.evaluation_video_ids = None

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                if (
                    self.video_id_filter is not None
                    and value['subset'].lower() not in self.split
                ):
                    continue
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            if (
                self.video_id_filter is not None
                and key not in self.video_id_filter
            ):
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict
