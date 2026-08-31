from types import SimpleNamespace

import numpy as np
import pytest

from tools.bata.profile_zoomtoken_ordered_video_reuse_r1_k100_cost import (
    EXPECTED_VIDEO_COUNT,
    EXPECTED_WINDOW_COUNT,
    FROZEN_EVALUATOR_VECTOR,
    PROFILE_ORDER,
    OrderedVideoDecordInit,
    _assert_metric_parity,
    patch_frame_index_parity_tap,
    patch_ordered_video_pipeline,
)


class FakeBatch:
    def __init__(self, array):
        self.array = array

    def asnumpy(self):
        return self.array


class FakeReader:
    def __init__(self, filename, num_threads):
        self.filename = filename
        self.num_threads = num_threads
        self.requests = []

    def __len__(self):
        return 32

    def get_avg_fps(self):
        return 25.0

    def get_batch(self, indices):
        checked = [int(index) for index in indices]
        self.requests.append(checked)
        frames = np.stack(
            [np.full((2, 3, 3), index, dtype=np.uint8) for index in checked],
            axis=0,
        )
        return FakeBatch(frames)


def _window(ordinal, start, end, video="v1"):
    return {
        "video_name": video,
        "filename": f"/video/{video}.mp4",
        "window_ordinal": ordinal,
        "feature_start_idx": start,
        "feature_end_idx": end,
    }


def test_frozen_population_and_pass_order():
    assert EXPECTED_VIDEO_COUNT == 211
    assert EXPECTED_WINDOW_COUNT == 792
    assert PROFILE_ORDER == ("K100", "R1", "R1", "K100", "R1", "K100", "K100", "R1")
    assert PROFILE_ORDER.count("K100") == PROFILE_ORDER.count("R1") == 4


def test_overlap_frames_are_requested_from_source_once_and_buffer_is_bounded():
    readers = []

    def factory(filename, num_threads):
        reader = FakeReader(filename, num_threads)
        readers.append(reader)
        return reader

    transform = OrderedVideoDecordInit(reader_factory=factory)
    first = transform(_window(0, 0, 3))
    first["video_reader"].get_batch([0, 1, 2, 3, 3])
    second = transform(_window(1, 2, 5))
    decoded = second["video_reader"].get_batch([2, 3, 4, 5])

    assert readers[0].requests == [[0, 1, 2, 3], [4, 5]]
    assert decoded.asnumpy()[:, 0, 0, 0].tolist() == [2, 3, 4, 5]
    assert transform.stats().source_frame_request_count == 6
    assert transform.stats().buffered_frames == 4
    assert transform.stats().maximum_buffered_frames <= 4


def test_evicted_source_frame_cannot_be_requested_again():
    transform = OrderedVideoDecordInit(reader_factory=FakeReader)
    transform(_window(0, 0, 3))["video_reader"].get_batch([0, 1, 2, 3])
    transform(_window(1, 2, 5))["video_reader"].get_batch([2, 3, 4, 5])
    with pytest.raises(RuntimeError, match="requested twice"):
        transform(_window(2, 3, 6))["video_reader"].get_batch([1, 3, 4, 5])


def test_video_and_explicit_reset_do_not_carry_decode_state():
    readers = []

    def factory(filename, num_threads):
        reader = FakeReader(filename, num_threads)
        readers.append(reader)
        return reader

    transform = OrderedVideoDecordInit(reader_factory=factory)
    transform(_window(0, 0, 1))["video_reader"].get_batch([0, 1])
    transform(_window(1, 0, 1, video="v2"))["video_reader"].get_batch([0, 1])
    transform.reset()
    transform(_window(0, 0, 1, video="v2"))["video_reader"].get_batch([0, 1])

    assert len(readers) == 3
    assert [reader.requests for reader in readers] == [[[0, 1]], [[0, 1]], [[0, 1]]]


def test_discontinuous_population_fails_closed_after_reset():
    transform = OrderedVideoDecordInit(reader_factory=FakeReader)
    transform(_window(0, 0, 3))
    with pytest.raises(ValueError, match="discontinuous"):
        transform(_window(2, 8, 11))
    assert transform.stats().video_name is None
    assert transform.stats().buffered_frames == 0


def test_pipeline_patch_replaces_only_decord_init():
    DecordInit = type("DecordInit", (), {"num_threads": 4})
    LoadFrames = type("LoadFrames", (), {})
    DecordDecode = type("DecordDecode", (), {})
    Resize = type("Resize", (), {})
    original_decode = DecordDecode()
    original_resize = Resize()
    dataset = SimpleNamespace(
        pipeline=SimpleNamespace(
            transforms=[DecordInit(), LoadFrames(), original_decode, original_resize]
        )
    )

    replacement = patch_ordered_video_pipeline(dataset, reader_factory=FakeReader)

    assert dataset.pipeline.transforms == [
        replacement,
        dataset.pipeline.transforms[1],
        original_decode,
        original_resize,
    ]
    assert isinstance(replacement, OrderedVideoDecordInit)


def test_frame_index_tap_binds_dtype_shape_and_values():
    LoadFrames = type("LoadFrames", (), {})
    DecordDecode = type("DecordDecode", (), {})
    Collect = type("Collect", (), {})
    collect = Collect()
    collect.meta_keys = []
    dataset = SimpleNamespace(
        pipeline=SimpleNamespace(transforms=[LoadFrames(), DecordDecode(), collect])
    )

    patch_frame_index_parity_tap(dataset)
    tap = dataset.pipeline.transforms[1]
    left = tap({"frame_inds": np.asarray([1, 2, 3], dtype=np.int64)})
    right = tap({"frame_inds": np.asarray([1, 2, 4], dtype=np.int64)})

    assert left["frame_indices_dtype"] == "int64"
    assert left["frame_indices_shape"] == [3]
    assert left["frame_indices_sha256"] != right["frame_indices_sha256"]
    assert set(collect.meta_keys) == {
        "frame_indices_dtype",
        "frame_indices_shape",
        "frame_indices_sha256",
    }


def test_evaluator_vector_requires_exact_raw_identity():
    exact = dict(FROZEN_EVALUATOR_VECTOR["K100"])
    assert _assert_metric_parity("K100", exact)["status"] == "EXACT_RAW_EVALUATOR_VECTOR_MATCH"
    changed = dict(exact)
    changed["mAP@0.7"] += 1e-15
    with pytest.raises(RuntimeError, match="raw vector differs"):
        _assert_metric_parity("K100", changed)
