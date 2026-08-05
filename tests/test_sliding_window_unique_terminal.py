from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip(
        "local Windows torch/c10.dll import is unstable; Linux remote runs this suite",
        allow_module_level=True,
    )

from opentad.datasets.base.sliding_dataset import SlidingWindowDataset


def _legacy_starts(snippet_num: int, *, window_size: int, window_stride: int) -> list[int]:
    starts = []
    for index in range(max(1, snippet_num // window_stride)):
        start = index * window_stride
        end = start + window_size
        terminal = end > snippet_num
        if terminal:
            end = snippet_num
            start = max(0, end - window_size)
        starts.append(start)
        if terminal:
            break
    return starts


def _split_starts(
    snippet_num: int,
    *,
    window_size: int,
    window_stride: int,
    snippet_stride: int = 1,
) -> list[int]:
    dataset = object.__new__(SlidingWindowDataset)
    dataset.fps = -1
    dataset.snippet_stride = snippet_stride
    dataset.window_size = window_size
    dataset.window_stride = window_stride
    dataset.ioa_thresh = 0.0
    dataset.include_background_windows = True
    rows = dataset.split_video_to_windows(
        "video",
        {
            "frame": snippet_num * snippet_stride,
            "duration": float(snippet_num * snippet_stride),
        },
        {},
    )
    return [int(row[3][0]) for row in rows]


def test_stage_a_exact_terminal_alignment_is_emitted_once() -> None:
    assert _legacy_starts(
        2688,
        window_size=768,
        window_stride=384,
    ).count(1920) == 2
    starts = _split_starts(2688, window_size=768, window_stride=384)

    assert starts == [0, 384, 768, 1152, 1536, 1920]
    assert len(starts) == len(set(starts))
    frame_starts = _split_starts(
        2688,
        window_size=768,
        window_stride=384,
        snippet_stride=4,
    )
    assert frame_starts.count(7680) == 1


@pytest.mark.parametrize(
    ("snippet_num", "expected"),
    [
        (231, [0]),
        (768, [0]),
        (769, [0, 1]),
        (1000, [0, 232]),
        (1152, [0, 384]),
        (1153, [0, 384, 385]),
    ],
)
def test_canonical_window_starts_cover_terminal_without_duplicates(
    snippet_num: int,
    expected: list[int],
) -> None:
    starts = _split_starts(snippet_num, window_size=768, window_stride=384)

    assert starts == expected
    assert starts[0] == 0
    assert starts[-1] == max(0, snippet_num - 768)
    assert all(right > left for left, right in zip(starts, starts[1:]))


def test_canonical_window_start_properties_hold_through_four_windows() -> None:
    window_size, window_stride = 48, 24
    for snippet_num in range(1, 4 * window_size + 1):
        starts = _split_starts(
            snippet_num,
            window_size=window_size,
            window_stride=window_stride,
        )
        assert starts[0] == 0
        assert starts[-1] == max(0, snippet_num - window_size)
        assert all(right > left for left, right in zip(starts, starts[1:]))
        covered = set()
        for start in starts:
            covered.update(range(start, min(start + window_size, snippet_num)))
        assert covered == set(range(snippet_num))
