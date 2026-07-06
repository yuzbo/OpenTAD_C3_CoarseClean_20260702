from __future__ import annotations

import copy
from typing import Any, Mapping

from tools.bata import paction_source_samples


DETECTOR_DEPLOY_FORBIDDEN_PAYLOAD_KEYS = tuple(
    dict.fromkeys(
        paction_source_samples.STRICT_DEPLOY_PAYLOAD_KEYS
        + (
            "teacher_utility",
            "teacher_utility_provenance",
            "frame_utility",
            "signed_frame_utility",
            "marginal_gain_frame_utility",
            "teacher_dense_points",
            "dense_teacher_points",
            "teacher_proposals",
            "actionformer_proposals",
            "actionformer_scores",
            "dense_teacher_logits",
        )
    )
)

DETECTOR_DEPLOY_FORBIDDEN_TRUE_FLAGS = tuple(
    dict.fromkeys(
        paction_source_samples.STRICT_DEPLOY_FORBIDDEN_TRUE_FLAGS
        + (
            "load_from_raw_predictions",
            "allow_raw_prediction_cache",
            "uses_raw_prediction_cache",
        )
    )
)

_FORBIDDEN_PAYLOAD_KEY_SET = {key.lower() for key in DETECTOR_DEPLOY_FORBIDDEN_PAYLOAD_KEYS}
_FORBIDDEN_TRUE_FLAG_SET = {key.lower() for key in DETECTOR_DEPLOY_FORBIDDEN_TRUE_FLAGS}
_STRIPPED_VALUE = object()
DETECTOR_DEPLOY_FORBIDDEN_KEY_FRAGMENTS = (
    "teacher_utility",
    "dense_teacher",
    "teacher_dense",
    "teacher_proposal",
    "teacher_prediction",
    "teacher_logits",
    "teacher_scores",
    "teacher_saliency",
    "frame_utility",
    "signed_frame_utility",
    "marginal_gain_frame_utility",
    "oracle",
    "ground_truth",
    "prediction_cache",
    "raw_prediction",
    "raw_predictions",
    "proposal_cache",
)
DETECTOR_DEPLOY_FORBIDDEN_VALUE_FRAGMENTS = DETECTOR_DEPLOY_FORBIDDEN_KEY_FRAGMENTS


def _has_forbidden_fragment(text: str, fragments: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(fragment in normalized for fragment in fragments)


def _key_is_forbidden_payload(normalized_key: str) -> bool:
    if normalized_key in _FORBIDDEN_PAYLOAD_KEY_SET:
        return True
    if normalized_key in _FORBIDDEN_TRUE_FLAG_SET:
        return False
    return _has_forbidden_fragment(normalized_key, DETECTOR_DEPLOY_FORBIDDEN_KEY_FRAGMENTS)


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _path_child(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _path_index(parent: str, index: int) -> str:
    return f"{parent}[{index}]" if parent else f"[{index}]"


def find_detector_deploy_forbidden_paths(value: Any, *, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            item_path = _path_child(path, key_text)
            normalized = key_text.lower()
            if _key_is_forbidden_payload(normalized):
                paths.append(item_path)
                continue
            if normalized in _FORBIDDEN_TRUE_FLAG_SET and _is_true(item):
                paths.append(item_path)
                continue
            paths.extend(find_detector_deploy_forbidden_paths(item, path=item_path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            paths.extend(find_detector_deploy_forbidden_paths(item, path=_path_index(path, idx)))
    elif isinstance(value, tuple):
        for idx, item in enumerate(value):
            paths.extend(find_detector_deploy_forbidden_paths(item, path=_path_index(path, idx)))
    elif isinstance(value, str):
        if _has_forbidden_fragment(value, DETECTOR_DEPLOY_FORBIDDEN_VALUE_FRAGMENTS):
            paths.append(path or "<value>")
    return paths


def reject_detector_deploy_forbidden_payloads(value: Any, *, source_name: str) -> None:
    paths = find_detector_deploy_forbidden_paths(value)
    if paths:
        first = paths[0]
        raise ValueError(
            f"{source_name}: forbidden detector-aware deploy payload key at {first}"
        )


def strip_detector_deploy_forbidden_payloads(row: Mapping[str, Any]) -> dict[str, Any]:
    def strip_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                normalized = key_text.lower()
                if _key_is_forbidden_payload(normalized):
                    continue
                if normalized in _FORBIDDEN_TRUE_FLAG_SET and _is_true(item):
                    continue
                stripped_item = strip_value(item)
                if stripped_item is _STRIPPED_VALUE:
                    continue
                out[key] = stripped_item
            return out
        if isinstance(value, list):
            return [
                item
                for item in (strip_value(item) for item in value)
                if item is not _STRIPPED_VALUE
            ]
        if isinstance(value, tuple):
            return tuple(
                item
                for item in (strip_value(item) for item in value)
                if item is not _STRIPPED_VALUE
            )
        if isinstance(value, str) and _has_forbidden_fragment(value, DETECTOR_DEPLOY_FORBIDDEN_VALUE_FRAGMENTS):
            return _STRIPPED_VALUE
        return copy.deepcopy(value)

    return strip_value(row)
