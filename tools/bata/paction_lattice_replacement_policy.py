from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


MOVE25_STRATEGY = "paction_lattice_replace_score_only_move25"
MOVE50_STRATEGY = "paction_lattice_replace_score_only_move50"
MOVE75_STRATEGY = "paction_lattice_replace_score_only_move75"
NO_PROTECT_STRATEGY = "paction_lattice_replace_score_only_no_protect"
DEFAULT_BUDGET = 384
_PROTECTED_COUNTS = {
    MOVE25_STRATEGY: 288,
    MOVE50_STRATEGY: 192,
    MOVE75_STRATEGY: 96,
    NO_PROTECT_STRATEGY: 0,
}


@dataclass(frozen=True)
class LatticeReplacementResult:
    selected_positions: list[int]
    diagnostics: dict[str, Any]


def _coerce_scores(
    *,
    frame_values: Sequence[Any] | None,
    score: Sequence[Any] | None,
    paction_score: Sequence[Any] | None,
) -> list[float]:
    provided = [item for item in (frame_values, score, paction_score) if item is not None]
    if len(provided) != 1:
        raise ValueError("provide exactly one of frame_values, score, or paction_score")
    values = [float(item) for item in provided[0]]
    if not values:
        raise ValueError("score vector must not be empty")
    if any(not math.isfinite(item) for item in values):
        raise ValueError("score vector must contain only finite values")
    return values


def _valid_positions(
    *,
    length: int,
    valid: Sequence[Any] | None,
    valid_positions: Sequence[Any] | None,
) -> list[int]:
    if valid is not None and valid_positions is not None:
        raise ValueError("provide either valid or valid_positions, not both")
    if valid_positions is not None:
        positions = sorted({int(item) for item in valid_positions})
        if any(item < 0 or item >= int(length) for item in positions):
            raise ValueError("valid_positions contains out-of-range positions")
        return positions
    if valid is None:
        return list(range(int(length)))
    if len(valid) != int(length):
        raise ValueError("valid mask length must match score vector length")
    return [idx for idx, is_valid in enumerate(valid) if bool(is_valid)]


def _uniform_lattice(positions: Sequence[int], budget: int) -> list[int]:
    valid = list(positions)
    target = min(max(0, int(budget)), len(valid))
    if target <= 0:
        return []
    if len(valid) <= target:
        return list(valid)
    if target == 1:
        return [valid[0]]
    ranks = [round(idx * (len(valid) - 1) / float(target - 1)) for idx in range(target)]
    selected: list[int] = []
    used: set[int] = set()
    for rank in ranks:
        rank = int(max(0, min(len(valid) - 1, rank)))
        if rank in used:
            left = rank - 1
            right = rank + 1
            while left >= 0 or right < len(valid):
                if right < len(valid) and right not in used:
                    rank = right
                    break
                if left >= 0 and left not in used:
                    rank = left
                    break
                left -= 1
                right += 1
        used.add(rank)
        selected.append(valid[rank])
    return sorted(selected)


def _gap_stats(selected: Sequence[int]) -> dict[str, float]:
    if len(selected) <= 1:
        return {"max": 0.0, "p95": 0.0, "cv": 0.0}
    gaps = [float(int(selected[idx]) - int(selected[idx - 1])) for idx in range(1, len(selected))]
    mean = sum(gaps) / float(len(gaps))
    variance = sum((gap - mean) ** 2 for gap in gaps) / float(len(gaps)) if gaps else 0.0
    sorted_gaps = sorted(gaps)
    p95_rank = int(math.ceil(0.95 * len(sorted_gaps))) - 1
    return {
        "max": float(max(gaps)),
        "p95": float(sorted_gaps[max(0, min(len(sorted_gaps) - 1, p95_rank))]),
        "cv": 0.0 if mean == 0.0 else float(math.sqrt(variance) / mean),
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / float(len(union))


def _phase_shift_uniform_similarity(selected: set[int], valid_positions: Sequence[int], budget: int) -> float:
    if len(valid_positions) <= 1 or not selected:
        return 1.0
    shifted = _uniform_lattice(valid_positions[1:] + valid_positions[:1], budget)
    return _jaccard(selected, set(shifted))


def _protected_count_for_budget(variant: str, budget: int) -> int:
    base = int(_PROTECTED_COUNTS[str(variant)])
    if base <= 0:
        return 0
    return int(round(float(budget) * float(base) / float(DEFAULT_BUDGET)))


def _best_local_victim(
    candidate: int,
    replaceable: set[int],
    current: set[int],
    scores: Sequence[float],
    *,
    local_radius: int,
    distance_penalty: float,
    geometry_distortion_penalty: float,
) -> tuple[int, float, int] | None:
    victims = [item for item in replaceable if abs(int(candidate) - int(item)) <= int(local_radius)]
    if not victims:
        return None
    normalizer = float(max(1, int(local_radius)))
    options: list[tuple[float, float, int, int]] = []
    for victim in victims:
        if int(victim) not in current:
            continue
        normalized_distance = abs(int(candidate) - int(victim)) / normalizer
        gain = (
            float(scores[int(candidate)])
            - float(scores[int(victim)])
            - float(distance_penalty) * normalized_distance
            - float(geometry_distortion_penalty) * normalized_distance * normalized_distance
        )
        options.append((gain, -normalized_distance, -int(victim), int(victim)))
    if not options:
        return None
    gain, _neg_distance, _neg_victim, victim = max(options)
    if gain <= 0.0:
        return None
    return int(victim), float(gain), abs(int(candidate) - int(victim))


def _repair_large_gaps(
    *,
    current: set[int],
    replacements: list[dict[str, Any]],
    max_allowed_gap: float,
) -> int:
    repair_count = 0
    while replacements:
        gap_after = _gap_stats(sorted(current))["max"]
        if gap_after <= float(max_allowed_gap):
            break
        victim = min(replacements, key=lambda item: (float(item["gain"]), -int(item["candidate"])))
        replacements.remove(victim)
        current.discard(int(victim["candidate"]))
        current.add(int(victim["victim"]))
        repair_count += 1
    return repair_count


def decode_paction_lattice_replacement(
    *,
    frame_values: Sequence[Any] | None = None,
    score: Sequence[Any] | None = None,
    paction_score: Sequence[Any] | None = None,
    valid: Sequence[Any] | None = None,
    valid_positions: Sequence[Any] | None = None,
    variant: str = MOVE50_STRATEGY,
    budget: int = DEFAULT_BUDGET,
    local_radius: int = 2,
    distance_penalty: float = 0.0,
    geometry_distortion_penalty: float = 0.0,
    max_gap_growth: int | None = None,
) -> LatticeReplacementResult:
    scores = _coerce_scores(frame_values=frame_values, score=score, paction_score=paction_score)
    if str(variant) not in _PROTECTED_COUNTS:
        raise ValueError(f"unknown lattice replacement variant: {variant}")
    if int(budget) <= 0:
        raise ValueError("budget must be positive")
    if int(local_radius) < 0:
        raise ValueError("local_radius must be non-negative")
    if float(distance_penalty) < 0.0 or float(geometry_distortion_penalty) < 0.0:
        raise ValueError("distance penalties must be non-negative")

    valid_pos = _valid_positions(length=len(scores), valid=valid, valid_positions=valid_positions)
    if not valid_pos:
        raise ValueError("at least one valid position is required")
    valid_set = set(valid_pos)
    base_uniform = _uniform_lattice(valid_pos, int(budget))
    protected_target = min(_protected_count_for_budget(str(variant), len(base_uniform)), len(base_uniform))
    protected_uniform = set(_uniform_lattice(base_uniform, protected_target))
    replaceable_uniform = set(base_uniform) - protected_uniform
    current = set(base_uniform)
    before_gap = _gap_stats(base_uniform)
    allowed_growth = int(max_gap_growth) if max_gap_growth is not None else max(1, 2 * int(local_radius))
    max_allowed_gap = before_gap["max"] + float(allowed_growth)

    replacements: list[dict[str, Any]] = []
    candidates = [
        idx
        for idx in sorted(valid_set - current, key=lambda item: (scores[int(item)], -int(item)), reverse=True)
    ]
    for candidate in candidates:
        local = _best_local_victim(
            int(candidate),
            replaceable_uniform,
            current,
            scores,
            local_radius=int(local_radius),
            distance_penalty=float(distance_penalty),
            geometry_distortion_penalty=float(geometry_distortion_penalty),
        )
        if local is None:
            continue
        victim, gain, distance = local
        current.remove(int(victim))
        current.add(int(candidate))
        replaceable_uniform.remove(int(victim))
        replacements.append(
            {
                "candidate": int(candidate),
                "victim": int(victim),
                "gain": float(gain),
                "distance": int(distance),
            }
        )

    safety_repair_count = _repair_large_gaps(
        current=current,
        replacements=replacements,
        max_allowed_gap=max_allowed_gap,
    )
    selected = sorted(current)
    if len(selected) > int(budget):
        raise RuntimeError("lattice replacement produced more than budget positions")
    if len(selected) != len(set(selected)):
        raise RuntimeError("lattice replacement produced duplicate positions")
    if any(item not in valid_set for item in selected):
        raise RuntimeError("lattice replacement produced invalid positions")

    after_gap = _gap_stats(selected)
    topk = set(sorted(valid_pos, key=lambda item: (scores[int(item)], -int(item)), reverse=True)[: len(selected)])
    distances = [float(item["distance"]) for item in replacements]
    distances_sorted = sorted(distances)
    p95_distance = 0.0
    if distances_sorted:
        p95_rank = int(math.ceil(0.95 * len(distances_sorted))) - 1
        p95_distance = distances_sorted[max(0, min(len(distances_sorted) - 1, p95_rank))]

    diagnostics = {
        "strategy_name": str(variant),
        "budget": int(budget),
        "protected_uniform_count": int(len(protected_uniform)),
        "replaceable_uniform_count": int(len(set(base_uniform) - protected_uniform)),
        "inserted_candidate_count": int(len(replacements)),
        "replaced_uniform_count": int(len(replacements)),
        "safety_repair_count": int(safety_repair_count),
        "selected_count": int(len(selected)),
        "max_gap_before": before_gap["max"],
        "p95_gap_before": before_gap["p95"],
        "gap_cv_before": before_gap["cv"],
        "max_gap_after": after_gap["max"],
        "p95_gap_after": after_gap["p95"],
        "gap_cv_after": after_gap["cv"],
        "base_uniform_jaccard": _jaccard(set(selected), set(base_uniform)),
        "phase_shift_uniform_similarity": _phase_shift_uniform_similarity(set(selected), valid_pos, len(base_uniform)),
        "paction_topk_overlap": len(set(selected) & topk) / float(max(1, len(selected))),
        "replacement_distance_mean": sum(distances) / float(len(distances)) if distances else 0.0,
        "replacement_distance_p95": float(p95_distance),
        "replacement_distance_max": max(distances) if distances else 0.0,
    }
    return LatticeReplacementResult(selected_positions=selected, diagnostics=diagnostics)
