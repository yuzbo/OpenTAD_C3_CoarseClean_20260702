from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DenseObservations = list[list[list[float]]]
Mask = list[list[bool]]


@dataclass(frozen=True)
class AcquisitionResult:
    observations: DenseObservations
    mask: Mask
    selected_positions: list[list[int]]

    @property
    def selected_count(self) -> list[int]:
        return [sum(1 for flag in row if flag) for row in self.mask]


class ZeroShotActionnessSource:
    """Deploy-visible actionness from dense observations only."""

    def __call__(self, dense_observations: DenseObservations, valid_mask: Mask) -> list[list[float]]:
        if len(dense_observations) != len(valid_mask):
            raise ValueError("dense_observations and valid_mask batch sizes differ")
        all_scores: list[list[float]] = []
        for batch_idx, (sample, mask_row) in enumerate(zip(dense_observations, valid_mask)):
            if len(sample) != len(mask_row):
                raise ValueError(f"sample {batch_idx}: valid_mask length mismatch")
            raw_scores = [_rms(vector) for vector in sample]
            valid_scores = [score for score, valid in zip(raw_scores, mask_row) if valid]
            if not valid_scores:
                raise ValueError(f"sample {batch_idx}: at least one valid observation is required")
            center = sum(valid_scores) / len(valid_scores)
            scores = [_sigmoid(score - center) if valid else float("-inf") for score, valid in zip(raw_scores, mask_row)]
            all_scores.append(scores)
        return all_scores


class DucaAcquisitionAdapter:
    """Online DUCA plugin adapter: choose detector-consumed original-time positions."""

    def __init__(self, actionness_source: ZeroShotActionnessSource, budget: int = 384) -> None:
        self.actionness_source = actionness_source
        self.budget = int(budget)
        if self.budget <= 0:
            raise ValueError("budget must be positive")

    def __call__(self, dense_observations: DenseObservations, valid_mask: Mask) -> AcquisitionResult:
        actionness = self.actionness_source(dense_observations, valid_mask)
        selected_positions: list[list[int]] = []
        selected_observations: DenseObservations = []
        selected_mask: Mask = []
        for batch_idx, (sample, mask_row, score_row) in enumerate(zip(dense_observations, valid_mask, actionness)):
            valid_count = sum(1 for item in mask_row if item)
            if valid_count < self.budget:
                raise ValueError(f"sample {batch_idx}: valid_count={valid_count} is below fixed budget={self.budget}")
            positions = sorted(
                index for index, _ in sorted(enumerate(score_row), key=lambda item: item[1], reverse=True)[: self.budget]
            )
            selected_positions.append(positions)
            selected_observations.append([list(sample[index]) for index in positions])
            selected_mask.append([True] * len(positions))
        return AcquisitionResult(
            observations=selected_observations,
            mask=selected_mask,
            selected_positions=selected_positions,
        )


class DummyDetector:
    """Tiny detector stand-in that records the sequence length it consumes."""

    def __init__(self) -> None:
        self.last_input_length = 0

    def forward_train(
        self,
        observations: DenseObservations,
        mask: Mask,
        metas: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, float]:
        self.last_input_length = _common_detector_input_length(observations)
        denom = max(1, sum(sum(1 for flag in row if flag) for row in mask))
        loss = sum(_rms(vector) for sample in observations for vector in sample) / denom
        return {"loss_dummy_detector": float(loss), "cost": float(loss)}

    def forward_test(
        self,
        observations: DenseObservations,
        mask: Mask,
        metas: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.last_input_length = _common_detector_input_length(observations)
        return {
            "detector_input_length": self.last_input_length,
            "valid_count": [sum(1 for flag in row if flag) for row in mask],
            "video_names": [str(meta.get("video_name", idx)) for idx, meta in enumerate(metas)],
        }


def _with_selected_metadata(metas: list[dict[str, Any]], acquisition: AcquisitionResult) -> list[dict[str, Any]]:
    out = [dict(meta) for meta in metas]
    for idx, meta in enumerate(out):
        positions = [int(item) for item in acquisition.selected_positions[idx]]
        meta["selected_positions"] = positions
        meta["selected_positions_coordinate_system"] = "original_time_detector_consumed_position"
        meta["selected_count"] = len(positions)
        meta["uses_ledger_for_decision"] = False
    return out


def duca_forward_train(
    adapter: DucaAcquisitionAdapter,
    detector: DummyDetector,
    dense_observations: DenseObservations,
    valid_mask: Mask,
    metas: list[dict[str, Any]],
    teacher_utility: list[list[float]] | None = None,
) -> dict[str, Any]:
    acquisition = adapter(dense_observations, valid_mask)
    selected_metas = _with_selected_metadata(metas, acquisition)
    losses = detector.forward_train(acquisition.observations, acquisition.mask, selected_metas)
    if teacher_utility is not None:
        selected_teacher = [
            teacher_utility[batch_idx][position]
            for batch_idx, positions in enumerate(acquisition.selected_positions)
            for position in positions
        ]
        losses["duca_train_only_teacher_utility_probe"] = sum(selected_teacher) * 0.0
    return {"acquisition": acquisition, "losses": losses, "metas": selected_metas}


def duca_forward_test(
    adapter: DucaAcquisitionAdapter,
    detector: DummyDetector,
    dense_observations: DenseObservations,
    valid_mask: Mask,
    metas: list[dict[str, Any]],
) -> dict[str, Any]:
    for meta in metas:
        forbidden = [key for key in ("gt_segments", "gt_labels", "teacher_utility", "ledger_path") if key in meta]
        if forbidden:
            raise ValueError(f"DUCA online test forbids decision-time payloads: {forbidden}")
    acquisition = adapter(dense_observations, valid_mask)
    selected_metas = _with_selected_metadata(metas, acquisition)
    detector_outputs = detector.forward_test(acquisition.observations, acquisition.mask, selected_metas)
    return {"acquisition": acquisition, "detector_outputs": detector_outputs, "metas": selected_metas}


def run_smoke(seed: int = 7, batch_size: int = 2, dense_len: int = 768, channels: int = 8, budget: int = 384) -> dict[str, Any]:
    rng = random.Random(int(seed))
    dense_observations = _dummy_dense_observations(
        rng=rng,
        batch_size=int(batch_size),
        dense_len=int(dense_len),
        channels=int(channels),
    )
    valid_mask = [[True] * int(dense_len) for _ in range(int(batch_size))]
    metas = [{"video_name": f"duca_online_smoke_{idx}"} for idx in range(int(batch_size))]
    teacher_utility = [[rng.uniform(-1.0, 1.0) for _ in range(int(dense_len))] for _ in range(int(batch_size))]

    adapter = DucaAcquisitionAdapter(ZeroShotActionnessSource(), budget=int(budget))
    detector = DummyDetector()

    train_result = duca_forward_train(
        adapter,
        detector,
        dense_observations,
        valid_mask,
        metas,
        teacher_utility=teacher_utility,
    )
    test_result = duca_forward_test(adapter, detector, dense_observations, valid_mask, metas)

    acquisition = test_result["acquisition"]
    selected_counts = acquisition.selected_count
    budget_violation_rate = sum(1 for count in selected_counts if count > int(budget)) / max(1, len(selected_counts))
    detector_input_length = int(test_result["detector_outputs"]["detector_input_length"])

    return {
        "route": "DUCA_online_temporal_acquisition_plugin_smoke",
        "selected_count": selected_counts[0] if len(set(selected_counts)) == 1 else selected_counts,
        "budget": int(budget),
        "detector_input_length": detector_input_length,
        "budget_violation_rate": float(budget_violation_rate),
        "uses_ledger_for_decision": False,
        "teacher_utility_used_train_only": "duca_train_only_teacher_utility_probe" in train_result["losses"],
        "train_forward": "duca_forward_train",
        "test_forward": "duca_forward_test",
        "selected_positions_contract": "original-time detector-consumed positions",
    }


def _dummy_dense_observations(
    *,
    rng: random.Random,
    batch_size: int,
    dense_len: int,
    channels: int,
) -> DenseObservations:
    return [
        [
            [
                math.sin((time_idx + 1) * (channel_idx + 1) * 0.013)
                + 0.05 * rng.uniform(-1.0, 1.0)
                + 0.10 * batch_idx
                for channel_idx in range(channels)
            ]
            for time_idx in range(dense_len)
        ]
        for batch_idx in range(batch_size)
    ]


def _common_detector_input_length(observations: DenseObservations) -> int:
    lengths = {len(sample) for sample in observations}
    if len(lengths) != 1:
        raise ValueError(f"dummy detector expects a common input length, got {sorted(lengths)}")
    return next(iter(lengths))


def _rms(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector) / max(1, len(vector)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU smoke for the DUCA online acquisition plugin contract.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dense-len", type=int, default=768)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--budget", type=int, default=384)
    args = parser.parse_args(argv)

    summary = run_smoke(
        seed=args.seed,
        batch_size=args.batch_size,
        dense_len=args.dense_len,
        channels=args.channels,
        budget=args.budget,
    )
    text = json.dumps(summary, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
