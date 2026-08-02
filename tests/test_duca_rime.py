from __future__ import annotations

import hashlib
import json

import pytest
import torch
from mmengine.config import ConfigDict

import opentad.cores.test_engine as test_engine
from opentad.datasets.base import SlidingWindowDataset
from opentad.models.duca.rime import (
    RimeBudgetController,
    RimeCostLedger,
    build_cost_matched_mixed_k_cycle,
    calibrate_rime_price,
    decode_rime_exact_k,
)
from opentad.models.duca.structured_selection import (
    physical_exact_k_forward_backward,
    physical_exact_k_select,
    physical_exact_k_viterbi,
)
from opentad.models.selectors.duca_rime_frame_selector import (
    DucaRimeFrameSelector,
)


def _physical_inputs(temporal_len: int = 9):
    scores = torch.linspace(-1.0, 1.0, temporal_len)[None, :]
    seconds = torch.arange(temporal_len, dtype=torch.float64)[None, :]
    valid = torch.ones((1, temporal_len), dtype=torch.bool)
    return scores, seconds, valid


def _active_positions(output) -> set[int]:
    return set(
        int(value)
        for value in output.hard_positions[0, output.hard_slot_mask[0]].tolist()
    )


def test_required_physical_positions_are_hard_and_soft_mandatory():
    scores, seconds, valid = _physical_inputs(6)
    scores = scores.clone()
    scores[0, 3] = -100.0
    required = torch.zeros_like(valid)
    required[0, 3] = True

    hard = physical_exact_k_viterbi(
        scores,
        seconds,
        valid,
        k=3,
        required_mask=required,
    )
    soft = physical_exact_k_forward_backward(
        scores,
        seconds,
        valid,
        k=3,
        required_mask=required,
    )
    joint = physical_exact_k_select(
        scores.requires_grad_(),
        seconds,
        valid,
        k=3,
        required_mask=required,
    )

    assert 3 in hard.hard_positions[0, hard.hard_slot_mask[0]].tolist()
    assert soft.soft_occupancy[0, 3].item() == pytest.approx(1.0, abs=2.0e-4)
    assert joint.hard_occupancy[0, 3].item() == 1.0
    joint.soft_occupancy.sum().backward()
    assert scores.grad is not None


def test_independent_constant_evidence_is_canonical_exact_uniform():
    _, seconds, valid = _physical_inputs(9)
    output = decode_rime_exact_k(
        torch.zeros((1, 9)),
        seconds,
        valid,
        4,
        candidate_budgets=(2, 4),
        decoder_family="independent",
    )

    assert output.hard_positions.tolist() == [[0, 3, 5, 8]]
    assert output.constant_uniform_identity.tolist() == [True]
    assert output.ledger.requested_k == (4,)
    assert output.ledger.unique_k == (4,)
    assert output.ledger.backbone_input_k == (4,)
    assert output.ledger.padded_k == (4,)
    assert output.ledger.dynamic_compute_realized is True


def test_nested_and_weak_overlap_decoder_families_obey_their_contracts():
    scores, seconds, valid = _physical_inputs(9)
    scores = torch.tensor([[0.0, 3.0, 2.0, 9.0, 8.0, 1.0, 7.0, 6.0, 0.0]])
    nested_small = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        2,
        candidate_budgets=(2, 4),
        decoder_family="strict_nested",
    )
    nested_large = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        4,
        candidate_budgets=(2, 4),
        decoder_family="strict_nested",
    )
    weak_small = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        2,
        candidate_budgets=(2, 4),
        decoder_family="weak_overlap",
        weak_overlap_fraction=0.5,
    )
    weak_large = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        4,
        candidate_budgets=(2, 4),
        decoder_family="weak_overlap",
        weak_overlap_fraction=0.5,
    )

    assert _active_positions(nested_small) <= _active_positions(nested_large)
    assert len(_active_positions(weak_small) & _active_positions(weak_large)) >= 1


def test_controller_is_batch_invariant_and_no_risk_has_vector_fallback():
    torch.manual_seed(7)
    controller = RimeBudgetController(
        evidence_dim=3,
        candidate_budgets=(2, 4, 6),
        candidate_costs=(2.0, 4.0, 6.0),
        use_risk=False,
        frozen_price=0.2,
    ).eval()
    evidence = torch.randn(2, 7, 3)
    scores = torch.randn(2, 7)
    valid = torch.tensor(
        [[True] * 7, [True, True, True, True, False, False, False]]
    )

    together = controller(evidence, scores, valid)
    separate = [
        controller(evidence[index : index + 1], scores[index : index + 1], valid[index : index + 1])
        for index in range(2)
    ]

    assert together.fallback_to_kmax.shape == (2,)
    assert not together.fallback_to_kmax.any()
    for index, decision in enumerate(separate):
        assert torch.allclose(
            together.predicted_utility[index],
            decision.predicted_utility[0],
            atol=1.0e-6,
        )
        assert together.requested_k[index].item() == decision.requested_k[0].item()


def test_floor_protocol_forces_exact_kmin_without_disabling_budget_losses(
    tmp_path,
):
    protocol = tmp_path / "floor_protocol.json"
    payload = {
        "schema_version": "duca_rime_budget_protocol_v1",
        "fit_split": "train_only",
        "uses_validation_or_test_labels": False,
        "candidate_budgets": [16, 32],
        "candidate_costs": [16.0, 32.0],
        "target_mean_cost": 16.0,
        "frozen_price": 0.0,
        "allocation_mode": "fixed_floor_budget_position_only",
        "forced_budget": 16,
        "risk_used_for_allocation": False,
        "dynamic_budget_claim_allowed": False,
        "risk_weight": 1.0,
        "risk_threshold": 0.0,
        "decoder_family": "weak_overlap",
        "weak_overlap_fraction": 0.5,
    }
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    selector = DucaRimeFrameSelector(
        in_channels=3,
        rime_arm="rime_full",
        candidate_budgets=(16, 32),
        fixed_budget=32,
        dense_window_size=32,
        execution_quantum=16,
        budget_protocol_path=str(protocol),
        budget_protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        detector_bridge_gradient_scale=1.0,
        actionness_source_cfg=dict(
            probe_model="official-action-seg",
            official_action_seg_backend="official_asformer",
            frozen=False,
            trainable=True,
        ),
    )
    evidence = torch.randn(2, 32, selector.coarse_hidden_dim)
    scores = torch.randn(2, 32)
    valid = torch.ones((2, 32), dtype=torch.bool)
    decision = selector.budget_controller(evidence, scores, valid)
    forced = selector._apply_protocol_allocation_mode(decision)
    assert forced.requested_k.tolist() == [16, 16]
    assert forced.fallback_to_kmax.tolist() == [False, False]
    assert forced.policy_name == "rime_fixed_floor_budget_position_only"
    assert forced.predicted_utility is decision.predicted_utility
    assert forced.predicted_risk is decision.predicted_risk


def test_price_calibration_meets_attainable_mean_cost_without_test_labels():
    utility = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [0.0, 0.8, 1.1],
            [0.0, 0.3, 0.4],
            [0.0, 2.0, 2.1],
        ]
    )
    risk = torch.zeros_like(utility)
    cost = torch.tensor([[1.0, 2.0, 3.0]]).expand_as(utility)
    result = calibrate_rime_price(
        utility,
        risk,
        cost,
        target_mean_cost=2.0,
        risk_weight=0.0,
        risk_threshold=1.0,
        use_risk=False,
    )

    assert result["frozen_price"] >= 0.0
    assert result["realized_mean_cost"] <= 2.0
    assert len(result["selected_indices"]) == 4


def test_cost_ledger_fails_closed_on_pad_to_kmax():
    ledger = RimeCostLedger(
        requested_k=(2,),
        effective_k=(2,),
        unique_k=(2,),
        backbone_input_k=(4,),
        padded_k=(4,),
        risk_fallback=(False,),
        dynamic_compute_realized=False,
    )

    with pytest.raises(ValueError, match="padded"):
        ledger.validate(require_no_padding=True)


def test_mixed_k_batch_must_be_bucketed_before_heavy_execution():
    scores = torch.zeros((2, 6))
    seconds = torch.arange(6, dtype=torch.float64)[None, :].expand(2, -1)
    valid = torch.ones((2, 6), dtype=torch.bool)

    with pytest.raises(ValueError, match="homogeneous effective-K bucket"):
        decode_rime_exact_k(
            scores,
            seconds,
            valid,
            torch.tensor([2, 4]),
            candidate_budgets=(2, 4),
        )


def test_tail_window_quantizes_effective_k_without_padding_or_duplicates():
    temporal_len = 205
    scores = torch.zeros((1, temporal_len))
    seconds = torch.arange(temporal_len, dtype=torch.float64)[None, :]
    valid = torch.ones((1, temporal_len), dtype=torch.bool)

    output = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        256,
        candidate_budgets=(192, 256),
        force_uniform=True,
        execution_quantum=16,
    )

    assert output.requested_k.tolist() == [256]
    assert output.effective_k.tolist() == [192]
    assert output.hard_positions.shape == (1, 192)
    assert len(set(output.hard_positions[0].tolist())) == 192
    assert output.ledger.requested_k == (256,)
    assert output.ledger.effective_k == (192,)
    assert output.ledger.backbone_input_k == (192,)
    assert output.ledger.padded_k == (192,)


@pytest.mark.parametrize(
    ("valid_length", "expected"),
    (
        (16, (16, 16, 16, 16)),
        (17, (16, 16, 16, 16)),
        (223, (192, 208, 208, 208)),
        (224, (192, 224, 224, 224)),
        (225, (192, 224, 224, 224)),
        (231, (192, 224, 224, 224)),
        (239, (192, 224, 224, 224)),
        (240, (192, 240, 240, 240)),
    ),
)
def test_mixed_k_natural_window_requested_to_effective_boundaries(
    valid_length,
    expected,
):
    budgets = (192, 256, 384, 512)
    scores = torch.zeros((1, valid_length))
    seconds = torch.arange(valid_length, dtype=torch.float64)[None, :]
    valid = torch.ones((1, valid_length), dtype=torch.bool)
    observed = []
    for requested in budgets:
        output = decode_rime_exact_k(
            scores,
            seconds,
            valid,
            requested,
            candidate_budgets=budgets,
            force_uniform=True,
            execution_quantum=16,
        )
        observed.append(int(output.effective_k.item()))
        positions = output.hard_positions[0].tolist()
        assert positions == sorted(set(positions))
        assert all(0 <= value < valid_length for value in positions)
        assert output.ledger.backbone_input_k == (int(output.effective_k.item()),)
        assert output.ledger.padded_k == (int(output.effective_k.item()),)
    assert tuple(observed) == expected


def test_mixed_k_subquantum_natural_window_fails_closed():
    valid_length = 15
    with pytest.raises(ValueError, match="shorter than one heavy-backbone execution quantum"):
        decode_rime_exact_k(
            torch.zeros((1, valid_length)),
            torch.arange(valid_length, dtype=torch.float64)[None, :],
            torch.ones((1, valid_length), dtype=torch.bool),
            192,
            candidate_budgets=(192, 256, 384, 512),
            force_uniform=True,
            execution_quantum=16,
        )


def test_cost_matched_mixed_k_cycle_is_exact_and_deterministic():
    budgets = (192, 256, 384, 512)
    counts = (8, 12, 16, 24)
    first = build_cost_matched_mixed_k_cycle(
        budgets,
        counts,
        target_mean_cost=384.0,
        schedule_seed=3407,
    )
    second = build_cost_matched_mixed_k_cycle(
        budgets,
        counts,
        target_mean_cost=384.0,
        schedule_seed=3407,
    )
    assert first == second
    assert len(first) == 60
    assert tuple(first.count(value) for value in budgets) == counts
    assert sum(first) / len(first) == 384.0

    with pytest.raises(ValueError, match="mean cost"):
        build_cost_matched_mixed_k_cycle(
            budgets,
            (15, 15, 15, 15),
            target_mean_cost=384.0,
            schedule_seed=3407,
        )


def test_uniform_mixed_k_selector_uses_exact_per_video_stateless_schedule():
    selector = DucaRimeFrameSelector(
        in_channels=3,
        rime_arm="uniform_mixed_k",
        candidate_budgets=(192, 256, 384, 512),
        fixed_budget=256,
        dense_window_size=768,
        target_mean_cost=384.0,
        execution_quantum=16,
        require_frozen_protocol=False,
        mixed_k_schedule_counts=(8, 12, 16, 24),
        mixed_k_schedule_seed=3407,
        detector_bridge_gradient_scale=0.0,
        actionness_source_cfg=None,
    )
    scores = torch.zeros((1, 768))
    valid = torch.ones((1, 768), dtype=torch.bool)
    requested = []
    per_video = {sample_index: [] for sample_index in range(100)}
    selector.train()
    for epoch in range(60):
        for sample_index in range(100):
            value = selector._fixed_requested_k(
                scores,
                valid,
                [
                    {
                        "duca_stateless_epoch": epoch,
                        "duca_stateless_sample_index": sample_index,
                    }
                ],
                training=True,
            )
            requested_k = int(value.item())
            requested.append(requested_k)
            per_video[sample_index].append(requested_k)
            assert selector.after_optimizer_step()["updated"] is True
    assert tuple(
        requested.count(value) for value in selector.candidate_budgets
    ) == (
        800,
        1200,
        1600,
        2400,
    )
    assert sum(requested) / len(requested) == 384.0
    for values in per_video.values():
        assert tuple(
            values.count(value) for value in selector.candidate_budgets
        ) == (8, 12, 16, 24)
        assert sum(values) / len(values) == 384.0
    assert int(selector._loss_weight_schedule_step.item()) == 6000
    assert selector.raw_actionness_source is None
    assert selector._fixed_requested_k(
        scores,
        valid,
        [{}],
        training=False,
    ).tolist() == [256]
    short_valid = torch.cat(
        [
            torch.ones((1, 231), dtype=torch.bool),
            torch.zeros((1, 537), dtype=torch.bool),
        ],
        dim=1,
    )
    assert selector._fixed_requested_k(
        scores,
        short_valid,
        [{}],
        training=False,
    ).tolist() == [256]
    selector.eval()
    short = selector.forward_test(
        torch.arange(768, dtype=torch.float32)[None, None].expand(1, 3, -1),
        short_valid,
        [
            {
                "video_name": "natural_short",
                "window_start_frame": 0,
                "frame_inds": torch.arange(768)[:, None],
                "avg_fps": 1.0,
            }
        ],
    )
    assert short["selector_outputs"]["requested_k"].tolist() == [256]
    assert short["selector_outputs"]["selected_count"].tolist() == [224]
    assert short["inputs"].shape[2] == 224
    assert (
        short["metas"][0]["duca_effective_k"]
        == short["metas"][0]["duca_unique_k"]
        == short["metas"][0]["duca_backbone_input_k"]
        == short["metas"][0]["duca_padded_k"]
        == 224
    )

    with pytest.raises(ValueError, match="stateless epoch and sample index"):
        selector._fixed_requested_k(
            scores,
            valid,
            [{}],
            training=True,
        )


def _stage1_protocol(tmp_path):
    protocol = tmp_path / "stage1_source_protocol.json"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "duca_rime_budget_protocol_v1",
        "fit_split": "train_only",
        "uses_validation_or_test_labels": False,
        "candidate_budgets": [16, 32],
        "candidate_costs": [16.0, 32.0],
        "target_mean_cost": 24.0,
        "frozen_price": 0.1,
        "allocation_mode": "frozen_price_dynamic_budget",
        "forced_budget": None,
        "risk_used_for_allocation": True,
        "dynamic_budget_claim_allowed": True,
        "risk_weight": 1.0,
        "risk_threshold": 0.35,
        "decoder_family": "independent",
        "weak_overlap_fraction": 0.5,
    }
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    return protocol, hashlib.sha256(protocol.read_bytes()).hexdigest()


def _stage1_selector(tmp_path, arm):
    protocol, protocol_sha = _stage1_protocol(tmp_path)
    selector = DucaRimeFrameSelector(
        in_channels=3,
        rime_arm=arm,
        candidate_budgets=(16, 32),
        fixed_budget=32,
        dense_window_size=32,
        execution_quantum=16,
        budget_protocol_path=str(protocol),
        budget_protocol_sha256=protocol_sha,
        require_frozen_protocol=True,
        allow_oracle_replay=arm.startswith("hrime_stage1_"),
        detector_bridge_gradient_scale=1.0,
        actionness_source_cfg=dict(
            probe_model="official-action-seg",
            official_action_seg_backend="official_asformer",
            frozen=False,
            trainable=True,
        ),
    )
    return selector, protocol_sha


@pytest.mark.parametrize(
    "stage1_arm",
    (
        "hrime_stage1_learned_positions",
        "hrime_stage1_uniform_positions",
    ),
)
def test_stage1_replay_selectors_are_strict_checkpoint_architecture_equivalent(
    tmp_path,
    stage1_arm,
):
    source, _ = _stage1_selector(tmp_path / "source", "rime_full")
    replay, _ = _stage1_selector(tmp_path / stage1_arm, stage1_arm)
    source_state = source.state_dict()
    replay_state = replay.state_dict()
    assert tuple(source_state) == tuple(replay_state)
    assert {
        key: tuple(value.shape) for key, value in source_state.items()
    } == {
        key: tuple(value.shape) for key, value in replay_state.items()
    }
    incompatible = replay.load_state_dict(source_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert replay.raw_actionness_source is not None
    assert replay.transition_scorer is not None
    assert replay.budget_controller is not None


def test_stage1_uniform_replay_skips_policy_but_seals_short_window_budget_truth(
    tmp_path,
    monkeypatch,
):
    selector, protocol_sha = _stage1_selector(
        tmp_path / "selector",
        "hrime_stage1_uniform_positions",
    )
    ledger_root = tmp_path / "ledger"
    monkeypatch.setenv("DUCA_RIME_INFERENCE_LEDGER_ROOT", str(ledger_root))
    selector.raw_actionness_source.forward = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("uniform Stage-1 replay must not execute the coarse policy")
    )
    provenance = {
        "role": "hrime_stage1_uniform_same_total",
        "strategy": "uniform_same_total",
        "oracle_only": True,
        "deployment_candidate": False,
        "uses_official_final": False,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "uses_test_batch_composition": False,
        "position_policy": "exact_uniform",
        "assignment_sha256": "a" * 64,
    }
    meta = {
        "video_name": "video",
        "window_start_frame": 0,
        "frame_inds": list(range(32)),
        "avg_fps": 1.0,
        "rime_requested_k_replay": 32,
        "rime_effective_k_replay": 16,
        "rime_requested_k_replay_provenance": provenance,
    }
    masks = torch.tensor([[True] * 23 + [False] * 9])
    selector.eval()
    output = selector.forward_test(
        torch.randn(1, 3, 32),
        masks,
        metas=[meta],
    )
    assert output["masks"].shape == (1, 16)
    assert output["selector_outputs"]["requested_k"].tolist() == [32]
    assert output["selector_outputs"]["selected_count"].tolist() == [16]
    selector.record_backbone_execution(
        {
            "schema_version": "duca_dynamic_backbone_input_v1",
            "measurement_source": (
                "actual_backbone_wrapper_and_videomae_input_tensors"
            ),
            "wrapper_temporal_k": 16,
            "mask_temporal_k": 16,
            "inner_reconstructed_k": 16,
            "inner_temporal_chunk_k": 16,
            "num_segs": 1,
            "all_mask_active": True,
            "padding_or_repetition_observed": False,
        }
    )
    row = json.loads(
        (ledger_root / "inference_ledger.rank0000.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert row["budget_protocol_sha256"] == protocol_sha
    assert row["raw_budget"] == 32
    assert row["reachable_budget"] == 16
    assert row["realized_budget"] == 16
    assert row["projection_unused_budget"] == 16
    assert row["solver_unused_budget"] == 0
    assert row["budget_scope"] == "video_exact_total_window_assignment"
    assert row["backbone_input_k"] == 16
    assert row["backbone_input_measurement_source"] == (
        "actual_backbone_wrapper_and_videomae_input_tensors"
    )


def test_stage1_runtime_receipt_proves_full_window_merge_nms_evaluator_chain(
    tmp_path,
    monkeypatch,
):
    class FakeSlidingDataset(SlidingWindowDataset):
        def __init__(self):
            self.class_map = ["action"]

        def __len__(self):
            return 3

    class FakeLoader:
        dataset = FakeSlidingDataset()

        def __iter__(self):
            yield {
                "inputs": torch.zeros((2, 1)),
                "masks": torch.ones((2, 1), dtype=torch.bool),
                "metas": [
                    {"video_name": "video_a"},
                    {"video_name": "video_a"},
                ],
            }
            yield {
                "inputs": torch.zeros((1, 1)),
                "masks": torch.ones((1, 1), dtype=torch.bool),
                "metas": [{"video_name": "video_b"}],
            }

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **data):
            result = {}
            for meta in data["metas"]:
                result.setdefault(meta["video_name"], []).append(
                    {
                        "segment": [0.0, 1.0],
                        "label": "action",
                        "score": 0.5,
                    }
                )
            return result

    class FakeEvaluator:
        def evaluate(self):
            return {"average_mAP": 0.0}

        def logging(self, logger):
            logger.info("fake evaluator")

    class FakeLogger:
        def info(self, *_args, **_kwargs):
            return None

    def fake_all_gather_object(output, value):
        output[0] = value

    def fake_batched_nms(segments, scores, labels, **_kwargs):
        return segments, scores, labels

    monkeypatch.setattr(test_engine.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(test_engine, "batched_nms", fake_batched_nms)
    monkeypatch.setattr(
        test_engine,
        "build_evaluator",
        lambda _config: FakeEvaluator(),
    )
    cfg = ConfigDict(
        inference=ConfigDict(save_raw_prediction=False),
        post_processing=ConfigDict(
            nms=ConfigDict(
                use_soft_nms=True,
                sigma=0.7,
                max_seg_num=2000,
                multiclass=True,
                voting_thresh=0.7,
            ),
            save_dict=True,
        ),
        evaluation=ConfigDict(
            type="mAP",
            subset="training",
            ground_truth_filename=str(tmp_path / "annotation.json"),
            tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
        ),
        work_dir=str(tmp_path),
    )
    summary = test_engine.eval_one_epoch(
        FakeLoader(),
        FakeModel(),
        cfg,
        FakeLogger(),
        rank=0,
        world_size=0,
        not_eval=False,
    )
    receipt = summary["post_processing_execution"]
    assert receipt["window_counts"] == {"video_a": 2, "video_b": 1}
    assert receipt["pre_nms_result_count"] == 3
    assert receipt["post_nms_result_count"] == 3
    assert receipt["nms_call_count"] == 2
    assert receipt["result_sha256"] == hashlib.sha256(
        (tmp_path / "result_detection.json").read_bytes()
    ).hexdigest()
    assert receipt["pipeline_events"] == [
        "model_forward_loop_complete",
        "ddp_result_gather_complete",
        "cross_window_result_aggregation_complete",
        "sliding_window_nms_complete",
        "post_nms_prediction_saved",
        "official_evaluator_evaluate_called",
        "official_evaluator_evaluate_returned",
    ]
    assert receipt["content_sha256"] == test_engine._canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "content_sha256"
        }
    )
    assert receipt["full_detector_window_merge_nms_evaluation_completed"] is True
