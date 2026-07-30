from __future__ import annotations

import hashlib

import pytest
import torch

from tools.bata import georoute_gradient_decomposition as gradient


def test_seed_is_mechanically_derived_and_disjoint() -> None:
    digest = hashlib.sha256(gradient.STUDY_ID.encode("utf-8")).hexdigest()
    expected = 1000 + (int(digest[:8], 16) % 9000)
    assert gradient.SEED == expected == 7367
    assert gradient.SEED not in gradient.FORBIDDEN_SEEDS


def _differentiable_ordered_log_probability(
    logits: torch.Tensor,
    order: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    available = valid_mask.clone()
    joint = torch.zeros(logits.shape[:2], dtype=logits.dtype)
    for slot in range(order.shape[-1]):
        choice = order[..., slot]
        masked = (logits / temperature).masked_fill(~available, float("-inf"))
        joint = joint + torch.log_softmax(masked, dim=-1).gather(
            -1, choice.unsqueeze(-1)
        ).squeeze(-1)
        available.scatter_(-1, choice.unsqueeze(-1), False)
    return joint


def test_analytic_pl_score_matches_autograd_scaled_gradient() -> None:
    logits = torch.tensor(
        [[[0.2, -0.1, 0.7, 0.0], [0.5, -0.4, 0.1, 0.3]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    order = torch.tensor([[[2, 0], [0, 3]]], dtype=torch.long)
    valid = torch.ones_like(logits, dtype=torch.bool)
    temperature = 0.7
    analytic_logp, score, _ = gradient.ordered_pl_log_prob_and_score(
        logits=logits.detach(),
        ordered_indices=order,
        valid_mask=valid,
        temperature=temperature,
    )
    production_logp = _differentiable_ordered_log_probability(
        logits, order, valid, temperature
    )
    assert torch.allclose(analytic_logp, production_logp.detach())

    advantage = torch.tensor(3.25)
    loss_scale = 4096.0
    loss = advantage * production_logp.mean()
    (loss * loss_scale).backward()
    expected = gradient.expected_scaled_logit_gradient(
        score=score,
        advantage=advantage,
        weight=1.0,
        temporal_reduction="mean",
        loss_scale=loss_scale,
    )
    assert torch.allclose(logits.grad, expected, rtol=1e-5, atol=1e-5)
    dot = torch.sum(logits.grad * expected)
    assert float(dot.item()) > 0.0


def test_fp16_shadow_detects_cast_before_divide_overflow() -> None:
    telemetry = gradient.bucket_cast_telemetry(
        bucket_buffer=torch.tensor([70000.0, -12.0], dtype=torch.float32),
        loss_scale=65536.0,
        world_size=1,
    )
    assert telemetry["fp32_pre_hook"]["finite"] is True
    assert telemetry["hypothetical_unscaled"]["finite"] is True
    assert telemetry["fp16_shadow_cast_then_divide"]["finite"] is False
    assert telemetry["cast_introduced_nonfinite"] is True
    with pytest.raises(ValueError, match="one DDP process"):
        gradient.bucket_cast_telemetry(
            bucket_buffer=torch.ones(2),
            loss_scale=1.0,
            world_size=2,
        )


def _synthetic_receipt(
    *,
    arm: str,
    cuda_sequence: list[str],
    pl_failure: bool,
) -> dict:
    data = [f"data-{index}" for index in range(gradient.MAX_BATCHES)]
    cpu = [f"cpu-{index}" for index in range(gradient.MAX_BATCHES)]
    batches = []
    for index in range(gradient.MAX_BATCHES):
        failed = arm == gradient.ARMS[0] and pl_failure and index == 11
        batches.append(
            {
                "iter_idx": index,
                "scaler_result": {
                    "update_succeeded": not failed,
                    "failure_classification": (
                        {
                            "mechanism_class": "DDP_FP16_CAST_OVERFLOW",
                            "detector_only": False,
                        }
                        if failed
                        else None
                    ),
                },
            }
        )
    return {
        "status": gradient.PASS_STATUS,
        "arm": arm,
        "binding": {
            "source_config_sha256": "source",
            "manifest_file_sha256": "manifest",
            "development_annotation": {"sha256": "annotation"},
            "class_map_sha256": "class",
            "development_video_root": "/development",
            "pretrained_checkpoint_sha256": "pretrained",
            "official_reference_config_sha256": "official",
            "parent_evidence_sha256": "parent",
        },
        "summary": {
            "data_fingerprint_sha256_by_batch": data,
            "cpu_rng_sha256_by_batch": cpu,
            "cuda_rng_sha256_by_batch": cuda_sequence,
            "all_forward_losses_finite": True,
            "complete_bucket_telemetry": True,
            "all_pl_directions_positive": (
                True if arm == gradient.ARMS[0] else None
            ),
        },
        "batches": batches,
    }


def test_pair_classifier_allows_expected_post_batch_zero_cuda_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gradient,
        "validate_receipt",
        lambda payload, expected_arm=None, **_kwargs: payload,
    )
    initial = "cuda-initial"
    receipts = {
        gradient.ARMS[0]: _synthetic_receipt(
            arm=gradient.ARMS[0],
            cuda_sequence=[initial]
            + [f"pl-{index}" for index in range(1, gradient.MAX_BATCHES)],
            pl_failure=True,
        ),
        gradient.ARMS[1]: _synthetic_receipt(
            arm=gradient.ARMS[1],
            cuda_sequence=[initial]
            + [f"st-{index}" for index in range(1, gradient.MAX_BATCHES)],
            pl_failure=False,
        ),
    }
    classification = gradient.classify_pair(receipts)
    assert classification["decision"] == gradient.DECISION_REPAIR
    assert classification["repair_class"] == "DDP_FP16_CAST_OVERFLOW"
    assert classification["initial_cuda_rng_matched"] is True
    assert classification["full_cuda_rng_sequence_matched"] is False
    assert (
        classification["post_batch_zero_cuda_rng_divergence_expected"] is True
    )


def test_pair_classifier_holds_without_a_pl_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gradient,
        "validate_receipt",
        lambda payload, expected_arm=None, **_kwargs: payload,
    )
    cuda = [f"cuda-{index}" for index in range(gradient.MAX_BATCHES)]
    receipts = {
        arm: _synthetic_receipt(
            arm=arm,
            cuda_sequence=cuda,
            pl_failure=False,
        )
        for arm in gradient.ARMS
    }
    classification = gradient.classify_pair(receipts)
    assert classification["decision"] == gradient.DECISION_HOLD
    assert classification["repair_authorized"] is False


def test_observed_hook_records_before_calling_authoritative_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        gradient.observed_fp16_compress_hook.__annotations__["bucket"]
        is torch.distributed.GradBucket
    )
    order: list[str] = []

    class Observer:
        def record_ddp_bucket(self, **_kwargs) -> None:
            order.append("observe")

    class Bucket:
        def parameters(self):
            return []

        def index(self):
            return 0

        def buffer(self):
            return torch.ones(2)

    sentinel = object()

    def authoritative(_state, _bucket):
        order.append("authoritative")
        return sentinel

    monkeypatch.setattr(
        gradient.comm_hooks, "fp16_compress_hook", authoritative
    )
    state = gradient.ObservedFp16HookState(
        observer=Observer(),
        parameter_names={},
    )
    assert gradient.observed_fp16_compress_hook(state, Bucket()) is sentinel
    assert order == ["observe", "authoritative"]
